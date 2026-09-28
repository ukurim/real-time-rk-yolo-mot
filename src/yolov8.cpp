#include "aerial/yolov8.hpp"
#include "aerial/yolo_decode.hpp"
#include "aerial/rknn_input.hpp"
#include "rknn_api.h"

#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <mutex>
#include <stdexcept>

namespace aerial {
namespace {
using Clock = std::chrono::steady_clock;
double milliseconds(Clock::time_point begin, Clock::time_point end) {
    return std::chrono::duration<double,std::milli>(end-begin).count();
}
void check(int status, const char* operation) {
    if (status != RKNN_SUCC)
        throw std::runtime_error(std::string(operation)+" failed, RKNN status="+std::to_string(status));
}
struct Context {
    rknn_context value = 0;
    ~Context() { if (value) rknn_destroy(value); }
};
struct OutputLease {
    rknn_context context;
    std::vector<rknn_output>& outputs;
    bool acquired = false;
    ~OutputLease() {
        bool allocated = acquired;
        for (const auto& output : outputs) allocated = allocated || output.buf != nullptr;
        if (allocated) rknn_outputs_release(context,static_cast<uint32_t>(outputs.size()),outputs.data());
    }
};

YoloTensor tensor_metadata(const rknn_tensor_attr& attr) {
    if (attr.n_dims != 4 || attr.dims[0] != 1 || attr.fmt != RKNN_TENSOR_NCHW)
        throw std::invalid_argument("YOLO output "+std::to_string(attr.index)+" must be batch=1, four-dimensional NCHW");
    for (unsigned i = 1; i < 4; ++i)
        if (attr.dims[i] == 0 || attr.dims[i] > 65536)
            throw std::invalid_argument("Invalid YOLO tensor dimension");
    YoloTensor result;
    result.channels = static_cast<int>(attr.dims[1]);
    result.height = static_cast<int>(attr.dims[2]);
    result.width = static_cast<int>(attr.dims[3]);
    const auto expected_elements = static_cast<std::uint64_t>(result.channels)*result.height*result.width;
    if (attr.n_elems != expected_elements) throw std::invalid_argument("Inconsistent YOLO output element count");
    if (attr.type == RKNN_TENSOR_INT8 || attr.type == RKNN_TENSOR_UINT8) {
        if (attr.qnt_type != RKNN_TENSOR_QNT_AFFINE_ASYMMETRIC)
            throw std::invalid_argument("YOLO integer outputs require affine quantization");
        result.element = attr.type == RKNN_TENSOR_INT8 ? TensorElement::Int8 : TensorElement::UInt8;
        result.zero_point = attr.zp;
        result.scale = attr.scale;
    } else if (attr.type == RKNN_TENSOR_FLOAT16 || attr.type == RKNN_TENSOR_FLOAT32) {
        // RKNN 2.3.2 reports FLOAT16 outputs of the unquantized official YOLOv8n
        // as AFFINE(zp=0,scale=1). This is identity metadata, not an integer
        // buffer. Request FP32 from the runtime; never apply affine arithmetic
        // to the half-float bit pattern. Reject other float/qnt combinations.
        const bool identity_affine = attr.qnt_type == RKNN_TENSOR_QNT_AFFINE_ASYMMETRIC &&
                                     attr.zp == 0 && attr.scale == 1.f;
        if (attr.qnt_type != RKNN_TENSOR_QNT_NONE && !identity_affine)
            throw std::invalid_argument("YOLO float output "+std::to_string(attr.index)+
                                        " requires NONE or identity AFFINE quantization metadata");
        result.element = TensorElement::Float32; // rknn_outputs_get(want_float=1).
    } else throw std::invalid_argument("Unsupported YOLO output data type");
    return result;
}
} // namespace

struct YoloV8Detector::Impl {
    Context context;
    DetectorConfig config;
    cv::Size size;
    int classes = 0;
    std::vector<YoloTensor> tensors;
    std::vector<rknn_output> outputs;
    cv::Mat rgb, resized;
    std::unique_ptr<Fp16RgbInput> fp16_input;
    std::mutex mutex;

    explicit Impl(const DetectorConfig& requested) : config(requested) {
        if (config.model_path.empty()) throw std::invalid_argument("YOLOv8 RKNN model path is required");
        if (config.preprocess != "cpu")
            throw std::invalid_argument("This version supports preprocess=cpu; RGA letterbox has not been validated");
        if (config.input_mode != "uint8" && config.input_mode != "fp16_normalized")
            throw std::invalid_argument("input_mode must be uint8 or fp16_normalized");
        if (config.core_mask != 0 && config.core_mask != 1 && config.core_mask != 2 &&
            config.core_mask != 3 && config.core_mask != 4 && config.core_mask != 7)
            throw std::invalid_argument("Supported RKNN core masks: 0,1,2,3,4,7");
        if (!std::isfinite(config.confidence_threshold) || config.confidence_threshold < 0 || config.confidence_threshold > 1 ||
            !std::isfinite(config.nms_threshold) || config.nms_threshold < 0 || config.nms_threshold > 1 || config.max_detections <= 0)
            throw std::invalid_argument("Invalid detector thresholds/max_detections");
        std::ifstream file(config.model_path,std::ios::binary|std::ios::ate);
        if (!file) throw std::runtime_error("Cannot open YOLOv8 RKNN model: "+config.model_path);
        const auto length = file.tellg();
        if (length <= 0 || static_cast<std::uint64_t>(length) > std::numeric_limits<uint32_t>::max())
            throw std::runtime_error("Invalid RKNN model length");
        std::vector<char> model(static_cast<std::size_t>(length));
        file.seekg(0);
        if (!file.read(model.data(),static_cast<std::streamsize>(model.size())))
            throw std::runtime_error("Cannot read complete RKNN model");
        check(rknn_init(&context.value,model.data(),static_cast<uint32_t>(model.size()),0,nullptr),"rknn_init");
        check(rknn_set_core_mask(context.value,static_cast<rknn_core_mask>(config.core_mask)),"rknn_set_core_mask");
        rknn_input_output_num io{};
        check(rknn_query(context.value,RKNN_QUERY_IN_OUT_NUM,&io,sizeof(io)),"RKNN_QUERY_IN_OUT_NUM");
        if (io.n_input != 1 || (io.n_output != 6 && io.n_output != 9))
            throw std::invalid_argument("Model must have one RGB input and Rockchip YOLOv8 split 6/9 outputs");
        rknn_tensor_attr input{};
        input.index = 0;
        check(rknn_query(context.value,RKNN_QUERY_INPUT_ATTR,&input,sizeof(input)),"RKNN_QUERY_INPUT_ATTR");
        if (input.n_dims != 4 || input.dims[0] != 1 ||
            (input.fmt != RKNN_TENSOR_NCHW && input.fmt != RKNN_TENSOR_NHWC))
            throw std::invalid_argument("YOLO requires a static batch=1 NCHW or NHWC input");
        for (unsigned i = 1; i < 4; ++i)
            if (input.dims[i] == 0 || input.dims[i] > 8192) throw std::invalid_argument("Invalid YOLO input dimension");
        const bool nchw = input.fmt == RKNN_TENSOR_NCHW;
        if (input.dims[nchw ? 1 : 3] != 3) throw std::invalid_argument("YOLO input must have three RGB channels");
        size = {static_cast<int>(input.dims[nchw ? 3 : 2]),static_cast<int>(input.dims[nchw ? 2 : 1])};
        if (config.input_mode == "fp16_normalized") {
            rknn_tensor_attr native{};
            check(rknn_query(context.value,RKNN_QUERY_NATIVE_INPUT_ATTR,&native,sizeof(native)),"RKNN_QUERY_NATIVE_INPUT_ATTR");
            fp16_input.reset(new Fp16RgbInput(native));
            if (native.dims[2] != static_cast<uint32_t>(size.width) || native.dims[1] != static_cast<uint32_t>(size.height))
                throw std::invalid_argument("Native FP16 shape differs from logical model input");
        }
        tensors.reserve(io.n_output);
        outputs.resize(io.n_output);
        for (uint32_t i = 0; i < io.n_output; ++i) {
            rknn_tensor_attr attr{};
            attr.index = i;
            check(rknn_query(context.value,RKNN_QUERY_OUTPUT_ATTR,&attr,sizeof(attr)),"RKNN_QUERY_OUTPUT_ATTR");
            tensors.push_back(tensor_metadata(attr));
        }
        classes = validate_yolov8_outputs(tensors,size,false);
        std::cerr << "YOLOv8 split model: " << size.width << 'x' << size.height
                  << ", classes=" << classes << ", outputs=" << tensors.size()
                  << ", core_mask=" << config.core_mask << ", CPU RGB letterbox; input_mode=" << config.input_mode
                  << "; regular RKNN input API\n";
        if (fp16_input)
            std::cerr << "FP16 pass-through explicitly uses RGB/255; requires conversion manifest mean=0,std=255\n";
    }
};

YoloV8Detector::YoloV8Detector(const DetectorConfig& config) : impl_(new Impl(config)) {}
YoloV8Detector::~YoloV8Detector() = default;
cv::Size YoloV8Detector::input_size() const { return impl_->size; }
int YoloV8Detector::class_count() const { return impl_->classes; }

std::vector<Detection> YoloV8Detector::detect(const cv::Mat& bgr, StageTiming* timing) {
    auto& state = *impl_;
    std::lock_guard<std::mutex> lock(state.mutex);
    if (timing) *timing = {};
    const auto start = Clock::now();
    const auto transform = letterbox_rgb(bgr,state.size,state.rgb,state.resized);
    rknn_input input{};
    input.index = 0;
    input.type = RKNN_TENSOR_UINT8;
    input.fmt = RKNN_TENSOR_NHWC;
    input.size = static_cast<uint32_t>(state.rgb.total()*state.rgb.elemSize());
    input.buf = state.rgb.data;
    input.pass_through = 0; // Runtime performs model conversion/embedded /255 normalization.
    if (state.fp16_input) input = state.fp16_input->prepare(state.rgb);
    // Manual FP16 normalization belongs to preprocessing, not to an excluded
    // timing interval. inputs_set remains separately measured below.
    const auto preprocessed = Clock::now();
    check(rknn_inputs_set(state.context.value,1,&input),"rknn_inputs_set");
    const auto submitted = Clock::now();
    check(rknn_run(state.context.value,nullptr),"rknn_run");
    const auto executed = Clock::now();
    for (std::size_t i = 0; i < state.outputs.size(); ++i) {
        state.outputs[i] = {};
        state.outputs[i].index = static_cast<uint32_t>(i);
        state.outputs[i].want_float = state.tensors[i].element == TensorElement::Float32;
    }
    OutputLease lease{state.context.value,state.outputs};
    check(rknn_outputs_get(state.context.value,static_cast<uint32_t>(state.outputs.size()),state.outputs.data(),nullptr),"rknn_outputs_get");
    lease.acquired = true;
    const auto inferred = Clock::now();
    for (std::size_t i = 0; i < state.outputs.size(); ++i) {
        state.tensors[i].data = state.outputs[i].buf;
        state.tensors[i].bytes = state.outputs[i].size;
    }
    const YoloDecodeConfig decode{state.config.confidence_threshold,state.config.nms_threshold,state.config.max_detections};
    auto detections = decode_yolov8(state.tensors,transform,decode);
    const auto done = Clock::now();
    if (timing) {
        timing->preprocess_ms = milliseconds(start,preprocessed);
        timing->inference_ms = milliseconds(preprocessed,inferred);
        timing->input_submit_ms = milliseconds(preprocessed,submitted);
        timing->rknn_run_ms = milliseconds(submitted,executed);
        timing->output_get_ms = milliseconds(executed,inferred);
        timing->postprocess_ms = milliseconds(inferred,done);
    }
    return detections;
}

} // namespace aerial
