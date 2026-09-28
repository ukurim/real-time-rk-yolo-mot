#include "aerial/yolo_decode.hpp"

#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

namespace aerial {
namespace {
float read_element(const YoloTensor& t, std::size_t i) {
    switch (t.element) {
    case TensorElement::Float32: return static_cast<const float*>(t.data)[i];
    case TensorElement::Int8:
        return (static_cast<const std::int8_t*>(t.data)[i] - t.zero_point) * t.scale;
    case TensorElement::UInt8:
        return (static_cast<const std::uint8_t*>(t.data)[i] - t.zero_point) * t.scale;
    }
    throw std::invalid_argument("Unsupported YOLO tensor element type");
}

float intersection_over_union(const cv::Rect2f& a, const cv::Rect2f& b) {
    const float w = std::max(0.f, std::min(a.x+a.width, b.x+b.width)-std::max(a.x,b.x));
    const float h = std::max(0.f, std::min(a.y+a.height, b.y+b.height)-std::max(a.y,b.y));
    const float intersection = w*h;
    const float united = a.area()+b.area()-intersection;
    return united > 0 ? intersection/united : 0;
}

float dfl_distance(const YoloTensor& t, std::size_t cell, int side, int bins) {
    const std::size_t plane = static_cast<std::size_t>(t.width)*t.height;
    std::array<float, 64> logits{};
    float maximum = -std::numeric_limits<float>::infinity();
    for (int b = 0; b < bins; ++b) {
        logits[b] = read_element(t, (side*bins+b)*plane+cell);
        if (!std::isfinite(logits[b])) throw std::runtime_error("Non-finite YOLO DFL output");
        maximum = std::max(maximum, logits[b]);
    }
    // Subtract the maximum to avoid exp overflow on floating point models.
    float sum = 0, weighted = 0;
    for (int b = 0; b < bins; ++b) {
        const float probability = std::exp(logits[b]-maximum);
        sum += probability;
        weighted += b*probability;
    }
    return weighted/sum;
}
} // namespace

LetterboxTransform letterbox_transform(cv::Size original, cv::Size input) {
    if (original.width <= 0 || original.height <= 0 || input.width <= 0 || input.height <= 0)
        throw std::invalid_argument("Letterbox dimensions must be positive");
    const double scale = std::min(static_cast<double>(input.width)/original.width,
                                  static_cast<double>(input.height)/original.height);
    LetterboxTransform t;
    t.original_size = original;
    t.input_size = input;
    t.resized_size = {std::max(1, std::min(input.width, static_cast<int>(std::round(original.width*scale)))),
                      std::max(1, std::min(input.height, static_cast<int>(std::round(original.height*scale))))};
    t.left = (input.width-t.resized_size.width)/2;
    t.top = (input.height-t.resized_size.height)/2;
    t.scale_x = static_cast<float>(t.resized_size.width)/original.width;
    t.scale_y = static_cast<float>(t.resized_size.height)/original.height;
    return t;
}

cv::Rect2f LetterboxTransform::to_original(const cv::Rect2f& box) const {
    if (!(scale_x > 0) || !(scale_y > 0)) throw std::invalid_argument("Invalid letterbox scale");
    const auto clip = [](float x, float max) { return std::max(0.f,std::min(max,x)); };
    const float x1 = clip((box.x-left)/scale_x, static_cast<float>(original_size.width));
    const float y1 = clip((box.y-top)/scale_y, static_cast<float>(original_size.height));
    const float x2 = clip((box.x+box.width-left)/scale_x, static_cast<float>(original_size.width));
    const float y2 = clip((box.y+box.height-top)/scale_y, static_cast<float>(original_size.height));
    return {x1,y1,std::max(0.f,x2-x1),std::max(0.f,y2-y1)};
}

LetterboxTransform letterbox_rgb(const cv::Mat& bgr, cv::Size input,
                                cv::Mat& rgb, cv::Mat& resize_scratch) {
    if (bgr.empty() || bgr.type() != CV_8UC3)
        throw std::invalid_argument("YOLO input must be a nonempty BGR CV_8UC3 image");
    const auto t = letterbox_transform(bgr.size(),input);
    rgb.create(input,CV_8UC3);
    rgb.setTo(cv::Scalar(114,114,114));
    cv::resize(bgr,resize_scratch,t.resized_size,0,0,cv::INTER_LINEAR);
    cv::Mat region = rgb(cv::Rect(t.left,t.top,t.resized_size.width,t.resized_size.height));
    cv::cvtColor(resize_scratch,region,cv::COLOR_BGR2RGB);
    return t;
}

int validate_yolov8_outputs(const std::vector<YoloTensor>& outputs,
                           cv::Size input, bool require_buffers) {
    if (input.width <= 0 || input.height <= 0 || input.width%32 || input.height%32)
        throw std::invalid_argument("YOLO input dimensions must be positive multiples of 32");
    if (outputs.size() != 6 && outputs.size() != 9)
        throw std::invalid_argument("Expected Rockchip YOLOv8 split 6/9 outputs; standard combined exports are unsupported");
    const int per_branch = static_cast<int>(outputs.size()/3);
    int class_count = 0, bins = 0;
    for (std::size_t i = 0; i < outputs.size(); ++i) {
        const auto& t = outputs[i];
        const int branch = static_cast<int>(i)/per_branch;
        const int role = static_cast<int>(i)%per_branch;
        const int stride = 8 << branch;
        const std::string prefix = "YOLO output " + std::to_string(i) + ": ";
        if (t.channels <= 0 || t.width != input.width/stride || t.height != input.height/stride)
            throw std::invalid_argument(prefix+"expected compact NCHW at ordered strides 8,16,32");
        if (role == 0) {
            if (t.channels%4 || t.channels/4 < 2 || t.channels/4 > 64)
                throw std::invalid_argument(prefix+"invalid 4*reg_max DFL channels");
            if (branch == 0) bins = t.channels/4;
            else if (t.channels/4 != bins) throw std::invalid_argument(prefix+"inconsistent DFL bins");
        } else if (role == 1) {
            if (branch == 0) class_count = t.channels;
            else if (t.channels != class_count) throw std::invalid_argument(prefix+"inconsistent class count");
        } else if (t.channels != 1) throw std::invalid_argument(prefix+"score sum must have one channel");
        if (t.element != TensorElement::Float32 && t.element != TensorElement::Int8 && t.element != TensorElement::UInt8)
            throw std::invalid_argument(prefix+"unsupported tensor type");
        if (t.element != TensorElement::Float32 && (!std::isfinite(t.scale) || t.scale <= 0))
            throw std::invalid_argument(prefix+"invalid affine quantization scale");
        if ((t.element == TensorElement::Int8 && (t.zero_point < -128 || t.zero_point > 127)) ||
            (t.element == TensorElement::UInt8 && (t.zero_point < 0 || t.zero_point > 255)))
            throw std::invalid_argument(prefix+"invalid affine zero point");
        const std::size_t required = static_cast<std::size_t>(t.channels)*t.width*t.height*
                                     (t.element == TensorElement::Float32 ? sizeof(float) : 1);
        if (require_buffers && (!t.data || t.bytes < required))
            throw std::invalid_argument(prefix+"missing/truncated tensor buffer");
    }
    return class_count;
}

std::vector<Detection> decode_yolov8(const std::vector<YoloTensor>& outputs,
                                   const LetterboxTransform& transform,
                                   const YoloDecodeConfig& config) {
    const int classes = validate_yolov8_outputs(outputs,transform.input_size);
    if (!std::isfinite(config.confidence_threshold) || config.confidence_threshold < 0 || config.confidence_threshold > 1 ||
        !std::isfinite(config.nms_threshold) || config.nms_threshold < 0 || config.nms_threshold > 1 || config.max_detections <= 0)
        throw std::invalid_argument("Invalid YOLO confidence/NMS/max_detections configuration");
    const int per_branch = static_cast<int>(outputs.size()/3);
    std::vector<Detection> candidates;
    candidates.reserve(1024);
    for (int branch = 0; branch < 3; ++branch) {
        const auto& boxes = outputs[branch*per_branch];
        const auto& scores = outputs[branch*per_branch+1];
        const int bins = boxes.channels/4;
        const std::size_t plane = static_cast<std::size_t>(boxes.width)*boxes.height;
        const float stride = static_cast<float>(8 << branch);
        for (std::size_t cell = 0; cell < plane; ++cell) {
            if (per_branch == 3) {
                const float sum = read_element(outputs[branch*per_branch+2],cell);
                if (!std::isfinite(sum)) throw std::runtime_error("Non-finite YOLO score-sum output");
                const auto& sum_tensor = outputs[branch*per_branch+2];
                // Independently quantized sums and classes need not preserve
                // sum >= max(class). A sum-only tolerance cannot guarantee the
                // same threshold decisions as the 6-output decoder, especially
                // with different scales or clipping. Use this optimization only
                // when both tensors are floating point. Integer/mixed outputs
                // always decide from the actual dequantized class probabilities.
                if (sum_tensor.element == TensorElement::Float32 &&
                    scores.element == TensorElement::Float32 &&
                    sum < config.confidence_threshold) continue;
            }
            int class_id = -1;
            float confidence = -1;
            for (int c = 0; c < classes; ++c) {
                const float value = read_element(scores,c*plane+cell);
                if (!std::isfinite(value) || value < -0.05f || value > 1.05f)
                    throw std::runtime_error("YOLO classes must be sigmoid probabilities; unexpected logits/non-finite output");
                if (value > confidence) { confidence = value; class_id = c; }
            }
            if (confidence < config.confidence_threshold) continue;
            const float cx = (static_cast<float>(cell%boxes.width)+0.5f)*stride;
            const float cy = (static_cast<float>(cell/boxes.width)+0.5f)*stride;
            const float left = dfl_distance(boxes,cell,0,bins)*stride;
            const float top = dfl_distance(boxes,cell,1,bins)*stride;
            const float right = dfl_distance(boxes,cell,2,bins)*stride;
            const float bottom = dfl_distance(boxes,cell,3,bins)*stride;
            candidates.push_back({{cx-left,cy-top,left+right,top+bottom},
                                  std::max(0.f,std::min(1.f,confidence)),class_id});
        }
    }
    std::stable_sort(candidates.begin(),candidates.end(),[](const Detection& a,const Detection& b) { return a.score > b.score; });
    std::vector<Detection> selected;
    selected.reserve(std::min(candidates.size(),static_cast<std::size_t>(config.max_detections)));
    for (const auto& candidate : candidates) {
        bool suppressed = false;
        for (const auto& accepted : selected) {
            if (candidate.class_id == accepted.class_id &&
                intersection_over_union(candidate.bbox,accepted.bbox) > config.nms_threshold) {
                suppressed = true;
                break;
            }
        }
        if (!suppressed) {
            // Exclude padding-only boxes before they consume the detection limit.
            const cv::Rect2f original = transform.to_original(candidate.bbox);
            if (original.width <= 0 || original.height <= 0) continue;
            selected.push_back(candidate);
            if (static_cast<int>(selected.size()) >= config.max_detections) break;
        }
    }
    for (auto& detection : selected) detection.bbox = transform.to_original(detection.bbox);
    return selected;
}

} // namespace aerial
