// Compare RKNN input submission paths on identical RGB bytes. Each alternative
// is checked against the regular UINT8 input path; timing never replaces that
// correctness check. Output tensors are requested as FP32 in every mode.
#include "aerial/rknn_input.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using Clock = std::chrono::steady_clock;
void check(int ret, const char* operation) {
    if (ret != RKNN_SUCC) throw std::runtime_error(std::string(operation) + ": " + std::to_string(ret));
}
double ms(Clock::time_point begin, Clock::time_point end) {
    return std::chrono::duration<double, std::milli>(end - begin).count();
}
struct Context {
    rknn_context value = 0;
    rknn_tensor_mem* memory = nullptr;
    explicit Context(const std::string& path) {
        std::ifstream file(path, std::ios::binary | std::ios::ate);
        if (!file) throw std::runtime_error("Cannot read model " + path);
        const auto length = file.tellg();
        if (length <= 0 || static_cast<uint64_t>(length) > UINT32_MAX) throw std::runtime_error("Invalid model size");
        std::vector<char> bytes(static_cast<size_t>(length));
        file.seekg(0);
        if (!file.read(bytes.data(), length)) throw std::runtime_error("Incomplete model read");
        check(rknn_init(&value, bytes.data(), bytes.size(), 0, nullptr), "rknn_init");
    }
    ~Context() {
        if (memory) rknn_destroy_mem(value, memory);
        if (value) rknn_destroy(value);
    }
    Context(const Context&) = delete;
};
struct Sample { double conversion, input, run, output; };
struct Result {
    std::vector<std::vector<float>> tensors;
    std::vector<Sample> samples;
};
void attr(const char* label, const rknn_tensor_attr& a) {
    std::cout << label << " type=" << get_type_string(a.type) << " fmt=" << get_format_string(a.fmt) << " dims=";
    for (unsigned i = 0; i < a.n_dims; ++i) std::cout << (i ? "x" : "") << a.dims[i];
    std::cout << " stride=" << a.w_stride << " bytes=" << a.size << " stride_bytes=" << a.size_with_stride
              << " zp=" << a.zp << " scale=" << a.scale << '\n';
}
Result execute(const std::string& path, const std::string& mode, const std::vector<uint8_t>& pixels,
               int iterations, int warmup, int mask) {
    Context ctx(path);
    check(rknn_set_core_mask(ctx.value, static_cast<rknn_core_mask>(mask)), "rknn_set_core_mask");
    rknn_input_output_num io{};
    check(rknn_query(ctx.value, RKNN_QUERY_IN_OUT_NUM, &io, sizeof(io)), "query counts");
    if (io.n_input != 1 || !io.n_output) throw std::runtime_error("Expected single-input model");
    rknn_tensor_attr native{};
    check(rknn_query(ctx.value, RKNN_QUERY_NATIVE_INPUT_ATTR, &native, sizeof(native)), "query native input");
    attr(mode.c_str(), native);
    if (native.n_dims != 4 || native.dims[0] != 1 || native.fmt != RKNN_TENSOR_NHWC ||
        native.dims[1] != 640 || native.dims[2] != 640 || native.dims[3] != 3)
        throw std::runtime_error("Requires native batch=1 NHWC 640x640x3");
    const bool mapped = mode == "io_uint8";
    const bool half = mode == "fp16_raw" || mode == "fp16_normalized";
    if (half && (native.type != RKNN_TENSOR_FLOAT16 || (native.w_stride != 0 && native.w_stride != 640) ||
                 native.size_with_stride != pixels.size()*2))
        throw std::runtime_error("FP16 mode requires exact contiguous FP16 NHWC native tensor");
    std::vector<cv::float16_t> half_pixels(mode == "fp16_raw" ? pixels.size() : 0);
    std::unique_ptr<aerial::Fp16RgbInput> normalized;
    if (mode == "fp16_normalized") normalized.reset(new aerial::Fp16RgbInput(native));
    if (mapped) {
        const size_t stride = native.w_stride ? native.w_stride : 640;
        const size_t allocation = std::max<size_t>(native.size_with_stride, stride*640*3);
        ctx.memory = rknn_create_mem(ctx.value, allocation);
        if (!ctx.memory || !ctx.memory->virt_addr) throw std::runtime_error("rknn_create_mem failed");
        std::memset(ctx.memory->virt_addr, 0, allocation);
        auto buffer_attr = native;
        buffer_attr.type = RKNN_TENSOR_UINT8;
        buffer_attr.fmt = RKNN_TENSOR_NHWC;
        buffer_attr.pass_through = 0;
        check(rknn_set_io_mem(ctx.value, ctx.memory, &buffer_attr), "rknn_set_io_mem");
    }
    Result result;
    for (int iteration = -warmup; iteration < iterations; ++iteration) {
        const auto start = Clock::now();
        rknn_input input{};
        input.type = half ? RKNN_TENSOR_FLOAT16 : RKNN_TENSOR_UINT8;
        input.fmt = RKNN_TENSOR_NHWC;
        input.pass_through = half ? 1 : 0;
        input.buf = half ? static_cast<void*>(half_pixels.data()) : const_cast<uint8_t*>(pixels.data());
        input.size = half ? half_pixels.size()*sizeof(cv::float16_t) : pixels.size();
        if (normalized) {
            const cv::Mat rgb(640, 640, CV_8UC3, const_cast<uint8_t*>(pixels.data()));
            input = normalized->prepare(rgb);
        } else if (half) {
            for (size_t i = 0; i < pixels.size(); ++i) half_pixels[i] = cv::float16_t(pixels[i]);
        }
        if (mapped) {
            const size_t stride = native.w_stride ? native.w_stride : 640;
            for (size_t y = 0; y < 640; ++y)
                std::memcpy(static_cast<uint8_t*>(ctx.memory->virt_addr) + y*stride*3,
                            pixels.data() + y*640*3, 640*3);
        }
        const auto converted = Clock::now();
        if (!mapped) check(rknn_inputs_set(ctx.value, 1, &input), "rknn_inputs_set");
        const auto submitted = Clock::now();
        check(rknn_run(ctx.value, nullptr), "rknn_run");
        const auto executed = Clock::now();
        std::vector<rknn_output> outputs(io.n_output);
        for (unsigned i = 0; i < io.n_output; ++i) { outputs[i].index = i; outputs[i].want_float = 1; }
        check(rknn_outputs_get(ctx.value, io.n_output, outputs.data(), nullptr), "rknn_outputs_get");
        const auto done = Clock::now();
        if (iteration == iterations-1) {
            for (const auto& o : outputs) {
                const float* p = static_cast<const float*>(o.buf);
                result.tensors.emplace_back(p, p + o.size/sizeof(float));
            }
        }
        check(rknn_outputs_release(ctx.value, io.n_output, outputs.data()), "rknn_outputs_release");
        if (iteration >= 0) result.samples.push_back({ms(start, converted), ms(converted, submitted),
                                                     ms(submitted, executed), ms(executed, done)});
    }
    return result;
}
double percentile(std::vector<double> v, double p) {
    std::sort(v.begin(), v.end());
    const double at = p*(v.size()-1);
    const size_t lo = at, hi = std::min(lo+1, v.size()-1);
    return v[lo] + (v[hi]-v[lo])*(at-lo);
}
void report(const std::string& mode, const Result& r) {
    std::vector<double> convert, input, run, output, total;
    for (const auto& s : r.samples) {
        convert.push_back(s.conversion); input.push_back(s.input); run.push_back(s.run); output.push_back(s.output);
        total.push_back(s.conversion+s.input+s.run+s.output);
    }
    for (const auto& column : std::vector<std::pair<std::string, std::vector<double>>>{
             {"conversion/copy", convert}, {"inputs_set", input}, {"run", run}, {"outputs_get", output}, {"total", total}})
        std::cout << mode << ',' << column.first << ",p50_ms=" << percentile(column.second,.5)
                  << ",p95_ms=" << percentile(column.second,.95) << '\n';
}
bool compare(const Result& reference, const Result& other) {
    if (reference.tensors.size() != other.tensors.size()) throw std::runtime_error("Output count differs");
    bool equal = true;
    for (size_t t = 0; t < reference.tensors.size(); ++t) {
        const auto& a = reference.tensors[t]; const auto& b = other.tensors[t];
        if (a.size() != b.size()) throw std::runtime_error("Output tensor size differs");
        double max_diff = 0, sum = 0; size_t changed = 0;
        for (size_t i = 0; i < a.size(); ++i) {
            if (!std::isfinite(a[i]) || !std::isfinite(b[i])) throw std::runtime_error("Non-finite output");
            const double d = std::abs(a[i]-b[i]); max_diff = std::max(max_diff,d); sum += d; changed += d != 0;
        }
        equal = equal && max_diff == 0;
        std::cout << "output=" << t << ",elements=" << a.size() << ",changed=" << changed
                  << ",max_abs=" << max_diff << ",mean_abs=" << sum/a.size() << '\n';
    }
    return equal;
}
} // namespace

int main(int argc, char** argv) {
    try {
        std::string model = "model/RK3588/yolov8n.rknn", mode = "io_uint8", rgb;
        int iterations = 30, warmup = 5, mask = 0;
        for (int i = 1; i < argc; ++i) {
            const std::string key = argv[i];
            if (key == "--help") {
                std::cout << "profile_input [--model PATH] [--mode io_uint8|fp16_raw|fp16_normalized]\n"
                             "  [--rgb PATH] [--iterations N] [--warmup N] [--mask 0|1|2|3|4|7]\n"
                             "RGB is exactly 640*640*3 bytes, already letterboxed. Without RGB, use a\n"
                             "deterministic synthetic pattern. Compare native path to UINT8 regular API.\n"
                             "FP16 variants probe normalization semantics; do not assume either is correct.\n";
                return 0;
            }
            if (++i >= argc) throw std::runtime_error("Missing value for " + key);
            if (key == "--model") model = argv[i];
            else if (key == "--mode") mode = argv[i];
            else if (key == "--rgb") rgb = argv[i];
            else if (key == "--iterations") iterations = std::stoi(argv[i]);
            else if (key == "--warmup") warmup = std::stoi(argv[i]);
            else if (key == "--mask") mask = std::stoi(argv[i]);
            else throw std::runtime_error("Unknown option " + key);
        }
        if (iterations < 1 || iterations > 10000 || warmup < 0 || warmup > 1000) throw std::runtime_error("Invalid loop counts");
        if (mode != "io_uint8" && mode != "fp16_raw" && mode != "fp16_normalized") throw std::runtime_error("Invalid input mode");
        if (mask != 0 && mask != 1 && mask != 2 && mask != 3 && mask != 4 && mask != 7) throw std::runtime_error("Invalid core mask");
        std::vector<uint8_t> pixels(640*640*3);
        if (rgb.empty()) {
            for (size_t i = 0; i < pixels.size(); ++i) pixels[i] = static_cast<uint8_t>((i*37 + i/191) % 256);
        } else {
            std::ifstream file(rgb, std::ios::binary | std::ios::ate);
            if (!file || file.tellg() != static_cast<std::streamoff>(pixels.size())) throw std::runtime_error("Invalid RGB file size");
            file.seekg(0);
            if (!file.read(reinterpret_cast<char*>(pixels.data()), pixels.size())) throw std::runtime_error("Cannot read RGB file");
        }
        std::cout << std::fixed << std::setprecision(6);
        const auto regular = execute(model, "regular", pixels, iterations, warmup, mask);
        const auto candidate = execute(model, mode, pixels, iterations, warmup, mask);
        report("regular", regular); report(mode, candidate);
        const bool equal = compare(regular, candidate);
        std::cout << "outputs_bit_equal=" << equal << '\n';
        return equal ? 0 : 2;
    } catch (const std::exception& e) { std::cerr << "profile_input: " << e.what() << '\n'; return 1; }
}
