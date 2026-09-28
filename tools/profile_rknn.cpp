// Isolated, synchronous RKNN API timing. Synthetic input is not an accuracy test.
#include "rknn_api.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using Clock = std::chrono::steady_clock;
double ms(Clock::time_point a, Clock::time_point b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
}
void check(int status, const std::string& operation) {
    if (status != RKNN_SUCC)
        throw std::runtime_error(operation + " failed, status=" + std::to_string(status));
}

struct Options {
    std::string model = "model/RK3588/yolov8n.rknn";
    std::string detail_path, samples_path;
    int mask = 0, iterations = 100, warmup = 10, want_float = 1;
    bool collect_perf = false;
};
void usage() {
    std::cout << "Usage: profile_rknn [--model PATH] [--mask 0|1|2|3|4|7]\n"
                 "  [--iterations N] [--warmup N] [--want-float 0|1]\n"
                 "  [--collect-perf] [--perf-detail PATH] [--samples PATH]\n"
                 "Defaults: model/RK3588/yolov8n.rknn, mask=0, iterations=100, warmup=10, want-float=1.\n"
                 "--perf-detail requires --collect-perf. Performance collection affects timing;\n"
                 "use a separate run for operator/core inspection. Input is synthetic 640x640 RGB\n"
                 "uint8 (all channels 114); this tool does not test detection accuracy.\n";
}
int integer(const std::string& value, const std::string& argument) {
    size_t consumed = 0;
    int result = std::stoi(value, &consumed);
    if (consumed != value.size()) throw std::runtime_error("Invalid integer for " + argument);
    return result;
}
Options parse(int argc, char** argv) {
    Options result;
    for (int i = 1; i < argc; ++i) {
        const std::string key = argv[i];
        if (key == "--help") { usage(); std::exit(0); }
        if (key == "--collect-perf") { result.collect_perf = true; continue; }
        if (i + 1 >= argc) throw std::runtime_error("Missing value for " + key);
        const std::string value = argv[++i];
        if (key == "--model") result.model = value;
        else if (key == "--mask") result.mask = integer(value, key);
        else if (key == "--iterations") result.iterations = integer(value, key);
        else if (key == "--warmup") result.warmup = integer(value, key);
        else if (key == "--want-float") result.want_float = integer(value, key);
        else if (key == "--perf-detail") result.detail_path = value;
        else if (key == "--samples") result.samples_path = value;
        else throw std::runtime_error("Unknown argument " + key);
    }
    const std::vector<int> valid_masks{0, 1, 2, 3, 4, 7};
    if (std::find(valid_masks.begin(), valid_masks.end(), result.mask) == valid_masks.end())
        throw std::runtime_error("Unsupported core mask");
    if (result.iterations < 1 || result.iterations > 1000000 || result.warmup < 0 || result.warmup > 1000000)
        throw std::runtime_error("iterations must be 1..1000000, warmup 0..1000000");
    if (result.want_float != 0 && result.want_float != 1)
        throw std::runtime_error("want-float must be 0 or 1");
    if (!result.detail_path.empty() && !result.collect_perf)
        throw std::runtime_error("--perf-detail requires --collect-perf");
    return result;
}

class Context {
public:
    rknn_context value = 0;
    Context(const std::string& path, uint32_t flags) {
        std::ifstream file(path, std::ios::binary | std::ios::ate);
        if (!file) throw std::runtime_error("Cannot open model: " + path);
        const auto length = file.tellg();
        if (length <= 0 || static_cast<uint64_t>(length) > std::numeric_limits<uint32_t>::max())
            throw std::runtime_error("Invalid model length");
        std::vector<char> bytes(static_cast<size_t>(length));
        file.seekg(0);
        if (!file.read(bytes.data(), length)) throw std::runtime_error("Cannot read model");
        const int status = rknn_init(&value, bytes.data(), static_cast<uint32_t>(bytes.size()), flags, nullptr);
        if (status != RKNN_SUCC) {
            if (value) rknn_destroy(value);
            value = 0;
            check(status, "rknn_init");
        }
    }
    ~Context() {
        if (value) {
            const int status = rknn_destroy(value);
            if (status != RKNN_SUCC) std::cerr << "rknn_destroy failed, status=" << status << '\n';
        }
    }
    Context(const Context&) = delete;
    Context& operator=(const Context&) = delete;
};

class OutputLease {
public:
    OutputLease(rknn_context context, uint32_t count, int want_float)
        : context_(context), outputs(count) {
        for (uint32_t i = 0; i < count; ++i) {
            outputs[i].index = i;
            outputs[i].want_float = static_cast<uint8_t>(want_float);
        }
    }
    void get() {
        check(rknn_outputs_get(context_, static_cast<uint32_t>(outputs.size()), outputs.data(), nullptr),
              "rknn_outputs_get");
        acquired_ = true;
    }
    void release() {
        const int status = rknn_outputs_release(context_, static_cast<uint32_t>(outputs.size()), outputs.data());
        acquired_ = false;
        check(status, "rknn_outputs_release");
    }
    ~OutputLease() {
        if (acquired_) {
            const int status = rknn_outputs_release(context_, static_cast<uint32_t>(outputs.size()), outputs.data());
            if (status != RKNN_SUCC) std::cerr << "rknn_outputs_release failed, status=" << status << '\n';
        }
    }
private:
    rknn_context context_;
    bool acquired_ = false;
public:
    std::vector<rknn_output> outputs;
};

void print_attr(const char* label, const rknn_tensor_attr& attr) {
    std::cout << label << '[' << attr.index << "] name=" << attr.name << " dims=";
    for (uint32_t i = 0; i < attr.n_dims; ++i) std::cout << (i ? "x" : "") << attr.dims[i];
    std::cout << " format=" << get_format_string(attr.fmt)
              << " type=" << get_type_string(attr.type)
              << " quant=" << get_qnt_type_string(attr.qnt_type)
              << " zp=" << attr.zp << " scale=" << attr.scale
              << " bytes=" << attr.size << " bytes_with_stride=" << attr.size_with_stride << '\n';
}
struct Sample {
    double inputs_set, run, outputs_get, outputs_release, perf_run;
    double api_total() const { return inputs_set + run + outputs_get + outputs_release; }
};
double percentile(const std::vector<double>& sorted, double fraction) {
    const double index = fraction * (sorted.size() - 1);
    const size_t lo = static_cast<size_t>(index), hi = std::min(lo + 1, sorted.size() - 1);
    return sorted[lo] + (sorted[hi] - sorted[lo]) * (index - lo);
}
void report(const char* name, std::vector<double> values) {
    std::sort(values.begin(), values.end());
    std::cout << std::left << std::setw(24) << name << std::right
              << std::setw(12) << std::accumulate(values.begin(), values.end(), 0.0) / values.size()
              << std::setw(12) << percentile(values, 0.50)
              << std::setw(12) << percentile(values, 0.95)
              << std::setw(12) << percentile(values, 0.99) << '\n';
}
void write_detail(const std::string& path, const std::string& contents) {
    std::ofstream file(path, std::ios::binary);
    if (!file || !file.write(contents.data(), static_cast<std::streamsize>(contents.size())))
        throw std::runtime_error("Cannot write performance detail: " + path);
}
} // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parse(argc, argv);
        std::cout << "SYNTHETIC INPUT ONLY: 640x640 RGB uint8, constant 114; no accuracy claim.\n"
                  << "Single synchronous context; async mode disabled. Requested core mask=" << options.mask
                  << " (AUTO=0 does not request all three cores), want_float=" << options.want_float
                  << ", collect_perf=" << options.collect_perf << '\n';
        const uint32_t flags = options.collect_perf ? RKNN_FLAG_COLLECT_PERF_MASK : 0;
        Context context(options.model, flags);
        check(rknn_set_core_mask(context.value, static_cast<rknn_core_mask>(options.mask)), "rknn_set_core_mask");
        rknn_sdk_version sdk{};
        check(rknn_query(context.value, RKNN_QUERY_SDK_VERSION, &sdk, sizeof(sdk)), "query SDK_VERSION");
        std::cout << "SDK=" << sdk.api_version << " driver=" << sdk.drv_version << '\n';
        rknn_input_output_num count{};
        check(rknn_query(context.value, RKNN_QUERY_IN_OUT_NUM, &count, sizeof(count)), "query IN_OUT_NUM");
        if (count.n_input != 1 || !count.n_output) throw std::runtime_error("Expected one image input and at least one output");
        rknn_tensor_attr input_attr{};
        check(rknn_query(context.value, RKNN_QUERY_INPUT_ATTR, &input_attr, sizeof(input_attr)), "query INPUT_ATTR");
        print_attr("input", input_attr);
        const bool nchw = input_attr.fmt == RKNN_TENSOR_NCHW;
        const bool nhwc = input_attr.fmt == RKNN_TENSOR_NHWC;
        if (input_attr.n_dims != 4 || input_attr.dims[0] != 1 || (!nchw && !nhwc)
            || input_attr.dims[nchw ? 1 : 3] != 3 || input_attr.dims[nchw ? 2 : 1] != 640
            || input_attr.dims[nchw ? 3 : 2] != 640)
            throw std::runtime_error("This benchmark requires static batch=1 640x640 3-channel input");
        for (uint32_t i = 0; i < count.n_output; ++i) {
            rknn_tensor_attr attr{};
            attr.index = i;
            check(rknn_query(context.value, RKNN_QUERY_OUTPUT_ATTR, &attr, sizeof(attr)), "query OUTPUT_ATTR");
            print_attr("output", attr);
        }
        std::vector<uint8_t> pixels(640 * 640 * 3, 114);
        rknn_input input{};
        input.index = 0;
        input.buf = pixels.data();
        input.size = static_cast<uint32_t>(pixels.size());
        input.type = RKNN_TENSOR_UINT8;
        input.fmt = RKNN_TENSOR_NHWC;
        input.pass_through = 0;
        std::vector<Sample> samples;
        samples.reserve(options.iterations);
        std::string detail;
        uint64_t output_bytes = 0;
        for (int i = -options.warmup; i < options.iterations; ++i) {
            OutputLease outputs(context.value, count.n_output, options.want_float);
            const auto start = Clock::now();
            check(rknn_inputs_set(context.value, 1, &input), "rknn_inputs_set");
            const auto after_input = Clock::now();
            check(rknn_run(context.value, nullptr), "rknn_run");
            const auto after_run = Clock::now();
            outputs.get();
            const auto after_get = Clock::now();
            rknn_perf_run perf{};
            check(rknn_query(context.value, RKNN_QUERY_PERF_RUN, &perf, sizeof(perf)), "query PERF_RUN");
            if (perf.run_duration < 0) throw std::runtime_error("Negative PERF_RUN duration");
            if (i == options.iterations - 1) {
                for (const auto& output : outputs.outputs) output_bytes += output.size;
                if (options.collect_perf) {
                    rknn_perf_detail info{};
                    check(rknn_query(context.value, RKNN_QUERY_PERF_DETAIL, &info, sizeof(info)), "query PERF_DETAIL");
                    if (!info.perf_data || !info.data_len) throw std::runtime_error("Empty PERF_DETAIL");
                    detail.assign(info.perf_data, static_cast<size_t>(info.data_len));
                }
            }
            const auto before_release = Clock::now();
            outputs.release();
            const auto after_release = Clock::now();
            if (i >= 0) samples.push_back({ms(start, after_input), ms(after_input, after_run),
                                          ms(after_run, after_get), ms(before_release, after_release),
                                          perf.run_duration / 1000.0});
        }
        std::cout << "Measured iterations=" << samples.size() << " warmup=" << options.warmup
                  << " returned_output_bytes=" << output_bytes << '\n'
                  << "All table values are milliseconds. API total is the per-iteration sum; excludes\n"
                     "performance queries, allocation of descriptors, preprocessing and postprocessing.\n"
                     "PERF_RUN is the runtime-reported duration; it is not an independent NPU hardware timer.\n";
        if (options.collect_perf)
            std::cout << "WARNING: collect_perf enabled; use an uninstrumented run for latency comparisons.\n";
        std::cout << std::fixed << std::setprecision(4) << std::left << std::setw(24) << "stage"
                  << std::right << std::setw(12) << "mean" << std::setw(12) << "p50"
                  << std::setw(12) << "p95" << std::setw(12) << "p99" << '\n';
        const auto column = [&](auto extract) {
            std::vector<double> values;
            for (const auto& sample : samples) values.push_back(extract(sample));
            return values;
        };
        report("inputs_set", column([](const Sample& s) { return s.inputs_set; }));
        report("run", column([](const Sample& s) { return s.run; }));
        report("outputs_get", column([](const Sample& s) { return s.outputs_get; }));
        report("outputs_release", column([](const Sample& s) { return s.outputs_release; }));
        report("api_total", column([](const Sample& s) { return s.api_total(); }));
        report("query_perf_run", column([](const Sample& s) { return s.perf_run; }));
        if (!options.samples_path.empty()) {
            std::ofstream csv(options.samples_path);
            if (!csv) throw std::runtime_error("Cannot open sample output: " + options.samples_path);
            csv << "iteration,inputs_set_ms,run_ms,outputs_get_ms,outputs_release_ms,api_total_ms,query_perf_run_ms\n";
            csv << std::fixed << std::setprecision(6);
            for (size_t i = 0; i < samples.size(); ++i) {
                const auto& s = samples[i];
                csv << i << ',' << s.inputs_set << ',' << s.run << ',' << s.outputs_get << ','
                    << s.outputs_release << ',' << s.api_total() << ',' << s.perf_run << '\n';
            }
            csv.close();
            if (!csv) throw std::runtime_error("Failed writing sample output");
        }
        if (!options.detail_path.empty()) {
            write_detail(options.detail_path, detail);
            std::cout << "Last measured iteration PERF_DETAIL written to " << options.detail_path << '\n';
        } else if (options.collect_perf) {
            std::cout << "Last measured iteration PERF_DETAIL:\n" << detail << '\n';
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "profile_rknn: " << error.what() << '\n';
        return 1;
    }
}
