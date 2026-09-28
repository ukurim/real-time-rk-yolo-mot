// Same-frame model agreement evaluation. Sequential inference, not a speed benchmark.
#include "aerial/capture.hpp"
#include "aerial/yolov8.hpp"
#include <glib.h>
#include <opencv2/core.hpp>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <csignal>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>

namespace {
std::atomic<bool> stop{false};
void interrupt(int) { stop.store(true); }
struct Options {
    std::string baseline, candidate, input, output = "build/detector_comparison.jsonl";
    std::string decoder = "auto";
    std::uint64_t start_frame = 0, stride = 30, samples = 100;
    int core_mask = 0, max_detections = 300;
    float confidence = 0.1f, nms = 0.7f;
    bool copy_input = true;
};
void usage() {
    std::cout << "Usage: compare_detectors --baseline FP16.rknn --candidate INT8.rknn --input VIDEO\n"
                 "  [--output build/detector_comparison.jsonl] [--start-frame 0] [--stride 30]\n"
                 "  [--samples 100] [--decoder auto|mpp] [--copy-input 0|1] [--core-mask 0]\n"
                 "  [--confidence 0.1] [--nms 0.7] [--max-detections 300]\n"
                 "Uses unpaced GstCapture and never drops decoded frames. Selected zero-based frame\n"
                 "indices are start_frame + k*stride. Both models receive exactly the same BGR pixels.\n"
                 "The JSONL contains model/video SHA256, source PTS, pixel SHA256, detections and\n"
                 "timings; timings include sequential model interference and are not a benchmark.\n"
                 "This is reference-model agreement, not ground-truth accuracy or mAP.\n";
}
std::uint64_t uint_value(const std::string& value, const std::string& name) {
    if (value.empty() || value.find_first_not_of("0123456789") != std::string::npos)
        throw std::runtime_error("Invalid nonnegative integer for " + name);
    size_t consumed = 0;
    const auto result = std::stoull(value, &consumed);
    if (consumed != value.size()) throw std::runtime_error("Invalid integer for " + name);
    return result;
}
int int_value(const std::string& value, const std::string& name) {
    const auto parsed = uint_value(value, name);
    if (parsed > std::uint64_t(std::numeric_limits<int>::max()))
        throw std::runtime_error("Integer too large for " + name);
    return static_cast<int>(parsed);
}
float probability(const std::string& value, const std::string& name) {
    size_t consumed = 0;
    const float parsed = std::stof(value, &consumed);
    if (consumed != value.size() || !std::isfinite(parsed) || parsed <= 0 || parsed > 1)
        throw std::runtime_error(name + " must be a finite value in (0,1]");
    return parsed;
}
Options parse(int argc, char** argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        const std::string key = argv[i];
        if (key == "--help") { usage(); std::exit(0); }
        if (i + 1 >= argc) throw std::runtime_error("Missing value for " + key);
        const std::string value = argv[++i];
        if (key == "--baseline") options.baseline = value;
        else if (key == "--candidate") options.candidate = value;
        else if (key == "--input") options.input = value;
        else if (key == "--output") options.output = value;
        else if (key == "--decoder") options.decoder = value;
        else if (key == "--start-frame") options.start_frame = uint_value(value, key);
        else if (key == "--stride") options.stride = uint_value(value, key);
        else if (key == "--samples") options.samples = uint_value(value, key);
        else if (key == "--core-mask") options.core_mask = int_value(value, key);
        else if (key == "--max-detections") options.max_detections = int_value(value, key);
        else if (key == "--confidence") options.confidence = probability(value, key);
        else if (key == "--nms") options.nms = probability(value, key);
        else if (key == "--copy-input") {
            if (value != "0" && value != "1") throw std::runtime_error("copy-input must be 0 or 1");
            options.copy_input = value == "1";
        } else throw std::runtime_error("Unknown option " + key);
    }
    if (options.baseline.empty() || options.candidate.empty() || options.input.empty())
        throw std::runtime_error("--baseline, --candidate and --input are required");
    if (options.output.empty() || options.output == "-")
        throw std::runtime_error("--output must be a file; runtime diagnostics may use stdout");
    if (!options.stride || !options.samples || options.samples > 1000000 || options.max_detections < 1)
        throw std::runtime_error("stride/samples/max-detections must be positive; samples <= 1000000");
    if (options.samples - 1 > (std::numeric_limits<std::uint64_t>::max() - options.start_frame) / options.stride)
        throw std::runtime_error("Frame selection overflows uint64");
    if (options.decoder != "auto" && options.decoder != "mpp")
        throw std::runtime_error("decoder must be auto or mpp");
    const std::vector<int> masks{0, 1, 2, 3, 4, 7};
    if (std::find(masks.begin(), masks.end(), options.core_mask) == masks.end())
        throw std::runtime_error("Unsupported core-mask");
    return options;
}
std::string json_string(const std::string& input) {
    std::ostringstream output;
    output << '"';
    for (const unsigned char ch : input) {
        if (ch == '"' || ch == '\\') output << '\\' << ch;
        else if (ch < 0x20) output << "\\u" << std::hex << std::setw(4) << std::setfill('0') << int(ch) << std::dec;
        else output << ch;
    }
    return output.str() + '"';
}
using Checksum = std::unique_ptr<GChecksum, decltype(&g_checksum_free)>;
Checksum new_checksum() {
    Checksum checksum(g_checksum_new(G_CHECKSUM_SHA256), &g_checksum_free);
    if (!checksum) throw std::runtime_error("Cannot create SHA256 checksum");
    return checksum;
}
std::string file_metadata(const std::string& path) {
    const auto absolute = std::filesystem::canonical(path);
    if (!std::filesystem::is_regular_file(absolute)) throw std::runtime_error("Not a regular file: " + path);
    std::ifstream input(absolute, std::ios::binary);
    if (!input) throw std::runtime_error("Cannot read " + path);
    auto checksum = new_checksum();
    char block[65536];
    while (input.read(block, sizeof(block)) || input.gcount())
        g_checksum_update(checksum.get(), reinterpret_cast<const guchar*>(block), input.gcount());
    if (!input.eof()) throw std::runtime_error("Failed reading " + path);
    return "{\"path\":" + json_string(absolute.string()) + ",\"bytes\":" +
        std::to_string(std::filesystem::file_size(absolute)) + ",\"sha256\":" +
        json_string(g_checksum_get_string(checksum.get())) + "}";
}
std::string pixel_checksum(const cv::Mat& frame) {
    auto checksum = new_checksum();
    // Hash exactly width*3 BGR bytes per row. Decoder stride padding is excluded.
    for (int row = 0; row < frame.rows; ++row)
        g_checksum_update(checksum.get(), frame.ptr<guchar>(row), frame.cols * frame.elemSize());
    return g_checksum_get_string(checksum.get());
}
void write_timing(std::ostream& out, const aerial::StageTiming& timing) {
    out << "{\"preprocess_ms\":" << timing.preprocess_ms
        << ",\"inference_ms\":" << timing.inference_ms
        << ",\"input_submit_ms\":" << timing.input_submit_ms
        << ",\"rknn_run_ms\":" << timing.rknn_run_ms
        << ",\"output_get_ms\":" << timing.output_get_ms
        << ",\"postprocess_ms\":" << timing.postprocess_ms << '}';
}
void write_detections(std::ostream& out, const std::vector<aerial::Detection>& detections) {
    out << '[';
    for (size_t i = 0; i < detections.size(); ++i) {
        const auto& detection = detections[i];
        const auto& box = detection.bbox;
        if (i) out << ',';
        out << "{\"class_id\":" << detection.class_id << ",\"score\":" << detection.score
            << ",\"bbox_xyxy\":[" << box.x << ',' << box.y << ',' << box.x + box.width << ',' << box.y + box.height << "]}";
    }
    out << ']';
}
} // namespace

int main(int argc, char** argv) {
    static_assert(std::atomic<bool>::is_always_lock_free, "Signal handler requires lock-free atomic");
    try {
        const Options options = parse(argc, argv);
        cv::setNumThreads(1);
        std::signal(SIGINT, interrupt);
        std::signal(SIGTERM, interrupt);
        // Hash source artifacts before loading either context, outside all timing.
        const std::string baseline_metadata = file_metadata(options.baseline);
        const std::string candidate_metadata = file_metadata(options.candidate);
        const std::string input_metadata = file_metadata(options.input);
        const auto output_absolute = std::filesystem::exists(options.output)
            ? std::filesystem::canonical(options.output)
            : std::filesystem::weakly_canonical(std::filesystem::absolute(options.output));
        for (const auto& path : {options.baseline, options.candidate, options.input})
            if (output_absolute == std::filesystem::canonical(path))
                throw std::runtime_error("Output cannot overwrite an input artifact");
        aerial::DetectorConfig detector_config;
        detector_config.model_path = options.baseline;
        detector_config.confidence_threshold = options.confidence;
        detector_config.nms_threshold = options.nms;
        detector_config.max_detections = options.max_detections;
        detector_config.core_mask = options.core_mask;
        aerial::YoloV8Detector baseline(detector_config);
        detector_config.model_path = options.candidate;
        aerial::YoloV8Detector candidate(detector_config);
        if (baseline.input_size() != candidate.input_size() || baseline.class_count() != candidate.class_count())
            throw std::runtime_error("Baseline and candidate must have equal input dimensions and class count");
        aerial::CaptureConfig capture_config;
        capture_config.type = "file";
        capture_config.path = options.input;
        capture_config.decoder = options.decoder;
        capture_config.realtime = false;
        aerial::GstCapture capture(capture_config);
        std::ofstream out(options.output);
        if (!out) throw std::runtime_error("Cannot open output " + options.output);
        out << std::setprecision(9);
        out << "{\"record_type\":\"run\",\"schema_version\":1,\"comparison\":\"reference_model_agreement_not_ground_truth\","
            << "\"baseline_model\":" << baseline_metadata << ",\"candidate_model\":" << candidate_metadata
            << ",\"source\":" << input_metadata << ",\"sampling\":{\"start_frame\":" << options.start_frame
            << ",\"stride\":" << options.stride << ",\"requested_samples\":" << options.samples
            << ",\"realtime\":false,\"decoder\":" << json_string(options.decoder)
            << ",\"copy_input\":" << (options.copy_input ? "true" : "false") << "},"
            << "\"detector\":{\"confidence_threshold\":" << options.confidence
            << ",\"nms_threshold\":" << options.nms << ",\"max_detections\":" << options.max_detections
            << ",\"core_mask\":" << options.core_mask << ",\"preprocess\":\"cpu_letterbox_rgb\",\"input_width\":"
            << baseline.input_size().width << ",\"input_height\":" << baseline.input_size().height
            << ",\"class_count\":" << baseline.class_count() << "},"
            << "\"inference_order\":\"baseline_then_candidate\",\"timing_is_benchmark\":false}\n";
        std::uint64_t decoded = 0, selected = 0;
        bool eof = false;
        while (!stop && selected < options.samples) {
            aerial::Frame frame;
            if (!capture.read(frame, stop)) { eof = !stop; break; }
            ++decoded;
            if (frame.id < options.start_frame || (frame.id - options.start_frame) % options.stride) continue;
            // Optional copy is identical for both models and is outside stage timings.
            if (options.copy_input) { frame.image = frame.image.clone(); frame.owner.reset(); }
            const auto pixels_sha256 = pixel_checksum(frame.image);
            aerial::StageTiming baseline_timing, candidate_timing;
            const auto baseline_detections = baseline.detect(frame.image, &baseline_timing);
            const auto candidate_detections = candidate.detect(frame.image, &candidate_timing);
            out << "{\"record_type\":\"frame\",\"sample_index\":" << selected
                << ",\"frame_id\":" << frame.id << ",\"image_timestamp_ns\":" << frame.image_timestamp_ns
                << ",\"timestamp_domain\":" << json_string(frame.timestamp_domain)
                << ",\"image_width\":" << frame.image.cols << ",\"image_height\":" << frame.image.rows
                << ",\"bgr_pixels_sha256\":" << json_string(pixels_sha256) << ",\"baseline\":";
            write_detections(out, baseline_detections);
            out << ",\"candidate\":";
            write_detections(out, candidate_detections);
            out << ",\"baseline_timing\":";
            write_timing(out, baseline_timing);
            out << ",\"candidate_timing\":";
            write_timing(out, candidate_timing);
            out << "}\n";
            if (!out) throw std::runtime_error("Failed writing output " + options.output);
            ++selected;
            if (selected % 10 == 0)
                std::cerr << "Compared " << selected << '/' << options.samples << " frames, last frame_id=" << frame.id << '\n';
        }
        out << "{\"record_type\":\"summary\",\"decoded_frames\":" << decoded << ",\"selected_frames\":" << selected
            << ",\"requested_samples\":" << options.samples << ",\"eof\":" << (eof ? "true" : "false")
            << ",\"interrupted\":" << (stop ? "true" : "false") << "}\n";
        out.close();
        if (!out) throw std::runtime_error("Failed closing output " + options.output);
        std::cerr << "Comparison saved: " << options.output << " (" << selected << " selected / " << decoded
                  << " decoded; model agreement only, no ground-truth accuracy claim)\n";
        return stop ? 130 : (selected ? 0 : 2);
    } catch (const std::exception& error) {
        std::cerr << "compare_detectors: " << error.what() << '\n';
        return 1;
    }
}
