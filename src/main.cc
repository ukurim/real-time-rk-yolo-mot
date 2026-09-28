#include "aerial/app_config.hpp"
#include "aerial/output.hpp"
#include <algorithm>
#include <atomic>
#include <csignal>
#include <deque>
#include <exception>
#include <iostream>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <thread>

namespace {
std::atomic<bool> stop_requested{false};
static_assert(std::atomic<bool>::is_always_lock_free, "Signal flag must be lock-free");
void stop_handler(int) { stop_requested.store(true); }
void usage() {
    std::cerr << "Usage: Aerial_detection_demo --config config/video.yaml [options]\n"
              << "  --model PATH          YOLOv8 RKNN model\n"
              << "  --input PATH          File input (use config for camera/custom GStreamer)\n"
              << "  --decoder auto|mpp    Auto decode or MPP direct BGR (H264 MOV/MP4)\n"
              << "  --copy-input | --no-copy-input  Copy selected decoder frames to CPU memory\n"
              << "  --calibration PATH    Real calibrated camera YAML\n"
              << "  --no-los              Explicit detection/tracking-only mode (los:null)\n"
              << "  --workers 1|2|3       Concurrent detector instances; benchmark before choosing\n"
              << "  --cv-threads N        OpenCV CPU parallelism (1..8); benchmark with workers\n"
              << "  --core-mask 0|1|2|4|7 RKNN auto-select/core0/core1/core2/combine-all\n"
              << "  --input-mode MODE     uint8 or fp16_normalized (verified /255 FP16 models)\n"
              << "  --realtime | --offline  Latest-frame replay or preserve all file frames\n"
              << "  --max-frames N        Stop after N received frames (0 = EOF)\n"
              << "  --output PATH|-       JSONL file or stdout\n"
              << "  --preview             Optional asynchronous window (off by default)\n";
}
unsigned unsigned_arg(const std::string& text) {
    std::size_t end = 0;
    if (text.empty() || text[0] == '-') throw std::runtime_error("Invalid nonnegative integer: " + text);
    const auto n = std::stoul(text, &end);
    if (end != text.size() || n > 1000000000) throw std::runtime_error("Invalid integer: " + text);
    return unsigned(n);
}
double percentile(std::deque<double> data, double q) {
    if (data.empty()) return 0;
    std::sort(data.begin(), data.end());
    return data[std::size_t(q * (data.size() - 1))];
}
void append(std::deque<double>& data, double value) {
    if (data.size() == 4096) data.pop_front();
    data.push_back(value);
}
}

int main(int argc, char** argv) {
    using namespace aerial;
    try {
        std::string config_path = "config/video.yaml";
        for (int i = 1; i < argc; ++i) {
            const std::string arg = argv[i];
            if (arg == "--help" || arg == "-h") { usage(); return 0; }
            if (arg == "--config") {
                if (++i == argc) throw std::runtime_error("Missing --config path");
                config_path = argv[i];
            }
        }
        AppConfig config = AppConfig::load(config_path);
        for (int i = 1; i < argc; ++i) {
            const std::string arg = argv[i];
            auto value = [&]() -> std::string {
                if (++i == argc) throw std::runtime_error("Missing value for " + arg);
                return argv[i];
            };
            if (arg == "--config") value();
            else if (arg == "--model") config.detector.model_path = value();
            else if (arg == "--input") { config.source.path = value(); config.source.type = "file"; }
            else if (arg == "--decoder") config.source.decoder = value();
            else if (arg == "--copy-input") config.source.copy_to_cpu = true;
            else if (arg == "--no-copy-input") config.source.copy_to_cpu = false;
            else if (arg == "--calibration") config.calibration = value();
            else if (arg == "--no-los") config.no_los = true;
            else if (arg == "--workers") config.workers = unsigned_arg(value());
            else if (arg == "--cv-threads") config.opencv_threads = unsigned_arg(value());
            else if (arg == "--core-mask") config.detector.core_mask = unsigned_arg(value());
            else if (arg == "--input-mode") config.detector.input_mode = value();
            else if (arg == "--max-frames") config.max_frames = unsigned_arg(value());
            else if (arg == "--output") config.jsonl = value();
            else if (arg == "--realtime") config.source.realtime = true;
            else if (arg == "--offline") config.source.realtime = false;
            else if (arg == "--preview") config.preview = true;
            else throw std::runtime_error("Unknown option: " + arg);
        }
        config.validate();
        std::signal(SIGINT, stop_handler);
        std::signal(SIGTERM, stop_handler);
        std::signal(SIGPIPE, SIG_IGN);
        cv::setNumThreads(config.opencv_threads);
        std::unique_ptr<LosProjector> los;
        if (!config.no_los) los = std::make_unique<LosProjector>(config.calibration);
        JsonPublisher publisher(config.jsonl, config.source.realtime, &stop_requested);
        BotSort tracker(config.tracker);
        std::vector<std::unique_ptr<YoloV8Detector>> detectors;
        for (unsigned i = 0; i < config.workers; ++i) {
            auto dc = config.detector;
            if (config.workers > 1 && dc.core_mask == 0) dc.core_mask = 1 << i;
            detectors.emplace_back(new YoloV8Detector(dc));
        }
        GstCapture capture(config.source);
        std::unique_ptr<Preview> preview;
        if (config.preview || !config.video_pipeline.empty())
            preview = std::make_unique<Preview>(config.preview, config.video_pipeline, config.source.fps);
        FrameSlot<Frame> input(config.source.realtime);
        CompletionQueue completed(config.source.realtime, config.workers);
        std::mutex error_mutex;
        std::exception_ptr error;
        auto fail = [&] {
            { std::lock_guard<std::mutex> lock(error_mutex); if (!error) error = std::current_exception(); }
            stop_requested = true;
            input.close(); completed.close();
        };
        std::atomic<unsigned> workers_left{config.workers};
        std::atomic<std::uint64_t> received{0}, expired{0};
        const auto started = monotonic_ns();
        std::vector<std::thread> workers;
        std::thread reader;
        std::uint64_t results_count = 0, targets_count = 0;
        std::deque<double> receiver_latency, source_latency;
        try {
            for (unsigned i = 0; i < config.workers; ++i) {
                workers.emplace_back([&, i] {
                    try {
                        Frame frame;
                        while (!stop_requested && input.get(frame)) {
                            DetectionPacket packet;
                            packet.frame = std::move(frame);
                            packet.inference_start_ns = monotonic_ns();
                            const auto origin = packet.frame.source_monotonic_ns > 0 ?
                                packet.frame.source_monotonic_ns : packet.frame.received_ns;
                            if (config.source.realtime && (packet.inference_start_ns - origin) / 1e6 > config.max_age_ms) {
                                ++expired; continue;
                            }
                            const auto copy_start = monotonic_ns();
                            if (config.source.copy_to_cpu) {
                                packet.frame.image = packet.frame.image.clone();
                                packet.frame.owner.reset();
                            }
                            const double copy_ms = config.source.copy_to_cpu ? (monotonic_ns() - copy_start) / 1e6 : 0;
                            packet.detections = detectors[i]->detect(packet.frame.image, &packet.timing);
                            packet.timing.preprocess_ms += copy_ms;
                            packet.detection_ready_ns = monotonic_ns();
                            if (!completed.put(std::move(packet))) break;
                        }
                    } catch (...) { fail(); }
                    if (--workers_left == 0) completed.close();
                });
            }
            reader = std::thread([&] {
                try {
                    Frame frame;
                    while (!stop_requested && (!config.max_frames || received < config.max_frames) && capture.read(frame, stop_requested)) {
                        ++received;
                        if (!input.put(std::move(frame))) break;
                    }
                    input.close();
                } catch (...) { fail(); }
            });
            DetectionPacket packet;
            while (!stop_requested && completed.get(packet)) {
                const auto track_start = monotonic_ns();
                const auto origin = packet.frame.source_monotonic_ns > 0 ? packet.frame.source_monotonic_ns : packet.frame.received_ns;
                if (config.source.realtime && (track_start - origin) / 1e6 > config.max_age_ms) { ++expired; continue; }
                ResultPacket result;
                result.completion_queue_ms = (track_start - packet.detection_ready_ns) / 1e6;
                result.detection = std::move(packet);
                result.tracks = tracker.update(result.detection.detections, result.detection.frame.image,
                                               result.detection.frame.tracking_time_seconds, &result.tracker_timing);
                const auto track_end = monotonic_ns();
                result.tracking_ms = (track_end - track_start) / 1e6;
                result.calibrated = bool(los);
                if (los) {
                    los->calibration().intrinsicsFor(result.detection.frame.image.size());
                    for (const auto& target : result.tracks)
                        result.los.push_back(los->project(target.bbox, result.detection.frame.image.size()));
                }
                result.ready_ns = monotonic_ns();
                result.los_ms = (result.ready_ns - track_end) / 1e6;
                // LOS becomes available before drawing, encoding, or network video output.
                if (!publisher.publish(result_json(result))) {
                    if (stop_requested) break;
                    throw std::runtime_error("JSON publisher stopped");
                }
                ++results_count;
                targets_count += result.tracks.size();
                append(receiver_latency, (result.ready_ns - result.detection.frame.received_ns) / 1e6);
                if (result.detection.frame.source_monotonic_ns > 0)
                    append(source_latency, (result.ready_ns - result.detection.frame.source_monotonic_ns) / 1e6);
                if (preview) preview->publish(result);
            }
        } catch (...) { fail(); }
        if (stop_requested) { input.close(); completed.close(); }
        if (reader.joinable()) reader.join();
        for (auto& worker : workers) if (worker.joinable()) worker.join();
        publisher.finish();
        if (preview) preview->finish();
        std::cerr << "Summary: received=" << received << " results=" << results_count
                  << " targets=" << targets_count << " capture_mailbox_dropped=" << input.dropped()
                  << " completion_dropped=" << completed.dropped() << " expired=" << expired
                  << " json_dropped=" << publisher.dropped() << " workers=" << config.workers
                  << " opencv_threads=" << config.opencv_threads
                  << " elapsed_s=" << (monotonic_ns() - started) / 1e9 << '\n';
        std::cerr << "Receiver->result ms (last <=4096): p50=" << percentile(receiver_latency, .50)
                  << " p95=" << percentile(receiver_latency, .95) << " p99=" << percentile(receiver_latency, .99) << '\n';
        if (!source_latency.empty())
            std::cerr << "Source PTS->result ESTIMATE ms: p50=" << percentile(source_latency, .50)
                      << " p95=" << percentile(source_latency, .95) << " p99=" << percentile(source_latency, .99) << '\n';
        if (error) std::rethrow_exception(error);
        if (publisher.failed()) return 1;
        if (!results_count && !stop_requested) throw std::runtime_error("No usable results produced");
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << '\n';
        return 1;
    }
}
