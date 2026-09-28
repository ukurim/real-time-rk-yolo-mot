#include "aerial/app_config.hpp"
#include <opencv2/core/persistence.hpp>
#include <filesystem>
#include <cmath>
#include <stdexcept>

namespace aerial {
namespace {
template<class T> void read(const cv::FileNode& node, const char* key, T& value) {
    if (!node[key].empty()) node[key] >> value;
}
void read_bool(const cv::FileNode& node, const char* key, bool& value) {
    int integer = value;
    read(node, key, integer);
    if (integer != 0 && integer != 1) throw std::runtime_error(std::string(key) + " must be 0 or 1");
    value = integer;
}
std::string resolve(const std::filesystem::path& base, const std::string& path) {
    if (path.empty() || path == "-") return path;
    return std::filesystem::absolute(base / path).lexically_normal().string();
}
}
AppConfig AppConfig::load(const std::string& path) {
    cv::FileStorage file(path, cv::FileStorage::READ);
    if (!file.isOpened()) throw std::runtime_error("Cannot open application config: " + path);
    const auto base = std::filesystem::absolute(path).parent_path();
    AppConfig c;
    const auto s = file["source"];
    read(s, "type", c.source.type); read(s, "path", c.source.path);
    read(s, "pipeline", c.source.pipeline); read(s, "decoder", c.source.decoder);
    read_bool(s, "realtime", c.source.realtime);
    read_bool(s, "copy_to_cpu", c.source.copy_to_cpu);
    read(s, "width", c.source.width); read(s, "height", c.source.height);
    read(s, "fps", c.source.fps); read(s, "timeout_ms", c.source.timeout_ms);
    if (c.source.type == "file") c.source.path = resolve(base, c.source.path);
    const auto d = file["detector"];
    read(d, "model_path", c.detector.model_path);
    c.detector.model_path = resolve(base, c.detector.model_path);
    read(d, "confidence_threshold", c.detector.confidence_threshold);
    read(d, "nms_threshold", c.detector.nms_threshold);
    read(d, "max_detections", c.detector.max_detections);
    read(d, "core_mask", c.detector.core_mask);
    read(d, "preprocess", c.detector.preprocess);
    read(d, "input_mode", c.detector.input_mode);
    const auto t = file["tracker"];
    read(t, "high_thresh", c.tracker.high_thresh); read(t, "low_thresh", c.tracker.low_thresh);
    read(t, "new_track_thresh", c.tracker.new_track_thresh); read(t, "match_thresh", c.tracker.match_thresh);
    read(t, "track_buffer_seconds", c.tracker.track_buffer_seconds);
    read(t, "prediction_output_seconds", c.tracker.prediction_output_seconds);
    read(t, "nominal_fps", c.tracker.nominal_fps); read(t, "gmc_method", c.tracker.gmc_method);
    const auto p = file["pipeline"];
    int workers = c.workers, frames = c.max_frames;
    read(p, "workers", workers); read(p, "max_frames", frames);
    if (workers < 1 || workers > 3 || frames < 0) throw std::runtime_error("Invalid workers/max_frames");
    c.workers = workers; c.max_frames = frames;
    read(p, "max_age_ms", c.max_age_ms);
    read(p, "opencv_threads", c.opencv_threads);
    const auto o = file["output"];
    read(o, "jsonl", c.jsonl); c.jsonl = resolve(base, c.jsonl);
    read_bool(o, "preview", c.preview); read(o, "video_pipeline", c.video_pipeline);
    read(file.root(), "calibration", c.calibration); c.calibration = resolve(base, c.calibration);
    return c;
}
void AppConfig::validate() const {
    if (workers < 1 || workers > 3) throw std::runtime_error("workers must be between 1 and 3");
    if (opencv_threads < 1 || opencv_threads > 8) throw std::runtime_error("opencv_threads must be between 1 and 8");
    if (!std::isfinite(max_age_ms) || max_age_ms <= 0) throw std::runtime_error("max_age_ms must be positive and finite");
    if (source.decoder != "auto" && source.decoder != "mpp") throw std::runtime_error("decoder must be auto or mpp");
    if (source.type == "camera" && !source.realtime) throw std::runtime_error("Camera input requires realtime: 1");
    if (source.width <= 0 || source.height <= 0 || source.fps <= 0 || source.timeout_ms <= 0)
        throw std::runtime_error("Invalid source dimensions, rate, or timeout");
    if (source.type == "pipeline" && source.pipeline.empty()) throw std::runtime_error("Empty source.pipeline");
    if (calibration.empty() && !no_los) throw std::runtime_error("Provide a calibrated camera YAML or explicitly use --no-los for detection/tracking only");
    if (!calibration.empty() && no_los) throw std::runtime_error("--no-los and calibration are mutually exclusive");
    if (detector.model_path.empty()) throw std::runtime_error("A YOLOv8 RKNN model is required");
    if (detector.confidence_threshold > tracker.low_thresh)
        throw std::runtime_error("Detector confidence_threshold must be <= tracker.low_thresh to retain second-pass candidates");
    if (jsonl.empty()) throw std::runtime_error("output.jsonl must be a filename or '-'");
}
}
