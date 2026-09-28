#pragma once
#include "aerial/capture.hpp"
#include "aerial/bot_sort.hpp"
#include "aerial/yolov8.hpp"
#include <string>

namespace aerial {
struct AppConfig {
    CaptureConfig source;
    DetectorConfig detector;
    TrackerConfig tracker;
    std::string calibration;
    std::string jsonl = "results.jsonl";
    std::string video_pipeline;
    bool preview = false;
    bool no_los = false;
    unsigned workers = 1;
    int opencv_threads = 1;
    unsigned max_frames = 0;
    double max_age_ms = 250;
    static AppConfig load(const std::string& path);
    void validate() const;
};
}
