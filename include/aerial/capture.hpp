#pragma once
#include "aerial/pipeline.hpp"
#include <atomic>
#include <memory>
#include <string>

namespace aerial {
struct CaptureConfig {
    std::string type = "file";
    std::string path = "1080p60hz.mov";
    std::string pipeline;
    std::string decoder = "auto"; // auto or mpp (H264 MP4/MOV, or MJPEG camera)
    bool realtime = false;
    bool copy_to_cpu = false; // Copy only selected frames, useful for uncached decoder buffers.
    int width = 1920, height = 1080, fps = 30;
    int timeout_ms = 5000;
};
class GstCapture {
public:
    explicit GstCapture(const CaptureConfig& config);
    ~GstCapture();
    GstCapture(const GstCapture&) = delete;
    GstCapture& operator=(const GstCapture&) = delete;
    bool read(Frame& frame, const std::atomic<bool>& stop);
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
} // namespace aerial
