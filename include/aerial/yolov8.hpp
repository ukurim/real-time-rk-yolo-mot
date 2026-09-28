#pragma once

#include "aerial/types.hpp"
#include <memory>
#include <string>

namespace aerial {

struct DetectorConfig {
    std::string model_path;
    float confidence_threshold = 0.1f;
    float nms_threshold = 0.7f;
    int max_detections = 300;
    int core_mask = 0; // AUTO=0 selects a core, not all three; cores01=3, all=7.
    std::string preprocess = "cpu";
    // fp16_normalized explicitly requires conversion mean=0,std=255 and a
    // contiguous native NHWC FP16 input. UINT8 keeps runtime normalization.
    std::string input_mode = "uint8";
};

// A context has one in-flight frame. Parallel workers must own separate instances.
class YoloV8Detector {
public:
    explicit YoloV8Detector(const DetectorConfig& config);
    ~YoloV8Detector();
    YoloV8Detector(const YoloV8Detector&) = delete;
    YoloV8Detector& operator=(const YoloV8Detector&) = delete;
    std::vector<Detection> detect(const cv::Mat& bgr, StageTiming* timing = nullptr);
    cv::Size input_size() const;
    int class_count() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace aerial
