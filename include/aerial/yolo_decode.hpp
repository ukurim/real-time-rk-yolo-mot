#pragma once

#include "aerial/types.hpp"
#include <cstddef>
#include <cstdint>
#include <vector>

namespace aerial {

// The transform records the actual integer resize and padding, not an ideal scale.
struct LetterboxTransform {
    cv::Size original_size;
    cv::Size input_size;
    cv::Size resized_size;
    int left = 0;
    int top = 0;
    float scale_x = 0;
    float scale_y = 0;
    cv::Rect2f to_original(const cv::Rect2f& box) const;
};

LetterboxTransform letterbox_transform(cv::Size original, cv::Size input);
LetterboxTransform letterbox_rgb(const cv::Mat& bgr, cv::Size input,
                                cv::Mat& rgb, cv::Mat& resize_scratch);

enum class TensorElement { Float32, Int8, UInt8 };

// Compact NCHW batch=1 views. RKNN padding/native layouts must be converted by
// the runtime before creating this view. Float16 is requested as float32 there.
struct YoloTensor {
    int channels = 0;
    int height = 0;
    int width = 0;
    TensorElement element = TensorElement::Float32;
    int zero_point = 0;
    float scale = 1;
    const void* data = nullptr;
    std::size_t bytes = 0;
};

struct YoloDecodeConfig {
    float confidence_threshold = 0.1f;
    float nms_threshold = 0.7f;
    int max_detections = 300;
};

// Rockchip split outputs: [DFL, sigmoid class scores, optional score sum]
// for strides 8,16,32, in that order. Returns the metadata-derived class count.
int validate_yolov8_outputs(const std::vector<YoloTensor>& outputs,
                           cv::Size input, bool require_buffers = true);
std::vector<Detection> decode_yolov8(const std::vector<YoloTensor>& outputs,
                                   const LetterboxTransform& transform,
                                   const YoloDecodeConfig& config = {});

} // namespace aerial
