#pragma once

#include <opencv2/core.hpp>
#include <cstdint>
#include <vector>

namespace aerial {

// All boxes use floating point xywh in the original, distorted input image.
struct Detection {
    cv::Rect2f bbox;
    float score = 0;
    int class_id = -1;
};

struct TrackResult {
    std::int64_t track_id = 0;
    cv::Rect2f bbox;
    float score = 0;
    int class_id = -1;
    bool detection_updated = false;
    double seconds_since_update = 0;
};

struct StageTiming {
    double preprocess_ms = 0;
    double inference_ms = 0;
    // Wall-clock API durations; run/get may include waiting, not pure kernels.
    double input_submit_ms = 0;
    double rknn_run_ms = 0;
    double output_get_ms = 0;
    double postprocess_ms = 0;
};

} // namespace aerial
