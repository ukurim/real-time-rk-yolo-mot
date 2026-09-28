#pragma once

#include "aerial/types.hpp"
#include <memory>
#include <string>

namespace aerial {

struct TrackerConfig {
    float high_thresh = 0.5f;
    float low_thresh = 0.1f;
    float new_track_thresh = 0.6f;
    float match_thresh = 0.8f;
    double track_buffer_seconds = 1.0;
    double prediction_output_seconds = 0.2;
    double nominal_fps = 30.0;
    std::string gmc_method = "sparseOptFlow"; // sparseOptFlow or none; ReID disabled.
};

struct TrackerTiming {
    double gmc_ms = 0;
    double association_ms = 0; // KF, matching, lifecycle and output; excludes GMC.
};

// One instance per source, called in strictly increasing image-time order.
// Input/output boxes are floating xywh in the original distorted image.
// See docs/BOTSORT_PROVENANCE.md for the pinned upstream and adaptations.
class BotSort {
public:
    explicit BotSort(const TrackerConfig& config = {});
    ~BotSort();
    BotSort(BotSort&&) noexcept;
    BotSort& operator=(BotSort&&) noexcept;
    BotSort(const BotSort&) = delete;
    BotSort& operator=(const BotSort&) = delete;

    std::vector<TrackResult> update(const std::vector<Detection>& detections,
                                    const cv::Mat& frame,
                                    double image_time_seconds,
                                    TrackerTiming* timing = nullptr);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace aerial
