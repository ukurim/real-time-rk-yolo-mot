// C++ port of Nir Aharon's BoT-SORT, MIT License (c) 2022 Nir Aharon.
// Upstream: 251985436d6712aaf682aaaf5f71edb4987224bd.
// See third_party/botsort/LICENSE and docs/BOTSORT_PROVENANCE.md.
#include "aerial/bot_sort.hpp"
#include <chrono>

#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/video/tracking.hpp>
#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <utility>

namespace aerial {
namespace {

using Mean = cv::Matx<double, 8, 1>;
using Covariance = cv::Matx<double, 8, 8>;
enum class State { Tracked, Lost, Removed };

struct Track {
    std::int64_t id = 0;
    Mean mean = Mean::zeros();
    Covariance covariance = Covariance::zeros();
    State state = State::Tracked;
    bool confirmed = false;
    bool detection_updated = false;
    float score = 0;
    int class_id = -1;
    double start_time = 0;
    double last_detection_time = 0;

    cv::Rect2f bbox() const {
        const double width = std::max(1e-3, mean(2));
        const double height = std::max(1e-3, mean(3));
        return {static_cast<float>(mean(0) - width * 0.5),
                static_cast<float>(mean(1) - height * 0.5),
                static_cast<float>(width), static_cast<float>(height)};
    }
};

double axis_extent(const Mean& mean, int i) {
    return std::max(1e-3, mean((i % 2) == 0 ? 2 : 3));
}

void initiate(Track& track, const Detection& detection) {
    const auto& b = detection.bbox;
    track.mean(0) = b.x + b.width * 0.5;
    track.mean(1) = b.y + b.height * 0.5;
    track.mean(2) = b.width;
    track.mean(3) = b.height;
    for (int i = 0; i < 8; ++i) {
        const double stddev = axis_extent(track.mean, i) * (i < 4 ? 2.0 / 20.0 : 10.0 / 160.0);
        track.covariance(i, i) = stddev * stddev;
    }
}

// One upstream KF step at dt=1; fractional final steps use Q*dt. Repeated
// steps preserve the upstream discrete noise model when frames are skipped.
void predict_step(Track& track, double dt) {
    Covariance motion = Covariance::eye();
    Covariance noise = Covariance::zeros();
    for (int i = 0; i < 4; ++i) motion(i, i + 4) = dt;
    for (int i = 0; i < 8; ++i) {
        const double stddev = axis_extent(track.mean, i) * (i < 4 ? 1.0 / 20.0 : 1.0 / 160.0);
        noise(i, i) = stddev * stddev * dt;
    }
    track.mean = motion * track.mean;
    track.covariance = motion * track.covariance * motion.t() + noise;
    track.mean(2) = std::max(track.mean(2), 1e-3);
    track.mean(3) = std::max(track.mean(3), 1e-3);
}

void predict(Track& track, double elapsed_frames) {
    if (track.state != State::Tracked) {
        track.mean(6) = 0;
        track.mean(7) = 0;
    }
    while (elapsed_frames > 1.0) {
        predict_step(track, 1.0);
        elapsed_frames -= 1.0;
    }
    if (elapsed_frames > 0) predict_step(track, elapsed_frames);
}

void correct(Track& track, const Detection& detection, double timestamp) {
    cv::Matx<double, 4, 4> projected;
    cv::Matx<double, 8, 4> cross;
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) projected(i, j) = track.covariance(i, j);
        const double sigma = axis_extent(track.mean, i) / 20.0;
        projected(i, i) += sigma * sigma;
    }
    for (int i = 0; i < 8; ++i)
        for (int j = 0; j < 4; ++j) cross(i, j) = track.covariance(i, j);
    const auto gain = cross * projected.inv(cv::DECOMP_CHOLESKY);
    const auto& box = detection.bbox;
    cv::Matx<double, 4, 1> residual(box.x + box.width * 0.5 - track.mean(0),
                                     box.y + box.height * 0.5 - track.mean(1),
                                     box.width - track.mean(2), box.height - track.mean(3));
    track.mean += gain * residual;
    track.covariance -= gain * projected * gain.t();
    track.covariance = (track.covariance + track.covariance.t()) * 0.5;
    track.mean(2) = std::max(track.mean(2), 1e-3);
    track.mean(3) = std::max(track.mean(3), 1e-3);
    track.state = State::Tracked;
    track.confirmed = true;
    track.detection_updated = true;
    track.score = detection.score;
    track.last_detection_time = timestamp;
}

void compensate(Track& track, const cv::Matx23d& warp) {
    Covariance transform = Covariance::zeros();
    for (int block = 0; block < 4; ++block)
        for (int r = 0; r < 2; ++r)
            for (int c = 0; c < 2; ++c)
                transform(2 * block + r, 2 * block + c) = warp(r, c);
    track.mean = transform * track.mean;
    track.mean(0) += warp(0, 2);
    track.mean(1) += warp(1, 2);
    track.covariance = transform * track.covariance * transform.t();
    track.mean(2) = std::max(track.mean(2), 1e-3);
    track.mean(3) = std::max(track.mean(3), 1e-3);
}

double iou(const cv::Rect2f& a, const cv::Rect2f& b) {
    const double width = std::max(0.0f, std::min(a.x + a.width, b.x + b.width) - std::max(a.x, b.x));
    const double height = std::max(0.0f, std::min(a.y + a.height, b.y + b.height) - std::max(a.y, b.y));
    const double intersection = width * height;
    const double combined = a.area() + b.area() - intersection;
    return combined > 0 ? intersection / combined : 0;
}

struct Assignment {
    std::vector<std::pair<int, int>> matches;
    std::vector<int> unmatched_tracks;
    std::vector<int> unmatched_detections;
};

// Exact minimum-cost assignment with cost-limit dummies, equivalent objective
// to lapjv(extend_cost=true,cost_limit=threshold), not greedy IoU matching.
Assignment associate(const std::vector<Track*>& tracks,
                     const std::vector<const Detection*>& detections,
                     double threshold, bool fuse_score) {
    Assignment result;
    const int rows = static_cast<int>(tracks.size());
    const int columns = static_cast<int>(detections.size());
    const int size = rows + columns;
    if (rows == 0 || columns == 0) {
        result.unmatched_tracks.resize(rows);
        result.unmatched_detections.resize(columns);
        std::iota(result.unmatched_tracks.begin(), result.unmatched_tracks.end(), 0);
        std::iota(result.unmatched_detections.begin(), result.unmatched_detections.end(), 0);
        return result;
    }
    const double forbidden = 1e6;
    std::vector<std::vector<double>> costs(size, std::vector<double>(size, 0));
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            if (i < rows && j < columns) {
                double cost = forbidden;
                if (tracks[i]->class_id == detections[j]->class_id) {
                    const double similarity = iou(tracks[i]->bbox(), detections[j]->bbox);
                    cost = 1.0 - similarity * (fuse_score ? detections[j]->score : 1.0);
                }
                costs[i][j] = cost <= threshold ? cost : forbidden;
            } else if (i < rows || j < columns) {
                costs[i][j] = threshold * 0.5;
            }
        }
    }

    // Primal-dual Hungarian algorithm; dummy assignments allow unmatched
    // rows AND columns without changing the cost-limited objective.
    std::vector<double> u(size + 1), v(size + 1);
    std::vector<int> p(size + 1), way(size + 1);
    for (int i = 1; i <= size; ++i) {
        p[0] = i;
        int j0 = 0;
        std::vector<double> minimum(size + 1, std::numeric_limits<double>::infinity());
        std::vector<bool> used(size + 1, false);
        do {
            used[j0] = true;
            const int i0 = p[j0];
            int j1 = 0;
            double delta = std::numeric_limits<double>::infinity();
            for (int j = 1; j <= size; ++j) {
                if (used[j]) continue;
                const double reduced = costs[i0 - 1][j - 1] - u[i0] - v[j];
                if (reduced < minimum[j]) { minimum[j] = reduced; way[j] = j0; }
                if (minimum[j] < delta) { delta = minimum[j]; j1 = j; }
            }
            for (int j = 0; j <= size; ++j) {
                if (used[j]) { u[p[j]] += delta; v[j] -= delta; }
                else minimum[j] -= delta;
            }
            j0 = j1;
        } while (p[j0] != 0);
        do {
            const int j1 = way[j0];
            p[j0] = p[j1];
            j0 = j1;
        } while (j0 != 0);
    }
    std::vector<bool> used_tracks(rows, false), used_detections(columns, false);
    for (int j = 1; j <= columns; ++j) {
        const int i = p[j] - 1;
        if (i >= 0 && i < rows && costs[i][j - 1] <= threshold) {
            result.matches.emplace_back(i, j - 1);
            used_tracks[i] = true;
            used_detections[j - 1] = true;
        }
    }
    for (int i = 0; i < rows; ++i) if (!used_tracks[i]) result.unmatched_tracks.push_back(i);
    for (int j = 0; j < columns; ++j) if (!used_detections[j]) result.unmatched_detections.push_back(j);
    return result;
}

bool valid_detection(const Detection& d) {
    return std::isfinite(d.bbox.x) && std::isfinite(d.bbox.y) &&
           std::isfinite(d.bbox.width) && std::isfinite(d.bbox.height) &&
           d.bbox.width > 0 && d.bbox.height > 0 && std::isfinite(d.score) &&
           d.score >= 0 && d.score <= 1 && d.class_id >= 0;
}

class SparseGmc {
public:
    cv::Matx23d apply(const cv::Mat& input) {
        const cv::Matx23d identity(1, 0, 0, 0, 1, 0);
        cv::Mat gray, small;
        if (input.channels() == 3) cv::cvtColor(input, gray, cv::COLOR_BGR2GRAY);
        else if (input.channels() == 4) cv::cvtColor(input, gray, cv::COLOR_BGRA2GRAY);
        else gray = input;
        cv::resize(gray, small, cv::Size(std::max(1, gray.cols / 2), std::max(1, gray.rows / 2)));
        std::vector<cv::Point2f> points;
        cv::goodFeaturesToTrack(small, points, 1000, 0.01, 1, cv::noArray(), 3, false, 0.04);
        cv::Matx23d warp = identity;
        if (!previous_.empty() && previous_.size() == small.size() && previous_points_.size() > 4) {
            std::vector<cv::Point2f> next, before, after;
            std::vector<unsigned char> status;
            std::vector<float> error;
            cv::calcOpticalFlowPyrLK(previous_, small, previous_points_, next, status, error);
            for (std::size_t i = 0; i < status.size(); ++i) {
                if (status[i] && std::isfinite(next[i].x) && std::isfinite(next[i].y)) {
                    before.push_back(previous_points_[i]);
                    after.push_back(next[i]);
                }
            }
            if (before.size() > 4) {
                cv::Mat estimated = cv::estimateAffinePartial2D(before, after, cv::noArray(), cv::RANSAC);
                if (!estimated.empty() && cv::checkRange(estimated)) {
                    // Convert reduced-image transform to the exact original
                    // coordinate system, including odd source dimensions.
                    const double sx = static_cast<double>(input.cols) / small.cols;
                    const double sy = static_cast<double>(input.rows) / small.rows;
                    warp = cv::Matx23d(estimated.at<double>(0, 0), estimated.at<double>(0, 1) * sx / sy,
                                      estimated.at<double>(0, 2) * sx,
                                      estimated.at<double>(1, 0) * sy / sx, estimated.at<double>(1, 1),
                                      estimated.at<double>(1, 2) * sy);
                }
            }
        }
        previous_ = small;
        previous_points_ = std::move(points);
        return warp;
    }
private:
    cv::Mat previous_;
    std::vector<cv::Point2f> previous_points_;
};

} // namespace

struct BotSort::Impl {
    explicit Impl(const TrackerConfig& input) : config(input) {
        if (!std::isfinite(config.low_thresh) || !std::isfinite(config.high_thresh) ||
            !std::isfinite(config.new_track_thresh) || !std::isfinite(config.match_thresh) ||
            config.low_thresh < 0 || config.low_thresh >= config.high_thresh || config.high_thresh > 1 ||
            config.new_track_thresh < config.high_thresh || config.new_track_thresh > 1 ||
            config.match_thresh <= 0 || config.match_thresh > 1 ||
            !std::isfinite(config.track_buffer_seconds) || config.track_buffer_seconds <= 0 ||
            !std::isfinite(config.prediction_output_seconds) || config.prediction_output_seconds < 0 ||
            config.prediction_output_seconds > config.track_buffer_seconds ||
            !std::isfinite(config.nominal_fps) || config.nominal_fps <= 0 ||
            config.nominal_fps * config.track_buffer_seconds > 10000 ||
            (config.gmc_method != "none" && config.gmc_method != "sparseOptFlow")) {
            throw std::invalid_argument("Invalid BoT-SORT thresholds, time limits, nominal_fps or GMC method");
        }
    }

    std::vector<TrackResult> update(const std::vector<Detection>& detections,
                                    const cv::Mat& frame, double timestamp, TrackerTiming* timing) {
        if (timing) *timing = {};
        if (!std::isfinite(timestamp) || (initialized && timestamp <= last_time))
            throw std::invalid_argument("BoT-SORT requires finite, strictly increasing image timestamps");
        if (config.gmc_method == "sparseOptFlow" &&
            (frame.empty() || frame.depth() != CV_8U ||
             (frame.channels() != 1 && frame.channels() != 3 && frame.channels() != 4)))
            throw std::invalid_argument("sparseOptFlow GMC requires a nonempty 8-bit gray/BGR/BGRA frame");
        for (const auto& detection : detections)
            if (!valid_detection(detection)) throw std::invalid_argument("Invalid BoT-SORT detection");

        const double elapsed_frames = initialized ? (timestamp - last_time) * config.nominal_fps : 0;
        const bool first_frame = !initialized;
        const auto gmc_start = std::chrono::steady_clock::now();
        const auto warp = config.gmc_method == "none" ? cv::Matx23d(1, 0, 0, 0, 1, 0) : gmc.apply(frame);
        const auto gmc_end = std::chrono::steady_clock::now();
        initialized = true;
        last_time = timestamp;
        for (auto& track : tracks) {
            track.detection_updated = false;
            // Expire before matching, including a long capture gap. An old
            // result must not revive an ID after its configured time limit.
            if (timestamp - track.last_detection_time > config.track_buffer_seconds)
                track.state = State::Removed;
        }
        std::vector<Track*> pool, unconfirmed;
        for (auto& track : tracks) {
            if (track.state == State::Removed) continue;
            if (track.state == State::Tracked && !track.confirmed) unconfirmed.push_back(&track);
            else pool.push_back(&track);
        }
        for (auto* track : pool) { predict(*track, elapsed_frames); compensate(*track, warp); }
        for (auto* track : unconfirmed) compensate(*track, warp);

        std::vector<const Detection*> high, low;
        for (const auto& detection : detections) {
            if (detection.score > config.high_thresh) high.push_back(&detection);
            else if (detection.score > config.low_thresh && detection.score < config.high_thresh)
                low.push_back(&detection);
        }
        const auto first = associate(pool, high, config.match_thresh, true);
        for (const auto& match : first.matches) correct(*pool[match.first], *high[match.second], timestamp);

        std::vector<Track*> remaining_tracked;
        for (int index : first.unmatched_tracks)
            if (pool[index]->state == State::Tracked) remaining_tracked.push_back(pool[index]);
        const auto second = associate(remaining_tracked, low, 0.5, false);
        for (const auto& match : second.matches)
            correct(*remaining_tracked[match.first], *low[match.second], timestamp);
        for (int index : second.unmatched_tracks) remaining_tracked[index]->state = State::Lost;

        std::vector<const Detection*> remaining_high;
        for (int index : first.unmatched_detections) remaining_high.push_back(high[index]);
        const auto tentative = associate(unconfirmed, remaining_high, 0.7, true);
        for (const auto& match : tentative.matches)
            correct(*unconfirmed[match.first], *remaining_high[match.second], timestamp);
        for (int index : tentative.unmatched_tracks) unconfirmed[index]->state = State::Removed;

        // No pointers into tracks are used after push_back (may reallocate).
        for (int index : tentative.unmatched_detections) {
            const auto& detection = *remaining_high[index];
            if (detection.score < config.new_track_thresh) continue;
            Track track;
            track.id = next_id++;
            track.confirmed = first_frame;
            track.detection_updated = true;
            track.score = detection.score;
            track.class_id = detection.class_id;
            track.start_time = timestamp;
            track.last_detection_time = timestamp;
            initiate(track, detection);
            tracks.push_back(track);
        }

        // Upstream removes tracked/lost duplicates with IoU > 0.85 and keeps
        // the longer-lived track. Collect removals before mutating states.
        std::vector<bool> duplicate(tracks.size(), false);
        for (std::size_t i = 0; i < tracks.size(); ++i) {
            if (tracks[i].state != State::Tracked) continue;
            for (std::size_t j = 0; j < tracks.size(); ++j) {
                if (tracks[j].state != State::Lost || tracks[i].class_id != tracks[j].class_id) continue;
                if (iou(tracks[i].bbox(), tracks[j].bbox()) <= 0.85) continue;
                const double age_i = tracks[i].last_detection_time - tracks[i].start_time;
                const double age_j = tracks[j].last_detection_time - tracks[j].start_time;
                duplicate[age_i > age_j ? j : i] = true;
            }
        }
        for (std::size_t i = 0; i < tracks.size(); ++i) if (duplicate[i]) tracks[i].state = State::Removed;
        tracks.erase(std::remove_if(tracks.begin(), tracks.end(), [](const Track& track) {
            return track.state == State::Removed;
        }), tracks.end());

        std::vector<TrackResult> output;
        for (const auto& track : tracks) {
            if (!track.confirmed) continue;
            const double since_update = timestamp - track.last_detection_time;
            if (!track.detection_updated && since_update > config.prediction_output_seconds) continue;
            output.push_back({track.id, track.bbox(), track.score, track.class_id,
                              track.detection_updated, since_update});
        }
        std::sort(output.begin(), output.end(), [](const TrackResult& a, const TrackResult& b) {
            return a.track_id < b.track_id;
        });
        if (timing) {
            timing->gmc_ms = std::chrono::duration<double, std::milli>(gmc_end - gmc_start).count();
            timing->association_ms = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - gmc_end).count();
        }
        return output;
    }

    TrackerConfig config;
    std::vector<Track> tracks;
    SparseGmc gmc;
    std::int64_t next_id = 1;
    bool initialized = false;
    double last_time = 0;
};

BotSort::BotSort(const TrackerConfig& config) : impl_(new Impl(config)) {}
BotSort::~BotSort() = default;
BotSort::BotSort(BotSort&&) noexcept = default;
BotSort& BotSort::operator=(BotSort&&) noexcept = default;
std::vector<TrackResult> BotSort::update(const std::vector<Detection>& detections,
                                       const cv::Mat& frame, double image_time_seconds, TrackerTiming* timing) {
    return impl_->update(detections, frame, image_time_seconds, timing);
}

} // namespace aerial
