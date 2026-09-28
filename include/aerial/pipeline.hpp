#pragma once

#include "aerial/types.hpp"
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <string>

namespace aerial {
inline std::int64_t monotonic_ns() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

struct Frame {
    std::uint64_t id = 0;
    cv::Mat image;
    // Keeps mapped GstSample memory alive; consumers must not modify image.
    std::shared_ptr<void> owner;
    std::int64_t image_timestamp_ns = 0;
    std::string timestamp_domain;
    std::int64_t received_ns = 0;
    std::int64_t source_monotonic_ns = 0; // 0 if source PTS cannot be mapped
    bool source_time_estimated = true;   // never claim sensor exposure time
    double tracking_time_seconds = 0;
};

// Exactly one pending item. Live mode replaces old pending work; file mode
// backpressures the producer to preserve every frame. close() wakes both sides.
template<class T> class FrameSlot {
public:
    explicit FrameSlot(bool latest) : latest_(latest) {}
    bool put(T value, const std::atomic<bool>* cancel = nullptr) {
        std::unique_lock<std::mutex> lock(mutex_);
        if (!latest_) {
            if (cancel) {
                while (!closed_ && value_ && !cancel->load())
                    cv_.wait_for(lock, std::chrono::milliseconds(50));
            } else cv_.wait(lock, [&]{ return closed_ || !value_; });
        }
        if (closed_ || (cancel && cancel->load())) return false;
        if (value_) ++dropped_;
        value_ = std::move(value);
        cv_.notify_all();
        return true;
    }
    bool get(T& value) {
        std::unique_lock<std::mutex> lock(mutex_);
        cv_.wait(lock, [&]{ return closed_ || value_.has_value(); });
        if (!value_) return false;
        value = std::move(*value_);
        value_.reset();
        cv_.notify_all();
        return true;
    }
    void close() {
        std::lock_guard<std::mutex> lock(mutex_);
        closed_ = true;
        cv_.notify_all();
    }
    std::uint64_t dropped() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return dropped_;
    }
private:
    bool latest_, closed_ = false;
    mutable std::mutex mutex_;
    std::condition_variable cv_;
    std::optional<T> value_;
    std::uint64_t dropped_ = 0;
};

struct DetectionPacket {
    Frame frame;
    std::vector<Detection> detections;
    StageTiming timing;
    std::int64_t inference_start_ns = 0;
    std::int64_t detection_ready_ns = 0;
    bool expired = false;
};

// Live completions are consumed in strictly increasing frame order without
// waiting for an older slow inference. File mode preserves all frames in order.
class CompletionQueue {
public:
    CompletionQueue(bool latest, unsigned workers) : latest_(latest), limit_(workers) {}
    bool put(DetectionPacket value) {
        std::unique_lock<std::mutex> lock(mutex_);
        const auto id = value.frame.id;
        if (latest_) {
            if (have_high_water_ && id <= high_water_) { ++dropped_; return true; }
            if (!values_.empty()) ++dropped_;
            values_.clear();
            high_water_ = id;
            have_high_water_ = true;
        } else {
            // Admit the required frame even when faster later frames filled the
            // bounded map. This prevents a reorder-buffer deadlock.
            cv_.wait(lock, [&]{ return closed_ || values_.size() < limit_ || id == next_; });
        }
        if (closed_) return false;
        values_.emplace(id, std::move(value));
        cv_.notify_all();
        return true;
    }
    bool get(DetectionPacket& value) {
        std::unique_lock<std::mutex> lock(mutex_);
        cv_.wait(lock, [&]{ return closed_ || (latest_ ? !values_.empty() : values_.count(next_) != 0); });
        auto it = latest_ ? values_.begin() : values_.find(next_);
        if (it == values_.end()) return false;
        value = std::move(it->second);
        values_.erase(it);
        if (!latest_) ++next_;
        cv_.notify_all();
        return true;
    }
    void close() {
        std::lock_guard<std::mutex> lock(mutex_);
        closed_ = true;
        cv_.notify_all();
    }
    std::uint64_t dropped() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return dropped_;
    }
private:
    bool latest_, closed_ = false, have_high_water_ = false;
    unsigned limit_;
    std::uint64_t next_ = 0, high_water_ = 0, dropped_ = 0;
    std::mutex mutable mutex_;
    std::condition_variable cv_;
    std::map<std::uint64_t, DetectionPacket> values_;
};
} // namespace aerial
