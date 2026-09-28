#pragma once
#include "aerial/pipeline.hpp"
#include "aerial/los.hpp"
#include "aerial/bot_sort.hpp"
#include <thread>
#include <atomic>

namespace aerial {
struct ResultPacket {
    DetectionPacket detection;
    std::vector<TrackResult> tracks;
    std::vector<LosResult> los;
    bool calibrated = false;
    double tracking_ms = 0, los_ms = 0;
    double completion_queue_ms = 0;
    TrackerTiming tracker_timing;
    std::int64_t ready_ns = 0;
};
std::string result_json(const ResultPacket& result);

class JsonPublisher {
public:
    JsonPublisher(const std::string& path, bool realtime, const std::atomic<bool>* cancel = nullptr);
    ~JsonPublisher();
    bool publish(std::string record);
    void finish();
    std::uint64_t dropped() const { return slot_.dropped(); }
    bool failed() const { return failed_; }
private:
    void run();
    int fd_ = -1;
    FrameSlot<std::string> slot_;
    std::thread thread_;
    std::atomic<bool> failed_{false};
    std::atomic<std::int64_t> closing_ns_{0};
    const std::atomic<bool>* cancel_;
};

// Optional display/encoding consumer with a one-result mailbox, never on the
// perception thread. It receives results only after LOS has been made available.
class Preview {
public:
    Preview(bool window, std::string pipeline, double fps);
    ~Preview();
    void publish(const ResultPacket& result);
    void finish();
    std::uint64_t dropped() const { return slot_.dropped(); }
private:
    void run();
    bool window_;
    std::string pipeline_;
    double fps_;
    FrameSlot<ResultPacket> slot_{true};
    std::thread thread_;
};
}
