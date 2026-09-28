#include "aerial/output.hpp"
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <cerrno>
#include <cmath>
#include <cstring>
#include <fcntl.h>
#include <iomanip>
#include <iostream>
#include <locale>
#include <poll.h>
#include <sstream>
#include <stdexcept>
#include <unistd.h>

namespace aerial {
std::string result_json(const ResultPacket& r) {
    const auto& f = r.detection.frame;
    std::ostringstream s;
    s.imbue(std::locale::classic());
    s << std::setprecision(12) << "{\"frame_id\":" << f.id
      << ",\"image_timestamp_ns\":" << f.image_timestamp_ns
      << ",\"image_timestamp_domain\":\"" << f.timestamp_domain
      << "\",\"received_monotonic_ns\":" << f.received_ns
      << ",\"result_ready_monotonic_ns\":" << r.ready_ns
      << ",\"image_size\":[" << f.image.cols << ',' << f.image.rows << ']'
      << ",\"calibrated\":" << (r.calibrated ? "true" : "false")
      << ",\"capture_latency_measured\":false,\"source_monotonic_ns\":";
    if (f.source_monotonic_ns > 0) s << f.source_monotonic_ns;
    else s << "null";
    s << ",\"latency_ms\":{\"receiver_to_result\":" << (r.ready_ns - f.received_ns) / 1e6
      << ",\"source_to_result_estimate\":";
    if (f.source_monotonic_ns > 0) s << (r.ready_ns - f.source_monotonic_ns) / 1e6;
    else s << "null";
    s << ",\"queue\":" << (r.detection.inference_start_ns - f.received_ns) / 1e6
      << ",\"completion_queue\":" << r.completion_queue_ms
      << ",\"preprocess\":" << r.detection.timing.preprocess_ms
      << ",\"inference\":" << r.detection.timing.inference_ms
      << ",\"input_submit\":" << r.detection.timing.input_submit_ms
      << ",\"rknn_run\":" << r.detection.timing.rknn_run_ms
      << ",\"output_get\":" << r.detection.timing.output_get_ms
      << ",\"postprocess\":" << r.detection.timing.postprocess_ms
      << ",\"tracking\":" << r.tracking_ms
      << ",\"tracking_gmc\":" << r.tracker_timing.gmc_ms
      << ",\"tracking_association\":" << r.tracker_timing.association_ms
      << ",\"los\":" << r.los_ms << "},\"targets\":[";
    for (std::size_t i = 0; i < r.tracks.size(); ++i) {
        if (i) s << ',';
        const auto& t = r.tracks[i];
        s << "{\"track_id\":" << t.track_id << ",\"class_id\":" << t.class_id
          << ",\"score\":" << t.score << ",\"bbox_xyxy\":[" << t.bbox.x << ',' << t.bbox.y
          << ',' << t.bbox.x + t.bbox.width << ',' << t.bbox.y + t.bbox.height << ']'
          << ",\"detection_updated\":" << (t.detection_updated ? "true" : "false")
          << ",\"observation\":\"" << (t.detection_updated ? "detection_update" : "prediction")
          << "\",\"seconds_since_update\":" << t.seconds_since_update
          << ",\"los_valid\":" << (r.calibrated && i < r.los.size() && r.los[i].valid ? "true" : "false")
          << ",\"los\":";
        if (!r.calibrated || i >= r.los.size() || !r.los[i].valid) s << "null";
        else {
            const auto& l = r.los[i];
            s << "{\"unit_vector_camera\":[" << l.unit_vector[0] << ',' << l.unit_vector[1] << ',' << l.unit_vector[2]
              << "],\"azimuth_rad\":" << l.azimuth_rad << ",\"elevation_rad\":" << l.elevation_rad << '}';
        }
        s << '}';
    }
    s << "]}\n";
    return s.str();
}

JsonPublisher::JsonPublisher(const std::string& path, bool realtime, const std::atomic<bool>* cancel)
    : slot_(realtime), cancel_(cancel) {
    if (path == "-") {
        // RKNN prints diagnostics on stdout. Preserve the original destination for
        // JSON, then route runtime printf() diagnostics to stderr.
        fd_ = dup(STDOUT_FILENO);
        if (fd_ < 0 || dup2(STDERR_FILENO, STDOUT_FILENO) < 0) {
            const std::string message = std::string("Cannot route JSON stdout: ") + std::strerror(errno);
            if (fd_ >= 0) close(fd_);
            fd_ = -1;
            throw std::runtime_error(message);
        }
    } else {
        // Set nonblocking at open, not just afterwards: opening a FIFO without
        // an attached reader would otherwise block application startup forever.
        fd_ = open(path.c_str(), O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC | O_NONBLOCK, 0644);
    }
    if (fd_ < 0) throw std::runtime_error("Cannot open JSON output: " + path + ": " + std::strerror(errno));
    const int flags = fcntl(fd_, F_GETFL);
    if (flags < 0 || fcntl(fd_, F_SETFL, flags | O_NONBLOCK) < 0) {
        const std::string message = std::string("Cannot make JSON output nonblocking: ") + std::strerror(errno);
        close(fd_);
        fd_ = -1;
        throw std::runtime_error(message);
    }
    try { thread_ = std::thread(&JsonPublisher::run, this); }
    catch (...) { close(fd_); fd_ = -1; throw; }
}
JsonPublisher::~JsonPublisher() { finish(); if (fd_ >= 0) close(fd_); }
bool JsonPublisher::publish(std::string record) { return !failed_ && slot_.put(std::move(record), cancel_); }
void JsonPublisher::finish() {
    if (!closing_ns_) closing_ns_ = monotonic_ns();
    slot_.close();
    if (thread_.joinable()) thread_.join();
}
void JsonPublisher::run() {
    std::string record;
    while (slot_.get(record)) {
        // Separate compute completion from availability at the output boundary.
        // A consumer can also timestamp complete-record receipt on its own clock.
        record.insert(1, "\"publish_started_monotonic_ns\":" + std::to_string(monotonic_ns()) + ",");
        std::size_t offset = 0;
        while (offset < record.size()) {
            const auto n = write(fd_, record.data() + offset, record.size() - offset);
            if (n > 0) { offset += n; continue; }
            if (n < 0 && errno == EINTR) continue;
            if (n < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) {
                if (closing_ns_ && monotonic_ns() - closing_ns_ > 2000000000LL) {
                    std::cerr << "JSON consumer stalled at shutdown; output may end with a partial record\n";
                    failed_ = true; slot_.close(); return;
                }
                pollfd p{fd_, POLLOUT, 0};
                poll(&p, 1, 50);
                continue;
            }
            std::cerr << "JSON output failed: " << std::strerror(errno) << '\n';
            failed_ = true; slot_.close(); return;
        }
    }
}

Preview::Preview(bool window, std::string pipeline, double fps)
    : window_(window), pipeline_(std::move(pipeline)), fps_(fps), thread_(&Preview::run, this) {}
Preview::~Preview() { finish(); }
void Preview::publish(const ResultPacket& result) { slot_.put(result); }
void Preview::finish() { slot_.close(); if (thread_.joinable()) thread_.join(); }
void Preview::run() {
    try {
        cv::VideoWriter writer;
        ResultPacket result;
        while (slot_.get(result)) {
            auto image = result.detection.frame.image.clone();
            // A blocked encoder must not retain an active decoder buffer.
            result.detection.frame = Frame{};
            for (const auto& t : result.tracks) {
                auto color = t.detection_updated ? cv::Scalar(0,255,0) : cv::Scalar(0,165,255);
                cv::rectangle(image, t.bbox, color, 2);
                cv::putText(image, std::to_string(t.track_id) + (t.detection_updated ? "" : " predicted"),
                    cv::Point(int(t.bbox.x), std::max(15, int(t.bbox.y) - 5)), cv::FONT_HERSHEY_SIMPLEX, .5, color, 1);
            }
            if (window_) { cv::imshow("YOLOv8 BoT-SORT", image); cv::waitKey(1); }
            if (!pipeline_.empty()) {
                if (!writer.isOpened() && !writer.open(pipeline_, cv::CAP_GSTREAMER, 0, fps_, image.size(), true))
                    throw std::runtime_error("Cannot open optional video output pipeline");
                writer.write(image); // Backpressure is confined to this consumer.
            }
        }
        if (window_) cv::destroyWindow("YOLOv8 BoT-SORT");
    } catch (const std::exception& e) { std::cerr << "Preview disabled: " << e.what() << '\n'; slot_.close(); }
}
}
