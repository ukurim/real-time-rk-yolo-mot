#include "aerial/capture.hpp"
#include <gst/app/gstappsink.h>
#include <gst/video/video.h>
#include <gst/gst.h>
#include <algorithm>
#include <filesystem>
#include <iostream>
#include <stdexcept>

namespace aerial {
namespace {
std::string quote(const std::string& value) {
    gchar* escaped = g_strescape(value.c_str(), nullptr);
    std::string result = std::string("\"") + escaped + "\"";
    g_free(escaped);
    return result;
}
struct SampleOwner {
    GstSample* sample = nullptr;
    GstVideoFrame mapped{};
    bool is_mapped = false;
    ~SampleOwner() {
        if (is_mapped) gst_video_frame_unmap(&mapped);
        if (sample) gst_sample_unref(sample);
    }
};
}
struct GstCapture::Impl {
    CaptureConfig config;
    GstElement* pipeline = nullptr;
    GstAppSink* sink = nullptr;
    GstBus* bus = nullptr;
    std::uint64_t sequence = 0;
    std::int64_t previous_pts = -1;
    int time_mode = 0; // fixed per stream: 1=PTS, 2=receive monotonic
    cv::Size frame_size;
    ~Impl() {
        if (pipeline) gst_element_set_state(pipeline, GST_STATE_NULL);
        if (bus) gst_object_unref(bus);
        if (sink) gst_object_unref(sink);
        if (pipeline) gst_object_unref(pipeline);
    }
    void check_error() {
        GstMessage* msg = gst_bus_pop_filtered(bus, GST_MESSAGE_ERROR);
        if (!msg) return;
        GError* error = nullptr;
        gchar* debug = nullptr;
        gst_message_parse_error(msg, &error, &debug);
        std::string message = error ? error->message : "GStreamer error";
        if (debug) message += std::string(" (") + debug + ")";
        g_clear_error(&error);
        g_free(debug);
        gst_message_unref(msg);
        throw std::runtime_error(message);
    }
};

GstCapture::GstCapture(const CaptureConfig& config) : impl_(new Impl) {
    gst_init(nullptr, nullptr);
    impl_->config = config;
    std::string pipeline;
    if (config.type == "pipeline") {
        pipeline = config.pipeline; // Caller supplies source through BGR appsink named frames.
    } else {
        if (config.type == "file") {
            const auto path = std::filesystem::absolute(config.path).string();
            if (!std::filesystem::is_regular_file(path)) throw std::runtime_error("Input file not found: " + path);
            if (config.decoder == "mpp") {
                pipeline = "filesrc location=" + quote(path) + " ! qtdemux ! h264parse ! mppvideodec format=BGR";
            } else {
                gchar* uri = gst_filename_to_uri(path.c_str(), nullptr);
                if (!uri) throw std::runtime_error("Cannot construct file URI");
                pipeline = "uridecodebin uri=" + quote(uri);
                g_free(uri);
            }
        } else if (config.type == "camera") {
            pipeline = "v4l2src device=" + quote(config.path) + " do-timestamp=true ! image/jpeg,width=" +
                std::to_string(config.width) + ",height=" + std::to_string(config.height) +
                ",framerate=" + std::to_string(config.fps) + "/1 ! " +
                (config.decoder == "mpp" ? "mppjpegdec format=BGR" : "jpegdec");
        } else throw std::runtime_error("source.type must be file, camera, or pipeline");
        // MPP's format property uses its hardware conversion path. Software
        // videoconvert needs parallel conversion to keep up with 1080p60 here.
        if (config.decoder != "mpp") pipeline += " ! videoconvert n-threads=4";
        pipeline += " ! video/x-raw,format=BGR ! appsink name=frames";
    }
    GError* error = nullptr;
    impl_->pipeline = gst_parse_launch(pipeline.c_str(), &error);
    if (error) {
        std::string message = error->message;
        g_clear_error(&error);
        throw std::runtime_error("Input pipeline: " + message);
    }
    if (!impl_->pipeline || !GST_IS_BIN(impl_->pipeline)) throw std::runtime_error("Input must be a GStreamer pipeline");
    GstElement* sink = gst_bin_get_by_name(GST_BIN(impl_->pipeline), "frames");
    if (!sink || !GST_IS_APP_SINK(sink)) {
        if (sink) gst_object_unref(sink);
        throw std::runtime_error("Input pipeline requires appsink name=frames with BGR caps");
    }
    impl_->sink = GST_APP_SINK(sink);
    GstCaps* caps = gst_caps_from_string("video/x-raw,format=BGR");
    gst_app_sink_set_caps(impl_->sink, caps);
    gst_caps_unref(caps);
    gst_app_sink_set_max_buffers(impl_->sink, 1);
    gst_app_sink_set_drop(impl_->sink, config.realtime);
    gst_app_sink_set_emit_signals(impl_->sink, false);
    // Pace file replay when requested; a live camera should not wait on a sink clock.
    g_object_set(sink, "sync", config.realtime && config.type == "file", "wait-on-eos", FALSE,
                 "enable-last-sample", FALSE, nullptr);
    impl_->bus = gst_element_get_bus(impl_->pipeline);
    if (gst_element_set_state(impl_->pipeline, GST_STATE_PLAYING) == GST_STATE_CHANGE_FAILURE) {
        impl_->check_error();
        throw std::runtime_error("Cannot start input pipeline");
    }
    std::cerr << "Input: " << pipeline << " (appsink max-buffers=1, drop=" << config.realtime << ")\n";
}
GstCapture::~GstCapture() = default;

bool GstCapture::read(Frame& frame, const std::atomic<bool>& stop) {
    const auto deadline = monotonic_ns() + std::int64_t(impl_->config.timeout_ms) * 1000000;
    while (!stop) {
        if (monotonic_ns() > deadline) throw std::runtime_error("Input timed out waiting for a fresh timestamped frame");
        impl_->check_error();
        GstSample* sample = gst_app_sink_try_pull_sample(impl_->sink, 100 * GST_MSECOND);
        if (!sample) {
            impl_->check_error();
            if (gst_app_sink_is_eos(impl_->sink)) return false;
            if (monotonic_ns() > deadline) throw std::runtime_error("Input timed out waiting for a frame");
            continue;
        }
        const auto received_ns = monotonic_ns();
        auto owner = std::make_shared<SampleOwner>();
        owner->sample = sample;
        GstVideoInfo info;
        GstCaps* caps = gst_sample_get_caps(sample);
        GstBuffer* buffer = gst_sample_get_buffer(sample);
        if (!caps || !buffer || !gst_video_info_from_caps(&info, caps) ||
            GST_VIDEO_INFO_FORMAT(&info) != GST_VIDEO_FORMAT_BGR)
            throw std::runtime_error("Input is not a valid BGR video frame");
        const cv::Size size(GST_VIDEO_INFO_WIDTH(&info), GST_VIDEO_INFO_HEIGHT(&info));
        if (impl_->frame_size.area() != 0 && impl_->frame_size != size)
            throw std::runtime_error("Input dimensions changed; restart tracking with matching calibration");
        impl_->frame_size = size;
        if (!gst_video_frame_map(&owner->mapped, &info, buffer, GST_MAP_READ))
            throw std::runtime_error("Cannot map decoded frame");
        owner->is_mapped = true;
        frame = Frame{};
        frame.received_ns = received_ns;
        frame.image = cv::Mat(GST_VIDEO_INFO_HEIGHT(&info), GST_VIDEO_INFO_WIDTH(&info), CV_8UC3,
            GST_VIDEO_FRAME_PLANE_DATA(&owner->mapped, 0), GST_VIDEO_FRAME_PLANE_STRIDE(&owner->mapped, 0));
        frame.owner = owner;
        GstClockTime pts = GST_BUFFER_PTS(buffer);
        const bool have_pts = GST_CLOCK_TIME_IS_VALID(pts) && pts <= std::uint64_t(INT64_MAX);
        if (impl_->time_mode == 0) impl_->time_mode = have_pts ? 1 : 2;
        if (impl_->time_mode == 1) {
            if (!have_pts) throw std::runtime_error("Source lost PTS; refusing to switch tracker time domains");
            // A seek/reset needs a fresh tracker and clock mapping; never silently
            // mix epochs or spend minutes waiting for the previous PTS to recur.
            if (std::int64_t(pts) < impl_->previous_pts)
                throw std::runtime_error("Source PTS moved backwards; restart the stream/tracker for a new epoch");
            if (std::int64_t(pts) == impl_->previous_pts) continue;
            impl_->previous_pts = pts;
            frame.image_timestamp_ns = pts;
            frame.timestamp_domain = "stream_pts";
            frame.tracking_time_seconds = double(pts) / 1e9;
            const GstSegment* segment = gst_sample_get_segment(sample);
            GstClockTime running = segment ? gst_segment_to_running_time(segment, GST_FORMAT_TIME, pts) : GST_CLOCK_TIME_NONE;
            GstClock* clock = gst_element_get_clock(impl_->pipeline);
            if (clock && GST_CLOCK_TIME_IS_VALID(running)) {
                const auto clock_now = gst_clock_get_time(clock);
                const auto base = gst_element_get_base_time(impl_->pipeline);
                const auto mono_now = monotonic_ns();
                if (GST_CLOCK_TIME_IS_VALID(base) && base <= clock_now && running <= clock_now - base) {
                    frame.source_monotonic_ns = mono_now - std::int64_t(clock_now - base - running);
                }
            }
            if (clock) gst_object_unref(clock);
        } else {
            if (!impl_->config.realtime)
                throw std::runtime_error("Offline input requires source PTS; receiver time is not media time");
            frame.image_timestamp_ns = frame.received_ns;
            frame.timestamp_domain = "receiver_monotonic_no_source_pts";
            frame.tracking_time_seconds = double(frame.received_ns) / 1e9;
        }
        // An unpaced recording has media time, not a measurable capture wall time.
        if (!impl_->config.realtime) frame.source_monotonic_ns = 0;
        frame.id = impl_->sequence++;
        return true;
    }
    return false;
}
} // namespace aerial
