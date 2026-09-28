#include "aerial/capture.hpp"
#include "aerial/bot_sort.hpp"
#include "aerial/output.hpp"
#include <cassert>
#include <iostream>
#include <limits>

int main(int argc, char** argv) {
    using namespace aerial;
    CaptureConfig c;
    c.type = "pipeline";
    c.pipeline = "videotestsrc num-buffers=5 pattern=ball ! video/x-raw,format=BGR,width=320,height=240,framerate=30/1 ! appsink name=frames";
    GstCapture input(c);
    std::atomic<bool> stop{false};
    Frame first, next;
    assert(input.read(first, stop));
    assert(first.id == 0 && first.image_timestamp_ns == 0 && first.timestamp_domain == "stream_pts");
    assert(first.source_monotonic_ns == 0); // unpaced file/test data is not capture wall-clock time
    auto snapshot = first.image.clone();
    for (unsigned i = 1; i < 5; ++i) {
        assert(input.read(next, stop));
        assert(next.id == i && next.image_timestamp_ns > first.image_timestamp_ns);
        assert(next.image.size() == cv::Size(320,240));
    }
    assert(!input.read(next, stop));
    assert(cv::norm(first.image, snapshot, cv::NORM_INF) == 0); // mapped buffer retained across later frames

    // End-to-end mathematical fixture: synthetic detection + explicit synthetic
    // calibration, not an accuracy or hardware-NPU benchmark.
    CameraCalibration calibration;
    calibration.calibrated = true;
    calibration.image_size = {320,240};
    calibration.camera_matrix = {200,0,160, 0,200,120, 0,0,1};
    calibration.distortion_coefficients = {0,0,0,0,0};
    LosProjector projector(calibration);
    TrackerConfig tc;
    tc.gmc_method = "none";
    BotSort tracker(tc);
    Detection d;
    d.bbox = {150,110,20,20}; d.score = .9f; d.class_id = 0;
    ResultPacket result;
    result.detection.frame = first;
    result.detection.inference_start_ns = first.received_ns;
    result.tracks = tracker.update({d}, first.image, 0);
    assert(result.tracks.size() == 1 && result.tracks[0].detection_updated);
    result.los = {projector.project(result.tracks[0].bbox, first.image.size())};
    result.calibrated = true;
    result.ready_ns = monotonic_ns();
    assert(result.los[0].valid && std::abs(result.los[0].azimuth_rad) < 1e-9);
    const auto json = result_json(result);
    assert(json.find("\"calibrated\":true") != std::string::npos);
    assert(json.find("\"unit_vector_camera\":[0,0,1]") != std::string::npos);
    result.tracks = tracker.update({}, first.image, 1.0 / 30);
    assert(result.tracks.size() == 1 && !result.tracks[0].detection_updated);
    result.calibrated = false; result.los.clear();
    assert(result_json(result).find("\"los\":null") != std::string::npos);
    std::cout << "Capture PTS/ownership/EOS and synthetic tracking-to-LOS integration passed\n";
    if (argc == 2) {
        CaptureConfig file;
        file.path = argv[1];
        GstCapture video(file);
        std::int64_t last = -1;
        for (int i = 0; i < 5; ++i) {
            assert(video.read(next, stop));
            assert(next.image_timestamp_ns > last);
            last = next.image_timestamp_ns;
        }
        std::cout << "Real file decoding/PTS smoke passed: " << next.image.cols << 'x' << next.image.rows << '\n';
    }
}
