#include "aerial/yolo_decode.hpp"
#include <opencv2/imgproc.hpp>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <functional>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace {
using namespace aerial;
bool near(float a, float b, float tolerance = 0.002f) { return std::abs(a-b) <= tolerance; }
void throws(const std::function<void()>& operation) {
    bool caught = false;
    try { operation(); } catch (const std::exception&) { caught = true; }
    assert(caught);
}

struct Fixture {
    int per_branch;
    std::vector<std::vector<float>> storage;
    std::vector<YoloTensor> tensors;
    explicit Fixture(bool score_sum = false) : per_branch(score_sum ? 3 : 2) {
        storage.resize(3*per_branch);
        tensors.resize(3*per_branch);
        for (int branch = 0; branch < 3; ++branch) {
            const int grid = 4 >> branch;
            for (int role = 0; role < per_branch; ++role) {
                const int index = branch*per_branch+role;
                auto& t = tensors[index];
                t.channels = role == 0 ? 64 : role == 1 ? 3 : 1;
                t.width = t.height = grid;
                storage[index].assign(t.channels*grid*grid, role == 0 ? -8.f : 0.f);
                if (role == 0) {
                    for (int side = 0; side < 4; ++side)
                        for (int cell = 0; cell < grid*grid; ++cell)
                            storage[index][(side*16+1)*grid*grid+cell] = 8.f;
                }
                t.data = storage[index].data();
                t.bytes = storage[index].size()*sizeof(float);
            }
        }
    }
    void score(int branch, int cell, int cls, float confidence) {
        const int index = branch*per_branch+1;
        const int plane = tensors[index].width*tensors[index].height;
        storage[index][cls*plane+cell] = confidence;
        if (per_branch == 3) storage[index+1][cell] = confidence;
    }
};

void test_letterbox() {
    const auto t = letterbox_transform({1920,1080},{640,640});
    assert(t.resized_size == cv::Size(640,360));
    assert(t.left == 0 && t.top == 140);
    const cv::Rect2f restored = t.to_original({64,176,320,180});
    assert(near(restored.x,192) && near(restored.y,108));
    assert(near(restored.width,960) && near(restored.height,540));
    const auto odd = letterbox_transform({641,359},{640,640});
    assert(odd.resized_size == cv::Size(640,358));
    assert(odd.top == 141);
    const cv::Rect2f full = odd.to_original({0,141,640,358});
    assert(near(full.width,641) && near(full.height,359));
    const auto asymmetric = letterbox_transform({5,2},{32,32});
    assert(asymmetric.resized_size == cv::Size(32,13) && asymmetric.top == 9);
    // Source can be a non-contiguous ROI, as happens with cropped camera buffers.
    cv::Mat parent(6,10,CV_8UC3,cv::Scalar(10,20,30));
    cv::Mat roi = parent(cv::Rect(1,1,5,2));
    assert(!roi.isContinuous());
    cv::Mat rgb,scratch;
    const auto actual = letterbox_rgb(roi,{32,32},rgb,scratch);
    assert(rgb.isContinuous() && rgb.size() == cv::Size(32,32));
    assert(rgb.at<cv::Vec3b>(0,0) == cv::Vec3b(114,114,114));
    assert(rgb.at<cv::Vec3b>(actual.top,actual.left) == cv::Vec3b(30,20,10));
    throws([] { letterbox_transform({0,1080},{640,640}); });
    throws([&] { letterbox_rgb(cv::Mat(2,2,CV_8UC1),{32,32},rgb,scratch); });
}

void test_float_decode_and_nms() {
    Fixture f;
    assert(validate_yolov8_outputs(f.tensors,{32,32}) == 3);
    const auto identity = letterbox_transform({32,32},{32,32});
    assert(decode_yolov8(f.tensors,identity).empty());
    f.score(0,5,2,0.9f); // centre(12,12), DFL distance=8 each side.
    auto results = decode_yolov8(f.tensors,identity);
    assert(results.size() == 1 && results[0].class_id == 2);
    assert(near(results[0].bbox.x,4) && near(results[0].bbox.y,4));
    assert(near(results[0].bbox.width,16) && near(results[0].bbox.height,16));
    // Low-confidence candidates must remain available to second-stage tracking.
    f.score(0,15,1,0.15f);
    f.score(0,6,2,0.8f); // overlaps class2 at IoU=1/3.
    YoloDecodeConfig config;
    config.nms_threshold = 0.3f;
    results = decode_yolov8(f.tensors,identity,config);
    assert(results.size() == 2 && near(results[1].score,0.15f));
    f.score(0,6,2,0.f);
    f.score(0,6,0,0.8f); // cross-class suppression must not happen.
    results = decode_yolov8(f.tensors,identity,config);
    assert(results.size() == 3 && results[1].class_id == 0);
    config.max_detections = 1;
    assert(decode_yolov8(f.tensors,identity,config).size() == 1);
    // Stable softmax must handle large logits without inf/NaN boxes.
    f.storage[0][(1)*16+5] = 1000.f;
    assert(std::isfinite(decode_yolov8(f.tensors,identity,config)[0].bbox.width));
    // Inverse mapping must remove padding and use the actual resized height.
    Fixture padded;
    padded.score(0,5,2,0.9f);
    const auto mapped = decode_yolov8(padded.tensors,letterbox_transform({64,32},{32,32}));
    assert(mapped.size() == 1);
    assert(near(mapped[0].bbox.x,8) && near(mapped[0].bbox.y,0));
    assert(near(mapped[0].bbox.width,32) && near(mapped[0].bbox.height,24));
}

void test_quantized_and_sum() {
    Fixture floating(true);
    floating.score(0,5,1,0.8f);
    const auto transform = letterbox_transform({32,32},{32,32});
    const auto expected = decode_yolov8(floating.tensors,transform);
    assert(expected.size() == 1);
    for (bool signed_data : {true,false}) {
        auto tensors = floating.tensors;
        std::vector<std::vector<std::uint8_t>> storage(tensors.size());
        for (std::size_t i = 0; i < tensors.size(); ++i) {
            auto& tensor = tensors[i];
            tensor.element = signed_data ? TensorElement::Int8 : TensorElement::UInt8;
            tensor.scale = i%3 == 0 ? 0.1f : 0.01f;
            tensor.zero_point = signed_data ? (i%3 == 0 ? 0 : -100) : 128;
            for (const float value : floating.storage[i]) {
                const int q = static_cast<int>(std::lround(value/tensor.scale))+tensor.zero_point;
                storage[i].push_back(static_cast<std::uint8_t>(q));
            }
            tensor.data = storage[i].data();
            tensor.bytes = storage[i].size();
        }
        const auto actual = decode_yolov8(tensors,transform);
        assert(actual.size() == 1 && actual[0].class_id == expected[0].class_id);
        assert(near(actual[0].score,expected[0].score));
        assert(near(actual[0].bbox.x,expected[0].bbox.x));
    }
    // Floating point nine-output contract can safely prefilter the score sum.
    floating.storage[2][5] = 0.01f;
    assert(decode_yolov8(floating.tensors,transform).empty());
}

void test_quantized_sum_threshold_boundary() {
    const auto transform = letterbox_transform({32,32},{32,32});
    for (bool signed_data : {true,false}) {
        // Scores and sums have different scales. The sum rounds below the
        // accepted class score: score=.100 or .102 while sum=.098. Even adding
        // one sum scale (.001) cannot safely reject these candidates.
        for (const auto& setting : std::vector<std::pair<float,int>>{{0.025f,4},{0.006f,17}}) {
            Fixture f(true);
            auto& scores = f.tensors[1];
            auto& sums = f.tensors[2];
            const int zp = signed_data ? -100 : 100;
            std::vector<std::uint8_t> score_data(3*16,static_cast<std::uint8_t>(zp));
            std::vector<std::uint8_t> sum_data(16,static_cast<std::uint8_t>(zp));
            scores.element = sums.element = signed_data ? TensorElement::Int8 : TensorElement::UInt8;
            scores.zero_point = sums.zero_point = zp;
            scores.scale = setting.first;
            sums.scale = 0.001f;
            score_data[16+5] = static_cast<std::uint8_t>(zp+setting.second);
            sum_data[5] = static_cast<std::uint8_t>(zp+98);
            scores.data = score_data.data(); scores.bytes = score_data.size();
            sums.data = sum_data.data(); sums.bytes = sum_data.size();
            // Compare with the same outputs after removing optional sums.
            std::vector<YoloTensor> split6;
            for (int branch = 0; branch < 3; ++branch) {
                split6.push_back(f.tensors[branch*3]);
                split6.push_back(f.tensors[branch*3+1]);
            }
            auto accepted = decode_yolov8(f.tensors,transform);
            const auto without_sum = decode_yolov8(split6,transform);
            assert(accepted.size() == 1 && without_sum.size() == 1);
            assert(accepted[0].score == without_sum[0].score);
            assert(accepted[0].class_id == 1);
            YoloDecodeConfig config;
            config.confidence_threshold = setting.first*setting.second;
            // Detection uses >= at the threshold; the tracker independently
            // owns its strictly-greater low/high association boundaries.
            assert(decode_yolov8(f.tensors,transform,config).size() == 1);
            config.confidence_threshold = std::nextafter(config.confidence_threshold,1.f);
            assert(decode_yolov8(f.tensors,transform,config).empty());
            // Mixed output element types must not re-enable the unsafe filter.
            sums.element = TensorElement::Float32;
            f.storage[2][5] = 0.098f;
            sums.data = f.storage[2].data(); sums.bytes = f.storage[2].size()*sizeof(float);
            assert(decode_yolov8(f.tensors,transform).size() == 1);
            scores.element = TensorElement::Float32;
            f.storage[1][16+5] = setting.first*setting.second;
            scores.data = f.storage[1].data(); scores.bytes = f.storage[1].size()*sizeof(float);
            sums.element = signed_data ? TensorElement::Int8 : TensorElement::UInt8;
            sums.data = sum_data.data(); sums.bytes = sum_data.size();
            assert(decode_yolov8(f.tensors,transform).size() == 1);
        }
    }
}

void test_invalid_contracts() {
    Fixture f;
    auto invalid = f.tensors;
    invalid.pop_back();
    throws([&] { validate_yolov8_outputs(invalid,{32,32}); });
    invalid = f.tensors;
    invalid[2].width = 4;
    throws([&] { validate_yolov8_outputs(invalid,{32,32}); });
    invalid = f.tensors;
    invalid[3].channels = 80;
    throws([&] { validate_yolov8_outputs(invalid,{32,32}); });
    invalid = f.tensors;
    invalid[0].bytes -= 1;
    throws([&] { validate_yolov8_outputs(invalid,{32,32}); });
    invalid = f.tensors;
    invalid[0].element = TensorElement::Int8;
    invalid[0].scale = 0;
    throws([&] { validate_yolov8_outputs(invalid,{32,32}); });
    f.score(0,5,1,4.f);
    throws([&] { decode_yolov8(f.tensors,letterbox_transform({32,32},{32,32})); });
    f.score(0,5,1,std::numeric_limits<float>::quiet_NaN());
    throws([&] { decode_yolov8(f.tensors,letterbox_transform({32,32},{32,32})); });
}
} // namespace

int main() {
    test_letterbox();
    test_float_decode_and_nms();
    test_quantized_and_sum();
    test_quantized_sum_threshold_boundary();
    test_invalid_contracts();
    std::cout << "detector: letterbox/RGB/ROI, float/INT8/UINT8 split decode, NMS and contract checks passed\n";
}
