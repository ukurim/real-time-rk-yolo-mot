#pragma once

#include "rknn_api.h"
#include <opencv2/core.hpp>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

namespace aerial {

// Explicit opt-in for models converted with mean=[0,0,0], std=[255,255,255].
// RKNN's attributes do not expose those conversion constants: the caller must
// verify the conversion manifest. pass_through=1 bypasses runtime normalization.
// This remains the ordinary copying rknn_inputs_set API, not shared IO memory.
class Fp16RgbInput {
public:
    explicit Fp16RgbInput(const rknn_tensor_attr& native) {
        const bool identity = native.qnt_type == RKNN_TENSOR_QNT_NONE ||
            (native.qnt_type == RKNN_TENSOR_QNT_AFFINE_ASYMMETRIC && native.zp == 0 && native.scale == 1.f);
        if (native.index != 0 || native.n_dims != 4 || native.dims[0] != 1 ||
            native.fmt != RKNN_TENSOR_NHWC || native.type != RKNN_TENSOR_FLOAT16 ||
            native.dims[3] != 3 || !identity || native.dims[1] == 0 || native.dims[2] == 0 ||
            native.dims[1] > 8192 || native.dims[2] > 8192)
            throw std::invalid_argument("fp16_normalized requires native batch=1 RGB NHWC FP16 with identity quantization");
        width_ = static_cast<int>(native.dims[2]);
        height_ = static_cast<int>(native.dims[1]);
        const uint64_t elements = uint64_t(width_) * height_ * 3;
        const uint64_t bytes = elements * sizeof(cv::float16_t);
        if (bytes > std::numeric_limits<uint32_t>::max() || native.n_elems != elements ||
            native.size != bytes || native.size_with_stride != bytes ||
            (native.w_stride != 0 && native.w_stride != native.dims[2]) ||
            (native.h_stride != 0 && native.h_stride != native.dims[1]))
            throw std::invalid_argument("fp16_normalized requires a contiguous native tensor without stride padding");
        pixels_.resize(static_cast<size_t>(elements));
    }

    rknn_input prepare(const cv::Mat& rgb) {
        if (rgb.type() != CV_8UC3 || rgb.cols != width_ || rgb.rows != height_ || !rgb.isContinuous())
            throw std::invalid_argument("fp16_normalized requires contiguous RGB uint8 matching native input dimensions");
        static_assert(sizeof(cv::float16_t) == 2, "RKNN requires IEEE binary16 storage");
        const auto* source = rgb.ptr<uint8_t>();
        constexpr float scale = 1.f/255.f;
        for (size_t i = 0; i < pixels_.size(); ++i)
            pixels_[i] = cv::float16_t(static_cast<float>(source[i]) * scale);
        rknn_input input{};
        input.index = 0;
        input.buf = pixels_.data();
        input.size = static_cast<uint32_t>(pixels_.size()*sizeof(cv::float16_t));
        input.pass_through = 1;
        input.type = RKNN_TENSOR_FLOAT16;
        input.fmt = RKNN_TENSOR_NHWC;
        return input;
    }

private:
    int width_ = 0, height_ = 0;
    std::vector<cv::float16_t> pixels_;
};

} // namespace aerial
