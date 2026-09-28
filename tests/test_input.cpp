#include "aerial/rknn_input.hpp"
#include <cassert>
#include <cmath>
#include <iostream>

namespace {
rknn_tensor_attr native() {
    rknn_tensor_attr a{};
    a.n_dims = 4;
    a.dims[0] = 1; a.dims[1] = 1; a.dims[2] = 256; a.dims[3] = 3;
    a.type = RKNN_TENSOR_FLOAT16; a.fmt = RKNN_TENSOR_NHWC;
    a.qnt_type = RKNN_TENSOR_QNT_AFFINE_ASYMMETRIC; a.scale = 1;
    a.n_elems = 256*3; a.size = a.size_with_stride = a.n_elems*2; a.w_stride = 256;
    return a;
}
template<class F> void rejects(F fn) {
    bool thrown = false;
    try { fn(); } catch (const std::invalid_argument&) { thrown = true; }
    assert(thrown);
}
}
int main() {
    const auto n = native();
    aerial::Fp16RgbInput input(n);
    cv::Mat rgb(1, 256, CV_8UC3);
    for (int x = 0; x < 256; ++x) rgb.at<cv::Vec3b>(0,x) = {uint8_t(x),uint8_t(255-x),uint8_t(114)};
    const auto prepared = input.prepare(rgb);
    assert(prepared.type == RKNN_TENSOR_FLOAT16 && prepared.fmt == RKNN_TENSOR_NHWC && prepared.pass_through == 1);
    assert(prepared.size == 256*3*2 && prepared.index == 0);
    const auto* fp = static_cast<const cv::float16_t*>(prepared.buf);
    for (int x = 0; x < 256; ++x) {
        // All possible RGB bytes and channel order; binary16 error <= half ULP.
        assert(std::abs(float(fp[x*3]) - x/255.f) <= 0.000245f);
        assert(std::abs(float(fp[x*3+1]) - (255-x)/255.f) <= 0.000245f);
        assert(fp[x*3+2].bits() == cv::float16_t(114.f/255.f).bits());
    }
    assert(fp[0].bits() == 0 && fp[255*3].bits() == 0x3c00);
    assert(fp[0*3+2].bits() == 0x3727); // 114/255 rounded to IEEE binary16.
    rgb.setTo(cv::Scalar(255,0,255));
    const auto reused = input.prepare(rgb);
    assert(reused.buf == prepared.buf && float(fp[0]) == 1.f && float(fp[1]) == 0.f);
    rejects([&] { auto a=n; a.type=RKNN_TENSOR_INT8; aerial::Fp16RgbInput t(a); });
    rejects([&] { auto a=n; a.fmt=RKNN_TENSOR_NCHW; aerial::Fp16RgbInput t(a); });
    rejects([&] { auto a=n; a.w_stride=257; aerial::Fp16RgbInput t(a); });
    rejects([&] { auto a=n; a.size_with_stride+=2; aerial::Fp16RgbInput t(a); });
    rejects([&] { auto a=n; a.n_elems-=1; aerial::Fp16RgbInput t(a); });
    rejects([&] { auto a=n; a.scale=.5f; aerial::Fp16RgbInput t(a); });
    rejects([&] { auto a=n; a.zp=1; aerial::Fp16RgbInput t(a); });
    rejects([&] { auto a=n; a.qnt_type=RKNN_TENSOR_QNT_DFP; aerial::Fp16RgbInput t(a); });
    rejects([&] { auto a=n; a.dims[3]=4; aerial::Fp16RgbInput t(a); });
    rejects([&] { input.prepare(cv::Mat(2,128,CV_8UC3)); });
    rejects([&] { input.prepare(cv::Mat(1,256,CV_32FC3)); });
    std::cout << "FP16 RGB conversion and native-layout rejection checks passed\n";
}
