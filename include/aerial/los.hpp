#pragma once

#include <opencv2/core.hpp>

#include <string>
#include <vector>

namespace aerial {

enum class DistortionModel { Pinhole, Fisheye };
enum class IntrinsicsResizePolicy { Reject, ScaleFullFrame };

struct CameraCalibration {
    // Deliberately false until the caller supplies an actual calibration.
    bool calibrated = false;
    cv::Size image_size;
    cv::Matx33d camera_matrix = cv::Matx33d::eye();
    std::vector<double> distortion_coefficients;
    DistortionModel distortion_model = DistortionModel::Pinhole;
    IntrinsicsResizePolicy resize_policy = IntrinsicsResizePolicy::Reject;
    std::string image_geometry = "full_frame";

    static CameraCalibration load(const std::string& path);
    void validate() const;
    // Throws on a mismatched size unless full-frame rescaling was authorized.
    cv::Matx33d intrinsicsFor(cv::Size frame_size) const;
};

struct LosResult {
    cv::Vec3d unit_vector{0.0, 0.0, 0.0};
    double azimuth_rad = 0.0;
    double elevation_rad = 0.0;
    bool valid = false;
};

class LosProjector {
public:
    explicit LosProjector(const std::string& calibration_path);
    explicit LosProjector(CameraCalibration calibration);

    const CameraCalibration& calibration() const { return calibration_; }

    // bbox is floating-point xywh in the original DISTORTED input image;
    // detector letterbox/resize must already have been inverted.
    // Camera axes: +X right, +Y down, +Z forward. Angles are radians;
    // azimuth is positive right, elevation positive up.
    // Invalid observations return valid=false and finite zero fields.
    // Invalid frame geometry/configuration throws std::invalid_argument.
    LosResult project(const cv::Rect2f& bbox, cv::Size frame_size) const;

private:
    CameraCalibration calibration_;
};

} // namespace aerial
