#include "aerial/los.hpp"

#include <opencv2/calib3d.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

namespace aerial {
namespace {

double requiredNumber(const cv::FileStorage& file, const char* key) {
    const cv::FileNode node = file[key];
    if (node.empty() || (!node.isInt() && !node.isReal())) {
        throw std::invalid_argument(std::string("Calibration requires numeric '") + key + "'");
    }
    const double value = static_cast<double>(node);
    if (!std::isfinite(value)) {
        throw std::invalid_argument(std::string("Calibration '") + key + "' must be finite");
    }
    return value;
}

int requiredInt(const cv::FileStorage& file, const char* key) {
    const double value = requiredNumber(file, key);
    if (value != std::floor(value) || value < std::numeric_limits<int>::min() ||
        value > std::numeric_limits<int>::max()) {
        throw std::invalid_argument(std::string("Calibration '") + key + "' must be an integer");
    }
    return static_cast<int>(value);
}

std::string requiredString(const cv::FileStorage& file, const char* key) {
    const cv::FileNode node = file[key];
    if (node.empty() || !node.isString()) {
        throw std::invalid_argument(std::string("Calibration requires string '") + key + "'");
    }
    return static_cast<std::string>(node);
}

bool finitePoint(const cv::Point2d& point) {
    return std::isfinite(point.x) && std::isfinite(point.y);
}

} // namespace

CameraCalibration CameraCalibration::load(const std::string& path) {
    try {
        const cv::FileStorage file(path, cv::FileStorage::READ);
        if (!file.isOpened()) {
            throw std::invalid_argument("Cannot open camera calibration: " + path);
        }
        if (requiredInt(file, "schema_version") != 1) {
            throw std::invalid_argument("Unsupported camera calibration schema_version");
        }
        CameraCalibration calibration;
        if (requiredInt(file, "calibrated") != 1) {
            throw std::invalid_argument("Camera calibration is not confirmed: calibrated must be 1 after real calibration");
        }
        calibration.calibrated = true;
        calibration.image_size = {requiredInt(file, "image_width"), requiredInt(file, "image_height")};
        calibration.camera_matrix = cv::Matx33d(
            requiredNumber(file, "fx"), 0.0, requiredNumber(file, "cx"),
            0.0, requiredNumber(file, "fy"), requiredNumber(file, "cy"),
            0.0, 0.0, 1.0);
        const std::string model = requiredString(file, "distortion_model");
        if (model == "pinhole") calibration.distortion_model = DistortionModel::Pinhole;
        else if (model == "fisheye") calibration.distortion_model = DistortionModel::Fisheye;
        else throw std::invalid_argument("distortion_model must be pinhole or fisheye");
        const cv::FileNode coefficients = file["distortion_coefficients"];
        if (coefficients.empty() || !coefficients.isSeq()) {
            throw std::invalid_argument("distortion_coefficients must be an explicit numeric sequence");
        }
        for (const auto& coefficient : coefficients) {
            if (!coefficient.isInt() && !coefficient.isReal()) {
                throw std::invalid_argument("distortion_coefficients must contain numbers");
            }
            calibration.distortion_coefficients.push_back(static_cast<double>(coefficient));
        }
        calibration.image_geometry = requiredString(file, "image_geometry");
        const std::string resize = requiredString(file, "resize_policy");
        if (resize == "reject") calibration.resize_policy = IntrinsicsResizePolicy::Reject;
        else if (resize == "scale_full_frame") calibration.resize_policy = IntrinsicsResizePolicy::ScaleFullFrame;
        else throw std::invalid_argument("resize_policy must be reject or scale_full_frame");
        calibration.validate();
        return calibration;
    } catch (const cv::Exception& error) {
        throw std::invalid_argument("Invalid camera calibration " + path + ": " + error.what());
    }
}

void CameraCalibration::validate() const {
    if (!calibrated) throw std::invalid_argument("LOS requires a confirmed camera calibration");
    if (image_size.width <= 0 || image_size.height <= 0) {
        throw std::invalid_argument("Calibration image dimensions must be positive");
    }
    for (double value : camera_matrix.val) {
        if (!std::isfinite(value)) throw std::invalid_argument("Camera matrix must be finite");
    }
    if (camera_matrix(0, 0) <= 0 || camera_matrix(1, 1) <= 0 ||
        camera_matrix(0, 1) != 0 || camera_matrix(1, 0) != 0 ||
        camera_matrix(2, 0) != 0 || camera_matrix(2, 1) != 0 || camera_matrix(2, 2) != 1) {
        throw std::invalid_argument("Camera matrix requires positive fx/fy, zero skew, and last row [0,0,1]");
    }
    if (image_geometry != "full_frame") {
        throw std::invalid_argument("Only image_geometry: full_frame is supported; crop/rotation/mirroring requires matching calibration");
    }
    if (resize_policy != IntrinsicsResizePolicy::Reject &&
        resize_policy != IntrinsicsResizePolicy::ScaleFullFrame) {
        throw std::invalid_argument("Invalid intrinsics resize policy");
    }
    const auto count = distortion_coefficients.size();
    if (distortion_model == DistortionModel::Pinhole) {
        if (count != 4 && count != 5 && count != 8 && count != 12 && count != 14) {
            throw std::invalid_argument("Pinhole distortion requires 4, 5, 8, 12, or 14 coefficients; explicitly use zeros for calibrated zero distortion");
        }
    } else if (distortion_model == DistortionModel::Fisheye) {
        if (count != 4) throw std::invalid_argument("Fisheye distortion requires exactly 4 coefficients");
    } else {
        throw std::invalid_argument("Invalid distortion model");
    }
    for (double value : distortion_coefficients) {
        if (!std::isfinite(value)) throw std::invalid_argument("Distortion coefficients must be finite");
    }
}

cv::Matx33d CameraCalibration::intrinsicsFor(cv::Size frame_size) const {
    validate();
    if (frame_size.width <= 0 || frame_size.height <= 0) {
        throw std::invalid_argument("LOS frame dimensions must be positive");
    }
    if (frame_size == image_size) return camera_matrix;
    if (resize_policy != IntrinsicsResizePolicy::ScaleFullFrame) {
        throw std::invalid_argument("Input size differs from calibration; supply matching calibration or explicitly authorize scale_full_frame");
    }
    const double sx = static_cast<double>(frame_size.width) / image_size.width;
    const double sy = static_cast<double>(frame_size.height) / image_size.height;
    auto scaled = camera_matrix;
    scaled(0, 0) *= sx;
    scaled(0, 2) *= sx;
    scaled(1, 1) *= sy;
    scaled(1, 2) *= sy;
    return scaled;
}

LosProjector::LosProjector(const std::string& calibration_path)
    : LosProjector(CameraCalibration::load(calibration_path)) {}

LosProjector::LosProjector(CameraCalibration calibration) : calibration_(std::move(calibration)) {
    calibration_.validate();
}

LosResult LosProjector::project(const cv::Rect2f& bbox, cv::Size frame_size) const {
    // Configuration errors are explicit even if this particular observation is invalid.
    const cv::Matx33d intrinsic = calibration_.intrinsicsFor(frame_size);
    LosResult result;
    if (!std::isfinite(bbox.x) || !std::isfinite(bbox.y) ||
        !std::isfinite(bbox.width) || !std::isfinite(bbox.height) ||
        bbox.width <= 0 || bbox.height <= 0) return result;
    const cv::Point2d center(static_cast<double>(bbox.x) + 0.5 * bbox.width,
                             static_cast<double>(bbox.y) + 0.5 * bbox.height);
    // A predicted track may leave the observable image; do not extrapolate lens calibration.
    if (!finitePoint(center) || center.x < 0 || center.y < 0 ||
        center.x >= frame_size.width || center.y >= frame_size.height) return result;

    try {
        const std::vector<cv::Point2d> pixels{center};
        std::vector<cv::Point2d> normalized;
        const auto criteria = cv::TermCriteria(cv::TermCriteria::MAX_ITER | cv::TermCriteria::EPS, 30, 1e-12);
        if (calibration_.distortion_model == DistortionModel::Fisheye) {
            cv::fisheye::undistortPoints(pixels, normalized, intrinsic,
                                        calibration_.distortion_coefficients,
                                        cv::noArray(), cv::noArray());
        } else {
            cv::undistortPoints(pixels, normalized, intrinsic,
                               calibration_.distortion_coefficients,
                               cv::noArray(), cv::noArray(), criteria);
        }
        if (normalized.size() != 1 || !finitePoint(normalized[0])) return result;

        // Inversion can fail with a finite sentinel or a nonconverged value.
        // Confirm the ray projects back to the measured pixel before publishing it.
        std::vector<cv::Point2d> reprojected;
        if (calibration_.distortion_model == DistortionModel::Fisheye) {
            cv::fisheye::distortPoints(normalized, reprojected, intrinsic,
                                      calibration_.distortion_coefficients);
        } else {
            const std::vector<cv::Point3d> ray{{normalized[0].x, normalized[0].y, 1.0}};
            cv::projectPoints(ray, cv::Vec3d::all(0), cv::Vec3d::all(0), intrinsic,
                              calibration_.distortion_coefficients, reprojected);
        }
        if (reprojected.size() != 1 || !finitePoint(reprojected[0]) ||
            std::hypot(reprojected[0].x - center.x, reprojected[0].y - center.y) > 0.05) return result;

        const double x = normalized[0].x;
        const double y = normalized[0].y;
        const double length = std::hypot(std::hypot(x, y), 1.0);
        if (!std::isfinite(length) || length <= 0) return result;
        result.unit_vector = cv::Vec3d(x / length, y / length, 1.0 / length);
        result.azimuth_rad = std::atan2(x, 1.0);
        result.elevation_rad = std::atan2(-y, std::hypot(x, 1.0));
        result.valid = true;
    } catch (const cv::Exception&) {
        // Invalid per-observation inverse distortion must not leak NaNs downstream.
        return LosResult{};
    }
    return result;
}

} // namespace aerial
