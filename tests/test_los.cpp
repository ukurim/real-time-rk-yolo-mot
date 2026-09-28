#include "aerial/los.hpp"

#include <opencv2/calib3d.hpp>

#include <cmath>
#include <cstdio>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <unistd.h>

namespace {

void check(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

void near(double actual, double expected, double tolerance, const char* message) {
    if (!std::isfinite(actual) || std::abs(actual - expected) > tolerance) {
        throw std::runtime_error(std::string(message) + ": actual=" + std::to_string(actual) +
                                 " expected=" + std::to_string(expected));
    }
}

template <class Function> void rejected(Function function, const char* message) {
    try { function(); } catch (const std::invalid_argument&) { return; }
    throw std::runtime_error(message);
}

// Synthetic optics for mathematical tests only; never used by the application.
aerial::CameraCalibration syntheticCalibration() {
    aerial::CameraCalibration calibration;
    calibration.calibrated = true;
    calibration.image_size = {1280, 720};
    calibration.camera_matrix = cv::Matx33d(800, 0, 640, 0, 820, 360, 0, 0, 1);
    calibration.distortion_coefficients = {0, 0, 0, 0, 0};
    return calibration;
}

cv::Rect2f boxAt(double x, double y) {
    return {static_cast<float>(x - 4), static_cast<float>(y - 6), 8, 12};
}

void checkRay(const aerial::LosResult& result, double x, double y, double tolerance = 1e-6) {
    check(result.valid, "LOS should be valid");
    const double length = std::sqrt(x * x + y * y + 1);
    near(result.unit_vector[0], x / length, tolerance, "ray X");
    near(result.unit_vector[1], y / length, tolerance, "ray Y");
    near(result.unit_vector[2], 1 / length, tolerance, "ray Z");
    near(cv::norm(result.unit_vector), 1, 1e-12, "unit length");
    near(result.azimuth_rad, std::atan2(x, 1), tolerance, "azimuth radians");
    near(result.elevation_rad, std::atan2(-y, std::hypot(x, 1)), tolerance, "elevation radians");
}

void testAxes() {
    const aerial::LosProjector projector(syntheticCalibration());
    const cv::Size size(1280, 720);
    checkRay(projector.project(boxAt(640, 360), size), 0, 0);
    checkRay(projector.project(boxAt(840, 360), size), 0.25, 0);
    checkRay(projector.project(boxAt(440, 360), size), -0.25, 0);
    checkRay(projector.project(boxAt(640, 155), size), 0, -0.25);
    checkRay(projector.project(boxAt(640, 565), size), 0, 0.25);
    checkRay(projector.project(boxAt(840, 565), size), 0.25, 0.25);
}

void testDistortion() {
    const std::vector<double> coefficients{-0.13, 0.03, 0.001, -0.002, 0.004,
                                            0.002, -0.001, 0.0003, 0.0001,
                                            -0.0002, 0.0002, 0.0001, 0.001, -0.001};
    for (std::size_t count : {4u, 5u, 8u, 12u, 14u}) {
        auto calibration = syntheticCalibration();
        calibration.distortion_coefficients.assign(coefficients.begin(), coefficients.begin() + count);
        const aerial::LosProjector projector(calibration);
        for (const cv::Point3d ray : {cv::Point3d(-0.55, -0.3, 1), cv::Point3d(0.6, 0.3, 1)}) {
            std::vector<cv::Point2d> pixels;
            cv::projectPoints(std::vector<cv::Point3d>{ray}, cv::Vec3d::all(0), cv::Vec3d::all(0),
                              calibration.camera_matrix, calibration.distortion_coefficients, pixels);
            checkRay(projector.project(boxAt(pixels[0].x, pixels[0].y), calibration.image_size), ray.x, ray.y);
        }
    }
    auto fisheye = syntheticCalibration();
    fisheye.distortion_model = aerial::DistortionModel::Fisheye;
    fisheye.distortion_coefficients = {-0.02, 0.003, -0.0004, 0.00005};
    const aerial::LosProjector projector(fisheye);
    for (const cv::Point2d ray : {cv::Point2d(0, 0), cv::Point2d(-0.7, -0.25), cv::Point2d(0.7, 0.25)}) {
        std::vector<cv::Point2d> pixels;
        cv::fisheye::distortPoints(std::vector<cv::Point2d>{ray}, pixels,
                                  fisheye.camera_matrix, fisheye.distortion_coefficients);
        checkRay(projector.project(boxAt(pixels[0].x, pixels[0].y), fisheye.image_size), ray.x, ray.y);
    }
}

void testResizePolicy() {
    auto calibration = syntheticCalibration();
    rejected([&] { aerial::LosProjector(calibration).project(boxAt(420, 282.5), {640, 360}); },
             "mismatched image size must be rejected without explicit permission");
    calibration.resize_policy = aerial::IntrinsicsResizePolicy::ScaleFullFrame;
    const aerial::LosProjector projector(calibration);
    checkRay(projector.project(boxAt(420, 282.5), {640, 360}), 0.25, 0.25);
    // Same complete image scaled differently along each axis; not a crop.
    checkRay(projector.project(boxAt(420, 565), {640, 720}), 0.25, 0.25);
    rejected([&] { projector.project(boxAt(640, 360), {0, 720}); }, "zero frame width must fail");
}

void testValidation() {
    auto calibration = syntheticCalibration();
    calibration.calibrated = false;
    rejected([&] { aerial::LosProjector projector(calibration); }, "uncalibrated config must fail");
    calibration = syntheticCalibration();
    calibration.camera_matrix(0, 0) = 0;
    rejected([&] { calibration.validate(); }, "zero focal length must fail");
    calibration = syntheticCalibration();
    calibration.camera_matrix(0, 1) = 1;
    rejected([&] { calibration.validate(); }, "unsupported skew must fail");
    calibration = syntheticCalibration();
    calibration.camera_matrix(0, 2) = std::numeric_limits<double>::quiet_NaN();
    rejected([&] { calibration.validate(); }, "NaN intrinsic must fail");
    calibration = syntheticCalibration();
    calibration.distortion_coefficients[0] = std::numeric_limits<double>::infinity();
    rejected([&] { calibration.validate(); }, "infinite distortion must fail");
    calibration = syntheticCalibration();
    calibration.distortion_coefficients = {0, 0, 0};
    rejected([&] { calibration.validate(); }, "invalid pinhole coefficient count must fail");
    calibration = syntheticCalibration();
    calibration.distortion_model = aerial::DistortionModel::Fisheye;
    rejected([&] { calibration.validate(); }, "invalid fisheye coefficient count must fail");
    calibration = syntheticCalibration();
    calibration.image_geometry = "cropped";
    rejected([&] { calibration.validate(); }, "undeclared crop transform must fail");
    calibration = syntheticCalibration();
    calibration.image_size.height = -1;
    rejected([&] { calibration.validate(); }, "negative calibration dimensions must fail");

    const aerial::LosProjector projector(syntheticCalibration());
    for (const cv::Rect2f bbox : {cv::Rect2f(0, 0, 0, 1), cv::Rect2f(0, 0, 1, -1),
                                boxAt(-1, 360), boxAt(1280, 360), boxAt(640, 720),
                                cv::Rect2f(std::numeric_limits<float>::quiet_NaN(), 0, 1, 1),
                                cv::Rect2f(0, 0, std::numeric_limits<float>::infinity(), 1)}) {
        const auto result = projector.project(bbox, {1280, 720});
        check(!result.valid, "invalid bbox must not produce valid LOS");
        check(std::isfinite(result.azimuth_rad) && std::isfinite(result.elevation_rad), "invalid LOS must not leak NaN");
    }
}

class TempCalibration {
public:
    TempCalibration() {
        char name[] = "/tmp/aerial-test-los-XXXXXX";
        const int descriptor = mkstemp(name);
        if (descriptor < 0) throw std::runtime_error("Cannot create temporary calibration");
        close(descriptor);
        path = name;
    }
    ~TempCalibration() { std::remove(path.c_str()); }
    std::string path;
};

void writeCalibration(const std::string& path, int calibrated, const std::string& model = "pinhole") {
    cv::FileStorage file(path, cv::FileStorage::WRITE | cv::FileStorage::FORMAT_YAML);
    file << "schema_version" << 1 << "calibrated" << calibrated
         << "image_width" << 1280 << "image_height" << 720
         << "fx" << 800.0 << "fy" << 820.0 << "cx" << 640.0 << "cy" << 360.0
         << "distortion_model" << model
         << "distortion_coefficients" << std::vector<double>{0, 0, 0, 0, 0}
         << "image_geometry" << "full_frame" << "resize_policy" << "reject";
}

void testFileLoading() {
    TempCalibration file;
    writeCalibration(file.path, 1);
    checkRay(aerial::LosProjector(file.path).project(boxAt(640, 360), {1280, 720}), 0, 0);
    writeCalibration(file.path, 0);
    rejected([&] { aerial::CameraCalibration::load(file.path); }, "template guard must reject calibrated=0");
    writeCalibration(file.path, 2);
    rejected([&] { aerial::CameraCalibration::load(file.path); }, "calibrated must be exactly 1");
    writeCalibration(file.path, 1, "unknown");
    rejected([&] { aerial::CameraCalibration::load(file.path); }, "unknown model must fail");
    {
        cv::FileStorage incomplete(file.path, cv::FileStorage::WRITE | cv::FileStorage::FORMAT_YAML);
        incomplete << "schema_version" << 1 << "calibrated" << 1;
    }
    rejected([&] { aerial::CameraCalibration::load(file.path); }, "missing required fields must fail");
}

} // namespace

int main() {
    try {
        testAxes();
        testDistortion();
        testResizePolicy();
        testValidation();
        testFileLoading();
        std::cout << "LOS tests passed (synthetic intrinsics only)\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "LOS test failed: " << error.what() << '\n';
        return 1;
    }
}
