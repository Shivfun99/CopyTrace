#include "Detector.hpp"

#include <algorithm>
#include <stdexcept>

cv::Mat preprocessImage(const cv::Mat& bgr, int maxDimension) {
    if (bgr.empty() || bgr.depth() != CV_8U || maxDimension < 1) {
        throw std::runtime_error("Preprocessing needs an 8-bit image and a positive size limit");
    }

    cv::Mat gray;
    if (bgr.channels() == 3) {
        cv::cvtColor(bgr, gray, cv::COLOR_BGR2GRAY);
    } else if (bgr.channels() == 4) {
        cv::cvtColor(bgr, gray, cv::COLOR_BGRA2GRAY);
    } else if (bgr.channels() == 1) {
        gray = bgr.clone();
    } else {
        throw std::runtime_error("Unsupported number of image channels");
    }

    // Keep the aspect ratio and limit the work needed for overlapping blocks.
    int largest = std::max(gray.rows, gray.cols);
    if (largest > maxDimension) {
        double scale = static_cast<double>(maxDimension) / largest;
        cv::Size size(std::max(1, cvRound(gray.cols * scale)),
                      std::max(1, cvRound(gray.rows * scale)));
        cv::resize(gray, gray, size, 0, 0, cv::INTER_AREA);
    }
    return gray;
}
