#include "Detector.hpp"

#include <stdexcept>

LocalizationResult localizeForgery(const cv::Mat& bgr, const cv::Mat& dctMask,
                                  const cv::Mat& orbMask) {
    if (dctMask.size() != bgr.size() || orbMask.size() != bgr.size()) {
        throw std::runtime_error("Image and mask dimensions do not match");
    }
    LocalizationResult output;

    // Merge the two methods, join small gaps, and remove isolated specks.
    cv::bitwise_or(dctMask, orbMask, output.mask);
    auto closing = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(9, 9));
    auto opening = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3));
    cv::morphologyEx(output.mask, output.mask, cv::MORPH_CLOSE, closing);
    cv::morphologyEx(output.mask, output.mask, cv::MORPH_OPEN, opening);

    cv::Mat labels, stats, centroids;
    int count = cv::connectedComponentsWithStats(output.mask, labels, stats, centroids);
    cv::Mat cleaned = cv::Mat::zeros(output.mask.size(), CV_8U);
    for (int i = 1; i < count; ++i) {
        if (stats.at<int>(i, cv::CC_STAT_AREA) < 180) {
            continue;
        }
        cleaned.setTo(255, labels == i);
        output.boxes.emplace_back(stats.at<int>(i, cv::CC_STAT_LEFT),
                                  stats.at<int>(i, cv::CC_STAT_TOP),
                                  stats.at<int>(i, cv::CC_STAT_WIDTH),
                                  stats.at<int>(i, cv::CC_STAT_HEIGHT));
    }
    output.mask = cleaned;

    // Tint the detected pixels red and show a box around each connected region.
    cv::Mat highlighted = bgr.clone();
    highlighted.setTo(cv::Scalar(30, 30, 255), output.mask);
    cv::addWeighted(bgr, 0.55, highlighted, 0.45, 0, output.annotated);
    for (const auto& box : output.boxes) {
        cv::rectangle(output.annotated, box, cv::Scalar(0, 255, 255), 2);
    }
    return output;
}
