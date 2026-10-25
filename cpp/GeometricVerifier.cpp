#include "Detector.hpp"

#include <algorithm>
#include <cmath>

namespace {

bool coversRegion(const std::vector<cv::Point2f>& points) {
    if (points.size() < 8) {
        return false;
    }
    float minX = points[0].x, maxX = minX;
    float minY = points[0].y, maxY = minY;
    for (const auto& point : points) {
        minX = std::min(minX, point.x);
        maxX = std::max(maxX, point.x);
        minY = std::min(minY, point.y);
        maxY = std::max(maxY, point.y);
    }
    std::vector<cv::Point2f> hull;
    cv::convexHull(points, hull);
    return maxX - minX >= 20 && maxY - minY >= 20 && cv::contourArea(hull) >= 400;
}

} // namespace

VerifiedMatches verifyGeometry(const std::vector<cv::KeyPoint>& keypoints,
                              const std::vector<cv::DMatch>& candidates) {
    VerifiedMatches result;
    if (candidates.size() < 8) {
        return result;
    }

    std::vector<cv::Point2f> source, target;
    for (const auto& match : candidates) {
        source.push_back(keypoints[match.queryIdx].pt);
        target.push_back(keypoints[match.trainIdx].pt);
    }

    // RANSAC looks for one rotation, scale and translation shared by the matches.
    cv::Mat inliers;
    cv::Mat transform = cv::estimateAffinePartial2D(source, target, inliers,
                                                   cv::RANSAC, 4.0, 3000, 0.995, 10);
    if (transform.empty()) {
        return result;
    }
    double scale = std::hypot(transform.at<double>(0, 0), transform.at<double>(1, 0));
    if (scale < 0.55 || scale > 1.8 || cv::countNonZero(inliers) < 8) {
        return result;
    }

    for (size_t i = 0; i < candidates.size(); ++i) {
        if (inliers.at<uchar>(static_cast<int>(i))) {
            result.source.push_back(source[i]);
            result.target.push_back(target[i]);
            result.matches.push_back(candidates[i]);
        }
    }
    // A small cluster or a straight line of points is weak evidence for a region.
    if (!coversRegion(result.source) || !coversRegion(result.target)) {
        return VerifiedMatches{};
    }
    return result;
}
