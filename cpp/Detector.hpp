#pragma once

#include <opencv2/opencv.hpp>
#include <vector>

// Resize large inputs and convert them to grayscale for both detectors.
cv::Mat preprocessImage(const cv::Mat& bgr, int maxDimension = 768);

struct DCTResult {
    cv::Mat mask;
    cv::Mat visualization;
    int candidatePairs = 0;
    int consistentPairs = 0;
    int displacementGroups = 0;
};

DCTResult detectDCT(const cv::Mat& gray);

struct ORBResult {
    cv::Mat mask;
    cv::Mat visualization;
    int candidateMatches = 0;
    int inliers = 0;
};

ORBResult detectORB(const cv::Mat& gray);

struct VerifiedMatches {
    std::vector<cv::Point2f> source;
    std::vector<cv::Point2f> target;
    std::vector<cv::DMatch> matches;
};

VerifiedMatches verifyGeometry(const std::vector<cv::KeyPoint>& keypoints,
                              const std::vector<cv::DMatch>& candidates);

struct LocalizationResult {
    cv::Mat mask;
    cv::Mat annotated;
    std::vector<cv::Rect> boxes;
};

LocalizationResult localizeForgery(const cv::Mat& bgr, const cv::Mat& dctMask,
                                  const cv::Mat& orbMask);
