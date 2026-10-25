#include "Detector.hpp"

#include <algorithm>
#include <limits>
#include <set>
#include <utility>

ORBResult detectORB(const cv::Mat& gray) {
    ORBResult output;
    output.mask = cv::Mat::zeros(gray.size(), CV_8U);
    cv::Mat color;
    cv::cvtColor(gray, color, cv::COLOR_GRAY2BGR);
    cv::hconcat(color, color, output.visualization);
    if (gray.rows < 32 || gray.cols < 32) {
        return output;
    }

    // ORB describes corners, so this branch can also find rotated or scaled copies.
    auto orb = cv::ORB::create(2500);
    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;
    orb->detectAndCompute(gray, cv::noArray(), keypoints, descriptors);
    if (descriptors.rows < 8) {
        return output;
    }

    cv::BFMatcher matcher(cv::NORM_HAMMING);
    std::vector<std::vector<cv::DMatch>> neighbors;
    matcher.knnMatch(descriptors, descriptors, neighbors, std::min(12, descriptors.rows));

    // Ignore the point itself and nearby corners. A clear nearest match is less
    // likely to come from an ordinary repeated texture.
    std::vector<int> best(keypoints.size(), -1);
    std::vector<float> distances(keypoints.size(), 0);
    for (size_t i = 0; i < neighbors.size(); ++i) {
        int nearest = -1;
        float first = std::numeric_limits<float>::infinity();
        float second = std::numeric_limits<float>::infinity();
        for (const auto& match : neighbors[i]) {
            if (match.queryIdx == match.trainIdx ||
                cv::norm(keypoints[i].pt - keypoints[match.trainIdx].pt) < 48) {
                continue;
            }
            if (nearest < 0) {
                nearest = match.trainIdx;
                first = match.distance;
            } else {
                second = match.distance;
                break;
            }
        }
        if (nearest >= 0 && first <= 48 && first < 0.8f * second) {
            best[i] = nearest;
            distances[i] = first;
        }
    }

    std::vector<cv::DMatch> candidates;
    for (int a = 0; a < static_cast<int>(best.size()); ++a) {
        int b = best[a];
        if (b <= a || best[b] != a) {
            continue;
        }
        // Keep the same spatial direction in the transform and its visualization.
        int source = a;
        int target = b;
        const auto& p = keypoints[source].pt;
        const auto& q = keypoints[target].pt;
        if (std::make_pair(p.x, p.y) > std::make_pair(q.x, q.y)) {
            std::swap(source, target);
        }
        candidates.emplace_back(source, target, distances[a]);
    }

    // ORB can detect the same corner at several scales. Count each small area once.
    std::sort(candidates.begin(), candidates.end(), [](const auto& a, const auto& b) {
        return a.distance < b.distance;
    });
    std::set<std::pair<int, int>> sourceCells;
    std::set<std::pair<int, int>> targetCells;
    std::vector<cv::DMatch> distinct;
    for (const auto& match : candidates) {
        auto p = keypoints[match.queryIdx].pt;
        auto q = keypoints[match.trainIdx].pt;
        auto sourceCell = std::make_pair(static_cast<int>(p.x) / 8, static_cast<int>(p.y) / 8);
        auto targetCell = std::make_pair(static_cast<int>(q.x) / 8, static_cast<int>(q.y) / 8);
        if (sourceCells.count(sourceCell) || targetCells.count(targetCell)) {
            continue;
        }
        sourceCells.insert(sourceCell);
        targetCells.insert(targetCell);
        distinct.push_back(match);
    }
    output.candidateMatches = static_cast<int>(distinct.size());

    auto verified = verifyGeometry(keypoints, distinct);
    output.inliers = static_cast<int>(verified.matches.size());
    for (size_t i = 0; i < verified.source.size(); ++i) {
        cv::circle(output.mask, verified.source[i], 13, cv::Scalar(255), cv::FILLED);
        cv::circle(output.mask, verified.target[i], 13, cv::Scalar(255), cv::FILLED);
    }
    cv::drawMatches(gray, keypoints, gray, keypoints, verified.matches, output.visualization,
                    cv::Scalar(0, 220, 0), cv::Scalar::all(-1), std::vector<char>(),
                    cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
    return output;
}
