#include "Detector.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <map>
#include <set>
#include <utility>
#include <vector>

namespace {

constexpr int BLOCK = 16;
constexpr int STRIDE = 4;
constexpr int FEATURES = 15;
constexpr int NEIGHBORS = 14;
constexpr int MIN_PAIRS = 12;

struct BlockPair {
    cv::Point a, b;
    float distance;
};

// Read low-frequency AC coefficients in zigzag order; skip average brightness.
std::array<cv::Point, FEATURES> frequencyPositions() {
    std::array<cv::Point, FEATURES> positions{};
    int count = 0;
    for (int sum = 1; count < FEATURES; ++sum) {
        for (int k = 0; k <= sum && count < FEATURES; ++k) {
            int x = sum % 2 == 0 ? k : sum - k;
            positions[count++] = cv::Point(x, sum - x);
        }
    }
    return positions;
}

bool coversRegion(const std::vector<BlockPair>& pairs) {
    std::vector<cv::Point> source, target;
    for (const auto& pair : pairs) {
        source.push_back(pair.a);
        target.push_back(pair.b);
    }
    auto a = cv::boundingRect(source);
    auto b = cv::boundingRect(target);
    return a.width > 20 && a.height > 20 && b.width > 20 && b.height > 20;
}

} // namespace

DCTResult detectDCT(const cv::Mat& gray) {
    DCTResult output;
    output.mask = cv::Mat::zeros(gray.size(), CV_8U);
    cv::Mat color;
    cv::cvtColor(gray, color, cv::COLOR_GRAY2BGR);
    cv::hconcat(color, color, output.visualization);
    if (gray.cols < BLOCK || gray.rows < BLOCK) {
        return output;
    }

    const auto positions = frequencyPositions();
    std::vector<cv::Point> blocks;
    cv::Mat descriptors;
    for (int y = 0; y + BLOCK <= gray.rows; y += STRIDE) {
        for (int x = 0; x + BLOCK <= gray.cols; x += STRIDE) {
            cv::Mat patch = gray(cv::Rect(x, y, BLOCK, BLOCK));
            cv::Scalar mean, deviation;
            cv::meanStdDev(patch, mean, deviation);
            if (deviation[0] < 12) {
                continue; // Flat areas do not have enough detail to compare reliably.
            }
            cv::Mat input, transformed;
            patch.convertTo(input, CV_32F);
            cv::dct(input, transformed);
            cv::Mat descriptor(1, FEATURES, CV_32F);
            for (int j = 0; j < FEATURES; ++j) {
                descriptor.at<float>(0, j) = transformed.at<float>(positions[j]);
            }
            double length = cv::norm(descriptor);
            if (length < 1) {
                continue;
            }
            descriptor /= length;
            descriptors.push_back(descriptor);
            blocks.emplace_back(x, y);
        }
    }
    if (descriptors.rows < MIN_PAIRS * 2) {
        return output;
    }

    // Nearby descriptors describe similar blocks. Group distant matches by shift.
    cv::FlannBasedMatcher matcher(cv::makePtr<cv::flann::KDTreeIndexParams>(4),
                                 cv::makePtr<cv::flann::SearchParams>(64));
    std::vector<std::vector<cv::DMatch>> nearest;
    matcher.knnMatch(descriptors, descriptors, nearest, NEIGHBORS);
    std::map<std::pair<int, int>, std::vector<BlockPair>> groups;
    for (const auto& neighbors : nearest) {
        int accepted = 0;
        for (const auto& match : neighbors) {
            cv::Point a = blocks[match.queryIdx];
            cv::Point b = blocks[match.trainIdx];
            if (match.distance >= 0.30f || cv::norm(a - b) < 32 ||
                std::make_pair(a.x, a.y) >= std::make_pair(b.x, b.y)) {
                continue;
            }
            cv::Point shift = b - a;
            auto group = std::make_pair(cvRound(shift.x / 8.0), cvRound(shift.y / 8.0));
            groups[group].push_back({a, b, match.distance});
            ++output.candidatePairs;
            if (++accepted == 3) {
                break;
            }
        }
    }

    std::vector<BlockPair> verified;
    for (auto& entry : groups) {
        auto& pairs = entry.second;
        if (pairs.size() < MIN_PAIRS) {
            continue;
        }
        std::sort(pairs.begin(), pairs.end(), [](const auto& a, const auto& b) {
            return a.distance < b.distance;
        });

        // Count each source and target block once in this displacement group.
        std::set<std::pair<int, int>> usedSource, usedTarget;
        std::vector<BlockPair> distinct;
        cv::Mat occupied = cv::Mat::zeros((gray.rows - BLOCK) / STRIDE + 1,
                                         (gray.cols - BLOCK) / STRIDE + 1, CV_8U);
        for (const auto& pair : pairs) {
            auto a = std::make_pair(pair.a.x, pair.a.y);
            auto b = std::make_pair(pair.b.x, pair.b.y);
            if (usedSource.count(a) || usedTarget.count(b)) {
                continue;
            }
            usedSource.insert(a);
            usedTarget.insert(b);
            distinct.push_back(pair);
            occupied.at<uchar>(pair.a.y / STRIDE, pair.a.x / STRIDE) = 255;
        }
        if (distinct.size() < MIN_PAIRS) {
            continue;
        }

        // Matching shifts must form a connected patch, not scattered coincidences.
        cv::Mat labels;
        int count = cv::connectedComponents(occupied, labels, 8);
        std::vector<std::vector<BlockPair>> components(count);
        for (const auto& pair : distinct) {
            int label = labels.at<int>(pair.a.y / STRIDE, pair.a.x / STRIDE);
            components[label].push_back(pair);
        }
        bool keptGroup = false;
        for (int label = 1; label < count; ++label) {
            const auto& component = components[label];
            if (component.size() < MIN_PAIRS || !coversRegion(component)) {
                continue;
            }
            keptGroup = true;
            verified.insert(verified.end(), component.begin(), component.end());
        }
        output.displacementGroups += keptGroup ? 1 : 0;
    }

    output.consistentPairs = static_cast<int>(verified.size());
    for (const auto& pair : verified) {
        cv::rectangle(output.mask, cv::Rect(pair.a.x, pair.a.y, BLOCK, BLOCK),
                      cv::Scalar(255), cv::FILLED);
        cv::rectangle(output.mask, cv::Rect(pair.b.x, pair.b.y, BLOCK, BLOCK),
                      cv::Scalar(255), cv::FILLED);
    }
    // Show a sample of verified block centers so dense matches stay readable.
    size_t step = std::max<size_t>(1, (verified.size() + 149) / 150);
    for (size_t i = 0; i < verified.size(); i += step) {
        cv::Point a = verified[i].a + cv::Point(BLOCK / 2, BLOCK / 2);
        cv::Point b = verified[i].b + cv::Point(gray.cols + BLOCK / 2, BLOCK / 2);
        cv::line(output.visualization, a, b, cv::Scalar(0, 220, 0), 1, cv::LINE_AA);
        cv::circle(output.visualization, a, 2, cv::Scalar(0, 255, 255), cv::FILLED);
        cv::circle(output.visualization, b, 2, cv::Scalar(0, 255, 255), cv::FILLED);
    }
    return output;
}
