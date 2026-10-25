#include "Detector.hpp"

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {

struct Metrics {
    int truePositive = 0, falsePositive = 0, falseNegative = 0;
    double precision = 0, recall = 0, f1 = 0, iou = 0;
};

double divide(double numerator, double denominator) {
    return denominator == 0 ? 0 : numerator / denominator;
}

Metrics evaluate(const cv::Mat& prediction, const cv::Mat& truth) {
    cv::Mat predicted = prediction > 0;
    cv::Mat expected = truth > 0;
    Metrics result;
    result.truePositive = cv::countNonZero(predicted & expected);
    result.falsePositive = cv::countNonZero(predicted & ~expected);
    result.falseNegative = cv::countNonZero(~predicted & expected);
    double tp = result.truePositive, fp = result.falsePositive, fn = result.falseNegative;
    result.precision = divide(tp, tp + fp);
    result.recall = divide(tp, tp + fn);
    result.f1 = divide(2 * tp, 2 * tp + fp + fn);
    result.iou = divide(tp, tp + fp + fn);
    return result;
}

void writeMetrics(std::ostream& stream, const Metrics& scores) {
    stream << std::fixed << std::setprecision(4)
           << "Precision: " << scores.precision << '\n'
           << "Recall: " << scores.recall << '\n'
           << "F1: " << scores.f1 << '\n'
           << "IoU: " << scores.iou << '\n';
}

bool sameFile(const fs::path& a, const fs::path& b) {
    if (fs::weakly_canonical(a) == fs::weakly_canonical(b)) {
        return true;
    }
    return fs::exists(a) && fs::exists(b) && fs::equivalent(a, b);
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 2 || argc > 4 || (argc == 2 && std::string(argv[1]) == "--help")) {
        std::cout << "Usage: forgery_detector <input_image> [output_folder] [truth_mask]\n";
        return argc > 4 ? 1 : 0;
    }

    try {
        fs::path inputPath = argv[1];
        fs::path outputDirectory = argc >= 3 ? fs::path(argv[2]) : fs::path("results");
        cv::Mat bgr = cv::imread(inputPath.string(), cv::IMREAD_COLOR);
        if (bgr.empty()) {
            throw std::runtime_error("Cannot open input image: " + inputPath.string());
        }
        cv::Size originalSize = bgr.size();

        // A truth mask is optional. Any nonzero pixel means source or copied area.
        cv::Mat truth;
        if (argc == 4) {
            cv::Mat rawTruth = cv::imread(argv[3], cv::IMREAD_UNCHANGED);
            if (rawTruth.empty()) {
                throw std::runtime_error("Cannot open ground-truth mask");
            }
            if (rawTruth.size() != originalSize) {
                throw std::runtime_error("Ground-truth mask must match the original image dimensions");
            }
            // Preserve low-valued labels in 16-bit or color masks; ignore alpha.
            std::vector<cv::Mat> channels;
            cv::split(rawTruth, channels);
            truth = cv::Mat::zeros(originalSize, CV_8U);
            int colorChannels = rawTruth.channels() == 4 ? 3 : rawTruth.channels();
            for (int i = 0; i < colorChannels; ++i) {
                cv::bitwise_or(truth, channels[i] != 0, truth);
            }
        }

        const std::vector<std::string> filenames = {
            "01_input.png", "02_grayscale.png", "03_dct_mask.png", "04_orb_mask.png",
            "05_final_mask.png", "06_highlighted.png", "07_orb_matches.png",
            "08_dct_matches.png", "report.txt"
        };
        for (const auto& filename : filenames) {
            fs::path destination = outputDirectory / filename;
            if (sameFile(destination, inputPath) ||
                (argc == 4 && sameFile(destination, fs::path(argv[3])))) {
                throw std::runtime_error("Choose an output folder that will not overwrite the input or truth mask");
            }
        }
        fs::create_directories(outputDirectory);
        auto start = std::chrono::steady_clock::now();
        cv::setRNGSeed(0);

        // Both detectors and the saved results use the same working resolution.
        cv::Mat gray = preprocessImage(bgr);
        if (bgr.size() != gray.size()) {
            cv::resize(bgr, bgr, gray.size(), 0, 0, cv::INTER_AREA);
        }
        if (!truth.empty() && truth.size() != gray.size()) {
            cv::resize(truth, truth, gray.size(), 0, 0, cv::INTER_NEAREST);
        }
        DCTResult dct = detectDCT(gray);
        ORBResult orb = detectORB(gray);
        LocalizationResult localized = localizeForgery(bgr, dct.mask, orb.mask);
        double elapsedMs = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - start).count();

        // Save each stage so the detection can be inspected for the project report.
        std::vector<cv::Mat> images = {bgr, gray, dct.mask, orb.mask, localized.mask,
                                       localized.annotated, orb.visualization, dct.visualization};
        for (size_t i = 0; i < images.size(); ++i) {
            fs::path destination = outputDirectory / filenames[i];
            if (!cv::imwrite(destination.string(), images[i])) {
                throw std::runtime_error("Could not save: " + destination.string());
            }
        }

        double areaPercent = 100.0 * cv::countNonZero(localized.mask) / gray.total();
        std::string assessment = localized.boxes.empty()
            ? "NO STRONG COPY-MOVE INDICATORS" : "SUSPICIOUS MATCHED REGIONS";
        std::ofstream report(outputDirectory / "report.txt");
        if (!report) {
            throw std::runtime_error("Could not create report.txt");
        }
        report << std::fixed << std::setprecision(2)
               << "CLASSICAL DIP COPY-MOVE ANALYSIS\n"
               << "Image: " << inputPath.filename().string() << '\n'
               << "Original dimensions: " << originalSize.width << " x " << originalSize.height << '\n'
               << "Processed dimensions: " << gray.cols << " x " << gray.rows << '\n'
               << "DCT candidate pairs: " << dct.candidatePairs << '\n'
               << "DCT consistent pairs: " << dct.consistentPairs << '\n'
               << "DCT displacement groups: " << dct.displacementGroups << '\n'
               << "ORB candidate matches: " << orb.candidateMatches << '\n'
               << "ORB RANSAC inliers: " << orb.inliers << '\n'
               << "Suspicious connected regions: " << localized.boxes.size() << '\n'
               << "Flagged area: " << areaPercent << "%\n"
               << "Processing time (excluding image loading and saving): " << elapsedMs << " ms\n"
               << "Assessment: " << assessment << '\n'
               << "Bounding boxes (zero-based x y width height, processed pixels):\n";
        for (size_t i = 0; i < localized.boxes.size(); ++i) {
            const auto& box = localized.boxes[i];
            report << "  " << i + 1 << ": " << box.x << ' ' << box.y << ' '
                   << box.width << ' ' << box.height << '\n';
        }
        report << "Both source and copied regions may be marked. Natural repeated patterns can also match.\n"
               << "The result is an indicator, not proof of manipulation.\n";

        Metrics scores;
        if (!truth.empty()) {
            scores = evaluate(localized.mask, truth);
            report << "Ground-truth mask: " << fs::path(argv[3]).filename().string() << '\n'
                   << "Evaluation uses processed pixels and nearest-neighbor truth resizing.\n"
                   << "Zero denominators produce a score of 0, including two empty masks.\n"
                   << "TP: " << scores.truePositive << " FP: " << scores.falsePositive
                   << " FN: " << scores.falseNegative << '\n';
            writeMetrics(report, scores);
        } else {
            report << "Evaluation: no ground-truth mask supplied.\n";
        }
        report.close();
        if (!report) {
            throw std::runtime_error("Could not finish writing report.txt");
        }

        std::cout << std::fixed << std::setprecision(2)
                  << assessment << '\n'
                  << "DCT consistent pairs: " << dct.consistentPairs << '\n'
                  << "ORB inliers: " << orb.inliers << '\n'
                  << "Suspicious regions: " << localized.boxes.size() << '\n'
                  << "Flagged area: " << areaPercent << "%\n"
                  << "Elapsed time: " << elapsedMs << " ms\n";
        if (!truth.empty()) {
            writeMetrics(std::cout, scores);
        }
        std::cout << "Results saved in: " << fs::absolute(outputDirectory).string() << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << '\n';
        return 1;
    }
}
