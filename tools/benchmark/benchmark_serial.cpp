#include "image_filters.h"
#include <iostream>
#include <chrono>
#include <iomanip>
#include <fstream>

using namespace iproc;

struct BenchmarkResult {
    std::string operation;
    int image_size;
    double time_ms;
};

void runBenchmark(const std::string& operation, const cv::Mat& img, std::vector<BenchmarkResult>& results) {
    auto start = std::chrono::high_resolution_clock::now();
    
    if (operation == "gaussian_blur") {
        gaussianBlur(img, 5, 1.0);
    } else if (operation == "median_filter") {
        medianFilter(img, 5);
    } else if (operation == "box_blur") {
        boxBlur(img, 5);
    } else if (operation == "sobel") {
        cv::Mat gray = convertToGrayscale(img);
        sobelEdgeDetection(gray, 3);
    } else if (operation == "bilateral") {
        bilateralFilter(img, 5, 50.0, 50.0);
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    double duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    
    results.push_back({operation, img.rows * img.cols, duration_ms});
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <input_image> [output_csv]\n";
        return 1;
    }
    
    std::string input_path = argv[1];
    std::string output_csv = (argc >= 3) ? argv[2] : "benchmark_serial.csv";
    
    try {
        cv::Mat img = loadImage(input_path);
        std::vector<BenchmarkResult> results;
        
        std::cout << "Running serial benchmarks on " << img.rows << "x" << img.cols << " image...\n";
        
        std::vector<std::string> operations = {
            "gaussian_blur", "median_filter", "box_blur", "sobel", "bilateral"
        };
        
        for (const auto& op : operations) {
            std::cout << "  " << op << "... " << std::flush;
            runBenchmark(op, img, results);
            std::cout << results.back().time_ms << " ms\n";
        }
        
        // Write CSV
        std::ofstream csv(output_csv);
        csv << "Operation,ImageSize,Time_ms\n";
        for (const auto& result : results) {
            csv << result.operation << "," << result.image_size << "," << result.time_ms << "\n";
        }
        csv.close();
        
        std::cout << "\nResults saved to " << output_csv << "\n";
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
}
