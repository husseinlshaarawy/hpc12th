#include "image_filters.h"
#include "image_filters_omp.h"
#include <iostream>
#include <chrono>
#include <fstream>
#include <omp.h>

using namespace iproc;

struct BenchmarkResult {
    std::string operation;
    int num_threads;
    int image_size;
    double time_ms;
};

void runBenchmark(const std::string& operation, const cv::Mat& img, int threads, std::vector<BenchmarkResult>& results) {
    auto start = std::chrono::high_resolution_clock::now();
    
    if (operation == "gaussian_blur") {
        omp::gaussianBlur(img, 5, 1.0, threads);
    } else if (operation == "median_filter") {
        omp::medianFilter(img, 5, threads);
    } else if (operation == "box_blur") {
        omp::boxBlur(img, 5, threads);
    } else if (operation == "sobel") {
        cv::Mat gray = convertToGrayscale(img);
        omp::sobelEdgeDetection(gray, 3, threads);
    } else if (operation == "bilateral") {
        omp::bilateralFilter(img, 5, 50.0, 50.0, threads);
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    double duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    
    results.push_back({operation, threads, img.rows * img.cols, duration_ms});
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <input_image> [output_csv]\n";
        return 1;
    }
    
    std::string input_path = argv[1];
    std::string output_csv = (argc >= 3) ? argv[2] : "benchmark_openmp.csv";
    
    try {
        cv::Mat img = loadImage(input_path);
        std::vector<BenchmarkResult> results;
        
        int max_threads = omp_get_max_threads();
        std::cout << "Running OpenMP benchmarks on " << img.rows << "x" << img.cols << " image\n";
        std::cout << "Max threads: " << max_threads << "\n\n";
        
        std::vector<std::string> operations = {
            "gaussian_blur", "median_filter", "box_blur", "sobel", "bilateral"
        };
        
        std::vector<int> thread_counts = {1, 2, 4};
        if (max_threads >= 8) thread_counts.push_back(8);
        if (max_threads >= 16) thread_counts.push_back(16);
        
        for (const auto& op : operations) {
            for (int threads : thread_counts) {
                if (threads <= max_threads) {
                    std::cout << "  " << op << " with " << threads << " threads... " << std::flush;
                    runBenchmark(op, img, threads, results);
                    std::cout << results.back().time_ms << " ms\n";
                }
            }
        }
        
        // Write CSV
        std::ofstream csv(output_csv);
        csv << "Operation,NumThreads,ImageSize,Time_ms\n";
        for (const auto& result : results) {
            csv << result.operation << "," << result.num_threads << "," 
                << result.image_size << "," << result.time_ms << "\n";
        }
        csv.close();
        
        std::cout << "\nResults saved to " << output_csv << "\n";
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
}
