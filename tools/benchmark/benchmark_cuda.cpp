#include "image_filters.h"
#ifdef BUILD_CUDA
#include "image_filters_cuda.h"
#endif
#include <iostream>
#include <chrono>
#include <fstream>

using namespace iproc;

int main(int argc, char* argv[]) {
#ifndef BUILD_CUDA
    std::cerr << "CUDA support not compiled. Rebuild with -DBUILD_CUDA=ON\n";
    return 1;
#else
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <input_image> [output_csv]\n";
        return 1;
    }
    
    std::string input_path = argv[1];
    std::string output_csv = (argc >= 3) ? argv[2] : "benchmark_cuda.csv";
    
    try {
        if (!cuda::initialize()) {
            std::cerr << "Failed to initialize CUDA\n";
            return 1;
        }
        
        cv::Mat img = loadImage(input_path);
        std::cout << "Running CUDA benchmarks on " << img.rows << "x" << img.cols << " image\n";
        
        std::vector<std::pair<std::string, double>> results;
        std::vector<std::string> operations = {"gaussian_blur", "median_filter", "sobel"};
        
        for (const auto& op : operations) {
            auto start = std::chrono::high_resolution_clock::now();
            
            if (op == "gaussian_blur") {
                cuda::gaussianBlur(img, 5, 1.0);
            } else if (op == "median_filter") {
                cuda::medianFilter(img, 5);
            } else if (op == "sobel") {
                cuda::sobelEdgeDetection(img, 3);
            }
            
            auto end = std::chrono::high_resolution_clock::now();
            double duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
            
            results.push_back({op, duration_ms});
            std::cout << "  " << op << ": " << duration_ms << " ms\n";
        }
        
        std::ofstream csv(output_csv);
        csv << "Operation,Time_ms\n";
        for (const auto& [op, time] : results) {
            csv << op << "," << time << "\n";
        }
        csv.close();
        
        std::cout << "\nResults saved to " << output_csv << "\n";
        
        cuda::cleanup();
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        cuda::cleanup();
        return 1;
    }
#endif
}
