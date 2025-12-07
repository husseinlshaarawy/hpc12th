#include "image_filters.h"
#include "image_filters_omp.h"
#include <iostream>
#include <string>
#include <chrono>

using namespace iproc;

void printUsage(const char* program_name) {
    std::cout << "Usage: " << program_name << " <operation> <input> <output> <kernel_size> [threads]\n\n";
    std::cout << "OpenMP-accelerated operations:\n";
    std::cout << "  box-blur <kernel_size> [threads]\n";
    std::cout << "  gaussian-blur <kernel_size> <sigma> [threads]\n";
    std::cout << "  median <kernel_size> [threads]\n";
    std::cout << "  bilateral <d> <sigma_color> <sigma_space> [threads]\n";
    std::cout << "  sobel [kernel_size] [threads]\n";
    std::cout << "  prewitt [threads]\n";
}

int main(int argc, char* argv[]) {
    if (argc < 5) {
        printUsage(argv[0]);
        return 1;
    }
    
    std::string operation = argv[1];
    std::string input_path = argv[2];
    std::string output_path = argv[3];
    
    try {
        cv::Mat input = loadImage(input_path);
        cv::Mat result;
        
        auto start = std::chrono::high_resolution_clock::now();
        
        if (operation == "box-blur") {
            int kernel_size = std::stoi(argv[4]);
            int threads = (argc >= 6) ? std::stoi(argv[5]) : 0;
            result = omp::boxBlur(input, kernel_size, threads);
        }
        else if (operation == "gaussian-blur" && argc >= 6) {
            int kernel_size = std::stoi(argv[4]);
            double sigma = std::stod(argv[5]);
            int threads = (argc >= 7) ? std::stoi(argv[6]) : 0;
            result = omp::gaussianBlur(input, kernel_size, sigma, threads);
        }
        else if (operation == "median") {
            int kernel_size = std::stoi(argv[4]);
            int threads = (argc >= 6) ? std::stoi(argv[5]) : 0;
            result = omp::medianFilter(input, kernel_size, threads);
        }
        else if (operation == "bilateral" && argc >= 7) {
            int d = std::stoi(argv[4]);
            double sigma_color = std::stod(argv[5]);
            double sigma_space = std::stod(argv[6]);
            int threads = (argc >= 8) ? std::stoi(argv[7]) : 0;
            result = omp::bilateralFilter(input, d, sigma_color, sigma_space, threads);
        }
        else if (operation == "sobel") {
            int kernel_size = (argc >= 5) ? std::stoi(argv[4]) : 3;
            int threads = (argc >= 6) ? std::stoi(argv[5]) : 0;
            result = omp::sobelEdgeDetection(input, kernel_size, threads);
        }
        else if (operation == "prewitt") {
            int threads = (argc >= 5) ? std::stoi(argv[4]) : 0;
            result = omp::prewittEdgeDetection(input, threads);
        }
        else {
            std::cerr << "Unknown operation: " << operation << "\n";
            printUsage(argv[0]);
            return 1;
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        
        if (saveImage(output_path, result)) {
            std::cout << "Successfully processed image in " << duration.count() << " ms\n";
            std::cout << "Saved to " << output_path << "\n";
            return 0;
        } else {
            std::cerr << "Failed to save result\n";
            return 1;
        }
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
}
