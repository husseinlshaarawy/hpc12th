#include "image_filters.h"
#ifdef BUILD_CUDA
#include "image_filters_cuda.h"
#endif
#include <iostream>
#include <string>

using namespace iproc;

void printUsage(const char* program_name) {
    std::cout << "Usage: " << program_name << " <operation> <input> <output> <params...>\n\n";
#ifdef BUILD_CUDA
    std::cout << "CUDA-accelerated operations:\n";
    std::cout << "  gaussian-blur <kernel_size> <sigma>\n";
    std::cout << "  median <kernel_size>\n";
    std::cout << "  sobel\n";
    std::cout << "  rotate <angle>\n";
#else
    std::cout << "CUDA support not compiled. Rebuild with -DBUILD_CUDA=ON\n";
#endif
}

int main(int argc, char* argv[]) {
#ifndef BUILD_CUDA
    std::cerr << "This executable was built without CUDA support.\n";
    std::cerr << "Rebuild with -DBUILD_CUDA=ON to enable CUDA.\n";
    return 1;
#else
    if (argc < 4) {
        printUsage(argv[0]);
        return 1;
    }
    
    std::string operation = argv[1];
    std::string input_path = argv[2];
    std::string output_path = argv[3];
    
    try {
        if (!cuda::isAvailable()) {
            std::cerr << "CUDA is not available on this system.\n";
            return 1;
        }
        
        if (!cuda::initialize()) {
            std::cerr << "Failed to initialize CUDA.\n";
            return 1;
        }
        
        cv::Mat input = loadImage(input_path);
        cv::Mat result;
        
        if (operation == "gaussian-blur" && argc >= 6) {
            int kernel_size = std::stoi(argv[4]);
            double sigma = std::stod(argv[5]);
            result = cuda::gaussianBlur(input, kernel_size, sigma);
        }
        else if (operation == "median" && argc >= 5) {
            int kernel_size = std::stoi(argv[4]);
            result = cuda::medianFilter(input, kernel_size);
        }
        else if (operation == "sobel") {
            result = cuda::sobelEdgeDetection(input);
        }
        else if (operation == "rotate" && argc >= 5) {
            double angle = std::stod(argv[4]);
            result = cuda::rotate(input, angle);
        }
        else {
            std::cerr << "Unknown operation: " << operation << "\n";
            printUsage(argv[0]);
            cuda::cleanup();
            return 1;
        }
        
        if (saveImage(output_path, result)) {
            std::cout << "Successfully saved result to " << output_path << "\n";
        } else {
            std::cerr << "Failed to save result\n";
            cuda::cleanup();
            return 1;
        }
        
        cuda::cleanup();
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
#ifdef BUILD_CUDA
        cuda::cleanup();
#endif
        return 1;
    }
#endif
}
