#include "image_filters.h"
#include <iostream>
#include <string>
#include <map>
#include <functional>

using namespace iproc;

void printUsage(const char* program_name) {
    std::cout << "Usage: " << program_name << " <operation> <input_image> <output_image> [params...]\n\n";
    std::cout << "Operations:\n";
    std::cout << "  grayscale\n";
    std::cout << "  brightness <beta>           - Adjust brightness (-255 to 255)\n";
    std::cout << "  contrast <alpha>            - Adjust contrast (0.0 to 3.0)\n";
    std::cout << "  threshold <value>           - Binary threshold (0-255)\n";
    std::cout << "  invert\n";
    std::cout << "  gamma <gamma>               - Gamma correction\n";
    std::cout << "  salt-pepper <amount> [seed] - Add salt and pepper noise\n";
    std::cout << "  gaussian-noise <mean> <stddev> [seed]\n";
    std::cout << "  speckle-noise <variance> [seed]\n";
    std::cout << "  box-blur <kernel_size>\n";
    std::cout << "  gaussian-blur <kernel_size> <sigma>\n";
    std::cout << "  median <kernel_size>\n";
    std::cout << "  bilateral <d> <sigma_color> <sigma_space>\n";
    std::cout << "  sobel [kernel_size]\n";
    std::cout << "  canny <low> <high>\n";
    std::cout << "  sharpen\n";
    std::cout << "  prewitt\n";
    std::cout << "  laplacian [kernel_size]\n";
    std::cout << "  erosion <kernel_size> [iterations]\n";
    std::cout << "  dilation <kernel_size> [iterations]\n";
    std::cout << "  opening <kernel_size>\n";
    std::cout << "  closing <kernel_size>\n";
    std::cout << "  rotate <angle>\n";
    std::cout << "  scale <scale_x> <scale_y>\n";
    std::cout << "  translate <dx> <dy>\n";
    std::cout << "  flip-h                      - Flip horizontally\n";
    std::cout << "  flip-v                      - Flip vertically\n";
    std::cout << "  adjust-hsv <hue> <sat> <val>\n";
    std::cout << "  lab-tone <l_scale>\n";
    std::cout << "  otsu\n";
    std::cout << "  adaptive-gaussian <block_size> <C>\n";
    std::cout << "  sauvola <window_size> <k> [R]\n";
    std::cout << "  niblack <window_size> <k>\n";
}

int main(int argc, char* argv[]) {
    if (argc < 4) {
        printUsage(argv[0]);
        return 1;
    }
    
    std::string operation = argv[1];
    std::string input_path = argv[2];
    std::string output_path = argv[3];
    
    try {
        // Load input image
        cv::Mat input = loadImage(input_path);
        cv::Mat result;
        
        // Process based on operation
        if (operation == "grayscale") {
            result = convertToGrayscale(input);
        }
        else if (operation == "brightness" && argc >= 5) {
            int beta = std::stoi(argv[4]);
            result = adjustBrightness(input, beta);
        }
        else if (operation == "contrast" && argc >= 5) {
            double alpha = std::stod(argv[4]);
            result = adjustContrast(input, alpha);
        }
        else if (operation == "threshold" && argc >= 5) {
            int thresh = std::stoi(argv[4]);
            result = threshold(input, thresh);
        }
        else if (operation == "invert") {
            result = invert(input);
        }
        else if (operation == "gamma" && argc >= 5) {
            double gamma = std::stod(argv[4]);
            result = gammaCorrection(input, gamma);
        }
        else if (operation == "salt-pepper" && argc >= 5) {
            double amount = std::stod(argv[4]);
            unsigned int seed = (argc >= 6) ? std::stoul(argv[5]) : 0;
            result = addSaltPepperNoise(input, amount, seed);
        }
        else if (operation == "gaussian-noise" && argc >= 6) {
            double mean = std::stod(argv[4]);
            double stddev = std::stod(argv[5]);
            unsigned int seed = (argc >= 7) ? std::stoul(argv[6]) : 0;
            result = addGaussianNoise(input, mean, stddev, seed);
        }
        else if (operation == "speckle-noise" && argc >= 5) {
            double variance = std::stod(argv[4]);
            unsigned int seed = (argc >= 6) ? std::stoul(argv[5]) : 0;
            result = addSpeckleNoise(input, variance, seed);
        }
        else if (operation == "box-blur" && argc >= 5) {
            int kernel_size = std::stoi(argv[4]);
            result = boxBlur(input, kernel_size);
        }
        else if (operation == "gaussian-blur" && argc >= 6) {
            int kernel_size = std::stoi(argv[4]);
            double sigma = std::stod(argv[5]);
            result = gaussianBlur(input, kernel_size, sigma);
        }
        else if (operation == "median" && argc >= 5) {
            int kernel_size = std::stoi(argv[4]);
            result = medianFilter(input, kernel_size);
        }
        else if (operation == "bilateral" && argc >= 7) {
            int d = std::stoi(argv[4]);
            double sigma_color = std::stod(argv[5]);
            double sigma_space = std::stod(argv[6]);
            result = bilateralFilter(input, d, sigma_color, sigma_space);
        }
        else if (operation == "sobel") {
            int kernel_size = (argc >= 5) ? std::stoi(argv[4]) : 3;
            result = sobelEdgeDetection(input, kernel_size);
        }
        else if (operation == "canny" && argc >= 6) {
            double low = std::stod(argv[4]);
            double high = std::stod(argv[5]);
            result = cannyEdgeDetection(input, low, high);
        }
        else if (operation == "sharpen") {
            result = sharpenFilter(input);
        }
        else if (operation == "prewitt") {
            result = prewittEdgeDetection(input);
        }
        else if (operation == "laplacian") {
            int kernel_size = (argc >= 5) ? std::stoi(argv[4]) : 3;
            result = laplacianEdgeDetection(input, kernel_size);
        }
        else if (operation == "erosion" && argc >= 5) {
            int kernel_size = std::stoi(argv[4]);
            int iterations = (argc >= 6) ? std::stoi(argv[5]) : 1;
            result = erosion(input, kernel_size, iterations);
        }
        else if (operation == "dilation" && argc >= 5) {
            int kernel_size = std::stoi(argv[4]);
            int iterations = (argc >= 6) ? std::stoi(argv[5]) : 1;
            result = dilation(input, kernel_size, iterations);
        }
        else if (operation == "opening" && argc >= 5) {
            int kernel_size = std::stoi(argv[4]);
            result = opening(input, kernel_size);
        }
        else if (operation == "closing" && argc >= 5) {
            int kernel_size = std::stoi(argv[4]);
            result = closing(input, kernel_size);
        }
        else if (operation == "rotate" && argc >= 5) {
            double angle = std::stod(argv[4]);
            result = rotate(input, angle);
        }
        else if (operation == "scale" && argc >= 6) {
            double scale_x = std::stod(argv[4]);
            double scale_y = std::stod(argv[5]);
            result = scale(input, scale_x, scale_y);
        }
        else if (operation == "translate" && argc >= 6) {
            int dx = std::stoi(argv[4]);
            int dy = std::stoi(argv[5]);
            result = translate(input, dx, dy);
        }
        else if (operation == "flip-h") {
            result = flipHorizontal(input);
        }
        else if (operation == "flip-v") {
            result = flipVertical(input);
        }
        else if (operation == "adjust-hsv" && argc >= 7) {
            int hue = std::stoi(argv[4]);
            double sat = std::stod(argv[5]);
            double val = std::stod(argv[6]);
            result = adjustHSV(input, hue, sat, val);
        }
        else if (operation == "lab-tone" && argc >= 5) {
            double l_scale = std::stod(argv[4]);
            result = labToneMapping(input, l_scale);
        }
        else if (operation == "otsu") {
            result = otsuThreshold(input);
        }
        else if (operation == "adaptive-gaussian" && argc >= 6) {
            int block_size = std::stoi(argv[4]);
            double C = std::stod(argv[5]);
            result = adaptiveGaussianThreshold(input, block_size, C);
        }
        else if (operation == "sauvola" && argc >= 6) {
            int window_size = std::stoi(argv[4]);
            double k = std::stod(argv[5]);
            double R = (argc >= 7) ? std::stod(argv[6]) : 128.0;
            result = sauvolaThreshold(input, window_size, k, R);
        }
        else if (operation == "niblack" && argc >= 6) {
            int window_size = std::stoi(argv[4]);
            double k = std::stod(argv[5]);
            result = niblackThreshold(input, window_size, k);
        }
        else {
            std::cerr << "Unknown operation or missing parameters: " << operation << "\n";
            printUsage(argv[0]);
            return 1;
        }
        
        // Save result
        if (saveImage(output_path, result)) {
            std::cout << "Successfully saved result to " << output_path << "\n";
            return 0;
        } else {
            std::cerr << "Failed to save result to " << output_path << "\n";
            return 1;
        }
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
}
