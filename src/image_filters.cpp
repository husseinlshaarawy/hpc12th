#include "image_filters.h"
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <random>
#include <algorithm>
#include <cmath>
#include <queue>

namespace iproc {

// ============================================================================
// IMAGE I/O
// ============================================================================

cv::Mat loadImage(const std::string& filepath, int color_mode) {
    cv::Mat img = cv::imread(filepath, color_mode);
    if (img.empty()) {
        throw std::runtime_error("Failed to load image: " + filepath);
    }
    return img;
}

bool saveImage(const std::string& filepath, const cv::Mat& image) {
    return cv::imwrite(filepath, image);
}

// ============================================================================
// POINT OPERATIONS
// ============================================================================

cv::Mat convertToGrayscale(const cv::Mat& src) {
    if (src.channels() == 1) {
        return src.clone();
    }
    cv::Mat gray;
    cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);
    return gray;
}

cv::Mat adjustBrightness(const cv::Mat& src, int beta) {
    cv::Mat result;
    src.convertTo(result, -1, 1.0, beta);
    return result;
}

cv::Mat adjustContrast(const cv::Mat& src, double alpha) {
    cv::Mat result;
    src.convertTo(result, -1, alpha, 0);
    return result;
}

cv::Mat threshold(const cv::Mat& src, int threshold_value, int max_value) {
    cv::Mat result;
    cv::threshold(src, result, threshold_value, max_value, cv::THRESH_BINARY);
    return result;
}

cv::Mat invert(const cv::Mat& src) {
    cv::Mat result;
    cv::bitwise_not(src, result);
    return result;
}

cv::Mat gammaCorrection(const cv::Mat& src, double gamma) {
    // Build lookup table
    cv::Mat lut(1, 256, CV_8U);
    uchar* p = lut.ptr();
    for (int i = 0; i < 256; ++i) {
        p[i] = cv::saturate_cast<uchar>(std::pow(i / 255.0, gamma) * 255.0);
    }
    
    cv::Mat result;
    cv::LUT(src, lut, result);
    return result;
}

// ============================================================================
// NOISE GENERATION
// ============================================================================

cv::Mat addSaltPepperNoise(const cv::Mat& src, double amount, unsigned int seed) {
    cv::Mat result = src.clone();
    std::mt19937 rng(seed == 0 ? std::random_device{}() : seed);
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    
    for (int i = 0; i < result.rows; ++i) {
        for (int j = 0; j < result.cols; ++j) {
            double rand_val = dist(rng);
            if (rand_val < amount / 2.0) {
                // Salt (white)
                if (result.channels() == 1) {
                    result.at<uchar>(i, j) = 255;
                } else {
                    result.at<cv::Vec3b>(i, j) = cv::Vec3b(255, 255, 255);
                }
            } else if (rand_val < amount) {
                // Pepper (black)
                if (result.channels() == 1) {
                    result.at<uchar>(i, j) = 0;
                } else {
                    result.at<cv::Vec3b>(i, j) = cv::Vec3b(0, 0, 0);
                }
            }
        }
    }
    
    return result;
}

cv::Mat addGaussianNoise(const cv::Mat& src, double mean, double stddev, unsigned int seed) {
    cv::Mat noise(src.size(), src.type());
    std::mt19937 rng(seed == 0 ? std::random_device{}() : seed);
    std::normal_distribution<double> dist(mean, stddev);
    
    for (int i = 0; i < noise.rows; ++i) {
        for (int j = 0; j < noise.cols; ++j) {
            if (noise.channels() == 1) {
                noise.at<uchar>(i, j) = cv::saturate_cast<uchar>(dist(rng));
            } else {
                for (int c = 0; c < noise.channels(); ++c) {
                    noise.at<cv::Vec3b>(i, j)[c] = cv::saturate_cast<uchar>(dist(rng));
                }
            }
        }
    }
    
    cv::Mat result;
    cv::add(src, noise, result);
    return result;
}

cv::Mat addSpeckleNoise(const cv::Mat& src, double variance, unsigned int seed) {
    cv::Mat noise(src.size(), CV_32F);
    std::mt19937 rng(seed == 0 ? std::random_device{}() : seed);
    std::normal_distribution<double> dist(1.0, std::sqrt(variance));
    
    for (int i = 0; i < noise.rows; ++i) {
        for (int j = 0; j < noise.cols; ++j) {
            noise.at<float>(i, j) = static_cast<float>(dist(rng));
        }
    }
    
    cv::Mat src_float, result_float;
    src.convertTo(src_float, CV_32F);
    cv::multiply(src_float, noise, result_float);
    
    cv::Mat result;
    result_float.convertTo(result, src.type());
    return result;
}

// ============================================================================
// SMOOTHING FILTERS
// ============================================================================

cv::Mat boxBlur(const cv::Mat& src, int kernel_size) {
    cv::Mat result;
    cv::blur(src, result, cv::Size(kernel_size, kernel_size));
    return result;
}

cv::Mat gaussianBlur(const cv::Mat& src, int kernel_size, double sigma) {
    cv::Mat result;
    cv::GaussianBlur(src, result, cv::Size(kernel_size, kernel_size), sigma);
    return result;
}

cv::Mat medianFilter(const cv::Mat& src, int kernel_size) {
    cv::Mat result;
    cv::medianBlur(src, result, kernel_size);
    return result;
}

cv::Mat bilateralFilter(const cv::Mat& src, int d, double sigma_color, double sigma_space) {
    cv::Mat result;
    cv::bilateralFilter(src, result, d, sigma_color, sigma_space);
    return result;
}

// ============================================================================
// EDGE DETECTION & SHARPENING
// ============================================================================

cv::Mat sobelEdgeDetection(const cv::Mat& src, int kernel_size) {
    cv::Mat grad_x, grad_y;
    cv::Mat abs_grad_x, abs_grad_y;
    
    // Gradient X
    cv::Sobel(src, grad_x, CV_16S, 1, 0, kernel_size);
    cv::convertScaleAbs(grad_x, abs_grad_x);
    
    // Gradient Y
    cv::Sobel(src, grad_y, CV_16S, 0, 1, kernel_size);
    cv::convertScaleAbs(grad_y, abs_grad_y);
    
    // Combine gradients
    cv::Mat result;
    cv::addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, result);
    
    return result;
}

cv::Mat cannyEdgeDetection(const cv::Mat& src, double low_threshold, double high_threshold) {
    cv::Mat result;
    cv::Canny(src, result, low_threshold, high_threshold);
    return result;
}

cv::Mat sharpenFilter(const cv::Mat& src) {
    // Sharpen kernel: center = 5, neighbors = -1
    cv::Mat kernel = (cv::Mat_<float>(3, 3) << 
                      0, -1, 0,
                      -1, 5, -1,
                      0, -1, 0);
    
    cv::Mat result;
    cv::filter2D(src, result, src.depth(), kernel);
    return result;
}

cv::Mat prewittEdgeDetection(const cv::Mat& src) {
    // Prewitt kernels
    cv::Mat kernel_x = (cv::Mat_<float>(3, 3) << 
                        -1, 0, 1,
                        -1, 0, 1,
                        -1, 0, 1);
    
    cv::Mat kernel_y = (cv::Mat_<float>(3, 3) << 
                        -1, -1, -1,
                        0, 0, 0,
                        1, 1, 1);
    
    cv::Mat grad_x, grad_y;
    cv::filter2D(src, grad_x, CV_16S, kernel_x);
    cv::filter2D(src, grad_y, CV_16S, kernel_y);
    
    cv::Mat abs_grad_x, abs_grad_y;
    cv::convertScaleAbs(grad_x, abs_grad_x);
    cv::convertScaleAbs(grad_y, abs_grad_y);
    
    cv::Mat result;
    cv::addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, result);
    
    return result;
}

cv::Mat laplacianEdgeDetection(const cv::Mat& src, int kernel_size) {
    cv::Mat result;
    cv::Laplacian(src, result, CV_16S, kernel_size);
    cv::convertScaleAbs(result, result);
    return result;
}

// ============================================================================
// MORPHOLOGICAL OPERATIONS
// ============================================================================

cv::Mat erosion(const cv::Mat& src, int kernel_size, int iterations) {
    cv::Mat element = cv::getStructuringElement(cv::MORPH_RECT, 
                                                cv::Size(kernel_size, kernel_size));
    cv::Mat result;
    cv::erode(src, result, element, cv::Point(-1, -1), iterations);
    return result;
}

cv::Mat dilation(const cv::Mat& src, int kernel_size, int iterations) {
    cv::Mat element = cv::getStructuringElement(cv::MORPH_RECT, 
                                                cv::Size(kernel_size, kernel_size));
    cv::Mat result;
    cv::dilate(src, result, element, cv::Point(-1, -1), iterations);
    return result;
}

cv::Mat opening(const cv::Mat& src, int kernel_size) {
    cv::Mat element = cv::getStructuringElement(cv::MORPH_RECT, 
                                                cv::Size(kernel_size, kernel_size));
    cv::Mat result;
    cv::morphologyEx(src, result, cv::MORPH_OPEN, element);
    return result;
}

cv::Mat closing(const cv::Mat& src, int kernel_size) {
    cv::Mat element = cv::getStructuringElement(cv::MORPH_RECT, 
                                                cv::Size(kernel_size, kernel_size));
    cv::Mat result;
    cv::morphologyEx(src, result, cv::MORPH_CLOSE, element);
    return result;
}

cv::Mat morphologicalReconstruction(const cv::Mat& marker, const cv::Mat& mask, int connectivity) {
    cv::Mat result = marker.clone();
    cv::Mat prev;
    
    cv::Mat element;
    if (connectivity == 4) {
        element = (cv::Mat_<uchar>(3, 3) << 
                   0, 1, 0,
                   1, 1, 1,
                   0, 1, 0);
    } else {
        element = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
    }
    
    do {
        prev = result.clone();
        cv::dilate(result, result, element);
        cv::min(result, mask, result);
    } while (cv::countNonZero(result != prev) > 0);
    
    return result;
}

// ============================================================================
// GEOMETRIC TRANSFORMATIONS
// ============================================================================

cv::Mat rotate(const cv::Mat& src, double angle) {
    cv::Point2f center(src.cols / 2.0f, src.rows / 2.0f);
    cv::Mat rotation_matrix = cv::getRotationMatrix2D(center, angle, 1.0);
    
    cv::Mat result;
    cv::warpAffine(src, result, rotation_matrix, src.size());
    return result;
}

cv::Mat scale(const cv::Mat& src, double scale_x, double scale_y, int interpolation) {
    cv::Mat result;
    cv::resize(src, result, cv::Size(), scale_x, scale_y, interpolation);
    return result;
}

cv::Mat translate(const cv::Mat& src, int dx, int dy) {
    cv::Mat translation_matrix = (cv::Mat_<double>(2, 3) << 
                                  1, 0, dx,
                                  0, 1, dy);
    
    cv::Mat result;
    cv::warpAffine(src, result, translation_matrix, src.size());
    return result;
}

cv::Mat flipHorizontal(const cv::Mat& src) {
    cv::Mat result;
    cv::flip(src, result, 1);
    return result;
}

cv::Mat flipVertical(const cv::Mat& src) {
    cv::Mat result;
    cv::flip(src, result, 0);
    return result;
}

cv::Mat perspectiveTransform(const cv::Mat& src,
                            const std::vector<cv::Point2f>& src_points,
                            const std::vector<cv::Point2f>& dst_points,
                            int interpolation) {
    if (src_points.size() != 4 || dst_points.size() != 4) {
        throw std::invalid_argument("Perspective transform requires exactly 4 points");
    }
    
    cv::Mat transform_matrix = cv::getPerspectiveTransform(src_points, dst_points);
    
    cv::Mat result;
    cv::warpPerspective(src, result, transform_matrix, src.size(), interpolation);
    return result;
}

// ============================================================================
// COLOR/CHANNEL OPERATIONS
// ============================================================================

std::vector<cv::Mat> splitChannels(const cv::Mat& src) {
    std::vector<cv::Mat> channels;
    cv::split(src, channels);
    return channels;
}

cv::Mat mergeChannels(const std::vector<cv::Mat>& channels) {
    cv::Mat result;
    cv::merge(channels, result);
    return result;
}

cv::Mat adjustHSV(const cv::Mat& src, int hue_shift, double sat_scale, double val_scale) {
    cv::Mat hsv;
    cv::cvtColor(src, hsv, cv::COLOR_BGR2HSV);
    
    for (int i = 0; i < hsv.rows; ++i) {
        for (int j = 0; j < hsv.cols; ++j) {
            cv::Vec3b& pixel = hsv.at<cv::Vec3b>(i, j);
            
            // Adjust hue (wraps around at 180)
            int new_hue = (pixel[0] + hue_shift) % 180;
            if (new_hue < 0) new_hue += 180;
            pixel[0] = static_cast<uchar>(new_hue);
            
            // Adjust saturation
            pixel[1] = cv::saturate_cast<uchar>(pixel[1] * sat_scale);
            
            // Adjust value
            pixel[2] = cv::saturate_cast<uchar>(pixel[2] * val_scale);
        }
    }
    
    cv::Mat result;
    cv::cvtColor(hsv, result, cv::COLOR_HSV2BGR);
    return result;
}

cv::Mat labToneMapping(const cv::Mat& src, double l_scale) {
    cv::Mat lab;
    cv::cvtColor(src, lab, cv::COLOR_BGR2Lab);
    
    std::vector<cv::Mat> channels;
    cv::split(lab, channels);
    
    // Scale L channel
    channels[0].convertTo(channels[0], -1, l_scale, 0);
    
    cv::merge(channels, lab);
    
    cv::Mat result;
    cv::cvtColor(lab, result, cv::COLOR_Lab2BGR);
    return result;
}

// ============================================================================
// ADVANCED THRESHOLDING (BONUS)
// ============================================================================

cv::Mat otsuThreshold(const cv::Mat& src) {
    cv::Mat result;
    cv::threshold(src, result, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
    return result;
}

cv::Mat adaptiveGaussianThreshold(const cv::Mat& src, int block_size, double C) {
    cv::Mat result;
    cv::adaptiveThreshold(src, result, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C, 
                         cv::THRESH_BINARY, block_size, C);
    return result;
}

cv::Mat sauvolaThreshold(const cv::Mat& src, int window_size, double k, double R) {
    cv::Mat result = cv::Mat::zeros(src.size(), src.type());
    cv::Mat src_float;
    src.convertTo(src_float, CV_32F);
    
    int half_win = window_size / 2;
    
    for (int i = 0; i < src.rows; ++i) {
        for (int j = 0; j < src.cols; ++j) {
            // Calculate local mean and standard deviation
            double sum = 0.0, sum_sq = 0.0;
            int count = 0;
            
            for (int wi = std::max(0, i - half_win); wi <= std::min(src.rows - 1, i + half_win); ++wi) {
                for (int wj = std::max(0, j - half_win); wj <= std::min(src.cols - 1, j + half_win); ++wj) {
                    float val = src_float.at<float>(wi, wj);
                    sum += val;
                    sum_sq += val * val;
                    ++count;
                }
            }
            
            double mean = sum / count;
            double variance = (sum_sq / count) - (mean * mean);
            double stddev = std::sqrt(std::max(0.0, variance));
            
            // Sauvola threshold
            double threshold = mean * (1.0 + k * ((stddev / R) - 1.0));
            
            result.at<uchar>(i, j) = src.at<uchar>(i, j) > threshold ? 255 : 0;
        }
    }
    
    return result;
}

cv::Mat niblackThreshold(const cv::Mat& src, int window_size, double k) {
    cv::Mat result = cv::Mat::zeros(src.size(), src.type());
    cv::Mat src_float;
    src.convertTo(src_float, CV_32F);
    
    int half_win = window_size / 2;
    
    for (int i = 0; i < src.rows; ++i) {
        for (int j = 0; j < src.cols; ++j) {
            // Calculate local mean and standard deviation
            double sum = 0.0, sum_sq = 0.0;
            int count = 0;
            
            for (int wi = std::max(0, i - half_win); wi <= std::min(src.rows - 1, i + half_win); ++wi) {
                for (int wj = std::max(0, j - half_win); wj <= std::min(src.cols - 1, j + half_win); ++wj) {
                    float val = src_float.at<float>(wi, wj);
                    sum += val;
                    sum_sq += val * val;
                    ++count;
                }
            }
            
            double mean = sum / count;
            double variance = (sum_sq / count) - (mean * mean);
            double stddev = std::sqrt(std::max(0.0, variance));
            
            // Niblack threshold
            double threshold = mean + k * stddev;
            
            result.at<uchar>(i, j) = src.at<uchar>(i, j) > threshold ? 255 : 0;
        }
    }
    
    return result;
}

} // namespace iproc
