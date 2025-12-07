#ifndef IMAGE_FILTERS_CUDA_H
#define IMAGE_FILTERS_CUDA_H

#include "image_filters.h"
#include <opencv2/opencv.hpp>

namespace iproc {
namespace cuda {

/**
 * @brief Initialize CUDA resources (must be called once)
 * @return true if CUDA is available
 */
bool initialize();

/**
 * @brief Cleanup CUDA resources
 */
void cleanup();

/**
 * @brief Check if CUDA is available on this system
 * @return true if CUDA is available
 */
bool isAvailable();

/**
 * @brief CUDA-accelerated Gaussian blur with separable convolution
 * Uses pinned memory, async transfers, and shared memory
 * @param src Source image
 * @param kernel_size Size of Gaussian kernel (must be odd)
 * @param sigma Standard deviation
 * @return Blurred image
 */
cv::Mat gaussianBlur(const cv::Mat& src, int kernel_size, double sigma);

/**
 * @brief CUDA-accelerated median filter
 * Uses shared memory and thread block tiling
 * @param src Source image
 * @param kernel_size Size of median kernel (must be odd)
 * @return Filtered image
 */
cv::Mat medianFilter(const cv::Mat& src, int kernel_size);

/**
 * @brief CUDA-accelerated 2D convolution
 * Uses shared memory and constant memory for kernel
 * @param src Source image
 * @param kernel Convolution kernel
 * @return Convolved image
 */
cv::Mat convolve2D(const cv::Mat& src, const cv::Mat& kernel);

/**
 * @brief CUDA-accelerated Sobel edge detection
 * @param src Source grayscale image
 * @param kernel_size Size of Sobel kernel
 * @return Edge-detected image
 */
cv::Mat sobelEdgeDetection(const cv::Mat& src, int kernel_size = 3);

/**
 * @brief CUDA-accelerated rotation
 * Uses texture memory for interpolation
 * @param src Source image
 * @param angle Rotation angle in degrees
 * @return Rotated image
 */
cv::Mat rotate(const cv::Mat& src, double angle);

/**
 * @brief CUDA-accelerated scaling
 * @param src Source image
 * @param scale_x Horizontal scale factor
 * @param scale_y Vertical scale factor
 * @return Scaled image
 */
cv::Mat scale(const cv::Mat& src, double scale_x, double scale_y);

/**
 * @brief Chain multiple CUDA filters without host round-trips
 * Keeps data on GPU between operations
 * @param src Source image
 * @param operations Vector of operation descriptors
 * @return Final processed image
 */
cv::Mat chainFilters(const cv::Mat& src, const std::vector<std::string>& operations);

} // namespace cuda
} // namespace iproc

#endif // IMAGE_FILTERS_CUDA_H
