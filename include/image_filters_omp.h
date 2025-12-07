#ifndef IMAGE_FILTERS_OMP_H
#define IMAGE_FILTERS_OMP_H

#include "image_filters.h"
#include <opencv2/opencv.hpp>

namespace iproc {
namespace omp {

/**
 * @brief OpenMP-parallelized box blur
 * Uses row-wise parallelization with explicit data scoping
 * @param src Source image
 * @param kernel_size Size of box kernel (must be odd)
 * @param num_threads Number of OpenMP threads (0 = auto)
 * @return Blurred image
 */
cv::Mat boxBlur(const cv::Mat& src, int kernel_size, int num_threads = 0);

/**
 * @brief OpenMP-parallelized Gaussian blur
 * Uses separable convolution with row-wise parallelization
 * @param src Source image
 * @param kernel_size Size of Gaussian kernel (must be odd)
 * @param sigma Standard deviation
 * @param num_threads Number of OpenMP threads (0 = auto)
 * @return Blurred image
 */
cv::Mat gaussianBlur(const cv::Mat& src, int kernel_size, double sigma, int num_threads = 0);

/**
 * @brief OpenMP-parallelized median filter
 * Uses row-wise parallelization with thread-private buffers
 * @param src Source image
 * @param kernel_size Size of median kernel (must be odd)
 * @param num_threads Number of OpenMP threads (0 = auto)
 * @return Filtered image
 */
cv::Mat medianFilter(const cv::Mat& src, int kernel_size, int num_threads = 0);

/**
 * @brief OpenMP-parallelized bilateral filter
 * Uses tile-based parallelization for better cache locality
 * @param src Source image
 * @param d Diameter of pixel neighborhood
 * @param sigma_color Filter sigma in color space
 * @param sigma_space Filter sigma in coordinate space
 * @param num_threads Number of OpenMP threads (0 = auto)
 * @return Filtered image
 */
cv::Mat bilateralFilter(const cv::Mat& src, int d, double sigma_color, double sigma_space, int num_threads = 0);

/**
 * @brief OpenMP-parallelized Sobel edge detection
 * @param src Source grayscale image
 * @param kernel_size Size of Sobel kernel
 * @param num_threads Number of OpenMP threads (0 = auto)
 * @return Edge-detected image
 */
cv::Mat sobelEdgeDetection(const cv::Mat& src, int kernel_size = 3, int num_threads = 0);

/**
 * @brief OpenMP-parallelized Prewitt edge detection
 * @param src Source grayscale image
 * @param num_threads Number of OpenMP threads (0 = auto)
 * @return Edge-detected image
 */
cv::Mat prewittEdgeDetection(const cv::Mat& src, int num_threads = 0);

/**
 * @brief OpenMP-parallelized convolution (general purpose)
 * @param src Source image
 * @param kernel Convolution kernel
 * @param num_threads Number of OpenMP threads (0 = auto)
 * @return Convolved image
 */
cv::Mat convolve2D(const cv::Mat& src, const cv::Mat& kernel, int num_threads = 0);

} // namespace omp
} // namespace iproc

#endif // IMAGE_FILTERS_OMP_H
