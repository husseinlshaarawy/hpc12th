#ifndef IMAGE_FILTERS_H
#define IMAGE_FILTERS_H

#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

namespace iproc {

// ============================================================================
// IMAGE I/O
// ============================================================================

/**
 * @brief Load an image from file, preserving color/grayscale mode
 * @param filepath Path to image file (.jpg, .png, .bmp)
 * @param color_mode cv::IMREAD_COLOR, cv::IMREAD_GRAYSCALE, or cv::IMREAD_UNCHANGED
 * @return Loaded image as cv::Mat
 */
cv::Mat loadImage(const std::string& filepath, int color_mode = cv::IMREAD_UNCHANGED);

/**
 * @brief Save an image to file in specified format
 * @param filepath Output path with extension (.jpg, .png, .bmp)
 * @param image Image to save
 * @return true if successful
 */
bool saveImage(const std::string& filepath, const cv::Mat& image);

// ============================================================================
// POINT OPERATIONS
// ============================================================================

/**
 * @brief Convert color image to grayscale
 * @param src Source image (BGR or RGB)
 * @return Grayscale image
 */
cv::Mat convertToGrayscale(const cv::Mat& src);

/**
 * @brief Adjust image brightness
 * @param src Source image
 * @param beta Brightness offset (-255 to 255)
 * @return Brightness-adjusted image
 */
cv::Mat adjustBrightness(const cv::Mat& src, int beta);

/**
 * @brief Adjust image contrast
 * @param src Source image
 * @param alpha Contrast factor (0.0 to 3.0, 1.0 = no change)
 * @return Contrast-adjusted image
 */
cv::Mat adjustContrast(const cv::Mat& src, double alpha);

/**
 * @brief Apply global binary thresholding
 * @param src Source grayscale image
 * @param threshold Threshold value (0-255)
 * @param max_value Maximum value for thresholded pixels
 * @return Binary image
 */
cv::Mat threshold(const cv::Mat& src, int threshold, int max_value = 255);

/**
 * @brief Invert image colors
 * @param src Source image
 * @return Inverted image
 */
cv::Mat invert(const cv::Mat& src);

/**
 * @brief Apply gamma correction
 * @param src Source image
 * @param gamma Gamma value (< 1.0 brightens, > 1.0 darkens)
 * @return Gamma-corrected image
 */
cv::Mat gammaCorrection(const cv::Mat& src, double gamma);

// ============================================================================
// NOISE GENERATION
// ============================================================================

/**
 * @brief Add salt-and-pepper noise to image
 * @param src Source image
 * @param amount Noise density (0.0 to 1.0)
 * @param seed Random seed for reproducibility
 * @return Noisy image
 */
cv::Mat addSaltPepperNoise(const cv::Mat& src, double amount, unsigned int seed = 0);

/**
 * @brief Add Gaussian noise to image
 * @param src Source image
 * @param mean Mean of Gaussian distribution
 * @param stddev Standard deviation of Gaussian distribution
 * @param seed Random seed for reproducibility
 * @return Noisy image
 */
cv::Mat addGaussianNoise(const cv::Mat& src, double mean, double stddev, unsigned int seed = 0);

/**
 * @brief Add speckle noise to image (bonus feature)
 * @param src Source image
 * @param variance Variance of speckle noise
 * @param seed Random seed for reproducibility
 * @return Noisy image
 */
cv::Mat addSpeckleNoise(const cv::Mat& src, double variance, unsigned int seed = 0);

// ============================================================================
// SMOOTHING FILTERS
// ============================================================================

/**
 * @brief Apply box blur (mean filter)
 * @param src Source image
 * @param kernel_size Size of box kernel (must be odd)
 * @return Blurred image
 */
cv::Mat boxBlur(const cv::Mat& src, int kernel_size);

/**
 * @brief Apply Gaussian blur
 * @param src Source image
 * @param kernel_size Size of Gaussian kernel (must be odd)
 * @param sigma Standard deviation of Gaussian
 * @return Blurred image
 */
cv::Mat gaussianBlur(const cv::Mat& src, int kernel_size, double sigma);

/**
 * @brief Apply median filter
 * @param src Source image
 * @param kernel_size Size of median kernel (must be odd)
 * @return Filtered image
 */
cv::Mat medianFilter(const cv::Mat& src, int kernel_size);

/**
 * @brief Apply bilateral filter (bonus feature)
 * @param src Source image
 * @param d Diameter of pixel neighborhood
 * @param sigma_color Filter sigma in color space
 * @param sigma_space Filter sigma in coordinate space
 * @return Filtered image
 */
cv::Mat bilateralFilter(const cv::Mat& src, int d, double sigma_color, double sigma_space);

// ============================================================================
// EDGE DETECTION & SHARPENING
// ============================================================================

/**
 * @brief Apply Sobel edge detection
 * @param src Source grayscale image
 * @param kernel_size Size of Sobel kernel (1, 3, 5, 7)
 * @return Edge-detected image
 */
cv::Mat sobelEdgeDetection(const cv::Mat& src, int kernel_size = 3);

/**
 * @brief Apply Canny edge detection
 * @param src Source grayscale image
 * @param low_threshold Lower threshold for hysteresis
 * @param high_threshold Upper threshold for hysteresis
 * @return Edge-detected image
 */
cv::Mat cannyEdgeDetection(const cv::Mat& src, double low_threshold, double high_threshold);

/**
 * @brief Apply sharpen filter
 * @param src Source image
 * @return Sharpened image
 */
cv::Mat sharpenFilter(const cv::Mat& src);

/**
 * @brief Apply Prewitt edge detection (bonus feature)
 * @param src Source grayscale image
 * @return Edge-detected image
 */
cv::Mat prewittEdgeDetection(const cv::Mat& src);

/**
 * @brief Apply Laplacian edge detection (bonus feature)
 * @param src Source grayscale image
 * @param kernel_size Size of aperture (1, 3, 5, 7)
 * @return Edge-detected image
 */
cv::Mat laplacianEdgeDetection(const cv::Mat& src, int kernel_size = 3);

// ============================================================================
// MORPHOLOGICAL OPERATIONS
// ============================================================================

/**
 * @brief Apply morphological erosion
 * @param src Source binary image
 * @param kernel_size Size of structuring element
 * @param iterations Number of times to apply
 * @return Eroded image
 */
cv::Mat erosion(const cv::Mat& src, int kernel_size, int iterations = 1);

/**
 * @brief Apply morphological dilation
 * @param src Source binary image
 * @param kernel_size Size of structuring element
 * @param iterations Number of times to apply
 * @return Dilated image
 */
cv::Mat dilation(const cv::Mat& src, int kernel_size, int iterations = 1);

/**
 * @brief Apply morphological opening (erosion followed by dilation)
 * @param src Source binary image
 * @param kernel_size Size of structuring element
 * @return Opened image
 */
cv::Mat opening(const cv::Mat& src, int kernel_size);

/**
 * @brief Apply morphological closing (dilation followed by erosion)
 * @param src Source binary image
 * @param kernel_size Size of structuring element
 * @return Closed image
 */
cv::Mat closing(const cv::Mat& src, int kernel_size);

/**
 * @brief Morphological reconstruction (bonus feature)
 * @param marker Marker image
 * @param mask Mask image
 * @param connectivity 4 or 8 connectivity
 * @return Reconstructed image
 */
cv::Mat morphologicalReconstruction(const cv::Mat& marker, const cv::Mat& mask, int connectivity = 4);

// ============================================================================
// GEOMETRIC TRANSFORMATIONS
// ============================================================================

/**
 * @brief Rotate image by specified angle
 * @param src Source image
 * @param angle Rotation angle in degrees (positive = counter-clockwise)
 * @return Rotated image
 */
cv::Mat rotate(const cv::Mat& src, double angle);

/**
 * @brief Scale/resize image
 * @param src Source image
 * @param scale_x Horizontal scale factor
 * @param scale_y Vertical scale factor
 * @param interpolation Interpolation method (cv::INTER_*)
 * @return Scaled image
 */
cv::Mat scale(const cv::Mat& src, double scale_x, double scale_y, int interpolation = cv::INTER_LINEAR);

/**
 * @brief Translate/shift image
 * @param src Source image
 * @param dx Horizontal shift in pixels
 * @param dy Vertical shift in pixels
 * @return Translated image
 */
cv::Mat translate(const cv::Mat& src, int dx, int dy);

/**
 * @brief Flip image horizontally
 * @param src Source image
 * @return Horizontally flipped image
 */
cv::Mat flipHorizontal(const cv::Mat& src);

/**
 * @brief Flip image vertically
 * @param src Source image
 * @return Vertically flipped image
 */
cv::Mat flipVertical(const cv::Mat& src);

/**
 * @brief Apply perspective transformation (bonus feature)
 * @param src Source image
 * @param src_points Source quadrilateral corners
 * @param dst_points Destination quadrilateral corners
 * @param interpolation Interpolation method with subpixel accuracy
 * @return Transformed image
 */
cv::Mat perspectiveTransform(const cv::Mat& src, 
                            const std::vector<cv::Point2f>& src_points,
                            const std::vector<cv::Point2f>& dst_points,
                            int interpolation = cv::INTER_LINEAR);

// ============================================================================
// COLOR/CHANNEL OPERATIONS
// ============================================================================

/**
 * @brief Split image into separate channels
 * @param src Source multi-channel image
 * @return Vector of single-channel images
 */
std::vector<cv::Mat> splitChannels(const cv::Mat& src);

/**
 * @brief Merge channels into single image
 * @param channels Vector of single-channel images
 * @return Merged multi-channel image
 */
cv::Mat mergeChannels(const std::vector<cv::Mat>& channels);

/**
 * @brief Adjust HSV values
 * @param src Source BGR image
 * @param hue_shift Hue adjustment (-180 to 180)
 * @param sat_scale Saturation scale factor (0.0 to 2.0)
 * @param val_scale Value scale factor (0.0 to 2.0)
 * @return Adjusted BGR image
 */
cv::Mat adjustHSV(const cv::Mat& src, int hue_shift, double sat_scale, double val_scale);

/**
 * @brief Apply tone mapping in Lab color space (bonus feature)
 * @param src Source BGR image
 * @param l_scale Scale factor for L channel
 * @return Tone-mapped BGR image
 */
cv::Mat labToneMapping(const cv::Mat& src, double l_scale);

// ============================================================================
// ADVANCED THRESHOLDING (BONUS)
// ============================================================================

/**
 * @brief Apply Otsu's automatic thresholding
 * @param src Source grayscale image
 * @return Binary image
 */
cv::Mat otsuThreshold(const cv::Mat& src);

/**
 * @brief Apply adaptive Gaussian thresholding
 * @param src Source grayscale image
 * @param block_size Size of neighborhood (must be odd)
 * @param C Constant subtracted from weighted mean
 * @return Binary image
 */
cv::Mat adaptiveGaussianThreshold(const cv::Mat& src, int block_size, double C);

/**
 * @brief Apply Sauvola thresholding (bonus feature)
 * @param src Source grayscale image
 * @param window_size Size of local window
 * @param k Sauvola parameter (typically 0.2-0.5)
 * @param R Dynamic range of standard deviation (default 128)
 * @return Binary image
 */
cv::Mat sauvolaThreshold(const cv::Mat& src, int window_size, double k = 0.5, double R = 128.0);

/**
 * @brief Apply Niblack thresholding (bonus feature)
 * @param src Source grayscale image
 * @param window_size Size of local window
 * @param k Niblack parameter (typically -0.2)
 * @return Binary image
 */
cv::Mat niblackThreshold(const cv::Mat& src, int window_size, double k = -0.2);

} // namespace iproc

#endif // IMAGE_FILTERS_H
