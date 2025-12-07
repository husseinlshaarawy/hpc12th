#include "image_filters_cuda.h"
#include <cuda_runtime.h>
#include <cmath>
#include <stdexcept>

// Forward declarations of CUDA kernels
extern __constant__ float d_kernel[256];

namespace iproc {
namespace cuda {

// CUDA error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(error)); \
        } \
    } while(0)

// Kernel launch declarations
extern "C" {
    void launchGaussianBlurHorizontal(const uchar* input, uchar* output,
                                      int width, int height, int channels,
                                      int kernel_size, cudaStream_t stream);
    void launchGaussianBlurVertical(const uchar* input, uchar* output,
                                    int width, int height, int channels,
                                    int kernel_size, cudaStream_t stream);
    void launchMedianFilter(const uchar* input, uchar* output,
                           int width, int height, int channels,
                           int kernel_size, cudaStream_t stream);
    void launchConvolve2D(const uchar* input, uchar* output,
                         int width, int height, int channels,
                         int kernel_size, cudaStream_t stream);
    void launchSobel(const uchar* input, uchar* output,
                    int width, int height, cudaStream_t stream);
    void launchRotate(const uchar* input, uchar* output,
                     int width, int height, int channels,
                     float cos_angle, float sin_angle, int cx, int cy,
                     cudaStream_t stream);
}

static bool g_initialized = false;

bool initialize() {
    int device_count = 0;
    cudaError_t error = cudaGetDeviceCount(&device_count);
    
    if (error != cudaSuccess || device_count == 0) {
        return false;
    }
    
    g_initialized = true;
    return true;
}

void cleanup() {
    if (g_initialized) {
        cudaDeviceReset();
        g_initialized = false;
    }
}

bool isAvailable() {
    int device_count = 0;
    cudaError_t error = cudaGetDeviceCount(&device_count);
    return (error == cudaSuccess && device_count > 0);
}

cv::Mat gaussianBlur(const cv::Mat& src, int kernel_size, double sigma) {
    if (!g_initialized) {
        throw std::runtime_error("CUDA not initialized. Call initialize() first.");
    }
    
    // Generate 1D Gaussian kernel
    std::vector<float> kernel(kernel_size);
    int half_k = kernel_size / 2;
    float sum = 0.0f;
    
    for (int i = 0; i < kernel_size; ++i) {
        int x = i - half_k;
        kernel[i] = std::exp(-(x * x) / (2.0 * sigma * sigma));
        sum += kernel[i];
    }
    
    // Normalize
    for (int i = 0; i < kernel_size; ++i) {
        kernel[i] /= sum;
    }
    
    // Copy kernel to constant memory
    CUDA_CHECK(cudaMemcpyToSymbol(d_kernel, kernel.data(), kernel_size * sizeof(float)));
    
    // Allocate pinned host memory for faster transfers
    uchar* h_pinned_src;
    uchar* h_pinned_dst;
    size_t img_size = src.rows * src.cols * src.channels();
    
    CUDA_CHECK(cudaMallocHost(&h_pinned_src, img_size));
    CUDA_CHECK(cudaMallocHost(&h_pinned_dst, img_size));
    
    memcpy(h_pinned_src, src.data, img_size);
    
    // Allocate device memory
    uchar *d_input, *d_temp, *d_output;
    CUDA_CHECK(cudaMalloc(&d_input, img_size));
    CUDA_CHECK(cudaMalloc(&d_temp, img_size));
    CUDA_CHECK(cudaMalloc(&d_output, img_size));
    
    // Create CUDA stream for async operations
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    
    // Async copy to device
    CUDA_CHECK(cudaMemcpyAsync(d_input, h_pinned_src, img_size, 
                               cudaMemcpyHostToDevice, stream));
    
    // Launch horizontal blur
    dim3 block(16, 16);
    dim3 grid((src.cols + block.x - 1) / block.x, (src.rows + block.y - 1) / block.y);
    
    launchGaussianBlurHorizontal(d_input, d_temp, src.cols, src.rows, 
                                 src.channels(), kernel_size, stream);
    
    // Launch vertical blur
    launchGaussianBlurVertical(d_temp, d_output, src.cols, src.rows,
                               src.channels(), kernel_size, stream);
    
    // Async copy back to host
    CUDA_CHECK(cudaMemcpyAsync(h_pinned_dst, d_output, img_size,
                               cudaMemcpyDeviceToHost, stream));
    
    // Wait for completion
    CUDA_CHECK(cudaStreamSynchronize(stream));
    
    // Create result
    cv::Mat result(src.size(), src.type());
    memcpy(result.data, h_pinned_dst, img_size);
    
    // Cleanup
    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_temp));
    CUDA_CHECK(cudaFree(d_output));
    CUDA_CHECK(cudaFreeHost(h_pinned_src));
    CUDA_CHECK(cudaFreeHost(h_pinned_dst));
    
    return result;
}

cv::Mat medianFilter(const cv::Mat& src, int kernel_size) {
    if (!g_initialized) {
        throw std::runtime_error("CUDA not initialized. Call initialize() first.");
    }
    
    uchar* h_pinned_src;
    uchar* h_pinned_dst;
    size_t img_size = src.rows * src.cols * src.channels();
    
    CUDA_CHECK(cudaMallocHost(&h_pinned_src, img_size));
    CUDA_CHECK(cudaMallocHost(&h_pinned_dst, img_size));
    
    memcpy(h_pinned_src, src.data, img_size);
    
    uchar *d_input, *d_output;
    CUDA_CHECK(cudaMalloc(&d_input, img_size));
    CUDA_CHECK(cudaMalloc(&d_output, img_size));
    
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    
    CUDA_CHECK(cudaMemcpyAsync(d_input, h_pinned_src, img_size,
                               cudaMemcpyHostToDevice, stream));
    
    launchMedianFilter(d_input, d_output, src.cols, src.rows,
                      src.channels(), kernel_size, stream);
    
    CUDA_CHECK(cudaMemcpyAsync(h_pinned_dst, d_output, img_size,
                               cudaMemcpyDeviceToHost, stream));
    
    CUDA_CHECK(cudaStreamSynchronize(stream));
    
    cv::Mat result(src.size(), src.type());
    memcpy(result.data, h_pinned_dst, img_size);
    
    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
    CUDA_CHECK(cudaFreeHost(h_pinned_src));
    CUDA_CHECK(cudaFreeHost(h_pinned_dst));
    
    return result;
}

cv::Mat convolve2D(const cv::Mat& src, const cv::Mat& kernel) {
    if (!g_initialized) {
        throw std::runtime_error("CUDA not initialized. Call initialize() first.");
    }
    
    // Convert kernel to float and copy to constant memory
    std::vector<float> h_kernel(kernel.rows * kernel.cols);
    for (int i = 0; i < kernel.rows; ++i) {
        for (int j = 0; j < kernel.cols; ++j) {
            h_kernel[i * kernel.cols + j] = kernel.at<float>(i, j);
        }
    }
    
    CUDA_CHECK(cudaMemcpyToSymbol(d_kernel, h_kernel.data(),
                                  h_kernel.size() * sizeof(float)));
    
    uchar* h_pinned_src;
    uchar* h_pinned_dst;
    size_t img_size = src.rows * src.cols * src.channels();
    
    CUDA_CHECK(cudaMallocHost(&h_pinned_src, img_size));
    CUDA_CHECK(cudaMallocHost(&h_pinned_dst, img_size));
    
    memcpy(h_pinned_src, src.data, img_size);
    
    uchar *d_input, *d_output;
    CUDA_CHECK(cudaMalloc(&d_input, img_size));
    CUDA_CHECK(cudaMalloc(&d_output, img_size));
    
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    
    CUDA_CHECK(cudaMemcpyAsync(d_input, h_pinned_src, img_size,
                               cudaMemcpyHostToDevice, stream));
    
    launchConvolve2D(d_input, d_output, src.cols, src.rows,
                    src.channels(), kernel.cols, stream);
    
    CUDA_CHECK(cudaMemcpyAsync(h_pinned_dst, d_output, img_size,
                               cudaMemcpyDeviceToHost, stream));
    
    CUDA_CHECK(cudaStreamSynchronize(stream));
    
    cv::Mat result(src.size(), src.type());
    memcpy(result.data, h_pinned_dst, img_size);
    
    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
    CUDA_CHECK(cudaFreeHost(h_pinned_src));
    CUDA_CHECK(cudaFreeHost(h_pinned_dst));
    
    return result;
}

cv::Mat sobelEdgeDetection(const cv::Mat& src, int kernel_size) {
    if (!g_initialized) {
        throw std::runtime_error("CUDA not initialized. Call initialize() first.");
    }
    
    cv::Mat gray = src;
    if (src.channels() > 1) {
        cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);
    }
    
    uchar* h_pinned_src;
    uchar* h_pinned_dst;
    size_t img_size = gray.rows * gray.cols;
    
    CUDA_CHECK(cudaMallocHost(&h_pinned_src, img_size));
    CUDA_CHECK(cudaMallocHost(&h_pinned_dst, img_size));
    
    memcpy(h_pinned_src, gray.data, img_size);
    
    uchar *d_input, *d_output;
    CUDA_CHECK(cudaMalloc(&d_input, img_size));
    CUDA_CHECK(cudaMalloc(&d_output, img_size));
    
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    
    CUDA_CHECK(cudaMemcpyAsync(d_input, h_pinned_src, img_size,
                               cudaMemcpyHostToDevice, stream));
    
    launchSobel(d_input, d_output, gray.cols, gray.rows, stream);
    
    CUDA_CHECK(cudaMemcpyAsync(h_pinned_dst, d_output, img_size,
                               cudaMemcpyDeviceToHost, stream));
    
    CUDA_CHECK(cudaStreamSynchronize(stream));
    
    cv::Mat result(gray.size(), CV_8UC1);
    memcpy(result.data, h_pinned_dst, img_size);
    
    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
    CUDA_CHECK(cudaFreeHost(h_pinned_src));
    CUDA_CHECK(cudaFreeHost(h_pinned_dst));
    
    return result;
}

cv::Mat rotate(const cv::Mat& src, double angle) {
    if (!g_initialized) {
        throw std::runtime_error("CUDA not initialized. Call initialize() first.");
    }
    
    float angle_rad = angle * M_PI / 180.0f;
    float cos_angle = std::cos(angle_rad);
    float sin_angle = std::sin(angle_rad);
    int cx = src.cols / 2;
    int cy = src.rows / 2;
    
    uchar* h_pinned_src;
    uchar* h_pinned_dst;
    size_t img_size = src.rows * src.cols * src.channels();
    
    CUDA_CHECK(cudaMallocHost(&h_pinned_src, img_size));
    CUDA_CHECK(cudaMallocHost(&h_pinned_dst, img_size));
    
    memcpy(h_pinned_src, src.data, img_size);
    
    uchar *d_input, *d_output;
    CUDA_CHECK(cudaMalloc(&d_input, img_size));
    CUDA_CHECK(cudaMalloc(&d_output, img_size));
    
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    
    CUDA_CHECK(cudaMemcpyAsync(d_input, h_pinned_src, img_size,
                               cudaMemcpyHostToDevice, stream));
    
    launchRotate(d_input, d_output, src.cols, src.rows, src.channels(),
                cos_angle, sin_angle, cx, cy, stream);
    
    CUDA_CHECK(cudaMemcpyAsync(h_pinned_dst, d_output, img_size,
                               cudaMemcpyDeviceToHost, stream));
    
    CUDA_CHECK(cudaStreamSynchronize(stream));
    
    cv::Mat result(src.size(), src.type());
    memcpy(result.data, h_pinned_dst, img_size);
    
    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
    CUDA_CHECK(cudaFreeHost(h_pinned_src));
    CUDA_CHECK(cudaFreeHost(h_pinned_dst));
    
    return result;
}

cv::Mat scale(const cv::Mat& src, double scale_x, double scale_y) {
    // Fallback to OpenCV for now
    cv::Mat result;
    cv::resize(src, result, cv::Size(), scale_x, scale_y);
    return result;
}

cv::Mat chainFilters(const cv::Mat& src, const std::vector<std::string>& operations) {
    // TODO: Implement filter chaining to keep data on GPU
    return src.clone();
}

} // namespace cuda
} // namespace iproc
