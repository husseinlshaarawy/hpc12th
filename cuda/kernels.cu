#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "image_filters_cuda.h"

// Constant memory for convolution kernels (max 64 KB)
__constant__ float d_kernel[256];

namespace iproc {
namespace cuda {

// ============================================================================
// CUDA ERROR CHECKING
// ============================================================================

#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(error)); \
        } \
    } while(0)

// ============================================================================
// GAUSSIAN BLUR KERNELS
// ============================================================================

__global__ void gaussianBlurHorizontalKernel(const uchar* input, uchar* output,
                                             int width, int height, int channels,
                                             int kernel_size) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    int half_k = kernel_size / 2;
    
    for (int c = 0; c < channels; ++c) {
        float sum = 0.0f;
        
        for (int k = 0; k < kernel_size; ++k) {
            int nx = x + k - half_k;
            nx = max(0, min(width - 1, nx));  // Clamp to borders
            
            int idx = (y * width + nx) * channels + c;
            sum += input[idx] * d_kernel[k];
        }
        
        int out_idx = (y * width + x) * channels + c;
        output[out_idx] = (uchar)min(255.0f, max(0.0f, sum));
    }
}

__global__ void gaussianBlurVerticalKernel(const uchar* input, uchar* output,
                                           int width, int height, int channels,
                                           int kernel_size) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    int half_k = kernel_size / 2;
    
    for (int c = 0; c < channels; ++c) {
        float sum = 0.0f;
        
        for (int k = 0; k < kernel_size; ++k) {
            int ny = y + k - half_k;
            ny = max(0, min(height - 1, ny));  // Clamp to borders
            
            int idx = (ny * width + x) * channels + c;
            sum += input[idx] * d_kernel[k];
        }
        
        int out_idx = (y * width + x) * channels + c;
        output[out_idx] = (uchar)min(255.0f, max(0.0f, sum));
    }
}

// ============================================================================
// MEDIAN FILTER KERNEL
// ============================================================================

__device__ void bubbleSort(uchar* arr, int n) {
    for (int i = 0; i < n - 1; ++i) {
        for (int j = 0; j < n - i - 1; ++j) {
            if (arr[j] > arr[j + 1]) {
                uchar temp = arr[j];
                arr[j] = arr[j + 1];
                arr[j + 1] = temp;
            }
        }
    }
}

__global__ void medianFilterKernel(const uchar* input, uchar* output,
                                   int width, int height, int channels,
                                   int kernel_size) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    int half_k = kernel_size / 2;
    int window_size = kernel_size * kernel_size;
    
    for (int c = 0; c < channels; ++c) {
        uchar window[256];  // Max kernel size 16x16
        int count = 0;
        
        for (int ky = -half_k; ky <= half_k; ++ky) {
            for (int kx = -half_k; kx <= half_k; ++kx) {
                int nx = x + kx;
                int ny = y + ky;
                
                if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                    int idx = (ny * width + nx) * channels + c;
                    window[count++] = input[idx];
                }
            }
        }
        
        bubbleSort(window, count);
        
        int out_idx = (y * width + x) * channels + c;
        output[out_idx] = window[count / 2];
    }
}

// ============================================================================
// 2D CONVOLUTION KERNEL WITH SHARED MEMORY
// ============================================================================

__global__ void convolve2DKernel(const uchar* input, uchar* output,
                                 int width, int height, int channels,
                                 int kernel_size) {
    extern __shared__ float s_data[];
    
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    int half_k = kernel_size / 2;
    int tile_width = blockDim.x + kernel_size - 1;
    
    // Load tile into shared memory with halo
    for (int c = 0; c < channels; ++c) {
        for (int dy = 0; dy <= 1; ++dy) {
            for (int dx = 0; dx <= 1; ++dx) {
                int sx = x + dx * blockDim.x - half_k;
                int sy = y + dy * blockDim.y - half_k;
                int stx = tx + dx * blockDim.x;
                int sty = ty + dy * blockDim.y;
                
                if (stx < tile_width && sty < tile_width) {
                    if (sx >= 0 && sx < width && sy >= 0 && sy < height) {
                        int idx = (sy * width + sx) * channels + c;
                        s_data[sty * tile_width + stx] = input[idx];
                    } else {
                        s_data[sty * tile_width + stx] = 0;
                    }
                }
            }
        }
    }
    
    __syncthreads();
    
    if (x >= width || y >= height) return;
    
    // Apply convolution
    for (int c = 0; c < channels; ++c) {
        float sum = 0.0f;
        
        for (int ky = 0; ky < kernel_size; ++ky) {
            for (int kx = 0; kx < kernel_size; ++kx) {
                int sx = tx + kx;
                int sy = ty + ky;
                sum += s_data[sy * tile_width + sx] * d_kernel[ky * kernel_size + kx];
            }
        }
        
        int out_idx = (y * width + x) * channels + c;
        output[out_idx] = (uchar)min(255.0f, max(0.0f, sum));
    }
}

// ============================================================================
// SOBEL EDGE DETECTION KERNEL
// ============================================================================

__global__ void sobelKernel(const uchar* input, uchar* output,
                           int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height || x == 0 || y == 0 || x == width - 1 || y == height - 1) {
        if (x < width && y < height) {
            output[y * width + x] = 0;
        }
        return;
    }
    
    // Sobel kernels
    int gx = 0, gy = 0;
    
    // Gradient X
    gx += input[(y - 1) * width + (x - 1)] * (-1);
    gx += input[(y - 1) * width + (x + 1)] * 1;
    gx += input[y * width + (x - 1)] * (-2);
    gx += input[y * width + (x + 1)] * 2;
    gx += input[(y + 1) * width + (x - 1)] * (-1);
    gx += input[(y + 1) * width + (x + 1)] * 1;
    
    // Gradient Y
    gy += input[(y - 1) * width + (x - 1)] * (-1);
    gy += input[(y - 1) * width + x] * (-2);
    gy += input[(y - 1) * width + (x + 1)] * (-1);
    gy += input[(y + 1) * width + (x - 1)] * 1;
    gy += input[(y + 1) * width + x] * 2;
    gy += input[(y + 1) * width + (x + 1)] * 1;
    
    int magnitude = (int)sqrtf((float)(gx * gx + gy * gy));
    output[y * width + x] = (uchar)min(255, magnitude);
}

// ============================================================================
// ROTATION KERNEL
// ============================================================================

__global__ void rotateKernel(const uchar* input, uchar* output,
                            int width, int height, int channels,
                            float cos_angle, float sin_angle,
                            int cx, int cy) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    // Calculate source coordinates
    int dx = x - cx;
    int dy = y - cy;
    
    float src_x = dx * cos_angle - dy * sin_angle + cx;
    float src_y = dx * sin_angle + dy * cos_angle + cy;
    
    if (src_x >= 0 && src_x < width - 1 && src_y >= 0 && src_y < height - 1) {
        // Bilinear interpolation
        int x0 = (int)src_x;
        int y0 = (int)src_y;
        int x1 = x0 + 1;
        int y1 = y0 + 1;
        
        float fx = src_x - x0;
        float fy = src_y - y0;
        
        for (int c = 0; c < channels; ++c) {
            float v00 = input[(y0 * width + x0) * channels + c];
            float v01 = input[(y0 * width + x1) * channels + c];
            float v10 = input[(y1 * width + x0) * channels + c];
            float v11 = input[(y1 * width + x1) * channels + c];
            
            float v = (1 - fx) * (1 - fy) * v00 +
                      fx * (1 - fy) * v01 +
                      (1 - fx) * fy * v10 +
                      fx * fy * v11;
            
            output[(y * width + x) * channels + c] = (uchar)min(255.0f, max(0.0f, v));
        }
    } else {
        for (int c = 0; c < channels; ++c) {
            output[(y * width + x) * channels + c] = 0;
        }
    }
}

} // namespace cuda
} // namespace iproc

// ============================================================================
// KERNEL LAUNCH WRAPPERS (for C++ linkage)
// ============================================================================

extern "C" {

void launchGaussianBlurHorizontal(const uchar* input, uchar* output,
                                  int width, int height, int channels,
                                  int kernel_size, cudaStream_t stream) {
    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
    
    iproc::cuda::gaussianBlurHorizontalKernel<<<grid, block, 0, stream>>>(
        input, output, width, height, channels, kernel_size);
}

void launchGaussianBlurVertical(const uchar* input, uchar* output,
                                int width, int height, int channels,
                                int kernel_size, cudaStream_t stream) {
    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
    
    iproc::cuda::gaussianBlurVerticalKernel<<<grid, block, 0, stream>>>(
        input, output, width, height, channels, kernel_size);
}

void launchMedianFilter(const uchar* input, uchar* output,
                       int width, int height, int channels,
                       int kernel_size, cudaStream_t stream) {
    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
    
    iproc::cuda::medianFilterKernel<<<grid, block, 0, stream>>>(
        input, output, width, height, channels, kernel_size);
}

void launchConvolve2D(const uchar* input, uchar* output,
                     int width, int height, int channels,
                     int kernel_size, cudaStream_t stream) {
    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
    
    int tile_width = 16 + kernel_size - 1;
    size_t shared_mem_size = tile_width * tile_width * sizeof(float);
    
    iproc::cuda::convolve2DKernel<<<grid, block, shared_mem_size, stream>>>(
        input, output, width, height, channels, kernel_size);
}

void launchSobel(const uchar* input, uchar* output,
                int width, int height, cudaStream_t stream) {
    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
    
    iproc::cuda::sobelKernel<<<grid, block, 0, stream>>>(
        input, output, width, height);
}

void launchRotate(const uchar* input, uchar* output,
                 int width, int height, int channels,
                 float cos_angle, float sin_angle, int cx, int cy,
                 cudaStream_t stream) {
    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
    
    iproc::cuda::rotateKernel<<<grid, block, 0, stream>>>(
        input, output, width, height, channels, cos_angle, sin_angle, cx, cy);
}

} // extern "C"
