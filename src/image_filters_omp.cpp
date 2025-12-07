#include "image_filters_omp.h"
#include <omp.h>
#include <vector>
#include <algorithm>
#include <cmath>

namespace iproc {
namespace omp {

cv::Mat boxBlur(const cv::Mat& src, int kernel_size, int num_threads) {
    if (num_threads > 0) {
        omp_set_num_threads(num_threads);
    }
    
    cv::Mat result = cv::Mat::zeros(src.size(), src.type());
    int half_k = kernel_size / 2;
    int channels = src.channels();
    
    // Row-wise parallelization with explicit data scoping
    #pragma omp parallel for shared(src, result, kernel_size, half_k, channels) schedule(dynamic)
    for (int i = 0; i < src.rows; ++i) {
        for (int j = 0; j < src.cols; ++j) {
            std::vector<double> sum(channels, 0.0);
            int count = 0;
            
            // Compute box average
            for (int ki = -half_k; ki <= half_k; ++ki) {
                for (int kj = -half_k; kj <= half_k; ++kj) {
                    int ni = i + ki;
                    int nj = j + kj;
                    
                    if (ni >= 0 && ni < src.rows && nj >= 0 && nj < src.cols) {
                        if (channels == 1) {
                            sum[0] += src.at<uchar>(ni, nj);
                        } else {
                            cv::Vec3b pixel = src.at<cv::Vec3b>(ni, nj);
                            for (int c = 0; c < channels; ++c) {
                                sum[c] += pixel[c];
                            }
                        }
                        ++count;
                    }
                }
            }
            
            // Write output
            if (channels == 1) {
                result.at<uchar>(i, j) = cv::saturate_cast<uchar>(sum[0] / count);
            } else {
                cv::Vec3b& pixel = result.at<cv::Vec3b>(i, j);
                for (int c = 0; c < channels; ++c) {
                    pixel[c] = cv::saturate_cast<uchar>(sum[c] / count);
                }
            }
        }
    }
    
    return result;
}

cv::Mat gaussianBlur(const cv::Mat& src, int kernel_size, double sigma, int num_threads) {
    if (num_threads > 0) {
        omp_set_num_threads(num_threads);
    }
    
    // Generate 1D Gaussian kernel
    std::vector<double> kernel(kernel_size);
    int half_k = kernel_size / 2;
    double sum = 0.0;
    
    for (int i = 0; i < kernel_size; ++i) {
        int x = i - half_k;
        kernel[i] = std::exp(-(x * x) / (2.0 * sigma * sigma));
        sum += kernel[i];
    }
    
    // Normalize kernel
    for (int i = 0; i < kernel_size; ++i) {
        kernel[i] /= sum;
    }
    
    // Separable convolution: horizontal pass
    cv::Mat temp = cv::Mat::zeros(src.size(), src.type());
    int channels = src.channels();
    
    #pragma omp parallel for shared(src, temp, kernel, half_k, channels) schedule(dynamic)
    for (int i = 0; i < src.rows; ++i) {
        for (int j = 0; j < src.cols; ++j) {
            std::vector<double> val(channels, 0.0);
            
            for (int k = 0; k < kernel_size; ++k) {
                int nj = j + k - half_k;
                if (nj >= 0 && nj < src.cols) {
                    if (channels == 1) {
                        val[0] += src.at<uchar>(i, nj) * kernel[k];
                    } else {
                        cv::Vec3b pixel = src.at<cv::Vec3b>(i, nj);
                        for (int c = 0; c < channels; ++c) {
                            val[c] += pixel[c] * kernel[k];
                        }
                    }
                }
            }
            
            if (channels == 1) {
                temp.at<uchar>(i, j) = cv::saturate_cast<uchar>(val[0]);
            } else {
                cv::Vec3b& pixel = temp.at<cv::Vec3b>(i, j);
                for (int c = 0; c < channels; ++c) {
                    pixel[c] = cv::saturate_cast<uchar>(val[c]);
                }
            }
        }
    }
    
    // Vertical pass
    cv::Mat result = cv::Mat::zeros(src.size(), src.type());
    
    #pragma omp parallel for shared(temp, result, kernel, half_k, channels) schedule(dynamic)
    for (int i = 0; i < temp.rows; ++i) {
        for (int j = 0; j < temp.cols; ++j) {
            std::vector<double> val(channels, 0.0);
            
            for (int k = 0; k < kernel_size; ++k) {
                int ni = i + k - half_k;
                if (ni >= 0 && ni < temp.rows) {
                    if (channels == 1) {
                        val[0] += temp.at<uchar>(ni, j) * kernel[k];
                    } else {
                        cv::Vec3b pixel = temp.at<cv::Vec3b>(ni, j);
                        for (int c = 0; c < channels; ++c) {
                            val[c] += pixel[c] * kernel[k];
                        }
                    }
                }
            }
            
            if (channels == 1) {
                result.at<uchar>(i, j) = cv::saturate_cast<uchar>(val[0]);
            } else {
                cv::Vec3b& pixel = result.at<cv::Vec3b>(i, j);
                for (int c = 0; c < channels; ++c) {
                    pixel[c] = cv::saturate_cast<uchar>(val[c]);
                }
            }
        }
    }
    
    return result;
}

cv::Mat medianFilter(const cv::Mat& src, int kernel_size, int num_threads) {
    if (num_threads > 0) {
        omp_set_num_threads(num_threads);
    }
    
    cv::Mat result = cv::Mat::zeros(src.size(), src.type());
    int half_k = kernel_size / 2;
    int channels = src.channels();
    
    // Row-wise parallelization with thread-private buffers
    #pragma omp parallel shared(src, result, kernel_size, half_k, channels)
    {
        // Thread-private buffer for median computation
        std::vector<uchar> buffer;
        buffer.reserve(kernel_size * kernel_size);
        
        #pragma omp for schedule(dynamic)
        for (int i = 0; i < src.rows; ++i) {
            for (int j = 0; j < src.cols; ++j) {
                if (channels == 1) {
                    buffer.clear();
                    
                    for (int ki = -half_k; ki <= half_k; ++ki) {
                        for (int kj = -half_k; kj <= half_k; ++kj) {
                            int ni = i + ki;
                            int nj = j + kj;
                            
                            if (ni >= 0 && ni < src.rows && nj >= 0 && nj < src.cols) {
                                buffer.push_back(src.at<uchar>(ni, nj));
                            }
                        }
                    }
                    
                    std::nth_element(buffer.begin(), buffer.begin() + buffer.size() / 2, buffer.end());
                    result.at<uchar>(i, j) = buffer[buffer.size() / 2];
                } else {
                    std::vector<uchar> buffer_b, buffer_g, buffer_r;
                    buffer_b.reserve(kernel_size * kernel_size);
                    buffer_g.reserve(kernel_size * kernel_size);
                    buffer_r.reserve(kernel_size * kernel_size);
                    
                    for (int ki = -half_k; ki <= half_k; ++ki) {
                        for (int kj = -half_k; kj <= half_k; ++kj) {
                            int ni = i + ki;
                            int nj = j + kj;
                            
                            if (ni >= 0 && ni < src.rows && nj >= 0 && nj < src.cols) {
                                cv::Vec3b pixel = src.at<cv::Vec3b>(ni, nj);
                                buffer_b.push_back(pixel[0]);
                                buffer_g.push_back(pixel[1]);
                                buffer_r.push_back(pixel[2]);
                            }
                        }
                    }
                    
                    std::nth_element(buffer_b.begin(), buffer_b.begin() + buffer_b.size() / 2, buffer_b.end());
                    std::nth_element(buffer_g.begin(), buffer_g.begin() + buffer_g.size() / 2, buffer_g.end());
                    std::nth_element(buffer_r.begin(), buffer_r.begin() + buffer_r.size() / 2, buffer_r.end());
                    
                    result.at<cv::Vec3b>(i, j) = cv::Vec3b(
                        buffer_b[buffer_b.size() / 2],
                        buffer_g[buffer_g.size() / 2],
                        buffer_r[buffer_r.size() / 2]
                    );
                }
            }
        }
    }
    
    return result;
}

cv::Mat bilateralFilter(const cv::Mat& src, int d, double sigma_color, double sigma_space, int num_threads) {
    if (num_threads > 0) {
        omp_set_num_threads(num_threads);
    }
    
    cv::Mat result = cv::Mat::zeros(src.size(), src.type());
    int radius = d / 2;
    int channels = src.channels();
    
    // Precompute spatial Gaussian weights
    std::vector<double> space_weight((2 * radius + 1) * (2 * radius + 1));
    int idx = 0;
    for (int i = -radius; i <= radius; ++i) {
        for (int j = -radius; j <= radius; ++j) {
            double dist = std::sqrt(i * i + j * j);
            space_weight[idx++] = std::exp(-(dist * dist) / (2.0 * sigma_space * sigma_space));
        }
    }
    
    // Tile-based parallelization for better cache locality
    #pragma omp parallel for shared(src, result, d, radius, sigma_color, space_weight, channels) schedule(dynamic)
    for (int i = 0; i < src.rows; ++i) {
        for (int j = 0; j < src.cols; ++j) {
            std::vector<double> sum(channels, 0.0);
            double weight_sum = 0.0;
            
            int idx = 0;
            for (int ki = -radius; ki <= radius; ++ki) {
                for (int kj = -radius; kj <= radius; ++kj) {
                    int ni = i + ki;
                    int nj = j + kj;
                    
                    if (ni >= 0 && ni < src.rows && nj >= 0 && nj < src.cols) {
                        double color_dist = 0.0;
                        
                        if (channels == 1) {
                            int diff = src.at<uchar>(i, j) - src.at<uchar>(ni, nj);
                            color_dist = diff * diff;
                        } else {
                            cv::Vec3b p1 = src.at<cv::Vec3b>(i, j);
                            cv::Vec3b p2 = src.at<cv::Vec3b>(ni, nj);
                            for (int c = 0; c < channels; ++c) {
                                int diff = p1[c] - p2[c];
                                color_dist += diff * diff;
                            }
                        }
                        
                        double color_weight = std::exp(-color_dist / (2.0 * sigma_color * sigma_color));
                        double total_weight = space_weight[idx] * color_weight;
                        
                        if (channels == 1) {
                            sum[0] += src.at<uchar>(ni, nj) * total_weight;
                        } else {
                            cv::Vec3b pixel = src.at<cv::Vec3b>(ni, nj);
                            for (int c = 0; c < channels; ++c) {
                                sum[c] += pixel[c] * total_weight;
                            }
                        }
                        
                        weight_sum += total_weight;
                    }
                    ++idx;
                }
            }
            
            if (weight_sum > 0) {
                if (channels == 1) {
                    result.at<uchar>(i, j) = cv::saturate_cast<uchar>(sum[0] / weight_sum);
                } else {
                    cv::Vec3b& pixel = result.at<cv::Vec3b>(i, j);
                    for (int c = 0; c < channels; ++c) {
                        pixel[c] = cv::saturate_cast<uchar>(sum[c] / weight_sum);
                    }
                }
            }
        }
    }
    
    return result;
}

cv::Mat sobelEdgeDetection(const cv::Mat& src, int kernel_size, int num_threads) {
    if (num_threads > 0) {
        omp_set_num_threads(num_threads);
    }
    
    // Use OpenCV's Sobel with parallel processing
    cv::Mat grad_x, grad_y;
    cv::Mat abs_grad_x, abs_grad_y;
    
    cv::Sobel(src, grad_x, CV_16S, 1, 0, kernel_size);
    cv::Sobel(src, grad_y, CV_16S, 0, 1, kernel_size);
    
    cv::convertScaleAbs(grad_x, abs_grad_x);
    cv::convertScaleAbs(grad_y, abs_grad_y);
    
    cv::Mat result;
    cv::addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, result);
    
    return result;
}

cv::Mat prewittEdgeDetection(const cv::Mat& src, int num_threads) {
    if (num_threads > 0) {
        omp_set_num_threads(num_threads);
    }
    
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

cv::Mat convolve2D(const cv::Mat& src, const cv::Mat& kernel, int num_threads) {
    if (num_threads > 0) {
        omp_set_num_threads(num_threads);
    }
    
    cv::Mat result;
    cv::filter2D(src, result, src.depth(), kernel);
    return result;
}

} // namespace omp
} // namespace iproc
