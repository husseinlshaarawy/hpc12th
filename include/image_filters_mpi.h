#ifndef IMAGE_FILTERS_MPI_H
#define IMAGE_FILTERS_MPI_H

#include "image_filters.h"
#include <opencv2/opencv.hpp>
#include <mpi.h>

namespace iproc {
namespace mpi {

/**
 * @brief Image partitioning information for MPI
 */
struct ImagePartition {
    int rank;
    int size;
    int start_row;
    int num_rows;
    int halo_size;
    cv::Mat local_data;    // Local portion with halo
    cv::Mat local_output;  // Local output (no halo)
};

/**
 * @brief Partition image across MPI ranks with row-wise decomposition
 * Handles non-divisible heights, uses Scatterv
 * @param src Source image (only valid on rank 0)
 * @param halo_size Number of halo rows needed for stencil operations
 * @param comm MPI communicator
 * @return Local partition with halo
 */
ImagePartition partitionImage(const cv::Mat& src, int halo_size, MPI_Comm comm = MPI_COMM_WORLD);

/**
 * @brief Gather image from all ranks back to root
 * Uses Gatherv to handle non-uniform partitions
 * @param partition Local partition
 * @param comm MPI communicator
 * @return Complete image on rank 0, empty on others
 */
cv::Mat gatherImage(const ImagePartition& partition, MPI_Comm comm = MPI_COMM_WORLD);

/**
 * @brief Exchange halo regions with neighboring ranks (non-blocking)
 * Uses MPI_Isend/MPI_Irecv for overlap with computation
 * @param partition Partition to exchange halos for
 * @param requests Output vector of MPI_Request handles
 * @param comm MPI communicator
 */
void exchangeHalos(ImagePartition& partition, std::vector<MPI_Request>& requests, MPI_Comm comm = MPI_COMM_WORLD);

/**
 * @brief Wait for all halo exchanges to complete
 * @param requests Vector of MPI_Request handles from exchangeHalos
 */
void waitHalos(std::vector<MPI_Request>& requests);

/**
 * @brief MPI-parallelized Gaussian blur with halo exchange
 * @param src Source image (only valid on rank 0)
 * @param kernel_size Size of Gaussian kernel
 * @param sigma Standard deviation
 * @param comm MPI communicator
 * @return Blurred image on rank 0, empty on others
 */
cv::Mat gaussianBlur(const cv::Mat& src, int kernel_size, double sigma, MPI_Comm comm = MPI_COMM_WORLD);

/**
 * @brief MPI-parallelized median filter with halo exchange
 * @param src Source image (only valid on rank 0)
 * @param kernel_size Size of median kernel
 * @param comm MPI communicator
 * @return Filtered image on rank 0, empty on others
 */
cv::Mat medianFilter(const cv::Mat& src, int kernel_size, MPI_Comm comm = MPI_COMM_WORLD);

/**
 * @brief MPI-parallelized box blur with halo exchange
 * @param src Source image (only valid on rank 0)
 * @param kernel_size Size of box kernel
 * @param comm MPI communicator
 * @return Blurred image on rank 0, empty on others
 */
cv::Mat boxBlur(const cv::Mat& src, int kernel_size, MPI_Comm comm = MPI_COMM_WORLD);

} // namespace mpi
} // namespace iproc

#endif // IMAGE_FILTERS_MPI_H
