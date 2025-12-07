#include "image_filters_mpi.h"
#include <vector>
#include <algorithm>

namespace iproc {
namespace mpi {

ImagePartition partitionImage(const cv::Mat& src, int halo_size, MPI_Comm comm) {
    ImagePartition partition;
    MPI_Comm_rank(comm, &partition.rank);
    MPI_Comm_size(comm, &partition.size);
    partition.halo_size = halo_size;
    
    int total_rows = 0;
    int cols = 0;
    int type = 0;
    
    // Broadcast image metadata from rank 0
    if (partition.rank == 0) {
        total_rows = src.rows;
        cols = src.cols;
        type = src.type();
    }
    
    MPI_Bcast(&total_rows, 1, MPI_INT, 0, comm);
    MPI_Bcast(&cols, 1, MPI_INT, 0, comm);
    MPI_Bcast(&type, 1, MPI_INT, 0, comm);
    
    // Calculate row distribution (handle non-divisible heights)
    int base_rows = total_rows / partition.size;
    int remainder = total_rows % partition.size;
    
    // Assign extra rows to first 'remainder' ranks
    partition.num_rows = base_rows + (partition.rank < remainder ? 1 : 0);
    partition.start_row = partition.rank * base_rows + std::min(partition.rank, remainder);
    
    // Calculate send counts and displacements for Scatterv
    std::vector<int> send_counts(partition.size);
    std::vector<int> displs(partition.size);
    
    for (int r = 0; r < partition.size; ++r) {
        int rows = base_rows + (r < remainder ? 1 : 0);
        send_counts[r] = rows * cols * CV_MAT_CN(type);
        displs[r] = (r * base_rows + std::min(r, remainder)) * cols * CV_MAT_CN(type);
    }
    
    // Allocate local data with halo
    int local_rows_with_halo = partition.num_rows + 2 * halo_size;
    partition.local_data = cv::Mat::zeros(local_rows_with_halo, cols, type);
    partition.local_output = cv::Mat::zeros(partition.num_rows, cols, type);
    
    // Scatter image data
    uchar* send_buf = partition.rank == 0 ? src.data : nullptr;
    uchar* recv_buf = partition.local_data.ptr(halo_size);  // Start after halo
    
    MPI_Scatterv(send_buf, send_counts.data(), displs.data(), MPI_UNSIGNED_CHAR,
                 recv_buf, send_counts[partition.rank], MPI_UNSIGNED_CHAR,
                 0, comm);
    
    return partition;
}

cv::Mat gatherImage(const ImagePartition& partition, MPI_Comm comm) {
    int total_rows = 0;
    int base_rows = partition.num_rows;
    
    // Gather information about row counts
    std::vector<int> all_row_counts(partition.size);
    MPI_Allgather(&partition.num_rows, 1, MPI_INT,
                  all_row_counts.data(), 1, MPI_INT, comm);
    
    for (int count : all_row_counts) {
        total_rows += count;
    }
    
    cv::Mat result;
    if (partition.rank == 0) {
        result = cv::Mat::zeros(total_rows, partition.local_output.cols, partition.local_output.type());
    }
    
    // Calculate receive counts and displacements
    std::vector<int> recv_counts(partition.size);
    std::vector<int> displs(partition.size);
    int channels = CV_MAT_CN(partition.local_output.type());
    
    int offset = 0;
    for (int r = 0; r < partition.size; ++r) {
        recv_counts[r] = all_row_counts[r] * partition.local_output.cols * channels;
        displs[r] = offset;
        offset += recv_counts[r];
    }
    
    // Gather results
    uchar* recv_buf = partition.rank == 0 ? result.data : nullptr;
    MPI_Gatherv(partition.local_output.data, recv_counts[partition.rank], MPI_UNSIGNED_CHAR,
                recv_buf, recv_counts.data(), displs.data(), MPI_UNSIGNED_CHAR,
                0, comm);
    
    return result;
}

void exchangeHalos(ImagePartition& partition, std::vector<MPI_Request>& requests, MPI_Comm comm) {
    requests.clear();
    
    int cols = partition.local_data.cols;
    int channels = CV_MAT_CN(partition.local_data.type());
    int row_bytes = cols * channels;
    
    MPI_Request req;
    
    // Send top halo to rank-1, receive from rank-1
    if (partition.rank > 0) {
        // Send top rows
        uchar* send_buf = partition.local_data.ptr(partition.halo_size);
        MPI_Isend(send_buf, partition.halo_size * row_bytes, MPI_UNSIGNED_CHAR,
                 partition.rank - 1, 0, comm, &req);
        requests.push_back(req);
        
        // Receive into top halo
        uchar* recv_buf = partition.local_data.ptr(0);
        MPI_Irecv(recv_buf, partition.halo_size * row_bytes, MPI_UNSIGNED_CHAR,
                 partition.rank - 1, 1, comm, &req);
        requests.push_back(req);
    }
    
    // Send bottom halo to rank+1, receive from rank+1
    if (partition.rank < partition.size - 1) {
        // Send bottom rows
        int send_start = partition.halo_size + partition.num_rows - partition.halo_size;
        uchar* send_buf = partition.local_data.ptr(send_start);
        MPI_Isend(send_buf, partition.halo_size * row_bytes, MPI_UNSIGNED_CHAR,
                 partition.rank + 1, 1, comm, &req);
        requests.push_back(req);
        
        // Receive into bottom halo
        int recv_start = partition.halo_size + partition.num_rows;
        uchar* recv_buf = partition.local_data.ptr(recv_start);
        MPI_Irecv(recv_buf, partition.halo_size * row_bytes, MPI_UNSIGNED_CHAR,
                 partition.rank + 1, 0, comm, &req);
        requests.push_back(req);
    }
}

void waitHalos(std::vector<MPI_Request>& requests) {
    if (!requests.empty()) {
        MPI_Waitall(requests.size(), requests.data(), MPI_STATUSES_IGNORE);
    }
}

cv::Mat gaussianBlur(const cv::Mat& src, int kernel_size, double sigma, MPI_Comm comm) {
    int halo_size = kernel_size / 2;
    ImagePartition partition = partitionImage(src, halo_size, comm);
    
    // Exchange halos (non-blocking)
    std::vector<MPI_Request> requests;
    exchangeHalos(partition, requests, comm);
    
    // Process inner region while halo exchanges are in flight
    // (Simplified: we'll just wait for halos then process all)
    waitHalos(requests);
    
    // Apply Gaussian blur to local partition
    cv::Mat local_result = iproc::gaussianBlur(partition.local_data, kernel_size, sigma);
    
    // Extract result without halo
    local_result(cv::Rect(0, halo_size, local_result.cols, partition.num_rows))
        .copyTo(partition.local_output);
    
    // Gather results
    return gatherImage(partition, comm);
}

cv::Mat medianFilter(const cv::Mat& src, int kernel_size, MPI_Comm comm) {
    int halo_size = kernel_size / 2;
    ImagePartition partition = partitionImage(src, halo_size, comm);
    
    std::vector<MPI_Request> requests;
    exchangeHalos(partition, requests, comm);
    waitHalos(requests);
    
    cv::Mat local_result = iproc::medianFilter(partition.local_data, kernel_size);
    
    local_result(cv::Rect(0, halo_size, local_result.cols, partition.num_rows))
        .copyTo(partition.local_output);
    
    return gatherImage(partition, comm);
}

cv::Mat boxBlur(const cv::Mat& src, int kernel_size, MPI_Comm comm) {
    int halo_size = kernel_size / 2;
    ImagePartition partition = partitionImage(src, halo_size, comm);
    
    std::vector<MPI_Request> requests;
    exchangeHalos(partition, requests, comm);
    waitHalos(requests);
    
    cv::Mat local_result = iproc::boxBlur(partition.local_data, kernel_size);
    
    local_result(cv::Rect(0, halo_size, local_result.cols, partition.num_rows))
        .copyTo(partition.local_output);
    
    return gatherImage(partition, comm);
}

} // namespace mpi
} // namespace iproc
