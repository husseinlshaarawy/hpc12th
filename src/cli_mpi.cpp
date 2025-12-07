#include "image_filters.h"
#include "image_filters_mpi.h"
#include <mpi.h>
#include <iostream>
#include <string>

using namespace iproc;

void printUsage(const char* program_name) {
    std::cout << "Usage: mpirun -np <N> " << program_name << " <operation> <input> <output> <params...>\n\n";
    std::cout << "MPI-parallelized operations:\n";
    std::cout << "  box-blur <kernel_size>\n";
    std::cout << "  gaussian-blur <kernel_size> <sigma>\n";
    std::cout << "  median <kernel_size>\n";
}

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);
    
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    if (argc < 5) {
        if (rank == 0) {
            printUsage(argv[0]);
        }
        MPI_Finalize();
        return 1;
    }
    
    std::string operation = argv[1];
    std::string input_path = argv[2];
    std::string output_path = argv[3];
    
    try {
        cv::Mat input, result;
        
        // Only rank 0 loads the image
        if (rank == 0) {
            input = loadImage(input_path);
        }
        
        if (operation == "box-blur" && argc >= 5) {
            int kernel_size = std::stoi(argv[4]);
            result = mpi::boxBlur(input, kernel_size);
        }
        else if (operation == "gaussian-blur" && argc >= 6) {
            int kernel_size = std::stoi(argv[4]);
            double sigma = std::stod(argv[5]);
            result = mpi::gaussianBlur(input, kernel_size, sigma);
        }
        else if (operation == "median" && argc >= 5) {
            int kernel_size = std::stoi(argv[4]);
            result = mpi::medianFilter(input, kernel_size);
        }
        else {
            if (rank == 0) {
                std::cerr << "Unknown operation: " << operation << "\n";
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 1;
        }
        
        // Only rank 0 saves the result
        if (rank == 0) {
            if (saveImage(output_path, result)) {
                std::cout << "Successfully saved result to " << output_path << "\n";
            } else {
                std::cerr << "Failed to save result\n";
                MPI_Finalize();
                return 1;
            }
        }
        
    } catch (const std::exception& e) {
        if (rank == 0) {
            std::cerr << "Error: " << e.what() << "\n";
        }
        MPI_Finalize();
        return 1;
    }
    
    MPI_Finalize();
    return 0;
}
