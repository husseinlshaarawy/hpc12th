#include "image_filters.h"
#include "image_filters_mpi.h"
#include <mpi.h>
#include <iostream>
#include <chrono>
#include <fstream>

using namespace iproc;

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);
    
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    if (argc < 2) {
        if (rank == 0) {
            std::cerr << "Usage: mpirun -np <N> " << argv[0] << " <input_image> [output_csv]\n";
        }
        MPI_Finalize();
        return 1;
    }
    
    std::string input_path = argv[1];
    std::string output_csv = (argc >= 3) ? argv[2] : "benchmark_mpi.csv";
    
    try {
        cv::Mat img;
        if (rank == 0) {
            img = loadImage(input_path);
            std::cout << "Running MPI benchmarks with " << size << " processes\n";
        }
        
        std::vector<std::string> operations = {"gaussian_blur", "median_filter", "box_blur"};
        std::vector<std::pair<std::string, double>> results;
        
        for (const auto& op : operations) {
            auto start = std::chrono::high_resolution_clock::now();
            
            if (op == "gaussian_blur") {
                mpi::gaussianBlur(img, 5, 1.0);
            } else if (op == "median_filter") {
                mpi::medianFilter(img, 5);
            } else if (op == "box_blur") {
                mpi::boxBlur(img, 5);
            }
            
            auto end = std::chrono::high_resolution_clock::now();
            double duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
            
            if (rank == 0) {
                results.push_back({op, duration_ms});
                std::cout << "  " << op << ": " << duration_ms << " ms\n";
            }
        }
        
        if (rank == 0) {
            std::ofstream csv(output_csv);
            csv << "Operation,NumProcesses,Time_ms\n";
            for (const auto& [op, time] : results) {
                csv << op << "," << size << "," << time << "\n";
            }
            csv.close();
            std::cout << "\nResults saved to " << output_csv << "\n";
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
