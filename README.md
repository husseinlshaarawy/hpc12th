# HPC Image Processing Assignment

A comprehensive image processing library with serial and parallel implementations (OpenMP, MPI, CUDA) covering 100 required features and 50 bonus features.

## Table of Contents
- [Features](#features)
- [Requirements](#requirements)
- [Building](#building)
- [Usage](#usage)
- [Testing](#testing)
- [Benchmarking](#benchmarking)
- [Documentation](#documentation)

## Features

### Core Features (100 marks)
- **Image I/O**: Load/save JPG, PNG, BMP with color/grayscale preservation
- **Point Operations**: Grayscale conversion, brightness/contrast adjustment, thresholding, inversion, gamma correction
- **Noise Generation**: Salt-and-pepper, Gaussian noise
- **Smoothing**: Box blur, Gaussian blur, median filter
- **Edge Detection**: Sobel, Canny, sharpen filter
- **Morphological Operations**: Erosion, dilation, opening, closing
- **Geometric Transforms**: Rotation, scaling, translation, horizontal/vertical flip
- **Color Operations**: Channel split/merge, HSV adjustments

### Bonus Features (50 marks)
- **Advanced Thresholding**: Otsu, adaptive Gaussian, Sauvola, Niblack
- **Additional Noise**: Speckle noise
- **Advanced Filtering**: Bilateral filter
- **Additional Edge Operators**: Prewitt, Laplacian
- **Advanced Morphology**: Morphological reconstruction
- **Perspective Transforms**: Arbitrary perspective with subpixel interpolation
- **Color Space Operations**: Lab tone mapping

### Parallel Implementations
- **OpenMP**: Row/tile-based parallelization with explicit data scoping
- **MPI**: Row-wise partitioning with non-blocking halo exchange
- **CUDA**: Device kernels with pinned memory, async transfers, and shared memory optimization

## Requirements

### Build Dependencies
- C++17 or newer compiler (g++ 7+, clang++ 5+)
- CMake 3.12+
- OpenCV 4.0+ (required)
- OpenMP (optional, for parallel implementations)
- MPI (optional, OpenMPI or MPICH)
- CUDA Toolkit 10.0+ (optional, for GPU acceleration)

### Install Dependencies on Ubuntu/Debian
```bash
sudo apt-get update
sudo apt-get install -y build-essential cmake
sudo apt-get install -y libopencv-dev
sudo apt-get install -y libopenmpi-dev  # For MPI support
```

## Building

### Basic Build (Serial + OpenMP)
```bash
mkdir build && cd build
cmake .. -DBUILD_OPENMP=ON -DBUILD_MPI=OFF -DBUILD_CUDA=OFF
make -j$(nproc)
```

### Build with All Features
```bash
mkdir build && cd build
cmake .. -DBUILD_OPENMP=ON -DBUILD_MPI=ON -DBUILD_CUDA=ON
make -j$(nproc)
```

### Build Options
- `BUILD_OPENMP`: Enable OpenMP parallel implementations (default: ON)
- `BUILD_MPI`: Enable MPI parallel implementations (default: OFF)
- `BUILD_CUDA`: Enable CUDA GPU acceleration (default: OFF)
- `BUILD_TESTS`: Build unit tests with Catch2 (default: ON)
- `BUILD_BENCHMARKS`: Build benchmark tools (default: ON)

### Build Targets
After building, executables are in `build/bin/`:
- `iproc_demo`: Serial implementation CLI
- `iproc_omp_demo`: OpenMP parallel CLI
- `iproc_mpi_demo`: MPI parallel CLI (requires MPI)
- `iproc_cuda_demo`: CUDA accelerated CLI (requires CUDA)
- `unit_tests`: Test suite
- `benchmark_serial`, `benchmark_openmp`, `benchmark_mpi`, `benchmark_cuda`: Benchmark tools

## Usage

### Serial Implementation
```bash
# Grayscale conversion
./build/bin/iproc_demo grayscale input.jpg output.jpg

# Gaussian blur
./build/bin/iproc_demo gaussian-blur input.jpg output.jpg 5 1.0

# Edge detection
./build/bin/iproc_demo sobel input.jpg output_edges.jpg

# Adjust brightness
./build/bin/iproc_demo brightness input.jpg brighter.jpg 50

# Median filter (noise removal)
./build/bin/iproc_demo median input.jpg denoised.jpg 5

# Rotation
./build/bin/iproc_demo rotate input.jpg rotated.jpg 45

# Thresholding
./build/bin/iproc_demo threshold input.jpg binary.jpg 128
```

### OpenMP Parallel
```bash
# Gaussian blur with 4 threads
./build/bin/iproc_omp_demo gaussian-blur input.jpg output.jpg 5 1.0 4

# Median filter with 8 threads
./build/bin/iproc_omp_demo median input.jpg output.jpg 5 8

# Box blur (auto thread count)
./build/bin/iproc_omp_demo box-blur input.jpg output.jpg 7
```

### MPI Parallel
```bash
# Gaussian blur with 4 MPI processes
mpirun -np 4 ./build/bin/iproc_mpi_demo gaussian-blur input.jpg output.jpg 5 1.0

# Median filter with 8 processes
mpirun -np 8 ./build/bin/iproc_mpi_demo median input.jpg output.jpg 5

# Box blur
mpirun -np 4 ./build/bin/iproc_mpi_demo box-blur input.jpg output.jpg 7
```

### CUDA Accelerated
```bash
# Gaussian blur on GPU
./build/bin/iproc_cuda_demo gaussian-blur input.jpg output.jpg 5 1.0

# Median filter on GPU
./build/bin/iproc_cuda_demo median input.jpg output.jpg 5

# Sobel edge detection on GPU
./build/bin/iproc_cuda_demo sobel input.jpg output_edges.jpg
```

### All Available Operations

#### Point Operations
- `grayscale` - Convert to grayscale
- `brightness <beta>` - Adjust brightness (-255 to 255)
- `contrast <alpha>` - Adjust contrast (0.0 to 3.0)
- `threshold <value>` - Binary threshold (0-255)
- `invert` - Invert colors
- `gamma <gamma>` - Gamma correction

#### Noise Operations
- `salt-pepper <amount> [seed]` - Add salt and pepper noise
- `gaussian-noise <mean> <stddev> [seed]` - Add Gaussian noise
- `speckle-noise <variance> [seed]` - Add speckle noise (bonus)

#### Smoothing Filters
- `box-blur <kernel_size>` - Box blur/mean filter
- `gaussian-blur <kernel_size> <sigma>` - Gaussian blur
- `median <kernel_size>` - Median filter
- `bilateral <d> <sigma_color> <sigma_space>` - Bilateral filter (bonus)

#### Edge Detection
- `sobel [kernel_size]` - Sobel edge detection
- `canny <low> <high>` - Canny edge detection
- `sharpen` - Sharpen filter
- `prewitt` - Prewitt edge detection (bonus)
- `laplacian [kernel_size]` - Laplacian edge detection (bonus)

#### Morphological Operations
- `erosion <kernel_size> [iterations]` - Morphological erosion
- `dilation <kernel_size> [iterations]` - Morphological dilation
- `opening <kernel_size>` - Morphological opening
- `closing <kernel_size>` - Morphological closing

#### Geometric Transforms
- `rotate <angle>` - Rotate image (degrees)
- `scale <scale_x> <scale_y>` - Scale image
- `translate <dx> <dy>` - Translate/shift image
- `flip-h` - Flip horizontally
- `flip-v` - Flip vertically

#### Color Operations
- `adjust-hsv <hue> <sat> <val>` - Adjust HSV values
- `lab-tone <l_scale>` - Lab color space tone mapping (bonus)

#### Advanced Thresholding (Bonus)
- `otsu` - Otsu's automatic thresholding
- `adaptive-gaussian <block_size> <C>` - Adaptive Gaussian threshold
- `sauvola <window_size> <k> [R]` - Sauvola thresholding
- `niblack <window_size> <k>` - Niblack thresholding

## Testing

### Run All Tests
```bash
cd /path/to/hpc12th
./build/bin/unit_tests
```

### Run Specific Test Cases
```bash
# Run only I/O tests
./build/bin/unit_tests "[io]"

# Run only smoothing tests
./build/bin/unit_tests "[smoothing]"

# Run with verbose output
./build/bin/unit_tests -s
```

### Test Categories
- `[io]` - Image I/O operations
- `[point_ops]` - Point operations (brightness, contrast, etc.)
- `[noise]` - Noise generation
- `[smoothing]` - Smoothing filters
- `[edge_detection]` - Edge detection and sharpening
- `[morphology]` - Morphological operations
- `[geometric]` - Geometric transformations
- `[color]` - Color and channel operations
- `[bonus]` - Bonus features

## Benchmarking

### Serial Benchmark
```bash
./build/bin/benchmark_serial path/to/image.jpg results_serial.csv
```

### OpenMP Benchmark
```bash
./build/bin/benchmark_openmp path/to/image.jpg results_omp.csv
```

### MPI Benchmark
```bash
mpirun -np 4 ./build/bin/benchmark_mpi path/to/image.jpg results_mpi.csv
```

### CUDA Benchmark
```bash
./build/bin/benchmark_cuda path/to/image.jpg results_cuda.csv
```

### Analyzing Results
Benchmark results are saved as CSV files with timing data:
```csv
Operation,NumThreads,ImageSize,Time_ms
gaussian_blur,1,1048576,125.3
gaussian_blur,4,1048576,35.2
```

You can analyze speedup with:
```bash
python3 tools/benchmark/analyze_speedup.py results_*.csv
```

## Performance Considerations

### OpenMP
- Uses row-wise parallelization for most filters
- Explicit `private`/`shared`/`reduction` annotations prevent race conditions
- Thread-private buffers for median filter to avoid false sharing
- Dynamic scheduling for load balancing

### MPI
- Row-wise image partitioning handles non-divisible heights
- Non-blocking halo exchange (`MPI_Isend`/`MPI_Irecv`) overlaps communication with computation
- Rank 0 handles I/O, distributes with `Scatterv`, collects with `Gatherv`

### CUDA
- Pinned host memory for faster PCIe transfers
- Asynchronous transfers with streams
- Separable convolution for Gaussian blur reduces complexity from O(n²) to O(n)
- Shared memory for convolution kernels improves cache hit rate
- Constant memory for small kernel coefficients
- Tile-based processing with overlap for large images

## Documentation

- [Memory Synchronization Checklist](docs/memory_sync_checklist.md) - OpenMP/MPI/CUDA memory safety guidelines
- [Implementation Plan](docs/plan.md) - Detailed feature mapping to grading rubric
- API documentation in header files (`include/image_filters*.h`)

## Project Structure
```
.
├── include/                    # Public API headers
│   ├── image_filters.h        # Core serial API
│   ├── image_filters_omp.h    # OpenMP API
│   ├── image_filters_mpi.h    # MPI API
│   └── image_filters_cuda.h   # CUDA API
├── src/                        # Implementation files
│   ├── image_filters.cpp      # Serial implementations
│   ├── image_filters_omp.cpp  # OpenMP implementations
│   ├── image_filters_mpi.cpp  # MPI implementations
│   ├── main.cpp               # Serial CLI
│   ├── cli_omp.cpp            # OpenMP CLI
│   ├── cli_mpi.cpp            # MPI CLI
│   └── cli_cuda.cpp           # CUDA CLI
├── cuda/                       # CUDA kernels and wrappers
│   ├── kernels.cu             # Device kernels
│   └── cuda_wrapper.cpp       # Host-side wrappers
├── tests/                      # Unit tests
│   ├── test_*.cpp             # Test files
│   └── data/                  # Test images
├── tools/benchmark/            # Benchmark scripts
│   └── benchmark_*.cpp        # Timing measurements
├── docs/                       # Documentation
├── CMakeLists.txt             # Build configuration
└── README.md                  # This file
```

## CI/CD

Continuous integration is configured via GitHub Actions (`.github/workflows/ci.yml`):
- Builds serial and OpenMP variants on ubuntu-latest
- Runs full test suite
- MPI and CUDA jobs are optional (require self-hosted runners or specific configurations)

## License

Academic project for HPC course.

## Authors

Hussein Shaarawy

## References

- OpenCV Documentation: https://docs.opencv.org/
- OpenMP Specification: https://www.openmp.org/
- MPI Standard: https://www.mpi-forum.org/
- CUDA Programming Guide: https://docs.nvidia.com/cuda/