#!/bin/bash

# MPI Image Processing Run Script
# Usage: ./run_mpi.sh <operation> <input_image> <output_image> <num_processes> [params...]

set -e

if [ "$#" -lt 4 ]; then
    echo "Usage: $0 <operation> <input_image> <output_image> <num_processes> [params...]"
    echo ""
    echo "Operations:"
    echo "  gaussian-blur <kernel_size> <sigma>"
    echo "  median <kernel_size>"
    echo "  box-blur <kernel_size>"
    echo ""
    echo "Example:"
    echo "  $0 gaussian-blur input.jpg output.jpg 4 5 1.0"
    exit 1
fi

OPERATION=$1
INPUT_IMAGE=$2
OUTPUT_IMAGE=$3
NUM_PROCESSES=$4
shift 4

# Check if input image exists
if [ ! -f "$INPUT_IMAGE" ]; then
    echo "Error: Input image '$INPUT_IMAGE' not found"
    exit 1
fi

# Find the MPI executable
MPI_EXEC="./build/bin/iproc_mpi_demo"
if [ ! -f "$MPI_EXEC" ]; then
    echo "Error: MPI executable not found at $MPI_EXEC"
    echo "Please build the project first:"
    echo "  mkdir build && cd build"
    echo "  cmake .. -DBUILD_MPI=ON"
    echo "  make"
    exit 1
fi

# Run with MPI
echo "Running MPI image processing with $NUM_PROCESSES processes..."
echo "Operation: $OPERATION"
echo "Input: $INPUT_IMAGE"
echo "Output: $OUTPUT_IMAGE"
echo ""

mpirun -np $NUM_PROCESSES $MPI_EXEC $OPERATION $INPUT_IMAGE $OUTPUT_IMAGE "$@"

echo ""
echo "Done! Output saved to $OUTPUT_IMAGE"
