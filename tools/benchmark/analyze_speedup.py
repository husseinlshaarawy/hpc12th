#!/usr/bin/env python3
"""
Analyze speedup from benchmark CSV files

Requirements: Python 3.x (no external dependencies)
"""

import sys
import csv
from pathlib import Path
from collections import defaultdict

def check_requirements():
    """Check if Python version is sufficient"""
    if sys.version_info < (3, 6):
        print("Error: Python 3.6 or newer is required")
        sys.exit(1)

def load_benchmark(filepath):
    """Load benchmark CSV file"""
    data = []
    with open(filepath, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append(row)
    return data

def analyze_openmp(serial_file, openmp_file):
    """Analyze OpenMP speedup compared to serial"""
    serial_data = load_benchmark(serial_file)
    openmp_data = load_benchmark(openmp_file)
    
    # Build serial baseline
    serial_times = {}
    for row in serial_data:
        op = row['Operation']
        time_ms = float(row['Time_ms'])
        serial_times[op] = time_ms
    
    # Compute speedups
    speedups = defaultdict(list)
    for row in openmp_data:
        op = row['Operation']
        threads = int(row['NumThreads'])
        time_ms = float(row['Time_ms'])
        
        if op in serial_times:
            speedup = serial_times[op] / time_ms
            speedups[op].append((threads, speedup, time_ms))
    
    # Print results
    print("=" * 60)
    print("OpenMP Speedup Analysis")
    print("=" * 60)
    
    for op in sorted(speedups.keys()):
        print(f"\nOperation: {op}")
        print(f"  Serial time: {serial_times[op]:.2f} ms")
        print(f"  {'Threads':<10} {'Time (ms)':<12} {'Speedup':<10} {'Efficiency':<10}")
        print(f"  {'-'*10} {'-'*12} {'-'*10} {'-'*10}")
        
        for threads, speedup, time_ms in sorted(speedups[op]):
            efficiency = (speedup / threads) * 100
            print(f"  {threads:<10} {time_ms:<12.2f} {speedup:<10.2f} {efficiency:<10.1f}%")

def analyze_mpi(serial_file, mpi_file):
    """Analyze MPI speedup compared to serial"""
    serial_data = load_benchmark(serial_file)
    mpi_data = load_benchmark(mpi_file)
    
    # Build serial baseline
    serial_times = {}
    for row in serial_data:
        op = row['Operation']
        time_ms = float(row['Time_ms'])
        serial_times[op] = time_ms
    
    # Compute speedups
    print("=" * 60)
    print("MPI Speedup Analysis")
    print("=" * 60)
    
    for row in mpi_data:
        op = row['Operation']
        procs = int(row['NumProcesses'])
        time_ms = float(row['Time_ms'])
        
        if op in serial_times:
            speedup = serial_times[op] / time_ms
            efficiency = (speedup / procs) * 100
            
            print(f"\nOperation: {op}")
            print(f"  Serial time: {serial_times[op]:.2f} ms")
            print(f"  MPI time ({procs} processes): {time_ms:.2f} ms")
            print(f"  Speedup: {speedup:.2f}x")
            print(f"  Efficiency: {efficiency:.1f}%")

def analyze_cuda(serial_file, cuda_file):
    """Analyze CUDA speedup compared to serial"""
    serial_data = load_benchmark(serial_file)
    cuda_data = load_benchmark(cuda_file)
    
    # Build serial baseline
    serial_times = {}
    for row in serial_data:
        op = row['Operation']
        time_ms = float(row['Time_ms'])
        serial_times[op] = time_ms
    
    # Compute speedups
    print("=" * 60)
    print("CUDA Speedup Analysis")
    print("=" * 60)
    
    for row in cuda_data:
        op = row['Operation']
        time_ms = float(row['Time_ms'])
        
        if op in serial_times:
            speedup = serial_times[op] / time_ms
            
            print(f"\nOperation: {op}")
            print(f"  Serial time: {serial_times[op]:.2f} ms")
            print(f"  CUDA time: {time_ms:.2f} ms")
            print(f"  Speedup: {speedup:.2f}x")

def main():
    check_requirements()
    
    if len(sys.argv) < 2:
        print("Usage: python3 analyze_speedup.py <benchmark_files...>")
        print("\nExample:")
        print("  python3 analyze_speedup.py serial.csv openmp.csv")
        print("  python3 analyze_speedup.py serial.csv mpi.csv")
        print("  python3 analyze_speedup.py serial.csv cuda.csv")
        sys.exit(1)
    
    files = [Path(f) for f in sys.argv[1:]]
    
    # Check files exist
    for f in files:
        if not f.exists():
            print(f"Error: File not found: {f}")
            sys.exit(1)
    
    # Detect types and analyze
    serial_file = None
    openmp_file = None
    mpi_file = None
    cuda_file = None
    
    for f in files:
        name = f.stem.lower()
        if 'serial' in name:
            serial_file = f
        elif 'openmp' in name or 'omp' in name:
            openmp_file = f
        elif 'mpi' in name:
            mpi_file = f
        elif 'cuda' in name:
            cuda_file = f
    
    if not serial_file:
        print("Error: No serial benchmark file found")
        print("Please provide a file with 'serial' in the name")
        sys.exit(1)
    
    if openmp_file:
        analyze_openmp(serial_file, openmp_file)
    
    if mpi_file:
        print()
        analyze_mpi(serial_file, mpi_file)
    
    if cuda_file:
        print()
        analyze_cuda(serial_file, cuda_file)

if __name__ == '__main__':
    main()
