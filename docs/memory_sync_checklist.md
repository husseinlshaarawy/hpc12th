# Memory Synchronization Checklist

This document provides concrete guidelines for ensuring memory safety and correctness in OpenMP, MPI, and CUDA implementations.

## OpenMP Memory Safety

### Data Scoping Rules
✅ **ALWAYS** explicitly specify data-sharing attributes
- Use `private` for loop iteration variables and thread-local temporaries
- Use `shared` for read-only data and output arrays
- Use `reduction` for variables accumulated across threads
- ❌ **NEVER** rely on default data-sharing (implementation-dependent)

### Example: Correct OpenMP Scoping
```cpp
#pragma omp parallel for shared(src, result, kernel_size) private(sum, count) schedule(dynamic)
for (int i = 0; i < src.rows; ++i) {
    double sum = 0.0;  // Private to each thread
    int count = 0;
    // ... computation
    result.at<uchar>(i, j) = sum / count;  // result is shared, safe per-pixel write
}
```

### Avoiding False Sharing
✅ Use thread-private buffers for temporary data
```cpp
#pragma omp parallel shared(src, result)
{
    std::vector<uchar> thread_buffer;  // Private to each thread
    #pragma omp for
    for (int i = 0; i < src.rows; ++i) {
        // Use thread_buffer for computations
    }
}
```

### Reductions
✅ Use `reduction` clause for aggregating values
```cpp
double total = 0.0;
#pragma omp parallel for reduction(+:total)
for (int i = 0; i < n; ++i) {
    total += data[i];
}
```

### Common Pitfalls
❌ Writing to the same memory location from multiple threads without synchronization
❌ Using `static` variables inside parallel regions
❌ Relying on specific thread execution order

---

## MPI Memory Safety

### Halo Exchange Pattern
✅ Use non-blocking communication to overlap computation and communication
```cpp
// Start non-blocking sends/receives for halo regions
std::vector<MPI_Request> requests;
MPI_Isend(top_row, size, MPI_BYTE, rank-1, tag, comm, &req);
requests.push_back(req);
MPI_Irecv(top_halo, size, MPI_BYTE, rank-1, tag, comm, &req);
requests.push_back(req);

// Process interior rows while halos are exchanged
for (int i = halo_size; i < num_rows - halo_size; ++i) {
    // ... computation on interior
}

// Wait for halo exchange to complete
MPI_Waitall(requests.size(), requests.data(), MPI_STATUSES_IGNORE);

// Now process boundary rows
for (int i = 0; i < halo_size; ++i) {
    // ... computation on top boundary
}
```

### Buffer Management
✅ Allocate separate send and receive buffers
❌ Reusing buffers for send/receive before communication completes

### Collective Operations
✅ Ensure all ranks participate in collective operations (Scatterv, Gatherv, Allreduce)
✅ Match send/receive counts and data types exactly

### Data Consistency
✅ Only rank 0 performs I/O in typical patterns
✅ Broadcast image metadata (dimensions, type) before scattering data
```cpp
if (rank == 0) {
    rows = img.rows;
    cols = img.cols;
}
MPI_Bcast(&rows, 1, MPI_INT, 0, comm);
MPI_Bcast(&cols, 1, MPI_INT, 0, comm);
```

### Common Pitfalls
❌ Deadlocks from mismatched sends/receives
❌ Buffer overruns from incorrect size calculations
❌ Race conditions from simultaneous writes to overlapping regions

---

## CUDA Memory Safety

### Memory Transfer Pattern
✅ Use pinned memory for asynchronous transfers
```cpp
// Allocate pinned memory
uchar* h_pinned;
cudaMallocHost(&h_pinned, size);

// Async transfer to device
cudaMemcpyAsync(d_ptr, h_pinned, size, cudaMemcpyHostToDevice, stream);

// Launch kernel
kernel<<<grid, block, 0, stream>>>(d_ptr, ...);

// Async transfer back
cudaMemcpyAsync(h_pinned, d_ptr, size, cudaMemcpyDeviceToHost, stream);

// Synchronize
cudaStreamSynchronize(stream);
```

### Kernel Launch Safety
✅ Check for launch errors
```cpp
kernel<<<grid, block>>>(args);
cudaError_t err = cudaGetLastError();
if (err != cudaSuccess) {
    // Handle error
}
```

### Shared Memory Usage
✅ Declare shared memory with correct size and synchronize
```cpp
__global__ void kernel(float* input, float* output) {
    extern __shared__ float s_data[];
    
    // Load data into shared memory
    s_data[threadIdx.x] = input[...];
    __syncthreads();  // Wait for all threads to load
    
    // Use shared memory
    float sum = s_data[threadIdx.x] + s_data[threadIdx.x + 1];
    __syncthreads();  // If writing to shared memory again
    
    output[...] = sum;
}
```

### Constant Memory
✅ Use constant memory for read-only kernel parameters
```cpp
__constant__ float d_kernel[256];

// Copy to constant memory on host
cudaMemcpyToSymbol(d_kernel, h_kernel, size);

// Access in kernel (cached, broadcast to all threads)
__global__ void convolve(...) {
    float value = d_kernel[i] * input[...];
}
```

### Thread Synchronization
✅ Use `__syncthreads()` when threads depend on each other's results
❌ Using `__syncthreads()` in conditional code (causes deadlock)

### Memory Access Patterns
✅ Coalesce global memory accesses (stride-1 access pattern)
```cpp
// Good: coalesced
int idx = blockIdx.x * blockDim.x + threadIdx.x;
output[idx] = input[idx];

// Bad: strided access
int idx = threadIdx.x * stride + blockIdx.x;
```

### Common Pitfalls
❌ Not synchronizing before accessing results
❌ Race conditions in shared memory without `__syncthreads()`
❌ Out-of-bounds access (always check `if (idx < n)`)
❌ Using too much shared memory (exceeds limit per SM)

---

## General Guidelines

### Correctness Verification
1. Test with small inputs where results can be manually verified
2. Compare parallel results against serial reference implementation
3. Use fixed random seeds for reproducibility
4. Test edge cases (small images, odd dimensions, boundary conditions)

### Debugging Tools
- **OpenMP**: Use `OMP_NUM_THREADS=1` to isolate threading issues
- **MPI**: Use `mpirun -np 1` to test without distributed memory
- **CUDA**: Use `cuda-memcheck` to detect memory errors

### Performance vs. Correctness Trade-offs
✅ Prioritize correctness first, then optimize
✅ Document any approximations or numerical differences
✅ Use atomic operations when necessary (but prefer avoiding them for performance)

---

## Verification Checklist

### Before Submitting Code
- [ ] All OpenMP loops have explicit data-sharing clauses
- [ ] MPI halo exchanges use non-blocking communication
- [ ] CUDA kernels check bounds and synchronize appropriately
- [ ] Tests pass for serial, OpenMP, MPI, and CUDA variants
- [ ] Benchmark results show expected speedup
- [ ] No data races detected by sanitizers (ThreadSanitizer, cuda-memcheck)
- [ ] Code handles edge cases (1x1 images, prime dimensions, etc.)
