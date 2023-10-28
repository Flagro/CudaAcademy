/*
Compiling with: nvcc -O3 -gencode arch=compute_52,code=sm_52 -gencode arch=compute_60,code=sm_60 main.cu -o run -std=c++11
Run the program: ./run [vector size] {block size}
*/
 
#include <iostream>
#include <cstdlib>
#include <random>
#include <string>
#include <chrono>
 
using num_type = float;
using size_type = long long;
 
enum { BLOCK_SIZE = 1024, PRINT_MARGINS = 6 };
 
//const num_type EPS = 1e-12;
 
cudaEvent_t start, stop;
 
void start_timer() {
   // FIXME: ADD TIMING CODE, HERE, USE GLOBAL VARIABLES AS NEEDED.
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
}
 
float stop_timer() {
   // FIXME: ADD TIMING CODE, HERE, USE GLOBAL VARIABLES AS NEEDED.
    cudaDeviceSynchronize();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds_gpu = 0;
    cudaEventElapsedTime(&milliseconds_gpu, start, stop);
   return milliseconds_gpu;
}
 
num_type * gen_vector(size_type n) {
    num_type *vec = new num_type[n];
    for (size_type i = 0; i < n; ++i) {
        vec[i] = (num_type) rand() / rand();
    }
    return vec;
}
 
num_type * copy_vector(num_type *vec, size_type n) {
    num_type *new_vec = new num_type[n];
    for (size_type i = 0; i < n; ++i) {
        new_vec[i] = vec[i];
    }
    return new_vec;
}
void print_vector(num_type *vec, long long n)
{
    if (n <= 2 * PRINT_MARGINS) {
        for (int i = 0; i < n; ++i) {
            std::cout << vec[i] << ' ';
        }
    } else {
        for (size_type i = 0; i < PRINT_MARGINS; ++i) {
            std::cout << vec[i] << ' ';
        }
        std::cout << ". . . ";
    }
    std::cout << std::endl;
}
 
__global__ void cuda_vec_add(num_type *arr_a, num_type *arr_b, num_type *arr_c, size_type n, size_type offset) {
    size_type idx = blockIdx.x * blockDim.x + threadIdx.x + offset;
    if (idx < n) {
        arr_c[idx] = arr_a[idx] + arr_b[idx];
    }
}
 
float vec_add(size_type n) {
    num_type *h_a = gen_vector(n);
    num_type *h_b = gen_vector(n);
    std::cout << "First generated vector: ";
    print_vector(h_a, n);
    std::cout << "Second generated vector: ";
    print_vector(h_b, n);
    num_type *h_c = new num_type[n];
    // allocating memory for devices' vectors
    size_type bytes = n * sizeof(num_type);
    num_type *d_a, *d_b, *d_c;
    auto start = std::chrono::steady_clock::now();
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);
    // initializing devices' vectors
    cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);
    // initializing grid size
    long long grid_size = (n - 1) / BLOCK_SIZE + 1;
    // preparing for measuring GPU elapsed time:
    // calling CUDA kernel
    cuda_vec_add<<<grid_size, BLOCK_SIZE>>>(d_a, d_b, d_c, n, 0);
    // copying the results
    cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);
    // freeing allocated GPU's memory
    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<float> elapsed = end - start;
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    return elapsed.count();
}
 
float vec_add_mgpu(size_type n) {
    size_type bytes = n * sizeof(num_type);
    num_type *h_a, *h_b;
    cudaHostAlloc(&h_a, bytes, cudaHostAllocDefault);
    cudaHostAlloc(&h_b, bytes, cudaHostAllocDefault);
    for (size_type i = 0; i < n; ++i) {
        h_a[i] = (num_type) rand() / rand();
        h_b[i] = (num_type) rand() / rand();
    }
    std::cout << "First generated vector: ";
    print_vector(h_a, n);
    std::cout << "Second generated vector: ";
    print_vector(h_b, n);
    num_type *h_c;
    cudaMallocHost(&h_c, bytes);
 
    // allocating memory for devices' vectors
    const int device_count = 2;
    //cudaGetDeviceCount(&device_count);
    num_type *d_a[device_count];
    num_type *d_b[device_count];
    num_type *d_c[device_count];
    size_type dev_bytes = (n / device_count) * sizeof(num_type);
    auto start = std::chrono::steady_clock::now();
    for (size_type i = 0; i < device_count; ++i) {
        cudaSetDevice(i);
        cudaMalloc((void **)&(d_a[i]), dev_bytes);
        cudaMalloc((void **)&(d_b[i]), dev_bytes);
        cudaMalloc((void **)&(d_c[i]), dev_bytes);
    }
    dim3 block(BLOCK_SIZE);
    dim3 grid((n / device_count + BLOCK_SIZE - 1) / BLOCK_SIZE);
    // preparing for measuring GPU elapsed time:
    for (size_type i = 0; i < device_count; ++i) {
        int offset = i * (n / device_count);
        cudaSetDevice(i);
        cudaMemcpyAsync(&(d_a[i])[offset], &h_a[offset], dev_bytes, cudaMemcpyHostToDevice);
        cudaMemcpyAsync(&(d_b[i])[offset], &h_b[offset], dev_bytes, cudaMemcpyHostToDevice);
        cudaMemcpyAsync(&(d_c[i])[offset], &h_c[offset], dev_bytes, cudaMemcpyHostToDevice);;
        cuda_vec_add<<<grid, block>>>(d_a[i], d_b[i], d_c[i], n / device_count, offset);
        cudaMemcpyAsync(&h_c[offset], &(d_c[i])[offset], dev_bytes, cudaMemcpyDeviceToHost);
    }
    for (size_type i = 0; i < device_count; i++){
        cudaSetDevice(i);
        cudaDeviceSynchronize();
        cudaFree(d_a[i]);
        cudaFree(d_b[i]);
        cudaFree(d_c[i]);
    }
    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<float> elapsed = end - start;
 
    // freeing allocated GPU's memory
    std::cout << "The result: ";
    print_vector(h_c, n);
    cudaFreeHost(h_a);
    cudaFreeHost(h_b);
    cudaFreeHost(h_c);
    return elapsed.count();
}
 
int main(int argc, char *argv[]) {
    // initializing n - arr len, and block_size
    if (argc < 2) {
        std::cerr << "please specify the program type" << std::endl;
        return 1;
    }
    if (argc < 3) {
        std::cerr << "please specify an arr len" << std::endl;
        return 1;
    }
    long long n = std::strtoll(argv[2], NULL, 10);
    bool type = std::strtol(argv[1], NULL, 10);
    float elapsed;
    if (type) {
        elapsed = vec_add_mgpu(n);
    } else {
        elapsed = vec_add(n);
    }
    std::cout << "Total time: " << elapsed << std::endl;
    return 0;
}
