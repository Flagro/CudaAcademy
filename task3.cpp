/*
Compiling with: nvcc -O3 -gencode arch=compute_52,code=sm_52 -gencode arch=compute_60,code=sm_60 main.cu -o run -std=c++11
Run the program: ./run [vector size] {block size}
*/
 
#include <iostream>
#include <cstdlib>
#include <random>
#include <string>
#include <chrono>
 
using num_type = double;
using size_type = long long;
 
enum { BLOCK_SIZE = 1024, PRINT_MARGINS = 3 };
 
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
        for (size_type i = n - PRINT_MARGINS; i < n; ++i) {
            std::cout << vec[i] << ' ';
        }
    }
    std::cout << std::endl;
}
 
__global__ void cuda_vec_add(num_type *arr_a, num_type *arr_b, num_type *arr_c, size_type n) {
    size_type idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        arr_c[idx] = arr_a[idx] + arr_b[idx];
    }
}
 
__global__ void cuda_vec_add_divergence(num_type *arr_a, num_type *arr_b, num_type *arr_c, size_type n) {
    size_type idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        if ((idx + (size_type) arr_a[idx] + (size_type) (arr_a[idx] / 2) - 5) % 3 == 2) {
            arr_c[idx] = threadIdx.x * arr_a[idx] + 2 * arr_b[idx];
        } else {
            arr_c[idx] = 5 * arr_a[idx / 2] / blockIdx.x + 3 * arr_b[idx / 3];
        }
    }
}
 
float vec_add(num_type *h_a, num_type *h_b, num_type **h_c, size_type n, bool type) {
    *h_c = new num_type[n];
 
    // allocating memory for devices' vectors
    size_type bytes = n * sizeof(num_type);
    num_type *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);
 
    // initializing devices' vectors
    cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);
 
    // initializing grid size
    size_type grid_size = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
 
    // preparing for measuring GPU elapsed time:
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
 
    // calling CUDA kernel
    if (!type) {
        cuda_vec_add<<<grid_size, BLOCK_SIZE>>>(d_a, d_b, d_c, n);
    } else {
        cuda_vec_add_divergence<<<grid_size, BLOCK_SIZE>>>(d_a, d_b, d_c, n);
    }
 
    cudaDeviceSynchronize();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds_gpu = 0;
    cudaEventElapsedTime(&milliseconds_gpu, start, stop);
 
    // copying the results
    cudaMemcpy(*h_c, d_c, bytes, cudaMemcpyDeviceToHost);
 
    // freeing allocated GPU's memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    return milliseconds_gpu;
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
    num_type *h_a = gen_vector(n);
    num_type *h_b = gen_vector(n);
    num_type *ans = new num_type[n];
    std::cout << "First generated vector: ";
    print_vector(h_a, n);
    std::cout << "Second generated vector: ";
    print_vector(h_b, n);
    float elapsed = vec_add(h_a, h_b, &ans, n, type);
    std::cout << "The result: ";
    print_vector(ans, n);
    std::cout << "Elapsed time: " << elapsed << std::endl;
    delete[] h_a;
    delete[] h_b;
    delete[] ans;
    return 0;
}
