//
//  main.cpp
//  
//
//  Created by Elijah Afanasiev on 25.09.2018.
//
//
 
// System includes
#include <stdio.h>
#include <iostream>
#include <assert.h>
 
// CUDA runtime
#include <cuda.h>
#include <cuda_runtime.h>
 
#ifndef MAX
#define MAX(a,b) (a > b ? a : b)
#endif
 
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
 
__global__ void vectorAddGPU(float *a, float *b, float *c, int N)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    
    if (idx < N)
    {
        c[idx] = a[idx] + b[idx];
    }
}
 
void unified_samle(int size = 1048576)
{
    printf("======UNIFIED ALLOCATION======\n");
    int n = size;
    
    int nBytes = n*sizeof(float);
    
    float *a, *b;  // host data
    float *c;  // results
    
    dim3 block(256);
    dim3 grid((unsigned int)ceil(n/(float)block.x));
    
    start_timer();
    cudaMallocManaged(&a, nBytes);
    cudaMallocManaged(&b, nBytes);
    cudaMallocManaged(&c, nBytes);
    for(int i=0;i<n;i++)
    {
        a[i] = rand() / (float)RAND_MAX;
        b[i] = rand() / (float)RAND_MAX;
        c[i] = 0;
    }
 
    printf("Doing GPU Vector add\n");
    
    vectorAddGPU<<<grid, block>>>(a, b, c, n);
    float total_timer = stop_timer();
    
    printf("Total execution time: %f ms\n", total_timer);
    
    cudaThreadSynchronize();
    
    cudaFree(a);
    cudaFree(b);
    cudaFree(c);
}
 
void pinned_samle(int size = 1048576)
{
    printf("======PINNED ALLOCATION======\n");
    int n = size;
    
    int nBytes = n*sizeof(float);
    
    float *a, *b;  // host data
    float *c;  // results
    
    float *a_d,*b_d,*c_d;
    
    dim3 block(256);
    dim3 grid((unsigned int)ceil(n/(float)block.x));
    
    start_timer();
    cudaMallocHost(&a, nBytes);
    cudaMallocHost(&b, nBytes);
    cudaMallocHost(&c, nBytes);
    float malloc_cpy_timer = stop_timer();
    for(int i=0;i<n;i++)
    {
        a[i] = rand() / (float)RAND_MAX;
        b[i] = rand() / (float)RAND_MAX;
        c[i] = 0;
    }
     
    printf("Allocating device memory on host..\n");
 
    cudaMalloc((void **)&a_d,n*sizeof(float));
    cudaMalloc((void **)&b_d,n*sizeof(float));
    cudaMalloc((void **)&c_d,n*sizeof(float));
    
    printf("Copying to device..\n");
    
    start_timer();
    cudaMemcpy(a_d,a,n*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(b_d,b,n*sizeof(float), cudaMemcpyHostToDevice);
    
    float copy_time = stop_timer();
 
    printf("Doing GPU Vector add\n");
    
    start_timer();
    vectorAddGPU<<<grid, block>>>(a_d, b_d, c_d, n);
    float kernel_timer = stop_timer();
    
    printf("Kernel execution time: %f ms\n", kernel_timer);
    printf("Malloc execution time: %f ms\n", malloc_cpy_timer);
    printf("Copy to device execution time: %f ms\n", copy_time);
    
    cudaThreadSynchronize();
    
    cudaFreeHost(a);
    cudaFreeHost(b);
    cudaFreeHost(c);
 
    cudaFree(a_d);
    cudaFree(b_d);
    cudaFree(c_d);
}
 
void usual_sample(int size = 1048576)
{
    printf("======USUAL ALLOCATION======\n");
    int n = size;
    
    int nBytes = n*sizeof(float);
    
    float *a, *b;  // host data
    float *c;  // results
    
    float *a_d,*b_d,*c_d;
    
    dim3 block(256);
    dim3 grid((unsigned int)ceil(n/(float)block.x));
    
    start_timer();
    a = (float *)malloc(nBytes);
    b = (float *)malloc(nBytes);
    c = (float *)malloc(nBytes);
    float malloc_cpy_timer = stop_timer();
    for(int i=0;i<n;i++)
    {
        a[i] = rand() / (float)RAND_MAX;
        b[i] = rand() / (float)RAND_MAX;
        c[i] = 0;
    }
    
    printf("Allocating device memory on host..\n");
    start_timer();
 
    cudaMalloc((void **)&a_d,n*sizeof(float));
    cudaMalloc((void **)&b_d,n*sizeof(float));
    cudaMalloc((void **)&c_d,n*sizeof(float));
    
    printf("Copying to device..\n");
    
    cudaMemcpy(a_d,a,n*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(b_d,b,n*sizeof(float), cudaMemcpyHostToDevice);
    float copy_timer = stop_timer();
    
    printf("Doing GPU Vector add\n");
    
    start_timer();
    vectorAddGPU<<<grid, block>>>(a_d, b_d, c_d, n);
    float kernel_timer = stop_timer();
    
    printf("Kernel execution time: %f ms\n", kernel_timer);
    printf("Malloc execution time: %f ms\n", malloc_cpy_timer);
    printf("Copy execution time: %f ms\n", copy_timer);
    
    cudaThreadSynchronize();
    
    cudaFree(a_d);
    cudaFree(b_d);
    cudaFree(c_d);
    free(a);
    free(b);
    free(c);
}
 
 
int main(int argc, char **argv)
{
    usual_sample(atoi(argv[1]));
    pinned_samle(atoi(argv[1]));
    unified_samle(atoi(argv[1]));
    
    return 0;
}
