//一个block最多可以同时有1024个线程，一个线程最多使用255个register。
//block中smem不能使用过多，会导致比较差的warp利用率；寄存器使用过多也会影响线程的并行度
//optimize gemm
#include<stdio.h>
#include<stdlib.h>
#include"assert.h"
#include<iostream>
// CUDA runtime
#include <cuda_runtime.h>
#include <cublas_v2.h>
using namespace std;

#define checkCudaErrors(func){\
    cudaError_t e = (func);\
    if(e!= cudaSuccess)\
        printf("%s %d CUDA: %s\n", __FILE__, __LINE__, cudaGetErrorString(e));\
}

#define Offset(row ,col, col_Dim) row  * col_Dim + col
#define FLOAT4(value) (reinterpret_cast<float4*>(&(value))[0])

void getCpuResult(float*C, float * A, float *B, const int M, const int N, const int K){
    for(int i=0;i<M;i++){
        for(int j=0;j<N;j++){
            float sum =0.0;
            for(int k=0;k<K;k++){
                sum += A[i*K+k] * B[k*N+j];
            }
            C[i*N+j] = sum;
        }
    }
}

bool checkAccResult(float * host_result, float * device_result, const int M, const int N){
    double eps = 1.e-6;  // machine zero
    bool result = true;
    for (int i = 0; i < M * N; i++) {
        // int row = i / N;
        // int col = i % N;
        double abs_err = fabs(host_result[i] - device_result[i]);
        // std::cout<<host_result[i]<<device_result[i]<<std::endl;
        if (abs_err > eps) {
            std::cout << "Error at index " << i << ": " << host_result[i] << " != " << device_result[i] << std::endl;
            result = false;
            break;
        }
    }
    return result;
}

//每个thread使用了28个寄存器
__global__ void gemm_base(float * __restrict__ A,
                        float * __restrict__ B, 
                        float * __restrict__ C,
                        const int M,
                        const int N,
                        const int K){
    int m = blockIdx.y * blockDim.y + threadIdx.y;
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if(m < M && n < N){
        float sum = 0.0f;
        #pragma unroll
        for(int k=0;k<K;k++){
            sum+=A[m*K+k]*B[k*N+n];
        }
        C[m*N+n] = sum;
    }
}

int main(int argc ,char ** argv){
    if(argc!=4){
        printf("usage: ./gemm  [M] [K] [N]\n");
        exit(0);
    }

    size_t M = atoi(argv[1]);
    size_t K = atoi(argv[2]);
    size_t N = atoi(argv[3]);


    // assert( M%8==0 );
    // assert( N%8==0 );
    // assert( K%8==0 );
    // compute each matrix size
    size_t bytes_A = sizeof(float)*M*K;
    size_t bytes_B = sizeof(float)*K*N;
    size_t bytes_C = sizeof(float)*M*N;
    // allocate host memory
    float * host_A = (float*)malloc(bytes_A);
    float * host_B = (float*)malloc(bytes_B);
    float * host_C = (float*)malloc(bytes_C);
    float * host_Device = (float*)malloc(bytes_C);
    // allocate device memory
    float * device_A;
    float * device_B;
    float * device_C;
    checkCudaErrors(cudaMalloc(&device_A, bytes_A));
    checkCudaErrors(cudaMalloc(&device_B, bytes_B));
    checkCudaErrors(cudaMalloc(&device_C, bytes_C));
    //save performance
    double msecPerMatrixMul[2] = {0, 0};
    double gigaFlops[2] = {0, 0};
    double flopsPerMatrixMul = 2.0 * M * N * K;

    const int BLOCK_SIZE_M = 128;
    const int BLOCK_SIZE_K = 8;
    const int BLOCK_SIZE_N = 128;
    const int THREAD_SIZE_M = 8;
    const int THREAD_SIZE_N = 8;
    const bool Enable_Double_Buffer = false;

    //init A
    for(int i=0; i<M*K; i++)
        host_A[i] = 1;

    //init B
    for(int i=0;i<K*N;i++)
        host_B[i] = 2;

    //copy data from host to device
    checkCudaErrors(cudaMemcpy(device_A, host_A, bytes_A, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(device_B, host_B, bytes_B, cudaMemcpyHostToDevice));

    cudaEvent_t start, end;
    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&end));

    float msecTotal=0.0;
    int iter = 100;
    checkCudaErrors(cudaMemcpy(device_C, host_C, bytes_C, cudaMemcpyHostToDevice));

    checkCudaErrors(cudaEventRecord(start));
    for(int loop=0; loop<iter;loop++){
        dim3 grid((N + 32 -1) / 32, (M + 32 -1)/ 32);
        dim3 block(32, 32);//max thread block is 1024
        //这里要设置对，不然后面处理逻辑就错了。
        gemm_v1_smem<<<grid, block>>>(device_A, device_B, device_C, M, N,K);
        //在kernel中打印log，必须加显示同步才会输出到屏幕上
        // cudaDeviceSynchronize();
    }
    checkCudaErrors(cudaEventRecord(end));
    checkCudaErrors(cudaEventSynchronize(end));
    checkCudaErrors(cudaEventElapsedTime(&msecTotal, start, end));
    checkCudaErrors(cudaMemcpy(host_Device, device_C, bytes_C, cudaMemcpyDeviceToHost));
    msecPerMatrixMul[0] = msecTotal / iter;
    gigaFlops[0] = (flopsPerMatrixMul * 1.0e-9f) / (msecPerMatrixMul[0]/1000);
    printf("gemm_v1_smem performance is %.2f Gflops, time=%.3f msec, Size=%.0f ops,\n",
           gigaFlops[0],
           msecPerMatrixMul[0],
           flopsPerMatrixMul);
    getCpuResult(host_C, host_A, host_B, M, N, K);
    auto res = checkAccResult(host_C, host_Device,M,N);
    std::cout<<(res?"Result= PASS" : "Result= FAIL")<<std::endl;

    // Free Memory
    cudaFree(device_A);
    cudaFree(device_B);
    cudaFree(device_C);
    
    free(host_A);
    free(host_B);
    free(host_C);
    free(host_Device);
}
