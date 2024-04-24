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

//use shared mem
// SGEMM: Block Tile + K Tile, with smem
// Block Tile (BM, BN) + K Tile (BK=32)
// grid((N + BN - 1) / BN, (M + BM - 1) / BM), block(BN, BM)
// a: MxK, b: KxN, c: MxN, compute: c = a * b, all row major

//一个thread使用了36个寄存器，
//存在的问题，计算密度太低，bank 冲突。
//
__global__ void gemm_v1_smem(float * __restrict__ A,
                    float * __restrict__ B,
                    float * __restrict__ C,
                    const int M,
                    const int N,
                    const int K){
    constexpr int BM = 32;
    constexpr int BN = 32;
    constexpr int BK = 32;
    __shared__ float smem_a[BM][BK];
    __shared__ float smem_b[BK][BN];
    //load gmem to smem
    //sync
    //compute
    //sync
    //store
    // [1] Block Tile: 32x32的block处理c上一块32x32的元素计算
    // [2]     K Tile: 使用共享内存，并将K分块为BK大小的块 
    int idx = threadIdx.y * blockDim.x + threadIdx.x; // thread id in block
    int load_smem_a_m = idx / BK;// 0~31, tid / 32, tid / BM, threadIdx.y
    int load_smem_a_k = idx % BK;// 0~31, tid % 32, tid % BK, threadIdx.x
    int load_smem_b_k = idx / BN;// 0~31, tid / 32, tid / BK, threadIdx.y
    int load_smem_b_n = idx % BN;// 0~31, tid % 32, tid % BN, threadIdx.x
    int load_gmem_a_m = blockIdx.y * blockDim.y + load_smem_a_m;//global addr of c(m,n)
    int load_gmem_b_n = blockIdx.x * blockDim.x + load_smem_b_n;
    float sum=0.0f;
    for(int bk=0;bk<(K+BK-1)/BK;bk++) {
        //load smem_a
        int load_gmem_a_k = bk* BK + load_smem_a_k;
        int load_gmem_a_addr = load_gmem_a_m * K + load_gmem_a_k;// g_a * K + g_k
        smem_a[load_smem_a_m][load_smem_a_k]=A[load_gmem_a_addr];
        //load smem_b
        int load_gmem_bk = bk * BK + load_smem_b_k;
        int load_gmem_b_addr = load_gmem_bk * N + load_gmem_b_n;//
        smem_b[load_smem_b_k][load_smem_b_n]=B[load_gmem_b_addr];
        __syncthreads();
        #pragma unroll
        for(int k=0;k<BK;k++) {
            sum+=smem_a[load_smem_a_m][k]*smem_b[k][load_smem_b_n];//a没有bank冲突，b有bank冲突
        }
        __syncthreads();
    }
        int gmem_c_addr = load_gmem_a_m * N + load_gmem_b_n; 
        C[gmem_c_addr]=sum;
}
//smem + register
// SGEMM: Block Tile + Thread Tile + K Tile + Vec4, with smem
// BK:TILE_K=8 BM=BN=128
// TM=TN=8 增加计算密度 BM/TM=16 BN/TN=16和b
// dim3 blockDim(BN/TN, BM/TM);
// dim3 gridDim((N + BN - 1) / BN, (M + BM - 1) / BM)
__global__ void gemm_v2_register(float * __restrict__ A,
                    float * __restrict__ B,
                    float * __restrict__ C,
                    const int M,
                    const int N,
                    const int K) {
    // [1]  Block Tile: 一个16x16的block处理C上大小为128X128的一个目标块
    // [2] Thread Tile: 每个thread负责计算TM*TN(8*8)个元素，增加计算密度
    // [3]      K Tile: 将K分块，每块BK大小，迭代(K+BK-1/BK)次
    //                  每次计算TM*TN个元素各自的部分乘累加
    // [4]   Vectorize: 减少load和store指令，使用float4
    constexpr int BM=128;
    constexpr int BN=128;
    constexpr int BK=8;
    constexpr int TM=8;
    constexpr int TN=8;
    int idx = threadIdx.y*blockDim.x + threadIdx.x;
    __shared__ float smem_a[BM][BK],smem_b[BK][BN];// 2*128*8*4=8KB
    // 0. 先计算shared memory中的索引
    // tid和需要加载的smem s_a[BM][BK] 之间的索引关系 BM=128 BK=8 按行读取 A行主序
    // 对于s_a每行8个数据，每个线程读取4个，需要2个线程；总共128行，需要128x2刚好256线程
    int load_smem_a_m =  idx / 2;
    int load_smem_a_k = (idx % 2 == 0) ? 0 : 4;
    // tid和需要加载的smem s_b[BK][BN] 之间的索引关系 BK=8 BN=128 按行读取 B行主序
    // 对于s_b每行128个数据，每个线程读4个数据，需要32个线程；总共8行，需要32x8=256个线程
    int load_smem_b_k = idx /32;// tid/32, row of s_b 256/32=8 行 0~7
    int load_smem_b_n= (idx % 32) * 4;//(tid % 32) * 4, col of s_b 0,4,...,124
    // 1. 再计算全局内存中的索引
    // 要加载到s_a中的元素对应到A全局内存中的行数 每个block负责出C中大小为BM*BN的块
    int load_gmem_a_m = blockIdx.y * BM + load_smem_a_m; // global row of a and c
    int load_gmem_b_n = blockIdx.x * BN + load_smem_b_n; // global col of b and c
    float r_c[TM][TN] = {0.0}; // 8x8
    // 2. 先对K进行分块，每块BK大小
    for (int bk = 0; bk < (K + BK - 1) / BK; ++bk) {
        // get gmem ak address
        int load_gmem_a_k = bk * BK + load_smem_a_k;// global row of a
        int load_gmem_a_addr = load_gmem_a_m * K + load_gmem_a_k;
        FLOAT4(smem_a[load_smem_a_m][load_smem_a_k]) = FLOAT4(A[load_gmem_a_addr]);
        // get gmem b address
        int load_gmem_b_k = bk *BK + load_smem_b_k;// global row of b
        int load_gmem_b_addr = load_gmem_b_k * N + load_gmem_b_n;
        FLOAT4(smem_b[load_smem_b_k][load_smem_b_n])=FLOAT4(B[load_gmem_b_addr]);
        __syncthreads();//sync load
        #pragma unroll
        for(int k=0;k<BK;k++){
            // 3. 每个线程负责计算BM*BN(128x128)中的TM*TN(8x8)个元素
            #pragma unroll
            for(int i=0;i<TM;i++) {
                #pragma unroll
                for(int j=0;j<TN;j++){
                    r_c[i][j]+=smem_a[threadIdx.y * TM +i][k]*smem_b[k][threadIdx.x * TN + j];
                }
            }
        }
        //printf("run here");
        __syncthreads();//store
    }
    #pragma unroll
    for (int m = 0; m < TM; ++m) {
        int store_gmem_c_m = blockIdx.y * BM + threadIdx.y * TM + m;
        #pragma unroll
        for (int n = 0; n < TN; n += 4) {
            int store_gmem_c_n = blockIdx.x * BN + threadIdx.x * TN + n;
            int store_gmem_c_addr = store_gmem_c_m * N + store_gmem_c_n;
            FLOAT4(C[store_gmem_c_addr]) = FLOAT4(r_c[m][n]);
        }
    }
}

__global__ void sgemm_thread_tile_vec4(
  float* a, float* b, float* c, int M, int N, int K) {
  // [1]  Block Tile: 一个16x16的block处理C上大小为128X128的一个目标块
  // [2] Thread Tile: 每个thread负责计算TM*TN(8*8)个元素，增加计算密度
  // [3]      K Tile: 将K分块，每块BK大小，迭代(K+BK-1/BK)次，
  //                  每次计算TM*TN个元素各自的部分乘累加
  // [4]   Vectorize: 减少load和store指令，使用float4
  constexpr int BM = 128;
  constexpr int BN = 128;
  constexpr int BK = 8; 
  constexpr int TM = 8;
  constexpr int TN = 8;

  int bx = blockIdx.x;
  int by = blockIdx.y;
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int tid = threadIdx.y * blockDim.x + tx; // tid within the block
  __shared__ float s_a[BM][BK], s_b[BK][BN]; // 2*128*8*4=8KB
  
  // 0. 先计算shared memory中的索引
  // tid和需要加载的smem s_a[BM][BK] 之间的索引关系 BM=128 BK=8 按行读取 A行主序
  // 对于s_a每行8个数据，每个线程读取4个，需要2个线程；总共128行，需要128x2刚好256线程
  int load_smem_a_m = tid / 2; // tid/2 (128/8)*(128/8)=256 threads per block, tid/2->[0,128), BM=128 0~127
  int load_smem_a_k = (tid % 2 == 0) ? 0 : 4;  // (tid%2 == 0) ? 0 : 4, col of s_a 0,4
  // tid和需要加载的smem s_b[BK][BN] 之间的索引关系 BK=8 BN=128 按行读取 B行主序
  // 对于s_b每行128个数据，每个线程读4个数据，需要32个线程；总共8行，需要32x8=256个线程
  int load_smem_b_k = tid / 32; // tid/32, row of s_b 256/32=8 行 0~7
  int load_smem_b_n = (tid % 32) * 4;  // (tid % 32) * 4, col of s_b 0,4,...,124
  // 1. 再计算全局内存中的索引
  // 要加载到s_a中的元素对应到A全局内存中的行数 每个block负责出C中大小为BM*BN的块
  int load_gmem_a_m = by * BM + load_smem_a_m; // global row of a and c
  int load_gmem_b_n = bx * BN + load_smem_b_n; // global col of b and c
  
  float r_c[TM][TN] = {0.0}; // 8x8
  // 2. 先对K进行分块，每块BK大小
  for (int bk = 0; bk < (K + BK - 1) / BK; ++bk) {
    // 加载数据到共享内存smem s_a BM*BK 128*8 vectorize float4
    int load_gmem_a_k = bk * BK + load_smem_a_k; // global col of a
    int load_gmem_a_addr = load_gmem_a_m * K + load_gmem_a_k;
    FLOAT4(s_a[load_smem_a_m][load_smem_a_k]) = FLOAT4(a[load_gmem_a_addr]);
    // 加载数据到共享内存smem s_b BK*BN 8*128 vectorize float4
    int load_gmem_b_k = bk * BK + load_smem_b_k; // global row of b
    int load_gmem_b_addr = load_gmem_b_k * N + load_gmem_b_n; 
    FLOAT4(s_b[load_smem_b_k][load_smem_b_n]) = FLOAT4(b[load_gmem_b_addr]); 
    __syncthreads();
    #pragma unroll
    for (int k = 0; k < BK; k++) {
      // 3. 每个线程负责计算BM*BN(12x128)中的TM*TN(8x8)个元素
      #pragma unroll
      for (int m = 0; m < TM; m++) {
        #pragma unroll
        for (int n = 0; n < TN; n++) {
          // k from 0~7，0 ~ BK, ty and tx range from 0 to 15, 16x8=128
          int comp_smem_a_m = ty * TM + m;  // 128*8 128/TM(8)=16 M方向 16线程
          int comp_smem_b_n = tx * TN + n;  // 8*128 128/TN(8)=16 N方向 16线程
          r_c[m][n] += s_a[comp_smem_a_m][k] * s_b[k][comp_smem_b_n];
        }
      }
    }
    __syncthreads();
  }

  #pragma unroll
  for (int m = 0; m < TM; ++m) {
    int store_gmem_c_m = by * BM + ty * TM + m;
    #pragma unroll
    for (int n = 0; n < TN; n += 4) {
      int store_gmem_c_n = bx * BN + tx * TN + n;
      int store_gmem_c_addr = store_gmem_c_m * N + store_gmem_c_n;
      FLOAT4(c[store_gmem_c_addr]) = FLOAT4(r_c[m][n]);
    }
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
    int iter = 20;
    checkCudaErrors(cudaMemcpy(device_C, host_C, bytes_C, cudaMemcpyHostToDevice));

    checkCudaErrors(cudaEventRecord(start));
    for(int loop=0; loop<iter;loop++){
        dim3 grid((N + BLOCK_SIZE_N -1) / BLOCK_SIZE_N, (M + BLOCK_SIZE_M -1)/ BLOCK_SIZE_M);
        dim3 block(BLOCK_SIZE_N/THREAD_SIZE_N, BLOCK_SIZE_M/THREAD_SIZE_M);//max thread block is 1024
        //这里要设置对，不然后面处理逻辑就错了。
        gemm_v2_register<<<grid, block>>>(device_A, device_B, device_C, M, N,K);
        //在kernel中打印log，必须加显示同步才会输出到屏幕上
        // cudaDeviceSynchronize();
    }
    checkCudaErrors(cudaEventRecord(end));
    checkCudaErrors(cudaEventSynchronize(end));
    checkCudaErrors(cudaEventElapsedTime(&msecTotal, start, end));
    checkCudaErrors(cudaMemcpy(host_Device, device_C, bytes_C, cudaMemcpyDeviceToHost));
    msecPerMatrixMul[0] = msecTotal / iter;
    gigaFlops[0] = (flopsPerMatrixMul * 1.0e-9f) / (msecPerMatrixMul[0]/1000);
    printf("gemm_v2_register performance is %.2f Gflops, time=%.3f msec, Size=%.0f ops,\n",
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
