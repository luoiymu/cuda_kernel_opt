#include <iostream>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <assert.h>
using namespace std;

#define _DEBUG

inline
cudaError_t cudaCheck(cudaError_t result){
#ifdef _DEBUG
    if(result != cudaSuccess){
        assert(result == cudaSuccess);
    }
    return result;
#else
    printf("Not debugging\n");
    return result;
#endif
}

const int TILE_DIM = 32;
const int BLOCK_ROWS = 8; //TILE_DIM must be a multiple of BLOCK_ROWS

// Copy kernel, for warm up
__global__ void copy(float *idata, float *odata){
    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;
    int width = gridDim.x * TILE_DIM;

    for(int j = 0; j < TILE_DIM; j += BLOCK_ROWS){
        odata[(y + j) * width + x] = idata[(y + j) * width + x];
    }
}

// Base version
__global__ void transposeNative(float *idata, float *odata){
    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;
    int width = gridDim.x * TILE_DIM;

    for(int j = 0; j < TILE_DIM; j += BLOCK_ROWS){
        odata[x * width + (y + j)] = idata[(y + j) * width + x];
    }
    // To understand: 
    // odata[x * width + (y + 0)] = idata[(y + 0) * width + x];
    // odata[x * width + (y + 8)] = idata[(y + 8) * width + x];
    // odata[x * width + (y + 16)] = idata[(y + 16) * width + x];
    // odata[x * width + (y + 24)] = idata[(y + 24) * width + x];
}

// Shared memory version, but it has 32-way bank conflicts
__global__ void transposeSharedMem(float *idata, float *odata){
    __shared__ float sharedMem[TILE_DIM][TILE_DIM];

    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;
    int width = gridDim.x * TILE_DIM;

    //load data from global mem to shared mem
    for(int j = 0; j < TILE_DIM; j += BLOCK_ROWS){
        sharedMem[threadIdx.y + j][threadIdx.x] = idata[(y + j) * width + x];
    }
    __syncthreads();

    //write data
    x = blockIdx.y * TILE_DIM + threadIdx.x;
    y = blockIdx.x * TILE_DIM + threadIdx.y;
    for(int j = 0; j < TILE_DIM; j += BLOCK_ROWS){
        odata[(y + j) * width + x] = sharedMem[threadIdx.x][threadIdx.y + j];//32 way bank conflicts
    }
}

// Shared memory + no bank conflicts
__global__ void transposeNoBankConflicts(float *idata, float *odata){
    __shared__ float sharedMem[TILE_DIM][TILE_DIM + 1];//Add a padding

    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;
    int width = gridDim.x * TILE_DIM;

    for(int j = 0; j < TILE_DIM; j += BLOCK_ROWS){
        sharedMem[threadIdx.y + j][threadIdx.x] = idata[(y + j) * width + x];
    }
    __syncthreads();

    //write data
    x = blockIdx.y * TILE_DIM + threadIdx.x;
    y = blockIdx.x * TILE_DIM + threadIdx.y;
    for(int j = 0; j < TILE_DIM; j += BLOCK_ROWS){
        odata[(y + j) * width + x] = sharedMem[threadIdx.x][threadIdx.y + j];
    }
}

int main(){
    const int nx = 1024;
    const int ny = 1024;
    const int mem_size = nx * ny * sizeof(float);

    dim3 dimGrid(nx/TILE_DIM, ny/TILE_DIM, 1);
    dim3 dimBlock(TILE_DIM, BLOCK_ROWS, 1);

    float *h_idata = (float*)malloc(mem_size);
    float *h_cdata = (float*)malloc(mem_size);
    float *h_tdata = (float*)malloc(mem_size);
    float *ref = (float*)malloc(mem_size);

    float *d_idata, *d_cdata, *d_tdata;
    cudaCheck(cudaMalloc((void**)&d_idata, mem_size));
    cudaCheck(cudaMalloc((void**)&d_cdata, mem_size));
    cudaCheck(cudaMalloc((void**)&d_tdata, mem_size));

    for(int j = 0; j < ny; j++){
        for(int i = 0; i < nx; i++){
            h_idata[j * nx + i] = j * nx + i;
        }
    }

    for(int j = 0; j < ny; j++){
        for(int i = 0; i < nx; i++){
            ref[j * nx + i] = h_idata[i * nx + j];
        }
    }

    cudaCheck(cudaMemcpy(d_idata, h_idata, mem_size, cudaMemcpyHostToDevice));

    //warm up
    copy<<<dimGrid, dimBlock>>>(d_idata, d_cdata);

    //transposeNative<<<dimGrid, dimBlock>>>(d_idata, d_tdata);
    //transposeSharedMem<<<dimGrid, dimBlock>>>(d_idata, d_tdata);
    transposeNoBankConflicts<<<dimGrid, dimBlock>>>(d_idata, d_tdata);

    cudaCheck(cudaMemcpy(h_tdata, d_tdata, mem_size, cudaMemcpyDeviceToHost));
    for(int i = 0; i < nx * ny; i++){
        if(h_tdata[i] != ref[i]){
            printf("Error!");
            break;
        }
    }

    free(h_idata);
    free(h_cdata);
    free(h_tdata);
    free(ref);
    cudaCheck(cudaFree(d_idata));
    cudaCheck(cudaFree(d_cdata));
    cudaCheck(cudaFree(d_tdata));

    return 0;

}