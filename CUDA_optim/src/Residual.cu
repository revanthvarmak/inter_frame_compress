#include "Residual.h"
#include "constants.h"
#include <cuda_runtime.h>
#include <vector>

// Kernels
__global__ void kResidual(const uint8_t*cur,const uint8_t*ref, const MotionVector* MV, int W,int H, int C, uint8_t*res) {
    // Calculate pixel coordinates in the image
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H) return;

    // Determine the macroblock that this pixel is in and retrieve the motion vector
    int mbX = x / BLOCK_SIZE;
    int mbY = y / BLOCK_SIZE;

    // row major order calculatiojn
    int mbsX= (W + BLOCK_SIZE - 1) / BLOCK_SIZE;
    MotionVector mv = MV[mbY * mbsX + mbX];

    // Determine the reference pixel coordinates using the motion vectors
    int refX = min(max(x + mv.dx,0), W-1);
    int refY = min(max(y + mv.dy,0), H-1);

    for (int c = 0; c < C; c++) {
        int pixelIdx = (y * W + x) * C + c;
        int refIdx = (refY * W + refX) * C + c;
        
        int diff = int(cur[pixelIdx]) - int(ref[refIdx]);
        diff = max(-127, min(127, diff)); 
        res[pixelIdx] = uint8_t(diff + RESIDUAL_OFFSET);
    }
}

__global__ void kReconstruct(const uint8_t*ref,const uint8_t*res, const MotionVector* MV, int W,int H,int C, uint8_t*out) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H) return;

    int mbX = x / BLOCK_SIZE;
    int mbY = y / BLOCK_SIZE;
    int mbsX= (W + BLOCK_SIZE - 1) / BLOCK_SIZE;
    MotionVector mv = MV[mbY * mbsX + mbX];

    int refX = min(max(x + mv.dx,0), W-1);
    int refY = min(max(y + mv.dy,0), H-1);

    for (int c = 0; c < C; c++) {
        int pixelIdx = (y * W + x) * C + c;
        int refIdx = (refY * W + refX) * C + c;
        
        int pix = int(ref[refIdx]) + int(res[pixelIdx]) - RESIDUAL_OFFSET;
        pix = pix < 0 ? 0 : (pix > 255 ? 255 : pix);
        out[pixelIdx] = uint8_t(pix);
    }
}

// Host functions

// Calculate number of threads and blocks requried to cover the entire 2D frame
void launch2D(dim3 &grid, dim3 &block, int W, int H){
    // Since each warp of 32 threads execute together, and better performance is achieved if all threads in a warp access consecutive memory,
    // x dimension of block should be multiple of 32

    block = dim3(32, 16);  
    grid  = dim3((W+block.x-1)/block.x, (H+block.y-1)/block.y);
}

Frame calculateResidual(const Frame& cur,const Frame& ref, const std::vector<MotionVector>& MV) {
    int W = cur.width;
    int H = cur.height;
    int C = cur.channels;

    // Create CUDA stream
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    Frame residual(W, H, P_frame, C);

    // device buffers
    uint8_t *dCur, *dRef, *dRes; 
    MotionVector* dMV;

    size_t frameBytes = size_t(W) * H * C;

    cudaMalloc(&dCur,frameBytes);
    cudaMalloc(&dRef,frameBytes);
    cudaMalloc(&dRes,frameBytes);
    cudaMalloc(&dMV, MV.size() * sizeof(MotionVector));

    cudaMemcpyAsync(dCur, cur.data, frameBytes, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(dRef, ref.data, frameBytes, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(dMV, MV.data(), MV.size() * sizeof(MotionVector), cudaMemcpyHostToDevice, stream);

    dim3 grid, block; 
    launch2D(grid, block, W, H);
    kResidual<<<grid, block, 0, stream>>>(dCur, dRef, dMV, W, H, C, dRes);
    cudaMemcpyAsync(residual.data,dRes,frameBytes,cudaMemcpyDeviceToHost,stream);


    // Cleanup CUDA memory
    cudaStreamSynchronize(stream);
    cudaFree(dCur); 
    cudaFree(dRef); 
    cudaFree(dRes); 
    cudaFree(dMV);
    cudaStreamDestroy(stream);
    return residual;
}

Frame decodeP_Frame(const Frame& ref, const std::vector<MotionVector>& MV, const Frame& residual) {
    int W = ref.width;
    int H = ref.height;
    int C = ref.channels;
    Frame out(W, H, P_frame, C);

    // Create CUDA stream
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    uint8_t *dRef,*dRes,*dOut; 
    MotionVector* dMV;
    size_t frameBytes = size_t(W) * H * C;
    cudaMalloc(&dRef,frameBytes);
    cudaMalloc(&dRes,frameBytes);
    cudaMalloc(&dOut,frameBytes);
    cudaMalloc(&dMV ,MV.size() * sizeof(MotionVector));

    cudaMemcpyAsync(dRef, ref.data, frameBytes, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(dRes, residual.data, frameBytes, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(dMV, MV.data(), MV.size() * sizeof(MotionVector), cudaMemcpyHostToDevice, stream);

    dim3 grid,block; 
    launch2D(grid, block, W, H);
    kReconstruct<<<grid,block,0,stream>>>(dRef, dRes, dMV, W, H, C, dOut);
    
    cudaMemcpyAsync(out.data,dOut,frameBytes,cudaMemcpyDeviceToHost,stream);


    cudaStreamSynchronize(stream);
    cudaFree(dRef); 
    cudaFree(dRes); 
    cudaFree(dOut); 
    cudaFree(dMV);
    cudaStreamDestroy(stream);
    return out;
}
