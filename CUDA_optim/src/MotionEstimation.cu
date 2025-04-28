#include "frame.h"
#include "MotionVector.h"
#include "constants.h"
#include <cuda_runtime.h>
#include <vector>
#include <cstdio>


__global__ void motionKernel(const uint8_t *cur, const uint8_t *ref, int width, int height, int channels, MotionVector *outMV) {

    // Get the current block coordinates in the grid
    int mbX = blockIdx.x;
    int mbY = blockIdx.y;

    // Start position of first pixxel each macro block
    int startX = mbX * BLOCK_SIZE;
    int startY = mbY * BLOCK_SIZE;

    // Allocate shared memory to store the current macro block's pixel data
    __shared__ uint8_t sharedCurr[BLOCK_SIZE*BLOCK_SIZE * 3];

    // Allocate shared memory for reference frame's search area, search size includes block size plus the search range on each side
    __shared__ uint8_t sharedRef[(BLOCK_SIZE + 2*SEARCH_RANGE) * (BLOCK_SIZE + 2*SEARCH_RANGE) * 3];

    int tid = threadIdx.x;   
    if (tid < BLOCK_SIZE * BLOCK_SIZE) {

        // Convert the 1D thread index to 2D coordinates within the macro block
        int localX = tid % BLOCK_SIZE;
        int localY = tid / BLOCK_SIZE;

        // global pixel x and y coordinates
        int globalX = startX + localX;
        int globalY = startY + localY;

        // load current frame macroblock to shared memory
        if (globalX < width && globalY < height) {
            for (int c = 0; c < channels; c++) {
                sharedCurr[(localY * BLOCK_SIZE + localX) * channels + c] = cur[(globalY * width + globalX) * channels + c];
            }
        } else {
            for (int c = 0; c < channels; c++) {
                sharedCurr[(localY * BLOCK_SIZE + localX) * channels + c] = 0;
            }
        }
    }

    // Search window limits
    int minDx = -min(SEARCH_RANGE, startX);
    int maxDx =  min(SEARCH_RANGE, max(0, width  - (startX + BLOCK_SIZE)));
    int minDy = -min(SEARCH_RANGE, startY);
    int maxDy =  min(SEARCH_RANGE, max(0, height - (startY + BLOCK_SIZE)));
    
    // Calculate dimensions of the reference search area
    int refAreaWidth = BLOCK_SIZE + maxDx - minDx;
    int refAreaHeight = BLOCK_SIZE + maxDy - minDy;

    // Number of pixels in the reference search region
    int refPixelCount = refAreaWidth * refAreaHeight;

    
    // Load reference frame search region to shared memory
    for (int i = tid; i < refPixelCount; i += blockDim.x) {
        int localRefX = i % refAreaWidth;
        int localRefY = i / refAreaWidth;
        
        int globalRefX = startX + minDx + localRefX;
        int globalRefY = startY + minDy + localRefY;
        
        // Clamp coordinates to image boundaries
        globalRefX = min(max(globalRefX, 0), width - 1);
        globalRefY = min(max(globalRefY, 0), height - 1);
        
        for (int c = 0; c < channels; c++) {
            sharedRef[(localRefY * refAreaWidth + localRefX) * channels + c] = ref[(globalRefY * width + globalRefX) * channels + c];
        }
    }
    
    __syncthreads();

    // Number of motion vectors per row
    int candPerRow = maxDx - minDx + 1;

    // Number of motion vectors in total = Motion vectors per row x motion vectors per column
    int candidateCount = (candPerRow) * (maxDy - minDy + 1);

    // Metric to find best macro block in the reference frame
    unsigned bestSAD = 0xFFFFFFFF;

    // Store the displacement values from the current macro block
    int bestDx = 0, bestDy = 0;

    // Distributes the search candidates among all the threass in the block in the stride of blockDim
    for (int c = tid; c < candidateCount; c += blockDim.x) {
        int dx = (c % candPerRow) + minDx;
        int dy = (c / candPerRow) + minDy;

        unsigned sad = 0;
        // Unroll loops for instruction level parallelism
    #pragma unroll
        for (int by = 0; by < BLOCK_SIZE; ++by)
    #pragma unroll
            for (int bx = 0; bx < BLOCK_SIZE; ++bx) {
                int refLocalX = bx + dx - minDx;
                int refLocalY = by + dy - minDy;
                
                for (int c = 0; c < channels; c++) {
                    uint8_t refPix = sharedRef[(refLocalY * refAreaWidth + refLocalX) * channels + c];
                    uint8_t curPix = sharedCurr[(by * BLOCK_SIZE + bx) * channels + c];
                    sad += abs(int(curPix) - int(refPix));
                }
            }

        if (sad < bestSAD) { 
            bestSAD = sad; 
            bestDx = dx; 
            bestDy = dy; 
        }
    }
    
    // variables to store best SAD and best motion vector across all threads
    __shared__ unsigned gBestSAD;
    __shared__ int gBestDx, gBestDy;

    if (tid == 0) gBestSAD = 0xFFFFFFFF;
    __syncthreads();

    // sets minimum value to gBestSAD and returns the old value of gBestSAD to prev
    unsigned prev = atomicMin(&gBestSAD, bestSAD);
    if (bestSAD < prev) { 
        gBestDx = bestDx; 
        gBestDy = bestDy; 
    }
    __syncthreads();

    //  Only thread 0 writes the resultant motion vector to the global memory
    if (tid == 0) {
        int idx = mbY * gridDim.x + mbX;
        outMV[idx].dx = gBestDx;
        outMV[idx].dy = gBestDy;
    }
}

// host side function
std::vector<MotionVector> estimateMotion(const Frame &current, const Frame &reference) {
    int W = current.width;
    int H = current.height;
    int C = current.channels;
    // Calculate number of macroblocks in x and y direction of the image
    // Macro blocks are the small 16 x 16 shaped blocks we are dividing our current image into
    int mbsX = (W + BLOCK_SIZE - 1) / BLOCK_SIZE;
    int mbsY = (H + BLOCK_SIZE - 1) / BLOCK_SIZE;
    size_t frameBytes = size_t(W) * H * C;

    // Create CUDA streams
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // device buffers
    uint8_t *dCur, *dRef;          
    MotionVector *dMV;

    // Allocate memory for current and reference frame data
    cudaMalloc(&dCur, frameBytes);          
    cudaMalloc(&dRef, frameBytes);
    // Allocate memory for motion vectors
    cudaMalloc(&dMV, mbsX * mbsY * sizeof(MotionVector));

    // Use aysnc memory copies with stream
    cudaMemcpyAsync(dCur, current.data, frameBytes, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(dRef, reference.data, frameBytes, cudaMemcpyHostToDevice, stream);

    // Define the thread dimension
    dim3 grid(mbsX, mbsY);
    // One thread block per image macroblock. So each thread performs exhaustive search to find the candidate motion vector
    dim3 block(256);   
    motionKernel<<<grid, block, 0, stream>>>(dCur, dRef, W, H, C, dMV);

    std::vector<MotionVector> hostMV(mbsX * mbsY);
    cudaMemcpyAsync(hostMV.data(), dMV, mbsX * mbsY * sizeof(MotionVector), cudaMemcpyDeviceToHost, stream);

    // Wait until all operations in the stream are completed
    cudaStreamSynchronize(stream);

    cudaFree(dCur); 
    cudaFree(dRef); 
    cudaFree(dMV);
    cudaStreamDestroy(stream);
    return hostMV;
}
