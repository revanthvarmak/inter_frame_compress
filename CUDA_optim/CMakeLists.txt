cmake_minimum_required(VERSION 2.8.12.2)
project(cuda_video_codec LANGUAGES CXX CUDA)

set(OpenCV_DIR "/afs/ece.cmu.edu/usr/rkanthet/Private/build/")
find_package(OpenCV REQUIRED)


set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_ARCHITECTURES 75)
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include        
    ${OpenCV_INCLUDE_DIRS}                     
)

set(SRCS
    src/main.cpp
    src/MotionEstimation.cu
    src/Residual.cu
)

add_executable(cuda_video_codec ${SRCS})
target_link_libraries(cuda_video_codec
    ${OpenCV_LIBS}
    cuda
    cudart
)
