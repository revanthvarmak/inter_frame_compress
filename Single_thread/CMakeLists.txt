cmake_minimum_required(VERSION 2.8.12.2)
project(cuda_video_codec LANGUAGES CXX)

set(OpenCV_DIR "/afs/ece.cmu.edu/usr/rkanthet/Private/build/")
find_package(OpenCV REQUIRED)

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include        
    ${OpenCV_INCLUDE_DIRS}                     
)

set(SRCS
    src/main.cpp
    src/constants.cpp
    src/MotionEstimation.cpp
    src/Residual.cpp
)

add_executable(cuda_video_codec ${SRCS})
target_link_libraries(cuda_video_codec
    ${OpenCV_LIBS}
)
