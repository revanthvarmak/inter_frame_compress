#include "frame.h"
#include "MotionVector.h"
#include "Residual.h"
#include <iostream>
#include <vector>
#include <string>
#include <opencv2/opencv.hpp>
#include <chrono>


Frame convertMattoFrame(const cv::Mat &mat, FrameType type){
    int ch = mat.channels();
    Frame frame(mat.cols, mat.rows, type, ch);


    if(ch == 3){
        for(int y = 0; y < frame.height; y++){
            for(int x = 0; x < frame.width; x++){
                cv::Vec3b pix = mat.at<cv::Vec3b>(y, x);
                frame.setPixel(x, y, 0, pix[0]);
                frame.setPixel(x, y, 1, pix[1]);
                frame.setPixel(x, y, 2, pix[2]);
            }
        }
    }else{
        cv::Mat gray = mat.channels() == 1 ? mat : (cv::Mat)mat.clone();
        for(int y = 0; y < frame.height; y++){
            for(int x = 0; x < frame.width; x++){
                frame.setPixel(x, y, 0, gray.at<uchar>(y, x));
            }
        }
    }

    return frame;
}


cv::Mat convertFrametoMat(const Frame &frame){
    cv::Mat mat(frame.height, frame.width, frame.channels == 3 ? CV_8UC3 : CV_8UC1);
    for(int y = 0; y < frame.height; y++){
        for(int x = 0; x < frame.width; x++){
            if(frame.channels == 3){
                cv::Vec3b &out = mat.at<cv::Vec3b>(y, x);
                out[0]= frame.getPixel(x, y, 0);
                out[1]= frame.getPixel(x, y, 1);
                out[2]= frame.getPixel(x, y, 2);
            }else{
                mat.at<uchar>(y, x) = frame.getPixel(x, y, 0);
            }
        }
    }
    return mat;
}


int main(int argc, char** argv){
    if(argc < 3){
        std::cerr << "Provide input video and output directory" << std::endl;
        return -1;
    }

    std::string video_path = argv[1];
    std::string output_path = argv[2];

    cv::VideoCapture cap(video_path);
    if(!cap.isOpened()){
        std::cerr << "Cannot open file:" << video_path << std::endl;
        return -1;
    }

    int frameWidth = cap.get(cv::CAP_PROP_FRAME_WIDTH);
    int frameHeight = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
    int totalFrames = cap.get(cv::CAP_PROP_FRAME_COUNT);
    double fps = cap.get(cv::CAP_PROP_FPS);

    std::cout << "Video info: " << frameWidth << "x" << frameHeight << ", " << totalFrames << " frames, " << fps << " fps" << std::endl;

    cv::VideoWriter reconstructedWriter(output_path + "/reconstructed_video.mp4", cv::VideoWriter::fourcc('m', 'p', '4', 'v'), fps, cv::Size(frameWidth, frameHeight), true);
    
    double total_compression_time = 0.0;
    double total_decompression_time = 0.0;


    int frameCount = 0;
    cv::Mat first_mat;
    cap >> first_mat;

    if(first_mat.empty()){
        std::cerr << "Failed to read the first frame." << std::endl;
        return -1;
    }

    Frame reference_frame = convertMattoFrame(first_mat, I_frame);
    reconstructedWriter.write(convertFrametoMat(reference_frame));

    while(true){
        std::cout << "Processing frame " << frameCount + 1 << std::endl;
        cv::Mat current_mat;
        cap >> current_mat;

        if(current_mat.empty()){
            break;
        }

        Frame current_frame = convertMattoFrame(current_mat, P_frame);
        std::vector<MotionVector> motion_vectors;
        std::vector<std::vector<int>> residuals;

        auto compression_start = std::chrono::high_resolution_clock::now();
        encodeP_Frame(current_frame, reference_frame, motion_vectors, residuals);
        auto compression_end = std::chrono::high_resolution_clock::now();

        Frame reconstructed_frame(current_frame.width, current_frame.height, P_frame, reference_frame.channels);

        auto decompression_start = std::chrono::high_resolution_clock::now();
        decodeP_Frame(motion_vectors, residuals, reference_frame, reconstructed_frame);
        auto decompression_end = std::chrono::high_resolution_clock::now();

        std::chrono::duration<double, std::milli> compression_duration = compression_end - compression_start;
        std::chrono::duration<double, std::milli> decompression_duration = decompression_end - decompression_start;

        total_compression_time += compression_duration.count();
        total_decompression_time += decompression_duration.count();

        cv::Mat reconstructed_mat = convertFrametoMat(reconstructed_frame);
        reconstructedWriter.write(reconstructed_mat);

        reference_frame = std::move(reconstructed_frame);
        frameCount++;
        if (cv::waitKey(30) >= 0) break;
    }

    double avg_compression_time = total_compression_time / frameCount;
    double avg_decompression_time = total_decompression_time / frameCount;
    std::cout << "Total frames processed: " << frameCount << std::endl;
    std::cout << "Average compression time: " << avg_compression_time << " ms per frame" << std::endl;
    std::cout << "Average decompression time: " << avg_decompression_time << " ms per frame" << std::endl;
    std::cout << "Frames per second (compression): " 
                << (1000.0 / avg_compression_time) << std::endl;
    std::cout << "Frames per second (decompression): " 
                << (1000.0 / avg_decompression_time) << std::endl;

    cap.release();
    reconstructedWriter.release();
    std::cout << "Video processing done" << std::endl;
    return 0;
}