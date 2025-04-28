#include "frame.h"
#include "MotionVector.h"
#include "Residual.h"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <utility> 
#include <chrono>

Frame convertMattoFrame(const cv::Mat &mat, FrameType type) {
    int ch = mat.channels();
    Frame frame(mat.cols, mat.rows, type, ch);
    if (ch == 3) {
        for (int y = 0; y < frame.height; y++) {
            for (int x = 0; x < frame.width; x++) {
                // Access pixel coordinate at a location y,x of OpenCV Matrix
                cv::Vec3b pix = mat.at<cv::Vec3b>(y, x);
                frame.setPixel(x, y, pix[0], 0);
                frame.setPixel(x, y, pix[1], 1);
                frame.setPixel(x, y, pix[2], 2);
            }
        }
    } else {
        cv::Mat gray = mat.channels() == 1 ? mat : mat.clone();
        std::memcpy(frame.data, gray.data, gray.total() * gray.elemSize());
    }
    return frame;
}

cv::Mat convertFrametoMat(const Frame &frame) {
    cv::Mat mat;
    if (frame.channels == 3) {
        mat = cv::Mat(frame.height, frame.width, CV_8UC3);
        for (int y = 0; y < frame.height; y++) {
            for (int x = 0; x < frame.width; x++) {
                cv::Vec3b &pixel = mat.at<cv::Vec3b>(y, x);
                pixel[0] = frame.getPixel(x, y, 0);
                pixel[1] = frame.getPixel(x, y, 1);
                pixel[2] = frame.getPixel(x, y, 2);
            }
        }
    } else {
        mat = cv::Mat(frame.height, frame.width, CV_8UC1);
        std::memcpy(mat.data, frame.data, frame.width * frame.height);
    }
    return mat;
}

int main(int argc,char**argv)
{
    if(argc < 3){
        std::cerr << "Provide input video and output directory" << std::endl;
        return -1;
    }
    std::string video_path = argv[1];
    std::string output_path = argv[2];
    cv::VideoCapture cap(video_path);

    if(!cap.isOpened()){ 
        std::cerr<<"cannot open "<< video_path <<"\n"; 
        return -1; 
    }

    // Read video frame features
    int frameWidth = cap.get(cv::CAP_PROP_FRAME_WIDTH);
    int frameHeight = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
    double fps = cap.get(cv::CAP_PROP_FPS);

    cv::VideoWriter reconstructedWriter(output_path + "/reconstructed_video.mp4", cv::VideoWriter::fourcc('m', 'p', '4', 'v'), fps, cv::Size(frameWidth, frameHeight), true);

    double total_compression_time = 0.0;
    double total_decompression_time = 0.0;
    int frameCount = 0;
    cv::Mat first; 
    cap >> first;
    Frame ref = convertMattoFrame(first,I_frame);
    reconstructedWriter.write(convertFrametoMat(ref));

    while(true) {
        std::cout << "Processing frame " << frameCount + 1 << std::endl;
        cv::Mat mat; 
        cap >> mat; 
        if(mat.empty()) break;

        Frame cur = convertMattoFrame(mat,P_frame);

        auto compression_start = std::chrono::high_resolution_clock::now();
        std::vector<MotionVector> mv;
        Frame residual(cur.width, cur.height, P_frame, cur.channels);
        encodeP_Frame(cur, ref, mv, residual);
        auto compression_end = std::chrono::high_resolution_clock::now();

        Frame recon = decodeP_Frame(ref,mv,residual);
        auto decompression_end = std::chrono::high_resolution_clock::now();

        reconstructedWriter.write(convertFrametoMat(recon));
        ref = std::move(recon);

        total_compression_time += std::chrono::duration<double,std::milli>(compression_end-compression_start).count();
        total_decompression_time += std::chrono::duration<double,std::milli>(decompression_end-compression_end).count();
        ++frameCount;
        if (cv::waitKey(30) >= 0) break;
        std::cout<<"frame "<<frameCount<<" done\n";
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
