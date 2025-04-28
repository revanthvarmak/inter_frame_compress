#include "Residual.h"

void CalculateResidual(const Frame& current_frame, const Frame &reference_frame, MotionVector mv, int mbx, int mby, std::vector<int> &residualBlock){
    int current_block_x = mbx * BLOCK_SIZE;
    int current_block_y = mby * BLOCK_SIZE;
    int C = current_frame.channels;
    residualBlock.resize(BLOCK_SIZE * BLOCK_SIZE * C);
    
    for(int y = 0; y < BLOCK_SIZE; y++){
        for(int x = 0; x < BLOCK_SIZE; x++){
            for(int c = 0; c < C; c++){
                uint8_t curr_pixel = current_frame.getPixel(current_block_x + x, current_block_y + y, c);
                uint8_t ref_pixel = reference_frame.getPixel(current_block_x + x + mv.dx, current_block_y + y + mv.dy, c);
                residualBlock[(y * BLOCK_SIZE + x) * C + c] = static_cast<int>(curr_pixel) - static_cast<int>(ref_pixel); 
            }
        }
    }
}


void encodeP_Frame(const Frame& current_frame, const Frame& reference_frame, std::vector<MotionVector> &motionVectors, std::vector<std::vector<int>> &residuals){
    int mbWidth = (current_frame.width + BLOCK_SIZE - 1) / BLOCK_SIZE;
    int mbHeight = (current_frame.height + BLOCK_SIZE - 1) / BLOCK_SIZE;

    motionVectors.clear();
    residuals.clear();

    motionVectors.reserve(mbWidth * mbHeight);
    residuals.reserve(mbWidth * mbHeight);

    for(int mby = 0; mby < mbHeight; mby++){
        for(int mbx = 0; mbx < mbWidth; mbx++){
            MotionVector mv = estimateMotion(current_frame, reference_frame, mbx, mby);
            motionVectors.push_back(mv);
            std::vector<int> residualBlock;
            CalculateResidual(current_frame, reference_frame, mv, mbx, mby, residualBlock);
            residuals.push_back(residualBlock);
        }
    }
}

void decodeP_Frame(std::vector<MotionVector> &motionVectors, std::vector<std::vector<int>> &residuals, const Frame& reference_frame, Frame& output_frame){
    int mbWidth = (output_frame.width + BLOCK_SIZE - 1) / BLOCK_SIZE;
    int mbHeight = (output_frame.height + BLOCK_SIZE - 1) / BLOCK_SIZE;

    int block_idx = 0;

    for(int mby = 0; mby < mbHeight; mby++){
        for(int mbx = 0; mbx < mbWidth; mbx++){
            MotionVector mv = motionVectors[block_idx];
            std::vector<int> residualBlock = residuals[block_idx];
            block_idx++;

            int ref_block_x = mbx * BLOCK_SIZE;
            int ref_block_y = mby * BLOCK_SIZE;
            int C = output_frame.channels;
            for(int y = 0; y < BLOCK_SIZE; y++){
                for(int x = 0; x < BLOCK_SIZE; x++){
                    int global_x = ref_block_x + x;
                    int global_y = ref_block_y + y;
                    if(global_x >= output_frame.width || global_y >= output_frame.height){
                        continue;
                    }
                    for(int c = 0; c < C; c++){
                        int pred_x = global_x + mv.dx;
                        int pred_y = global_y + mv.dy;

                        uint8_t pred_pixel = reference_frame.getPixel(pred_x, pred_y, c);

                        int residual_pixel = residualBlock[(y * BLOCK_SIZE + x) * C + c];
                        int final_pixel = static_cast<int>(pred_pixel) + residual_pixel;

                        final_pixel = std::max(0, std::min(255, final_pixel));

                        output_frame.setPixel(global_x, global_y, c, final_pixel);
                    }
                }
            }
        }
    }
}