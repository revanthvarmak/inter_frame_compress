#include "frame.h"
#include "MotionVector.h"
#include <limits>
#include <cmath>


MotionVector estimateMotion(const Frame &current_frame, const Frame &reference_frame, int mbx, int mby){
    int current_block_x = mbx * BLOCK_SIZE;
    int current_block_y = mby * BLOCK_SIZE;

    MotionVector bestMV{0, 0};

    double best_correlation = std::numeric_limits<double>::min();

    for(int dy = -SEARCH_RANGE; dy <= SEARCH_RANGE; dy++){
        for(int dx = -SEARCH_RANGE; dx <= SEARCH_RANGE; dx++){
            int ref_block_x = current_block_x + dx;
            int ref_block_y = current_block_y + dy;

            if(ref_block_x < 0 || ref_block_y < 0 || ref_block_x + BLOCK_SIZE > reference_frame.width || ref_block_y + BLOCK_SIZE > reference_frame.height){
                continue;
            }
            // Calculate Sum of Absolute Difference (SAD) across all channels
            double sad = 0.0;
            int C = current_frame.channels;
            for(int j = 0; j < BLOCK_SIZE; j++){
                for(int i = 0; i < BLOCK_SIZE; i++){
                    for(int c = 0; c < C; c++){
                        double cv = current_frame.getPixel(current_block_x + i, current_block_y + j, c);
                        double rv = reference_frame.getPixel(ref_block_x + i, ref_block_y + j, c);
                        sad += std::abs(cv-rv);
                    }
                }
            }

            if(sad < best_correlation){
                best_correlation = sad;
                bestMV.dx = dx;
                bestMV.dy = dy;
            }
        }
    }
    return bestMV;
}