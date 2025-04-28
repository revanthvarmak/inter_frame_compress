#ifndef MOTION_VECTOR_H
#define MOTION_VECTOR_H


#include "frame.h"
extern int BLOCK_SIZE;
extern int SEARCH_RANGE;

class MotionVector{
    public:
        int dx;
        int dy;
        MotionVector(int dx, int dy): dx(dx), dy(dy){}
};

MotionVector estimateMotion(const Frame &current_frame, const Frame &reference_frame, int mbx, int mby);

#endif