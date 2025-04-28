#ifndef RESIDUAL_H
#define RESIDUAL_H

#include "frame.h"
#include "MotionVector.h"
#include <vector>

std::vector<MotionVector> estimateMotion(const Frame&, const Frame&);
Frame calculateResidual (const Frame& cur, const Frame& ref, const std::vector<MotionVector>& MV);
Frame decodeP_Frame (const Frame& ref, const std::vector<MotionVector>& MV, const Frame& residual);

inline void encodeP_Frame(const Frame& cur, const Frame& ref, std::vector<MotionVector>& mvOut, Frame& residualOut) {
    mvOut = estimateMotion(cur, ref);   // from MotionEstimation.cu
    residualOut = calculateResidual(cur, ref, mvOut);
}
#endif
