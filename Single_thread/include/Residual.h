#ifndef RESIDUAL_H
#define RESIDUAL_H

#include "frame.h"
#include "MotionVector.h"
#include <vector>

MotionVector estimateMotion(const Frame &current_frame, const Frame &reference_frame, int mbx, int mby);
void CalculateResidual(const Frame& current_frame, const Frame &reference_frame, MotionVector mv, int mbx, int mby, std::vector<int> &residualBlock);

void encodeP_Frame(const Frame& current_frame, const Frame& reference_frame, std::vector<MotionVector> &motionVectors, std::vector<std::vector<int>> &residuals);
void decodeP_Frame(std::vector<MotionVector> &motionVectors, std::vector<std::vector<int>> &residuals, const Frame& reference_frame, Frame& output_frame);

#endif