#pragma once

#include <opencv2/opencv.hpp>

cv::Mat applyCPUPencilFilter(const cv::Mat inputFrame, int kernelRadius);
cv::Mat applyCPURetroFilter(const cv::Mat inputFrame, int blockSize, int colorDepth);
cv::Mat applyCPUTransformations(const cv::Mat inputFrame, float rotX, float rotY, float posX, float posY, float scale);