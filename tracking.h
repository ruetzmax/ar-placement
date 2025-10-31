#pragma once

#include <opencv2/opencv.hpp>

void trackCamera(const std::vector<cv::Mat> &inputFrames, std::vector<cv::Mat> &outputFrames, GLFWwindow* window, std::string &processingTime, std::string &reprojectionError, int frameInterval = 0);
