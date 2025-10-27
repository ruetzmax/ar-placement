#pragma once

#include <opencv2/opencv.hpp>

void trackCamera(const std::vector<cv::Mat> &inputFrames, std::vector<cv::Mat> &outputFrames, std::vector<int> &frameIndeces, cv::Mat &cameraInternals, cv::Mat &cameraDistortion, cv::Mat &rotations, cv::Mat &translations, int frameInterval = 0);
