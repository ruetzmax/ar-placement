#include <opencv2/opencv.hpp>

cv::Mat performCPUTransforms(const cv::Mat &inputFrame)
{
    cv::Mat outputFrame;

    // tint green
    cv::Mat greenTint(inputFrame.size(), inputFrame.type(), cv::Scalar(0, 50, 0));
    cv::addWeighted(inputFrame, 0.7, greenTint, 0.3, 0.0, outputFrame);
    return outputFrame;
}