#include <opencv2/opencv.hpp>

cv::Mat applyCPUPencilFilter(const cv::Mat &inputFrame, int kernelRadius)
{
    cv::Mat outputFrame, grayScale;
    cv::cvtColor(inputFrame, grayScale, cv::COLOR_BGR2GRAY);
    outputFrame = 255 - grayScale;
    cv::GaussianBlur(outputFrame, outputFrame, cv::Size(kernelRadius, kernelRadius), 0);
    outputFrame = 255 - outputFrame;
    cv::divide(grayScale, outputFrame, outputFrame, 256.0);

    return outputFrame;
}