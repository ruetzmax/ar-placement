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

cv::Mat applyCPURetroFilter(const cv::Mat &inputFrame, int blockSize, int colorDepth)
{
    // pixelate image
    cv::Mat outputFrame = inputFrame.clone();
    for (int y = 0; y < inputFrame.rows; y += blockSize)
    {
        for (int x = 0; x < inputFrame.cols; x += blockSize)
        {
            cv::Rect rect(x, y, blockSize, blockSize);
            rect.width = std::min(rect.width, inputFrame.cols - x);
            rect.height = std::min(rect.height, inputFrame.rows - y);
            cv::Scalar color = cv::mean(inputFrame(rect));
            outputFrame(rect).setTo(color);
        }
    }

    // reduce color depth
    int levels = 1 << colorDepth;
    int step = 256 / levels;
    outputFrame /= step;
    outputFrame *= step;

    return outputFrame;
}