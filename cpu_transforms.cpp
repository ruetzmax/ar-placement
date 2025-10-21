#include <opencv2/opencv.hpp>

cv::Mat applyCPUPencilFilter(const cv::Mat inputFrame, int kernelSize)
{
    cv::Mat outputFrame, grayScale;
    cv::cvtColor(inputFrame, grayScale, cv::COLOR_BGR2GRAY);
    outputFrame = 255 - grayScale;
    cv::GaussianBlur(outputFrame, outputFrame, cv::Size(kernelSize, kernelSize), 0);
    outputFrame = 255 - outputFrame;
    cv::divide(grayScale, outputFrame, outputFrame, 256.0);

    return outputFrame;
}

cv::Mat applyCPURetroFilter(const cv::Mat inputFrame, int blockSize, int colorDepth)
{
    // pixelate image
    cv::Mat outputFrame = cv::Mat::zeros(inputFrame.size(), inputFrame.type());
    for (int y = 0; y < inputFrame.rows; y += blockSize)
    {
        for (int x = 0; x < inputFrame.cols; x += blockSize)
        {
            cv::Rect rect(x, y, blockSize, blockSize);
            rect.width = std::min(rect.width, inputFrame.cols - x);
            rect.height = std::min(rect.height, inputFrame.rows - y);
            cv::Scalar color = inputFrame.at<cv::Vec3b>(y, x);
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

cv::Mat applyCPUTransformations(const cv::Mat inputFrame, float rotX, float rotY, float posX, float posY, float scale){
    cv::Mat outputFrame;

    // use projective transformations for rotation
    float focalLength = 500.0f;

    float cosX = cos(rotX * CV_PI / 180.0);
    float sinX = sin(rotX * CV_PI / 180.0);
    float cosY = cos(rotY * CV_PI / 180.0);
    float sinY = sin(rotY * CV_PI / 180.0);

    double cx = inputFrame.cols / 2.0;
    double cy = inputFrame.rows / 2.0;

    cv::Mat rotXMat = (cv::Mat_<double>(3,3) << 
        1, 0, 0,
        0, cosX, -sinX,
        0, sinX, cosX);

    cv::Mat rotYMat = (cv::Mat_<double>(3,3) << 
        cosY, 0, sinY,
        0, 1, 0,
        -sinY, 0, cosY);
        
    cv::Mat rotMat = rotYMat * rotXMat;

    std::vector<cv::Mat> imageCornersCentered = {
        (cv::Mat_<double>(3,1) << -cx, -cy, 0),
        (cv::Mat_<double>(3,1) <<  cx, -cy, 0),
        (cv::Mat_<double>(3,1) <<  cx,  cy, 0),
        (cv::Mat_<double>(3,1) << -cx,  cy, 0)
    };

    std::vector<cv::Mat> imageCornersRotated = {
        rotMat * imageCornersCentered[0],
        rotMat * imageCornersCentered[1],
        rotMat * imageCornersCentered[2],
        rotMat * imageCornersCentered[3]
    };

    // move corners back for more stable numerical projection
    for (auto& corner : imageCornersRotated) {
        corner.at<double>(2, 0) += focalLength;
    }

    std::vector<cv::Point2f> imageCornersProjected;
    for (const auto& corner : imageCornersRotated){
        imageCornersProjected.push_back(cv::Point2f(
            corner.at<double>(0,0) * focalLength / (corner.at<double>(2,0) + 1e-5) + cx,
            corner.at<double>(1,0) * focalLength / (corner.at<double>(2,0) + 1e-5) + cy
        ));
    }

    std::vector<cv::Point2f> srcPoints = {
        cv::Point2f(0, 0),
        cv::Point2f(inputFrame.cols, 0),
        cv::Point2f(inputFrame.cols, inputFrame.rows),
        cv::Point2f(0, inputFrame.rows)
    };

    cv::Mat H = cv::getPerspectiveTransform(srcPoints, imageCornersProjected);
    cv::warpPerspective(inputFrame, outputFrame, H, inputFrame.size(), cv::INTER_LINEAR, cv::BORDER_CONSTANT);

    // use affine transformations for translation and scaling
    cv::Mat affineMat = (cv::Mat_<double>(2,3) << 
        scale, 0, posX,
        0, scale, posY
    );
    cv::warpAffine(outputFrame, outputFrame, affineMat, outputFrame.size());

    return outputFrame;
}