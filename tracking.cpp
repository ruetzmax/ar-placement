#include <opencv2/opencv.hpp>

using namespace cv;

int patternWidth = 9;
int patternHeight = 6;

void trackCamera(const std::vector<cv::Mat> &inputFrames, std::vector<cv::Mat> &outputFrames, std::vector<int> &frameIndeces, cv::Mat &cameraIntrinsics, cv::Mat &cameraDistortion, cv::Mat &rotations, cv::Mat &translations, int frameInterval = 0)
{
    outputFrames.clear();
    frameIndeces.clear();

    // construct 3D world points
    std::vector<cv::Point3f> objectPoints;
    objectPoints.reserve(patternWidth * patternHeight);
    for (int y = 0; y < patternHeight; ++y)
    {
        for (int x = 0; x < patternWidth; ++x)
        {
            objectPoints.emplace_back(static_cast<float>(x), static_cast<float>(y), 0.0f);
        }
    }

    std::vector<std::vector<cv::Point3f>> combinedObjectPoints;
    std::vector<std::vector<cv::Point2f>> combinedImagePoints;

    // track 2D image points
    for(int currentFrameIndex = 0; currentFrameIndex < inputFrames.size(); currentFrameIndex++)
    {
        auto frame = inputFrames[currentFrameIndex];
        cv::Mat greyScale;
        cv::cvtColor(frame, greyScale, cv::COLOR_BGR2GRAY);

        std::vector<cv::Point2f> imagePoints;
        cv::findChessboardCorners(greyScale, cv::Size(patternWidth, patternHeight), imagePoints, cv::CALIB_CB_ADAPTIVE_THRESH + cv::CALIB_CB_NORMALIZE_IMAGE);

        if (!imagePoints.empty())
        {
            combinedImagePoints.push_back(imagePoints);
            combinedObjectPoints.push_back(objectPoints);
            
            // cv::drawChessboardCorners(output, cv::Size(patternWidth, patternHeight), imagePoints, true);
            currentFrameIndex += frameInterval;
        }
        frameIndeces.push_back(currentFrameIndex);
    }

    // calibrate
    cv::calibrateCamera(combinedObjectPoints, combinedImagePoints, inputFrames[0].size(), cameraIntrinsics, cameraDistortion, rotations, translations);

    // undistort images
    for (auto &frame : inputFrames)
    {
        cv::Mat output;
        cv::undistort(frame, output, cameraIntrinsics, cameraDistortion);
        outputFrames.push_back(output);
    }

    // determine reprojection error
    double totalError = 0;
    for (size_t i = 0; i < combinedObjectPoints.size(); i++)
    {
        std::vector<cv::Point2f> projectedPoints;
        cv::projectPoints(combinedObjectPoints[i], rotations.row(i), translations.row(i), 
                          cameraIntrinsics, cameraDistortion, projectedPoints);
        
        totalError += cv::norm(combinedImagePoints[i], projectedPoints, cv::NORM_L2) / projectedPoints.size();
    }

    double reprojectionError = totalError / combinedObjectPoints.size();
    std::cout << "Reprojection error: " << reprojectionError << std::endl;
}