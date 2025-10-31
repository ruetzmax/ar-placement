#include <opencv2/opencv.hpp>
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <chrono>
#include "gpu_transforms.h"

using namespace cv;

int patternWidth = 9;
int patternHeight = 6;

void trackCamera(const std::vector<cv::Mat> &inputFrames, std::vector<cv::Mat> &outputFrames, GLFWwindow* window, std::string &processingTime, std::string &reprojectionError, int frameInterval = 0)
{
    std::chrono::milliseconds totalProcessingTime(0);
    auto trackingStartTime = std::chrono::high_resolution_clock::now();

    outputFrames.clear();
    glfwMakeContextCurrent(window);

    // Setup cube
    float cubeVertices[] = {
        -1.0f, -1.0f, -1.0f, // triangle 1 : begin
        -1.0f, -1.0f, 1.0f,
        -1.0f, 1.0f, 1.0f, // triangle 1 : end
        1.0f, 1.0f, -1.0f, // triangle 2 : begin
        -1.0f, -1.0f, -1.0f,
        -1.0f, 1.0f, -1.0f, // triangle 2 : end
        1.0f, -1.0f, 1.0f,
        -1.0f, -1.0f, -1.0f,
        1.0f, -1.0f, -1.0f,
        1.0f, 1.0f, -1.0f,
        1.0f, -1.0f, -1.0f,
        -1.0f, -1.0f, -1.0f,
        -1.0f, -1.0f, -1.0f,
        -1.0f, 1.0f, 1.0f,
        -1.0f, 1.0f, -1.0f,
        1.0f, -1.0f, 1.0f,
        -1.0f, -1.0f, 1.0f,
        -1.0f, -1.0f, -1.0f,
        -1.0f, 1.0f, 1.0f,
        -1.0f, -1.0f, 1.0f,
        1.0f, -1.0f, 1.0f,
        1.0f, 1.0f, 1.0f,
        1.0f, -1.0f, -1.0f,
        1.0f, 1.0f, -1.0f,
        1.0f, -1.0f, -1.0f,
        1.0f, 1.0f, 1.0f,
        1.0f, -1.0f, 1.0f,
        1.0f, 1.0f, 1.0f,
        1.0f, 1.0f, -1.0f,
        -1.0f, 1.0f, -1.0f,
        1.0f, 1.0f, 1.0f,
        -1.0f, 1.0f, -1.0f,
        -1.0f, 1.0f, 1.0f,
        1.0f, 1.0f, 1.0f,
        -1.0f, 1.0f, 1.0f,
        1.0f, -1.0f, 1.0f};

    unsigned int cubeVAO, cubeVBO;
    glGenVertexArrays(1, &cubeVAO);
    glGenBuffers(1, &cubeVBO);
    glBindVertexArray(cubeVAO);
    glBindBuffer(GL_ARRAY_BUFFER, cubeVBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(cubeVertices), cubeVertices, GL_STATIC_DRAW);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void *)0);
    glEnableVertexAttribArray(0);

    float vertices[] = {
        -1.0f, 1.0f, 0.0f, 1.0f,
        -1.0f, -1.0f, 0.0f, 0.0f,
        1.0f, -1.0f, 1.0f, 0.0f,

        -1.0f, 1.0f, 0.0f, 1.0f,
        1.0f, -1.0f, 1.0f, 0.0f,
        1.0f, 1.0f, 1.0f, 1.0f};

    unsigned int screenVAO, screenVBO;
    glGenVertexArrays(1, &screenVAO);
    glGenBuffers(1, &screenVBO);
    glBindVertexArray(screenVAO);
    glBindBuffer(GL_ARRAY_BUFFER, screenVBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices,
                 GL_STATIC_DRAW);

    // position attribute
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void *)0);
    glEnableVertexAttribArray(0);
    // texture coord attribute
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void *)(2 * sizeof(float)));
    glEnableVertexAttribArray(1);

    // Texture Setup
    unsigned int texture;
    glGenTextures(1, &texture);
    glBindTexture(GL_TEXTURE_2D, texture);
    // set texture parameters
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S,
                    GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T,
                    GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);



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
    std::vector<int> trackedFrameIndices;  // Store which frames were actually tracked
    std::vector<int> frameToCalibrationIndex;  // Map frame index to actually tracked frame index


    // track 2D image points
    int nextTrackingCandidate = 0;
    int adjustedStart = 0;
    int lastTrackedCalibrationIndex = -1;
    
    
    for(int currentFrameIndex = 0; currentFrameIndex < inputFrames.size(); currentFrameIndex++)
    {
        if (currentFrameIndex < nextTrackingCandidate)
        {
            // skip this frame for tracking, but map it to the last successful calibration
            frameToCalibrationIndex.push_back(lastTrackedCalibrationIndex);
            continue;
        }
        auto frame = inputFrames[currentFrameIndex];
        cv::Mat greyScale;
        cv::cvtColor(frame, greyScale, cv::COLOR_BGR2GRAY);

        std::vector<cv::Point2f> imagePoints;
        cv::findChessboardCorners(greyScale, cv::Size(patternWidth, patternHeight), imagePoints, cv::CALIB_CB_ADAPTIVE_THRESH + cv::CALIB_CB_NORMALIZE_IMAGE);

        if (!imagePoints.empty())
        {
            std::cout << "Tracked frame " << currentFrameIndex << " / " << inputFrames.size() << "\r" << std::flush;
            // tracking successful, add new imagePoints and set next tracking candidate
            combinedImagePoints.push_back(imagePoints);
            combinedObjectPoints.push_back(objectPoints);
            trackedFrameIndices.push_back(currentFrameIndex);
            lastTrackedCalibrationIndex = combinedImagePoints.size() - 1;
            frameToCalibrationIndex.push_back(lastTrackedCalibrationIndex);
            nextTrackingCandidate = currentFrameIndex + frameInterval;
        }
        else {
            if (combinedImagePoints.empty()) {
                // no previous successful tracking, skip frame and adjust start to skip untracked beginning frames
                adjustedStart++;
                continue;
            }

            // tracking failed, map this frame to the last successful calibration
            frameToCalibrationIndex.push_back(lastTrackedCalibrationIndex);
        }
    }

    // calibrate
    std::cout << "Calibrating camera with " << combinedImagePoints.size() << " tracked frames." << std::endl;
    cv::Mat cameraIntrinsics, cameraDistortion, rotations, translations;
    cv::calibrateCamera(combinedObjectPoints, combinedImagePoints, inputFrames[0].size(), cameraIntrinsics, cameraDistortion, rotations, translations);

    // expand rotations and translations to cover all frames
    std::vector<cv::Mat> allRotations(inputFrames.size() - adjustedStart);
    std::vector<cv::Mat> allTranslations(inputFrames.size() - adjustedStart);
    
    for (int frameIndex = adjustedStart; frameIndex < inputFrames.size(); frameIndex++)
    {
        int localIndex = frameIndex - adjustedStart;
        int calibrationIndex = frameToCalibrationIndex[localIndex];
        
        if (calibrationIndex >= 0) {
            allRotations[localIndex] = rotations.row(calibrationIndex).clone();
            allTranslations[localIndex] = translations.row(calibrationIndex).clone();
        }
    }

    auto trackingEndTime = std::chrono::high_resolution_clock::now();
    totalProcessingTime += std::chrono::duration_cast<std::chrono::milliseconds>(trackingEndTime - trackingStartTime);

    // undistort images and draw object
    for (int frameIndex = adjustedStart; frameIndex < inputFrames.size(); frameIndex++)
    {
        std::cout << "Processing frame " << frameIndex << " / " << inputFrames.size() << "\r" << std::flush;
        
        auto frameStartTime = std::chrono::high_resolution_clock::now();
        
        auto frame = inputFrames[frameIndex].clone();
        int localIndex = frameIndex - adjustedStart;
        cv::Mat rotationVec = allRotations[localIndex];
        cv::Mat translationVec = allTranslations[localIndex];

        // cv::undistort(frame, output, cameraIntrinsics, cameraDistortion);

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        cv::flip(frame, frame, 0);

        glBindTexture(GL_TEXTURE_2D, texture);
        cv::cvtColor(frame, frame, cv::COLOR_BGR2RGB);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, frame.cols, frame.rows,
                     0, GL_RGB, GL_UNSIGNED_BYTE, frame.data);

        glDisable(GL_DEPTH_TEST);
        glBindVertexArray(screenVAO);
        glBindBuffer(GL_ARRAY_BUFFER, screenVBO);
        glUseProgram(screenShaderProgram);
        glDrawArrays(GL_TRIANGLES, 0, 6);

        glEnable(GL_DEPTH_TEST);
        glUseProgram(objectShaderProgram);
        glBindVertexArray(cubeVAO);
        glBindBuffer(GL_ARRAY_BUFFER, cubeVBO);
        glm::mat4 viewMatrix = getViewMatrix(rotationVec, translationVec);
        glm::mat4 projectionMatrix = getProjectionMatrix(cameraIntrinsics);

        glUniformMatrix4fv(glGetUniformLocation(objectShaderProgram, "view"), 1, GL_FALSE, glm::value_ptr(viewMatrix));
        glUniformMatrix4fv(glGetUniformLocation(objectShaderProgram, "projection"), 1, GL_FALSE, glm::value_ptr(projectionMatrix));

        glDrawArrays(GL_TRIANGLES, 0, 36);

        // exclude reading out pixel values from timing, since this would not be part of real world application
        auto frameEndTime = std::chrono::high_resolution_clock::now();
        totalProcessingTime += std::chrono::duration_cast<std::chrono::milliseconds>(frameEndTime - frameStartTime);

        std::vector<unsigned char> pixels(frame.cols * frame.rows * 3);
        glReadPixels(0, 0, frame.cols, frame.rows, GL_RGB, GL_UNSIGNED_BYTE, pixels.data());
        cv::Mat output(frame.rows, frame.cols, CV_8UC3, pixels.data());
        cv::flip(output, output, 0);
        cv::cvtColor(output, output, cv::COLOR_RGB2BGR);

        outputFrames.push_back(output.clone());
    }

    processingTime = std::to_string(totalProcessingTime.count()) + " ms";

    // determine reprojection error (for all frames)
    std::vector<std::vector<cv::Point2f>> allFrameImagePoints(inputFrames.size());
    for (int frameIndex = 0; frameIndex < inputFrames.size(); frameIndex++)
    {
        std::cout << "Detecting frame " << frameIndex << " / " << inputFrames.size() << "\r" << std::flush;
        auto frame = inputFrames[frameIndex];
        cv::Mat greyScale;
        cv::cvtColor(frame, greyScale, cv::COLOR_BGR2GRAY);

        std::vector<cv::Point2f> imagePoints;
        cv::findChessboardCorners(greyScale, cv::Size(patternWidth, patternHeight), imagePoints, 
                                  cv::CALIB_CB_ADAPTIVE_THRESH + cv::CALIB_CB_NORMALIZE_IMAGE);
         
        allFrameImagePoints[frameIndex] = imagePoints;
    }

    double totalError = 0;
    int validFrameCount = 0;
    for (int frameIndex = adjustedStart; frameIndex < inputFrames.size(); frameIndex++)
    {
        // Skip frames where chessboard was not detected
        if (allFrameImagePoints[frameIndex].empty()) {
            continue;
        }

        int localIndex = frameIndex - adjustedStart;
        
        std::vector<cv::Point2f> projectedPoints;
        cv::projectPoints(objectPoints, allRotations[localIndex], allTranslations[localIndex], 
                          cameraIntrinsics, cameraDistortion, projectedPoints);
        
        totalError += cv::norm(allFrameImagePoints[frameIndex], projectedPoints, cv::NORM_L2) / projectedPoints.size();
        validFrameCount++;
    }

    if (validFrameCount > 0) {
        reprojectionError = std::to_string(totalError / validFrameCount);
    }
}