#pragma once

#include <string>
#include <GL/glew.h>
#include <opencv2/opencv.hpp>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

extern const std::string screenVertexShader;
extern const std::string screenFragmentShader;
extern const std::string objectVertexShader;
extern const std::string objectFragmentShader;

glm::mat4 getViewMatrix(const cv::Mat& rotationVec, const cv::Mat& translationVec);
glm::mat4 getProjectionMatrix(cv::Mat cameraIntrinsics);


unsigned int compileShader(unsigned int type, const std::string& source);
unsigned int createShaderProgram(const std::string& vertexShader, const std::string& fragmentShader);

extern unsigned int screenShaderProgram;
extern unsigned int objectShaderProgram;

void initShaderPrograms();
void cleanupShaderPrograms();