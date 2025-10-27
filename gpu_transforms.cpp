#include <iostream>
#include <string>
#include <map>
#include <utility>
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <opencv2/opencv.hpp>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

std::string screenVertexShader = R"(
#version 330 core
layout ( location = 0) in vec2 aPos ;
layout ( location = 1) in vec2 aTexCoord ;
out vec2 TexCoord ;
void main () {
gl_Position = vec4 ( aPos , 0.0 , 1.0) ;
TexCoord = aTexCoord ;
}
)";

std::string screenFragmentShader = R"(
#version 330 core
out vec4 FragColor ;
in vec2 TexCoord ;
uniform sampler2D texture1 ;
void main () {
FragColor = texture ( texture1 , vec2(TexCoord.x, TexCoord.y) ) ;
}
)";

std::string objectVertexShader = R"(
#version 330 core
layout ( location = 0) in vec3 aPos ;

uniform mat4 view ;
uniform mat4 projection ;

void main () {
gl_Position =  projection * view * vec4 ( aPos , 1.0) ;
}
)";

std::string objectFragmentShader = R"(
#version 330 core
out vec4 FragColor ;
void main () {
FragColor = vec4 (1.0 , 0.5 , 0.2 , 1.0) ;
}
)";

glm::mat4 getViewMatrix(const cv::Mat& rotationVec, const cv::Mat& translationVec)
{
    // cv::Mat rotationMat3;
    // cv::Rodrigues(rotationVec, rotationMat3);
    // rotationMat3.convertTo(rotationMat3, CV_64F);

    // cv::Mat rotationMat = cv::Mat::eye(4, 4, CV_64F);
    // for (int i = 0; i < 3; ++i)
    // {
    //     for (int j = 0; j < 3; ++j)
    //     {
    //         rotationMat.at<double>(i, j) = rotationMat3.at<double>(i, j);
    //     }
    // }

    // rotationMat = rotationMat.inv();


    // cv::Mat translationMat = cv::Mat::eye(4, 4, CV_64F);
    // for (int i = 0; i < 3; ++i)
    // {
    //     translationMat.at<double>(i, 3) = -translationVec.at<double>(0, i);
    // }


    // cv::Mat flip = (cv::Mat_<double>(4,4) <<
    //  1,  0,  0,  0,
    //  0, -1,  0,  0,
    //  0,  0, -1,  0,
    //  0,  0,  0,  1);

    // return rotationMat * translationMat;
    glm::mat4 view = glm::mat4(1.0f);
    // Apply translation
    view = glm::translate(view, glm::vec3(-translationVec.at<float>(0, 0), translationVec.at<float>(0, 1), translationVec.at<float>(0, 2)));
    view = glm::rotate(view, -rotationVec.at<float>(0, 0), glm::vec3(1.0f, 0.0f, 0.0f));
    view = glm::rotate(view, -rotationVec.at<float>(0, 1), glm::vec3(0.0f, -1.0f, 0.0f));
    view = glm::rotate(view, -rotationVec.at<float>(0, 2), glm::vec3(0.0f, 0.0f, -1.0f));
    return view;
}

glm::mat4 getProjectionMatrix(cv::Mat cameraIntrinsics){
    float fov = 45.0f;           // Field of view in degrees
    float aspect = 640.0f/480.0f; // Width / Height
    float near = 0.1f;
    float far = 100.0f;

    glm::mat4 projection = glm::perspective(glm::radians(fov), aspect, near, far);
    return projection;
}

unsigned int compileShader(unsigned int type, const std::string& source)
{
    unsigned int id = glCreateShader(type);
    const char* src = source.c_str();
    glShaderSource(id, 1, &src, nullptr);
    glCompileShader(id);

    // Error handling
    int result;
    glGetShaderiv(id, GL_COMPILE_STATUS, &result);
    if (result == GL_FALSE)
    {
        int length;
        glGetShaderiv(id, GL_INFO_LOG_LENGTH, &length);
        char* message = (char*)alloca(length * sizeof(char));
        glGetShaderInfoLog(id, length, &length, message);
        std::cout << "Failed to compile " << (type == GL_VERTEX_SHADER ? "vertex" : "fragment") << " shader!" << std::endl;
        std::cout << message << std::endl;
        glDeleteShader(id);
        return 0;
    }

    return id;
}

unsigned int createShaderProgram(const std::string& vertexShader, const std::string& fragmentShader)
{
    unsigned int program = glCreateProgram();
    unsigned int vs = compileShader(GL_VERTEX_SHADER, vertexShader);
    unsigned int fs = compileShader(GL_FRAGMENT_SHADER, fragmentShader);

    glAttachShader(program, vs);
    glAttachShader(program, fs);
    glLinkProgram(program);
    glValidateProgram(program);

    glDeleteShader(vs);
    glDeleteShader(fs);

    return program;
}

unsigned int screenShaderProgram;
unsigned int objectShaderProgram;

void initShaderPrograms()
{    
    screenShaderProgram = createShaderProgram(screenVertexShader, screenFragmentShader);
    objectShaderProgram = createShaderProgram(objectVertexShader, objectFragmentShader);
}

void cleanupShaderPrograms()
{
    glDeleteProgram(screenShaderProgram);
    glDeleteProgram(objectShaderProgram);
}
    