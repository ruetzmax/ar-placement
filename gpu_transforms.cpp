#include <iostream>
#include <string>
#include <map>
#include <utility>
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <opencv2/opencv.hpp>
#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>
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
FragColor = vec4 (0.0 , 0.0 , 1.0 , 1.0) ;
}
)";

glm::mat4 getViewMatrix(const cv::Mat& rotationVec, const cv::Mat& translationVec)
{
    cv::Mat R_cv;
    cv::Rodrigues(rotationVec, R_cv);
    
    cv::Mat view_cv = cv::Mat::eye(4,4,CV_64F);
    for (int r=0; r<3; r++)
        for (int c=0; c<3; c++)
            view_cv.at<double>(r,c) = R_cv.at<double>(r,c);
    
    for (int r=0; r<3; r++)
        view_cv.at<double>(r,3) = translationVec.at<double>(0, r);

    // convert to OpenGL coord system
    cv::Mat S = cv::Mat::eye(4,4,CV_64F);
    S.at<double>(1,1) = -1.0;
    S.at<double>(2,2) = -1.0;
    view_cv = S * view_cv;

    // convert row major to column major
    glm::mat4 view_gl;
    for (int r = 0; r < 4; r++) {
        for (int c = 0; c < 4; c++) {
            view_gl[c][r] = static_cast<float>(view_cv.at<double>(r, c));
        }
    }

    return view_gl;
}

glm::mat4 getProjectionMatrix(cv::Mat cameraIntrinsics){
    float fx = cameraIntrinsics.at<double>(0, 0);
    float fy = cameraIntrinsics.at<double>(1, 1);
    float cx = cameraIntrinsics.at<double>(0, 2);
    float cy = cameraIntrinsics.at<double>(1, 2);

    float fov = 2.0f * atan(cy / fy);
    float aspect = cx / cy;
    float near = 0.1f;
    float far = 100.0f;

    glm::mat4 projection = glm::perspective(fov, aspect, near, far);
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
    