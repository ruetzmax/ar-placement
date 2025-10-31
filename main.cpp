#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include "gpu_transforms.h"
#include "tracking.h"

#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#include <chrono>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

using namespace cv;

std::vector<cv::Mat> readVideo(const char *filename, double &videoFPS)
{
    cv::VideoCapture cap(filename);
    if (!cap.isOpened())
    {
        std::cerr << "Error: Could not open video file: " << filename << std::endl;
        return {};
    }

    videoFPS = cap.get(cv::CAP_PROP_FPS);
    if (videoFPS <= 0.0)
        videoFPS = 30.0;

    std::vector<cv::Mat> frames;
    cv::Mat frame;

    while (true)
    {
        bool ret = cap.read(frame);
        if (!ret)
            break;

        frames.push_back(frame.clone());
    }

    cap.release();
    return frames;
}

void saveImage(const char *filename, int width, int height)
{

    std::cout << "Saving image to " << filename << " (" << width << "x" << height << ")" << std::endl;
    std::vector<unsigned char> pixels(width * height * 3);

    glPixelStorei(GL_PACK_ALIGNMENT, 1);
    glReadBuffer(GL_BACK);
    glFinish();
    glReadPixels(0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE, pixels.data());

    cv::Mat img(height, width, CV_8UC3, pixels.data());

    cv::flip(img, img, 0);
    cv::cvtColor(img, img, cv::COLOR_RGB2BGR);

    try
    {
        cv::imwrite(filename, img);
        std::cout << "Image saved successfully!" << std::endl;
    }
    catch (const cv::Exception &e)
    {
        std::cerr << "OpenCV exception: " << e.what() << std::endl;
    }
}

int main()
{
    std::string imageSavePath = std::string(__FILE__).substr(0, std::string(__FILE__).find_last_of("/\\") + 1) + "images/";
    std::string videoPath = std::string(__FILE__).substr(0, std::string(__FILE__).find_last_of("/\\") + 1) + "videos/";

    int screenWidth, screenHeight;

    double videoFPS;
    std::vector<cv::Mat> unprocessedFrames = readVideo((videoPath + "tracker3.mp4").c_str(), videoFPS);
    std::vector<cv::Mat> processedFrames = unprocessedFrames;
    int currentFrameIndex = 0;
    bool videoIsPlaying = false;
    int frameTrackingInterval = 0;
    std::string processingTime = "Not tracked";
    std::string reprojectionError = "Not tracked";

    screenWidth = processedFrames[0].cols;
    screenHeight = processedFrames[0].rows;

    cv::Mat cameraIntrinsics, cameraDistortion, rotations, translations;
    std::vector<int> frameIndices;

    // -- Setup Window and OpenGL --
    if (!glfwInit())
        return -1;
    GLFWwindow *window = glfwCreateWindow(screenWidth, screenHeight, "AR Placement", NULL, NULL);
    if (!window)
    {
        glfwTerminate();
        return -1;
    }
    glfwMakeContextCurrent(window);
    glewInit();
    initShaderPrograms();

    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LESS);

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


    // -- Setup GUI --
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO &io = ImGui::GetIO();
    (void)io;
    ImGui::StyleColorsDark();
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init("#version 130");

    // -- Main Loop --

    while (!glfwWindowShouldClose(window))
    {
        // DRAW SCREEN
        glfwMakeContextCurrent(window);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        glBindVertexArray(screenVAO);
        glBindBuffer(GL_ARRAY_BUFFER, screenVBO);

        Mat frame = processedFrames[currentFrameIndex].clone();
        if (frame.empty())
            break;

        cv::flip(frame, frame, 0);

        // update texture
        glBindTexture(GL_TEXTURE_2D, texture);
        cv::cvtColor(frame, frame, cv::COLOR_BGR2RGB);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, frame.cols, frame.rows,
                     0, GL_RGB, GL_UNSIGNED_BYTE, frame.data);

        glUseProgram(screenShaderProgram);

        // render quad
        glDrawArrays(GL_TRIANGLES, 0, 6);

        glfwPollEvents();

        // GUI
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();
        ImGui::SetNextWindowSize(ImVec2(0, 0), ImGuiCond_Always);
        ImGui::Begin("Controls");

        // VIDEO CONTROLS
        ImGui::SliderInt("Current Frame", &currentFrameIndex, 0, processedFrames.size() - 1);
        if (ImGui::Button(videoIsPlaying ? "Pause Video" : "Play Video"))
        {
            videoIsPlaying = !videoIsPlaying;
        }
        ImGui::InputInt("Frame Tracking Interval", &frameTrackingInterval);
        if (ImGui::Button("Process Video"))
        {
            trackCamera(unprocessedFrames, processedFrames, window, processingTime, reprojectionError, frameTrackingInterval);

            currentFrameIndex = 0;
            if (!processedFrames.empty() && !processedFrames[0].empty())
            {
                screenWidth = processedFrames[0].cols;
                screenHeight = processedFrames[0].rows;
                glfwSetWindowSize(window, screenWidth, screenHeight);
            }
        }
        ImGui::Text("Processing Time: %s", processingTime.c_str());
        ImGui::Text("Reprojection Error: %s", reprojectionError.c_str());

        // Save image button
        if (ImGui::Button("Save Image"))
        {
            std::string filename = imageSavePath + "frame_" + std::to_string(std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now().time_since_epoch()).count()) + ".png";
            saveImage(filename.c_str(), screenWidth, screenHeight);
        }

        ImGui::End();
        ImGui::Render();
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

        glfwSwapBuffers(window);

        cv::waitKey(40);

        if (videoIsPlaying)
        {
            {
                static auto lastAdvance = std::chrono::high_resolution_clock::now();

                const double frameMs = 1000.0 / videoFPS;
                auto nowFrame = std::chrono::high_resolution_clock::now();
                double elapsedMs = std::chrono::duration<double, std::milli>(nowFrame - lastAdvance).count();

                if (elapsedMs >= frameMs)
                {
                    lastAdvance = nowFrame;
                    currentFrameIndex++;
                    if (currentFrameIndex >= static_cast<int>(processedFrames.size()))
                    {
                        currentFrameIndex = static_cast<int>(processedFrames.size()) - 1;
                        videoIsPlaying = false;
                    }
                }
            }
        }
    }
    cleanupShaderPrograms();
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    glfwTerminate();

    return 0;
}