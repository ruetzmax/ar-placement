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

std::vector<cv::Mat> trackCameraMotion(std::vector<cv::Mat> video, cv::Mat &cameraIntrinsics, cv::Mat &cameraDistortion, cv::Mat &rotations, cv::Mat &translations, std::vector<int> &frameIndices)
{
    std::vector<cv::Mat> outputFrames;
    trackCamera(video, outputFrames, frameIndices, cameraIntrinsics, cameraDistortion, rotations, translations, 0);
    return outputFrames;
}

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
    float fps = 0.0f;
    std::string avgFPS = "Not tracked";
    float timeWindowSeconds = 5.0f;
    std::chrono::time_point<std::chrono::high_resolution_clock> lastFrameTime;
    bool isTrackingFPS = false;
    int frameCount = 0;
    std::chrono::time_point<std::chrono::high_resolution_clock> trackingStartTime;

    std::string imageSavePath = std::string(__FILE__).substr(0, std::string(__FILE__).find_last_of("/\\") + 1) + "images/";
    std::string videoPath = std::string(__FILE__).substr(0, std::string(__FILE__).find_last_of("/\\") + 1) + "videos/";

    int screenWidth, screenHeight;

    double videoFPS;
    std::vector<cv::Mat> unprocessedFrames = readVideo((videoPath + "tracker3.mp4").c_str(), videoFPS);
    std::vector<cv::Mat> processedFrames = unprocessedFrames;
    int currentFrameIndex = 0;
    bool videoIsPlaying = false;
    std::string processingTime = "Not tracked";

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
        // TIME TRACKING
        auto now = std::chrono::high_resolution_clock::now();
        float deltaTime = std::chrono::duration<float>(now - lastFrameTime).count();
        fps = 1.0f / deltaTime;

        lastFrameTime = now;

        if (isTrackingFPS)
        {
            frameCount++;
            float elapsed = std::chrono::duration<float>(now - trackingStartTime).count();
            if (elapsed >= timeWindowSeconds)
            {
                float averageFPS = frameCount / elapsed;
                avgFPS = std::to_string(averageFPS).substr(0, 5);
                isTrackingFPS = false;
                frameCount = 0;
            }
        }

        // DRAW SCREEN
        glfwMakeContextCurrent(window);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        glDisable(GL_DEPTH_TEST);

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

        // DRAW OBJECT

        if (!frameIndices.empty())
        {

            glEnable(GL_DEPTH_TEST);
            glClear(GL_DEPTH_BUFFER_BIT);

            cv::Mat rotationVec, translationVec;

            // check if current frame was tracked
            // if so, use the actual tracked r+t
            auto it = std::find(frameIndices.begin(), frameIndices.end(), currentFrameIndex);
            if (it != frameIndices.end())
            {
                int index = std::distance(frameIndices.begin(), it);
                rotationVec = rotations.row(index);
                translationVec = translations.row(index);
            }
            // if not, interpolate linearly between previous and next tracked frame (if they exist)
            else
            {
                auto nextIt = std::find_if(frameIndices.begin(), frameIndices.end(), [&](int idx) { return idx > currentFrameIndex; });
                auto prevIt = (nextIt != frameIndices.begin()) ? std::prev(nextIt) : frameIndices.end();

                if (prevIt != frameIndices.end() && nextIt != frameIndices.end())
                {
                    int prevIndex = std::distance(frameIndices.begin(), prevIt);
                    int nextIndex = std::distance(frameIndices.begin(), nextIt);

                    float alpha = float(currentFrameIndex - *prevIt) / float(*nextIt - *prevIt);

                    cv::Mat prevRotVec = rotations.row(prevIndex);
                    cv::Mat nextRotVec = rotations.row(nextIndex);
                    cv::Mat prevTransVec = translations.row(prevIndex);
                    cv::Mat nextTransVec = translations.row(nextIndex);

                    rotationVec = (1.0f - alpha) * prevRotVec + alpha * nextRotVec;
                    translationVec = (1.0f - alpha) * prevTransVec + alpha * nextTransVec;
                }
                // for first or last frames, use the closest tracked frame
                else if (prevIt != frameIndices.end())
                {
                    int prevIndex = std::distance(frameIndices.begin(), prevIt);
                    rotationVec = rotations.row(prevIndex);
                    translationVec = translations.row(prevIndex);
                }
                else if (nextIt != frameIndices.end())
                {
                    int nextIndex = std::distance(frameIndices.begin(), nextIt);
                    rotationVec = rotations.row(nextIndex);
                    translationVec = translations.row(nextIndex);
                }
            }

            glUseProgram(objectShaderProgram);
            glBindVertexArray(cubeVAO);
            glBindBuffer(GL_ARRAY_BUFFER, cubeVBO);
            glm::mat4 viewMatrix = getViewMatrix(rotationVec, translationVec);
            glm::mat4 projectionMatrix = getProjectionMatrix(cameraIntrinsics);

            glUniformMatrix4fv(glGetUniformLocation(objectShaderProgram, "view"), 1, GL_FALSE, glm::value_ptr(viewMatrix));
            glUniformMatrix4fv(glGetUniformLocation(objectShaderProgram, "projection"), 1, GL_FALSE, glm::value_ptr(projectionMatrix));

            glDrawArrays(GL_TRIANGLES, 0, 36);
        }

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
        if (ImGui::Button("Track and Undistort"))
        {
            auto t0 = std::chrono::high_resolution_clock::now();
            processedFrames = trackCameraMotion(unprocessedFrames, cameraIntrinsics, cameraDistortion, rotations, translations, frameIndices);
            auto t1 = std::chrono::high_resolution_clock::now();
            double elapsedMs = std::chrono::duration<double, std::milli>(t1 - t0).count();
            processingTime = std::to_string(elapsedMs) + " ms";

            currentFrameIndex = 0;
            if (!processedFrames.empty() && !processedFrames[0].empty())
            {
                screenWidth = processedFrames[0].cols;
                screenHeight = processedFrames[0].rows;
                glfwSetWindowSize(window, screenWidth, screenHeight);
            }
        }
        ImGui::Text("Processing Time: %s", processingTime.c_str());

        // FPS
        ImGui::Text("Current FPS: %.1f", fps);
        ImGui::Text("Average FPS (%.1fs): %s", timeWindowSeconds, avgFPS.c_str());
        ImGui::InputFloat("Window Size (s)", &timeWindowSeconds);
        if (ImGui::Button("Track FPS"))
        {
            if (!isTrackingFPS)
            {
                isTrackingFPS = true;
                frameCount = 0;
                trackingStartTime = std::chrono::high_resolution_clock::now();
                avgFPS = "Tracking...";
            }
        }

        // Save image button
        if (ImGui::Button("Save Image"))
        {
            std::string filename = imageSavePath + "frame_" + std::to_string(std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count()) + ".png";
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