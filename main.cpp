#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include "gpu_transforms.h"
#include "cpu_transforms.h"

#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#include <chrono>

using namespace cv;

enum BackendMode
{
    CPU = 0,
    GPU = 1
};

enum ImageFilter
{
    NONE = 0,
    PENCIL = 1,
    RETRO = 2,
};

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
    BackendMode backend = CPU;
    ImageFilter filter = NONE;
    int screenWidth = 0;
    int screenHeight = 0;

    int pencilKernelRadius = 21;
    float pencilKernelWeights[pencilKernelRadius * pencilKernelRadius];
    float sigma = pencilKernelRadius / 3.0;

    float sum = 0.0;
    for (int y = -pencilKernelRadius / 2; y <= pencilKernelRadius / 2; y++)
    {
        for (int x = -pencilKernelRadius / 2; x <= pencilKernelRadius / 2; x++)
        {
            float weight = std::exp(-(x * x + y * y) / (2 * sigma * sigma));
            pencilKernelWeights[(y + pencilKernelRadius / 2) * pencilKernelRadius + (x + pencilKernelRadius / 2)] = weight;
            sum += weight;
        }
    }
    for (int y = 0; y < pencilKernelRadius; y++)
    {
        for (int x = 0; x < pencilKernelRadius; x++)
        {
            pencilKernelWeights[y * pencilKernelRadius + x] /= sum;
        }
    }

    int pixelBlockSize = 16;
    int pixelColorDepth = 4;

    float rotX = 0.0f;
    float rotY = 0.0f;

    float posX = 0.0f;
    float posY = 0.0f;

    static float scale = 1.0f;

    bool isInteractive = false;

    float avgFPS = 0.0f;
    const float timeWindow = 5.0f;
    std::deque<std::chrono::time_point<std::chrono::high_resolution_clock>> frameTimes;

    std::string imageSavePath = std::string(__FILE__).substr(0, std::string(__FILE__).find_last_of("/\\") + 1) + "images/";

    bool rightMouseDown = false, leftMouseDown = false;
    double lastMouseX = 0.0, lastMouseY = 0.0;

    // -- Setup Video Capture --
    VideoCapture cap(0);
    if (!cap.isOpened())
        return -1;

    cv ::Mat frame;
    cap >> frame;
    if (frame.empty())
    {
        std::cerr << " Error : couldn ’t capture an initial frame from camera. Exiting.\n";
        cap.release();
        glfwTerminate();
        return -1;
    }
    screenWidth = frame.cols;
    screenHeight = frame.rows;

    // -- Setup Window and OpenGL --
    if (!glfwInit())
        return -1;
    GLFWwindow *window = glfwCreateWindow(screenWidth, screenHeight, "VC2", NULL, NULL);
    if (!window)
    {
        glfwTerminate();
        return -1;
    }
    glfwMakeContextCurrent(window);
    glewInit();
    initShaderPrograms();

    float vertices[] = {
        -1.0f, 1.0f, 0.0f, 1.0f,
        -1.0f, -1.0f, 0.0f, 0.0f,
        1.0f, -1.0f, 1.0f, 0.0f,

        -1.0f, 1.0f, 0.0f, 1.0f,
        1.0f, -1.0f, 1.0f, 0.0f,
        1.0f, 1.0f, 1.0f, 1.0f};

    unsigned int VAO, VBO;
    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &VBO);
    glBindVertexArray(VAO);
    glBindBuffer(GL_ARRAY_BUFFER, VBO);
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
        // time tracking
        auto now = std::chrono::high_resolution_clock::now();
        frameTimes.push_back(now);

        auto cutoff = now - std::chrono::duration<float>(timeWindow);
        while (!frameTimes.empty() && frameTimes.front() < cutoff)
            frameTimes.pop_front();

        float elapsed = std::chrono::duration<float>(frameTimes.back() - frameTimes.front()).count();
        if (elapsed > 0.0f && frameTimes.size() > 1)
            avgFPS = (frameTimes.size() - 1) / elapsed;

        glfwMakeContextCurrent(window);
        glClear(GL_COLOR_BUFFER_BIT);

        // Pull frame from video stream
        Mat frame;
        cap >> frame;
        if (frame.empty())
            break;

        cv::resize(frame, frame, cv::Size(screenWidth, screenHeight));
        cv::flip(frame, frame, 0);

        if (backend == CPU)
        {
            if (filter == NONE){
                // do nothing
            }
            else if (filter == PENCIL){
                frame = applyCPUPencilFilter(frame, pencilKernelRadius);
            }
            else if (filter == RETRO){
                frame = applyCPURetroFilter(frame, pixelBlockSize, pixelColorDepth);
            }

            if (isInteractive)
            {
                frame = applyCPUTransformations(frame, rotX, rotY, posX, posY, scale);
            }
        }


        // update texture
        glBindTexture(GL_TEXTURE_2D, texture);
        cv::cvtColor(frame, frame, cv::COLOR_BGR2RGB);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, frame.cols, frame.rows,
                     0, GL_RGB, GL_UNSIGNED_BYTE, frame.data);

        unsigned int shaderProgram;
        if (backend == CPU)
        {
            shaderProgram = getShaderProgram(0, false);
        }
        else
        {
            if (filter == NONE){
                shaderProgram = getShaderProgram(0, isInteractive);
            }
            else if (filter == PENCIL){
                shaderProgram = getShaderProgram(1, isInteractive);
                glUniform1i(glGetUniformLocation(shaderProgram, "kernelRadius"), pencilKernelRadius);
                glUniform1fv(glGetUniformLocation(shaderProgram, "weights"), pencilKernelRadius * pencilKernelRadius, pencilKernelWeights);
            }
            else if (filter == RETRO){
                shaderProgram = getShaderProgram(2, isInteractive);
                glUniform1i(glGetUniformLocation(shaderProgram, "blockPixelSize"), pixelBlockSize);
                glUniform1i(glGetUniformLocation(shaderProgram, "colorDepth"), pixelColorDepth);
            }

            if (isInteractive)
            {
                glUniform1f(glGetUniformLocation(shaderProgram, "rotX"), rotX);
                glUniform1f(glGetUniformLocation(shaderProgram, "rotY"), rotY);
                glUniform1f(glGetUniformLocation(shaderProgram, "posX"), posX / frame.cols);
                glUniform1f(glGetUniformLocation(shaderProgram, "posY"), posY / frame.rows);
                glUniform1f(glGetUniformLocation(shaderProgram, "scale"), scale);
            }
        }
        glUseProgram(shaderProgram);

        // render quad
        glDrawArrays(GL_TRIANGLES, 0, 6);

        glfwPollEvents();

        // GUI
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();
        ImGui::SetNextWindowSize(ImVec2(0, 0), ImGuiCond_Always);
        ImGui::Begin("Controls");

        // Mode dropdown
        if (ImGui::BeginCombo("Backend Mode", "Backend Mode"))
        {
            if (ImGui::Selectable("CPU", false))
            {
                backend = BackendMode::CPU;
            }
            if (ImGui::Selectable("GPU", false))
            {
                backend = BackendMode::GPU;
            }
            ImGui::EndCombo();
        }

        //Filter dropdown
        if (ImGui::BeginCombo("Image Filter", "Image Filter"))
        {
            if (ImGui::Selectable("None", false))
            {
                filter = ImageFilter::NONE;
            }
            if (ImGui::Selectable("Pencil", false))
            {
                filter = ImageFilter::PENCIL;
            }
            if (ImGui::Selectable("Retro", false))
            {
                filter = ImageFilter::RETRO;
            }
            ImGui::EndCombo();
        }

        // Screen dimensions
        ImGui::InputInt("Screen Width", &screenWidth);
        ImGui::InputInt("Screen Height", &screenHeight);

        // Interactive toggle
        ImGui::Checkbox("Interactive Mode", &isInteractive);

        // FPS
        ImGui::Text("Average FPS (%.1fs): %.2f", timeWindow, avgFPS);

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

        // Input handling
        if (isInteractive){
            double mouseX, mouseY;
            glfwGetCursorPos(window, &mouseX, &mouseY);
            if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_RIGHT) == GLFW_PRESS) {
                rightMouseDown = true;
            }
            else if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_RIGHT) == GLFW_RELEASE) {
                rightMouseDown = false;
            }

            if(glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS) {
                leftMouseDown = true;
            }
            else if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_RELEASE) {
                leftMouseDown = false;
            }

            if (rightMouseDown) {
                float deltaX = static_cast<float>(mouseX - lastMouseX);
                float deltaY = static_cast<float>(mouseY - lastMouseY);
                rotY += deltaX;
                rotX += deltaY;
            }

            if (leftMouseDown) {
                float deltaX = static_cast<float>(mouseX - lastMouseX);
                float deltaY = static_cast<float>(mouseY - lastMouseY);
                posX += deltaX;
                posY -= deltaY;
            }

            // Define scale as static/global so it can be accessed in the callback
            auto scroll_callback = [](GLFWwindow *window, double xoffset, double yoffset) {
                if (scale)
                {
                    if (yoffset > 0)
                        scale *= 1.1f;
                    else
                        scale /= 1.1f;
                }
            };
            glfwSetScrollCallback(window, scroll_callback);

            lastMouseX = mouseX;
            lastMouseY = mouseY;
        }

        cv::waitKey(40);
    }
    cap.release();
    cleanupShaderPrograms();
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    glfwTerminate();

    return 0;
}