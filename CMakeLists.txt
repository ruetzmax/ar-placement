# Set the minimum required version of CMake
cmake_minimum_required(VERSION 3.10)

# Define the project name
project(OpenCV_Project)

set(CMAKE_INCLUDE_DIRECTORIES_PROJECT_BEFORE ON)

# Locate the OpenCV library package
# The REQUIRED flag ensures that CMake will error if OpenCV is not found.
find_package(OpenCV REQUIRED)
find_package(OpenGL REQUIRED)
find_package(GLEW REQUIRED)

find_package(PkgConfig REQUIRED)
pkg_search_module(GLFW REQUIRED glfw3)

# Add the OpenCV include directories to the compiler’s search path.
# This is necessary for resolving #include <opencv2/...> directives.
include_directories(${OpenCV_INCLUDE_DIRS})

# Define the executable target and its source files.
set(IMGUI_SOURCES
	${CMAKE_SOURCE_DIR}/imgui/imgui.cpp
	${CMAKE_SOURCE_DIR}/imgui/imgui_draw.cpp
	${CMAKE_SOURCE_DIR}/imgui/imgui_tables.cpp
	${CMAKE_SOURCE_DIR}/imgui/imgui_widgets.cpp
	${CMAKE_SOURCE_DIR}/imgui/imgui_demo.cpp
	${CMAKE_SOURCE_DIR}/imgui/backends/imgui_impl_glfw.cpp
	${CMAKE_SOURCE_DIR}/imgui/backends/imgui_impl_opengl3.cpp
)

add_library(imgui_static STATIC ${IMGUI_SOURCES})
target_include_directories(imgui_static PUBLIC ${CMAKE_SOURCE_DIR}/imgui ${CMAKE_SOURCE_DIR}/imgui/backends)
target_link_libraries(imgui_static PUBLIC glfw OpenGL::GL GLEW::GLEW)

# Define the executable target and its source files.
add_executable(${PROJECT_NAME} main.cpp gpu_transforms.cpp gpu_transforms.h)

# Link the executable against the required OpenCV libraries and ImGui.
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS} OpenGL::GL GLEW::GLEW glfw imgui_static)