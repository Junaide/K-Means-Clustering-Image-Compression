cmake_minimum_required(VERSION 3.10)
project(opencv_project)

# Set OpenCV_DIR to the directory containing OpenCVConfig.cmake
set(OpenCV_DIR "C:/opencv/build")  # Adjust to your OpenCV installation path

# Find OpenCV package
find_package(OpenCV REQUIRED)

# Add OpenCV include directories
include_directories(${OpenCV_INCLUDE_DIRS})

# Add the executable
add_executable(opencv_project src/main.cpp)

# Link OpenCV libraries
target_link_libraries(opencv_project ${OpenCV_LIBS})