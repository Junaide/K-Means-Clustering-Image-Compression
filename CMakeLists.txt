cmake_minimum_required(VERSION 3.10)
project(opencv_project)

# Set OpenCV_DIR to the directory containing OpenCVConfig.cmake
set(OpenCV_DIR "C:/opencv/build")  # Adjust to your OpenCV installation path

# Find OpenCV package
find_package(OpenCV REQUIRED)

# Include directories
include_directories(${CMAKE_SOURCE_DIR}/include)  # Include directory for your headers
include_directories(${OpenCV_INCLUDE_DIRS})      # Include OpenCV headers

# Add all source files
add_executable(opencv_project 
    src/main.cpp
    src/kmeans.cpp  # Include the kmeans implementation if it exists
)

# Link OpenCV libraries
target_link_libraries(opencv_project ${OpenCV_LIBS})
