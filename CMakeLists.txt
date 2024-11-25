cmake_minimum_required(VERSION 3.10)
project(kmeans_compression)


set(OpenCV_DIR "C:/opencv/build") 

# Find OpenCV package
find_package(OpenCV REQUIRED)

# Include directories
include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${OpenCV_INCLUDE_DIRS})     

# Add all source files
add_executable(kmeans_compression 
    src/main.cpp
    src/kmeans.cpp
)

# Link OpenCV libraries
target_link_libraries(kmeans_compression ${OpenCV_LIBS})
