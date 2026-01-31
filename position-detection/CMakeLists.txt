cmake_minimum_required(VERSION 3.10)
project(position_detector LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQQUIRED ON)

find_package(OpenCV REQUIRED)
find_package(fastrtps REQUIRED)
find_package(fastcdr REQUIRED)
find_package(realsense2 REQUIRED)

add_executable(position_detector
    src/main.cpp
)

target_link_libraries(position_detector
    ${OpenCV_LIBS}
    realsense2::realsense2
    apriltag
)