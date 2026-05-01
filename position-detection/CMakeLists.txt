cmake_minimum_required(VERSION 3.10)
project(position_detector LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# we love windows, if your paths are different, change here
if (WIN32)
    set(OpenCV_DIR "C:/opencv/opencv/build")
    set(realsense2_DIR "C:/librealsense/install/lib/cmake/realsense2")
    set(APRILTAG_DIR "C:/apriltag/apriltag/install")
    list(APPEND CMAKE_PREFIX_PATH "C:/vcpkg/installed/x64-windows")
    include_directories(${APRILTAG_DIR}/include)
    link_directories(${APRILTAG_DIR}/lib)
else()
    find_package(fastrtps REQUIRED)
    find_package(fastcdr REQUIRED)
endif()

if(MSVC)
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /O2 /Ob2 /DNDEBUG")
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_BINARY_DIR}/Release)
else()
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -DNDEBUG")
endif()

# this is fine if you have a normalish linux install
find_package(OpenCV REQUIRED)
find_package(realsense2 REQUIRED)
find_package(Eigen3 CONFIG REQUIRED)

add_executable(position_detector
    src/main.cpp
)

target_link_libraries(position_detector
    ${OpenCV_LIBS}
    realsense2::realsense2
    apriltag
    Eigen3::Eigen
)