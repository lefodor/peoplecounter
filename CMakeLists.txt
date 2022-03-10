cmake_minimum_required(VERSION 3.16.3)

project(peoplecounter)

set(OpenCV_DIR "~/opencv/build")

set(ENV{PKG_CONFIG_PATH} /usr/lib/x86_x64-linux-gnu/pkgconfig)
find_package(PkgConfig REQUIRED)
pkg_check_modules(OPENCV REQUIRED opencv4)
find_package(OpenCV REQUIRED COMPONENTS core imgproc highgui)

if(${OPENCV_FOUND})
    MESSAGE(${PKG_CONFIG_EXECUTABLE})
    MESSAGE(${OPENCV_LIBRARIES})
    MESSAGE(${OPENCV_LINK_LIBRARIES})
endif()

add_executable(${PROJECT_NAME} main.cpp)
target_link_libraries(peoplecounter ${OpenCV_LIBS})