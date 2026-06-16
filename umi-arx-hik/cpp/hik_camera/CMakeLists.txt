cmake_minimum_required(VERSION 3.14)
project(hik_camera_cpp)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(BUILD_PYTHON_MODULE "Build pybind11 Python module" ON)
option(BUILD_TEST_MAIN "Build standalone C++ test executable" ON)
option(BUILD_HIK_SERVER "Build Hik shared-memory server" ON)

include_directories(/opt/MVS/include)
link_directories(/opt/MVS/lib/64)

find_package(OpenCV REQUIRED)

if(BUILD_PYTHON_MODULE)
    find_package(pybind11 REQUIRED)

    pybind11_add_module(hik_camera_cpp
        hik_camera_core.cpp
        pybind_module.cpp
    )

    target_include_directories(hik_camera_cpp PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${OpenCV_INCLUDE_DIRS}
    )

    target_link_libraries(hik_camera_cpp PRIVATE
        ${OpenCV_LIBS}
        MvCameraControl
        pthread
    )
endif()

if(BUILD_TEST_MAIN)
    add_executable(test_hik_main
        hik_camera_core.cpp
        test_hik_main.cpp
    )

    target_include_directories(test_hik_main PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${OpenCV_INCLUDE_DIRS}
    )

    target_link_libraries(test_hik_main PRIVATE
        ${OpenCV_LIBS}
        MvCameraControl
        pthread
    )
endif()

if(BUILD_HIK_SERVER)
    add_executable(hik_server
        hik_camera_core.cpp
        hik_server_main.cpp
    )

    target_include_directories(hik_server PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${OpenCV_INCLUDE_DIRS}
    )

    target_link_libraries(hik_server PRIVATE
        ${OpenCV_LIBS}
        MvCameraControl
        pthread
        rt
    )
endif()