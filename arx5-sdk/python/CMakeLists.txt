cmake_minimum_required(VERSION 3.16.3)

# =========================
# Runtime: prefer ROS Noetic shared libraries
# =========================
set(CMAKE_SKIP_RPATH FALSE)
set(CMAKE_BUILD_RPATH "/opt/ros/noetic/lib")
set(CMAKE_INSTALL_RPATH "/opt/ros/noetic/lib")

# =========================
# Force ROS Noetic package configs for KDL/URDF stack
# =========================
set(orocos_kdl_DIR "/opt/ros/noetic/share/orocos_kdl/cmake")
set(kdl_parser_DIR "/opt/ros/noetic/share/kdl_parser/cmake")
set(urdf_DIR "/opt/ros/noetic/share/urdf/cmake")

# =========================
# Conda-provided packages we still want to use
# =========================
if(DEFINED ENV{CONDA_PREFIX})
    set(fmt_DIR "$ENV{CONDA_PREFIX}/lib/cmake/fmt")
    set(spdlog_DIR "$ENV{CONDA_PREFIX}/lib/cmake/spdlog")
    set(Boost_DIR "$ENV{CONDA_PREFIX}/lib/cmake/Boost-1.85.0")
endif()

find_package(pybind11 REQUIRED)
find_package(Threads REQUIRED)
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
find_package(Eigen3 REQUIRED)

find_package(orocos_kdl REQUIRED)
find_package(kdl_parser REQUIRED)
find_package(urdf REQUIRED)

find_package(fmt REQUIRED)
find_package(spdlog REQUIRED)
find_package(Boost REQUIRED COMPONENTS container)

set(KDL_PARSER_LIBRARY "/opt/ros/noetic/lib/libkdl_parser.so")
link_directories(/opt/ros/noetic/lib)

pybind11_add_module(arx5_interface
    arx5_pybind.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/app/joint_controller.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/app/cartesian_controller.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/app/controller_base.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/utils.cpp
)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../include/libcan)
include_directories(${EIGEN3_INCLUDE_DIRS})
include_directories(${orocos_kdl_INCLUDE_DIRS})
include_directories(${kdl_parser_INCLUDE_DIRS})
include_directories(${urdf_INCLUDE_DIRS})
include_directories(/opt/ros/noetic/include)

add_compile_options(-pthread)

if(CMAKE_SYSTEM_PROCESSOR MATCHES "arm.*|arm|aarch64")
    set(LIB_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../lib/aarch64)
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "(x86_64)|(AMD64)|(^i686)")
    set(LIB_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../lib/x86_64)
endif()

target_link_libraries(arx5_interface PRIVATE
    ${LIB_DIR}/libhardware.so
    ArxSolver
    spdlog::spdlog
    fmt::fmt
    Eigen3::Eigen
    Threads::Threads
    ${KDL_PARSER_LIBRARY}
    ${orocos_kdl_LIBRARIES}
    ${urdf_LIBRARIES}
    pthread
    soem
)

target_include_directories(arx5_interface PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
    ${EIGEN3_INCLUDE_DIRS}
    ${orocos_kdl_INCLUDE_DIRS}
    ${kdl_parser_INCLUDE_DIRS}
    ${urdf_INCLUDE_DIRS}
    /opt/ros/noetic/include
)

set_target_properties(arx5_interface PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    PREFIX ""
    OUTPUT_NAME "arx5_interface"
)