cmake_minimum_required(VERSION 3.16.3)

project(arx5-sdk)

set(CMAKE_BUILD_TYPE Debug)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

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

# =========================
# Dependencies
# =========================
find_package(Threads REQUIRED)
find_package(Eigen3 REQUIRED)

find_package(orocos_kdl REQUIRED)
find_package(kdl_parser REQUIRED)
find_package(urdf REQUIRED)

find_package(fmt REQUIRED)
find_package(spdlog REQUIRED)
find_package(Boost REQUIRED COMPONENTS container)

add_compile_definitions(SDK_ROOT="${CMAKE_BINARY_DIR}/..")
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# =========================
# Architecture-specific closed-source hardware library
# =========================
if(CMAKE_SYSTEM_PROCESSOR MATCHES "arm.*|arm|aarch64")
    set(LIB_DIR ${CMAKE_CURRENT_SOURCE_DIR}/lib/aarch64)
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "(x86_64)|(AMD64)|(^i686)")
    set(LIB_DIR ${CMAKE_CURRENT_SOURCE_DIR}/lib/x86_64)
endif()

# =========================
# Includes
# =========================
include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${EIGEN3_INCLUDE_DIRS})

# ROS include dirs from packages
include_directories(${orocos_kdl_INCLUDE_DIRS})
include_directories(${kdl_parser_INCLUDE_DIRS})
include_directories(${urdf_INCLUDE_DIRS})

# Keep a direct ROS include fallback
include_directories(/opt/ros/noetic/include)

link_directories(/opt/ros/noetic/lib)

# Hard-pin ROS Noetic kdl_parser shared library
set(KDL_PARSER_LIBRARY "/opt/ros/noetic/lib/libkdl_parser.so")

message(STATUS "Using ROS KDL_PARSER_LIBRARY: ${KDL_PARSER_LIBRARY}")
message(STATUS "Using orocos_kdl_DIR: ${orocos_kdl_DIR}")
message(STATUS "Using kdl_parser_DIR: ${kdl_parser_DIR}")
message(STATUS "Using urdf_DIR: ${urdf_DIR}")
message(STATUS "orocos_kdl include dirs: ${orocos_kdl_INCLUDE_DIRS}")
message(STATUS "kdl_parser include dirs: ${kdl_parser_INCLUDE_DIRS}")
message(STATUS "urdf include dirs: ${urdf_INCLUDE_DIRS}")
message(STATUS "Using fmt_DIR: ${fmt_DIR}")
message(STATUS "Using spdlog_DIR: ${spdlog_DIR}")
message(STATUS "Using Boost_DIR: ${Boost_DIR}")

# =========================
# Open-source solver replacement
# =========================
add_library(ArxSolver SHARED
    src/app/solver.cpp
)

target_link_libraries(ArxSolver
    Eigen3::Eigen
    ${KDL_PARSER_LIBRARY}
    ${orocos_kdl_LIBRARIES}
    ${urdf_LIBRARIES}
)

target_include_directories(ArxSolver PUBLIC
    ${CMAKE_SOURCE_DIR}/include
    ${EIGEN3_INCLUDE_DIRS}
    ${orocos_kdl_INCLUDE_DIRS}
    ${kdl_parser_INCLUDE_DIRS}
    ${urdf_INCLUDE_DIRS}
    /opt/ros/noetic/include
)

# =========================
# Controllers
# =========================
add_library(ArxJointController SHARED
    src/app/joint_controller.cpp
    src/app/controller_base.cpp
    src/utils.cpp
)

target_link_libraries(ArxJointController
    ${LIB_DIR}/libhardware.so
    ArxSolver
    Eigen3::Eigen
    Threads::Threads
    spdlog::spdlog
    fmt::fmt
    ${KDL_PARSER_LIBRARY}
    ${orocos_kdl_LIBRARIES}
    ${urdf_LIBRARIES}
    soem
)

target_include_directories(ArxJointController PUBLIC
    ${CMAKE_SOURCE_DIR}/include
    ${EIGEN3_INCLUDE_DIRS}
    ${orocos_kdl_INCLUDE_DIRS}
    ${kdl_parser_INCLUDE_DIRS}
    ${urdf_INCLUDE_DIRS}
    /opt/ros/noetic/include
)

add_library(ArxCartesianController SHARED
    src/app/cartesian_controller.cpp
    src/app/controller_base.cpp
    src/utils.cpp
)

target_link_libraries(ArxCartesianController
    ${LIB_DIR}/libhardware.so
    ArxSolver
    Eigen3::Eigen
    Threads::Threads
    spdlog::spdlog
    fmt::fmt
    ${KDL_PARSER_LIBRARY}
    ${orocos_kdl_LIBRARIES}
    ${urdf_LIBRARIES}
    soem
)

target_include_directories(ArxCartesianController PUBLIC
    ${CMAKE_SOURCE_DIR}/include
    ${EIGEN3_INCLUDE_DIRS}
    ${orocos_kdl_INCLUDE_DIRS}
    ${kdl_parser_INCLUDE_DIRS}
    ${urdf_INCLUDE_DIRS}
    /opt/ros/noetic/include
)

# =========================
# Examples / tests
# =========================
add_executable(test_joint_controller
    examples/test_joint_controller.cpp
)

target_link_libraries(test_joint_controller
    ${LIB_DIR}/libhardware.so
    ArxSolver
    ArxJointController
    Eigen3::Eigen
    Threads::Threads
    spdlog::spdlog
    fmt::fmt
    ${KDL_PARSER_LIBRARY}
    ${orocos_kdl_LIBRARIES}
    ${urdf_LIBRARIES}
    soem
)

target_include_directories(test_joint_controller PUBLIC
    ${CMAKE_SOURCE_DIR}/include
    ${EIGEN3_INCLUDE_DIRS}
    ${orocos_kdl_INCLUDE_DIRS}
    ${kdl_parser_INCLUDE_DIRS}
    ${urdf_INCLUDE_DIRS}
    /opt/ros/noetic/include
)

add_executable(test_cartesian_controller
    examples/test_cartesian_controller.cpp
)

target_link_libraries(test_cartesian_controller
    ${LIB_DIR}/libhardware.so
    ArxSolver
    ArxCartesianController
    spdlog::spdlog
    fmt::fmt
    Eigen3::Eigen
    Threads::Threads
    ${KDL_PARSER_LIBRARY}
    ${orocos_kdl_LIBRARIES}
    ${urdf_LIBRARIES}
    soem
)

target_include_directories(test_cartesian_controller PUBLIC
    ${CMAKE_SOURCE_DIR}/include
    ${EIGEN3_INCLUDE_DIRS}
    ${orocos_kdl_INCLUDE_DIRS}
    ${kdl_parser_INCLUDE_DIRS}
    ${urdf_INCLUDE_DIRS}
    /opt/ros/noetic/include
)

# =========================
# Python bindings
# =========================
add_subdirectory(python)

# =========================
# Install
# =========================
install(DIRECTORY ${CMAKE_SOURCE_DIR}/include/
    DESTINATION ${CMAKE_INSTALL_PREFIX}/include/${PROJECT_NAME}
)

install(FILES ${LIB_DIR}/libhardware.so
    DESTINATION ${CMAKE_INSTALL_PREFIX}/lib/${PROJECT_NAME}
)

install(TARGETS
    ArxSolver
    ArxJointController
    ArxCartesianController
    LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/lib/${PROJECT_NAME}
)