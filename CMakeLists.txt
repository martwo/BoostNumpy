cmake_minimum_required(VERSION 2.8.3)

# Set BoostNumpy_BINARY_DIR and BoostNumpy_SOURCE_DIR cmake variables.
project(BoostNumpy CXX)

message(STATUS "Configuring project \"${PROJECT_NAME}\"")
message(STATUS "+    project source dir: ${PROJECT_SOURCE_DIR}")


# Put our local cmake find scripts at the beginning of the cmake
# module search path.
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake/tools ${CMAKE_MODULE_PATH})

set(BUILD_SHARED_LIBS TRUE)

find_package(PythonInterp REQUIRED)
find_package(PythonLibs ${PYTHON_VERSION_STRING} REQUIRED)
find_package(NumPy REQUIRED)
find_package(boost REQUIRED)

list(APPEND ${PROJECT_NAME}_LIB_SOURCE_FILES
    src/boost/numpy/detail/iter.cpp
    src/boost/numpy/dtype.cpp
    src/boost/numpy/matrix.cpp
    src/boost/numpy/ndarray.cpp
    src/boost/numpy/numpy.cpp
    src/boost/numpy/scalars.cpp
)

#add_library(boostnumpy

#)
