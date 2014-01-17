cmake_minimum_required(VERSION 2.8.3)

# Set BoostNumpy_BINARY_DIR and BoostNumpy_SOURCE_DIR cmake variables.
project(BoostNumpy CXX)

# Put our local cmake find scripts at the beginning of the cmake
# module search path.
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake/tools ${CMAKE_MODULE_PATH})

find_package(PythonInterp REQUIRED)
find_package(PythonLibs ${PYTHON_VERSION_STRING} REQUIRED)
find_package(NumPy REQUIRED)