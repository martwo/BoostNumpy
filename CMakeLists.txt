cmake_minimum_required(VERSION 2.8.3)

# Set BoostNumpy_BINARY_DIR and BoostNumpy_SOURCE_DIR cmake variables.
project(BoostNumpy CXX)

message(STATUS "Configuring project \"${PROJECT_NAME}\"")
message(STATUS "+    project source dir: ${PROJECT_SOURCE_DIR}")

set(BOOSTNUMPY_CONFIGURED_SUCCESSFULLY TRUE
    CACHE BOOL "BoostNumpy configured successfully." FORCE)

include(${CMAKE_SOURCE_DIR}/cmake/config.cmake)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib/BoostNumpy-${BOOSTNUMPY_VERSION_STRING}
    CACHE PATH "Output directory for static libraries." FORCE)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib/BoostNumpy-${BOOSTNUMPY_VERSION_STRING}
    CACHE PATH "Output directory for shared libraries." FORCE)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/BoostNumpy-${BOOSTNUMPY_VERSION_STRING}
    CACHE PATH "Output directory for executables and DLL's." FORCE)

include(${CMAKE_SOURCE_DIR}/cmake/tools/python.cmake)
include(${CMAKE_SOURCE_DIR}/cmake/tools/NumPy.cmake)
include(${CMAKE_SOURCE_DIR}/cmake/tools/boost.cmake)

if((NOT PYTHON_FOUND) OR (NOT NUMPY_FOUND) OR (NOT BOOST_FOUND))
    set(BOOSTNUMPY_CONFIGURED_SUCCESSFULLY FALSE
        CACHE BOOL "BoostNumpy configured successfully." FORCE)
endif((NOT PYTHON_FOUND) OR (NOT NUMPY_FOUND) OR (NOT BOOST_FOUND))

if(BOOSTNUMPY_CONFIGURED_SUCCESSFULLY)

    list(APPEND ${PROJECT_NAME}_libboostnumpy_INCLUDE_DIRS
        ${PYTHON_INCLUDE_DIRS}
        ${NUMPY_INCLUDE_DIR}
        ${Boost_INCLUDE_DIR}
        ${PROJECT_SOURCE_DIR}/include
        ${PROJECT_SOURCE_DIR}/src
    )

    list(APPEND ${PROJECT_NAME}_libboostnumpy_SOURCE_FILES
        src/boost/numpy/detail/iter.cpp
        src/boost/numpy/dtype.cpp
        src/boost/numpy/matrix.cpp
        src/boost/numpy/ndarray.cpp
        src/boost/numpy/numpy.cpp
        src/boost/numpy/scalars.cpp
    )

    list(APPEND ${PROJECT_NAME}_libboostnumpy_LIBRARIES
        ${PYTHON_LIBRARIES}
        ${Boost_LIBRARIES}
    )

    include_directories(
        ${${PROJECT_NAME}_libboostnumpy_INCLUDE_DIRS}
    )
    add_library(boostnumpy
        ${${PROJECT_NAME}_libboostnumpy_SOURCE_FILES}
    )
    target_link_libraries(boostnumpy
        ${${PROJECT_NAME}_libboostnumpy_LIBRARIES}
    )
    get_target_property(${PROJECT_NAME}_libboostnumpy_LIBRARY_OUTPUT_DIRECTORY
        boostnumpy LIBRARY_OUTPUT_DIRECTORY)
    add_custom_command(TARGET boostnumpy
        PRE_LINK
        COMMAND mkdir -p ${${PROJECT_NAME}_libboostnumpy_LIBRARY_OUTPUT_DIRECTORY}
    )

    # Install all the public header files into the include directory.
    install(DIRECTORY ${PROJECT_SOURCE_DIR}/include/
        DESTINATION ${CMAKE_INSTALL_PREFIX}/include/BoostNumpy-${BOOSTNUMPY_VERSION_STRING}
        FILES_MATCHING PATTERN "*.hpp"
        PATTERN ".svn" EXCLUDE
    )

    # Install the shared libraries into the library directory.
    install(TARGETS boostnumpy
        LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/lib/BoostNumpy-${BOOSTNUMPY_VERSION_STRING}
    )

    message(STATUS "Project \"${PROJECT_NAME}\" configured.")
    message(STATUS "+    Now, type-in \"cd ${CMAKE_BINARY_DIR} && make && make install\"!")
    message(STATUS "+    \"${PROJECT_NAME}\" will be installed in ${CMAKE_INSTALL_PREFIX}.")

endif(BOOSTNUMPY_CONFIGURED_SUCCESSFULLY)