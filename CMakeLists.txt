cmake_minimum_required(VERSION 2.8.3)

# Choose CMAKE_BUILD_TYPE=Release if the user does not specify it.
if(DEFINED CMAKE_BUILD_TYPE)
   set(CMAKE_BUILD_TYPE ${CMAKE_BUILD_TYPE} CACHE STRING "Choose the type of build, options are: None Debug Release RelWithDebInfo MinSizeRel.")
else()
   set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build, options are: None Debug Release RelWithDebInfo MinSizeRel.")
endif()

# Set BoostNumpy_BINARY_DIR and BoostNumpy_SOURCE_DIR cmake variables.
project(BoostNumpy CXX)

message(STATUS "Configuring project \"${PROJECT_NAME}\"")
message(STATUS "+    project source dir: ${PROJECT_SOURCE_DIR}")

set(BOOSTNUMPY_CONFIGURED_SUCCESSFULLY TRUE
    CACHE BOOL "${PROJECT_NAME} configured successfully." FORCE)

include(${CMAKE_SOURCE_DIR}/cmake/config.cmake)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
    CACHE PATH "Output directory for shared libraries." FORCE)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
    CACHE PATH "Output directory for executables like tests.")

include(${CMAKE_SOURCE_DIR}/cmake/add_python_module.cmake)
include(${CMAKE_SOURCE_DIR}/cmake/tools/boost.cmake)
include(${CMAKE_SOURCE_DIR}/cmake/tools/python.cmake)
include(${CMAKE_SOURCE_DIR}/cmake/tools/NumPy.cmake)
include(${CMAKE_SOURCE_DIR}/cmake/tools/sphinx.cmake)

find_boost()
find_python()
find_numpy()
find_sphinx()

if((NOT PYTHON_FOUND) OR (NOT NUMPY_FOUND) OR (NOT BOOST_FOUND))
    set(BOOSTNUMPY_CONFIGURED_SUCCESSFULLY FALSE
        CACHE BOOL "${PROJECT_NAME} configured successfully." FORCE)
endif((NOT PYTHON_FOUND) OR (NOT NUMPY_FOUND) OR (NOT BOOST_FOUND))

if(BOOSTNUMPY_CONFIGURED_SUCCESSFULLY)

    list(APPEND ${PROJECT_NAME}_libboost_numpy_INCLUDE_DIRS
        ${PYTHON_INCLUDE_DIRS}
        ${NUMPY_INCLUDE_DIRS}
        ${BOOST_INCLUDE_DIRS}
        ${PROJECT_SOURCE_DIR}/include
        ${PROJECT_SOURCE_DIR}/src
    )

    list(APPEND ${PROJECT_NAME}_libboost_numpy_SOURCE_FILES
        src/boost/numpy/detail/iter.cpp
        src/boost/numpy/detail/logging.cpp
        src/boost/numpy/dtype.cpp
        src/boost/numpy/matrix.cpp
        src/boost/numpy/ndarray.cpp
        src/boost/numpy/numpy.cpp
        src/boost/numpy/scalars.cpp
    )

    list(APPEND ${PROJECT_NAME}_libboost_numpy_LIBRARIES
        ${PYTHON_LIBRARIES}
        ${Boost_LIBRARIES}
    )

    include_directories(
        ${${PROJECT_NAME}_libboost_numpy_INCLUDE_DIRS}
    )
    add_library(boost_numpy
        ${${PROJECT_NAME}_libboost_numpy_SOURCE_FILES}
    )
    target_link_libraries(boost_numpy
        ${${PROJECT_NAME}_libboost_numpy_LIBRARIES}
    )
    get_target_property(${PROJECT_NAME}_libboost_numpy_LIBRARY_OUTPUT_DIRECTORY
        boost_numpy LIBRARY_OUTPUT_DIRECTORY)
    add_custom_command(TARGET boost_numpy
        PRE_LINK
        COMMAND mkdir -p ${${PROJECT_NAME}_libboost_numpy_LIBRARY_OUTPUT_DIRECTORY}
    )

    # Install all the public header files into the include directory.
    install(DIRECTORY ${PROJECT_SOURCE_DIR}/include/boost
        DESTINATION ${CMAKE_INSTALL_PREFIX}/include/
        FILES_MATCHING PATTERN "*.hpp"
        PATTERN ".svn" EXCLUDE
    )

    # Install the shared libraries into the library directory.
    install(TARGETS boost_numpy
        LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/lib
    )

    add_subdirectory(doc)

    enable_testing()
    add_subdirectory(test)

    message(STATUS "Project \"${PROJECT_NAME}\" configured.")
    message(STATUS "+    Now, type-in \"cd ${CMAKE_BINARY_DIR} && make && make install\"!")
    message(STATUS "+    \"${PROJECT_NAME}\" will be installed in ${CMAKE_INSTALL_PREFIX}.")

endif(BOOSTNUMPY_CONFIGURED_SUCCESSFULLY)
