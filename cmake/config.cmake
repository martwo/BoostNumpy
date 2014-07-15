message(STATUS "Entering 'config.cmake'")

set(BUILD_SHARED_LIBS TRUE)

add_definitions(-fPIC)

link_libraries(stdc++)

set(BOOST_NUMPY_VERSION_STRING "0.1.1" CACHE STRING "The boost_numpy version." FORCE)

message(STATUS "+    BOOST_NUMPY_VERSION: ${BOOST_NUMPY_VERSION_STRING}")
message(STATUS "+    CMAKE_INSTALL_PREFIX: ${CMAKE_INSTALL_PREFIX}")
