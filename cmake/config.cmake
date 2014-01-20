message(STATUS "Entering 'config.cmake'")

set(BUILD_SHARED_LIBS TRUE)

add_definitions(-fPIC)

link_libraries(stdc++)

set(BOOSTNUMPY_VERSION_STRING "1.0.0" CACHE STRING "The BoostNumpy version." FORCE)

message(STATUS "+    BOOSTNUMPY_VERSION: ${BOOSTNUMPY_VERSION_STRING}")
message(STATUS "+    CMAKE_INSTALL_PREFIX: ${CMAKE_INSTALL_PREFIX}")
