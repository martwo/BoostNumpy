# CMake find script to search for the Boost library.

set(Boost_USE_STATIC_LIBS OFF)
set(Boost_USE_MULTITHREADED ON)
set(Boost_USE_STATIC_RUNTIME OFF)

find_package(Boost "1.38.0" REQUIRED python system thread)

if(Boost_FOUND)
    message(STATUS "Boost found.")
    message(STATUS "+    include path: ${Boost_INCLUDE_DIR}")
    message(STATUS "+    libraries: ${Boost_LIBRARIES}")
endif(Boost_FOUND)
