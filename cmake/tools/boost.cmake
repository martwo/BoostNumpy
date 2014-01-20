# CMake find script to search for the Boost library.

set(Boost_USE_STATIC_LIBS OFF)
set(Boost_USE_MULTITHREADED ON)
set(Boost_USE_STATIC_RUNTIME OFF)

find_package(Boost "1.38.0" REQUIRED python system thread)

if(Boost_FOUND)
    set(BOOST_FOUND TRUE CACHE BOOL "Boost found" FORCE)
    set(BOOST_INCLUDE_DIR ${Boost_INCLUDE_DIR} CACHE PATH "Boost include directory." FORCE)
    set(BOOST_LIBRARIES ${Boost_LIBRARIES} CACHE STRING "Boost libraries." FORCE)

    message(STATUS "Boost found.")
    message(STATUS "+    include path: ${BOOST_INCLUDE_DIR}")
    message(STATUS "+    libraries: ${BOOST_LIBRARIES}")
else(Boost_FOUND)
    set(BOOST_FOUND FALSE CACHE BOOL "Boost found" FORCE)

    message(STATUS "Error: Boost not found!")
endif(Boost_FOUND)
