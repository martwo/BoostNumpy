# CMake find script to search for the Boost library.
function(find_boost)
    set(Boost_USE_STATIC_LIBS OFF)
    set(Boost_USE_MULTITHREADED ON)
    set(Boost_USE_STATIC_RUNTIME OFF)

    find_package(Boost "1.38.0" REQUIRED python system thread)

    if(NOT Boost_FOUND)
        set(BOOST_FOUND FALSE CACHE BOOL "Boost found" FORCE)
        message(STATUS "Error: Boost not found!")
        return()
    endif()

    set(BOOST_FOUND TRUE
        CACHE BOOL "Boost found" FORCE)
    set(BOOST_INCLUDE_DIRS "${Boost_INCLUDE_DIR}"
        CACHE PATH "Boost include directory." FORCE)
    set(BOOST_LIBRARIES "${Boost_LIBRARIES}"
        CACHE STRING "Boost libraries." FORCE)

    message(STATUS "Found boost.")
    message(STATUS "+    include paths: ${BOOST_INCLUDE_DIRS}")
    message(STATUS "+    libraries: ${BOOST_LIBRARIES}")

endfunction(find_boost)
