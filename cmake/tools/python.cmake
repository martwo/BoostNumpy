function(find_python)

    find_package(PythonInterp REQUIRED)

    if(NOT PYTHONINTERP_FOUND)
        set(PYTHON_FOUND FALSE CACHE BOOL "Python found." FORCE)
        return()
    endif()

    set(PYTHON_FOUND TRUE CACHE BOOL "Python found." FORCE)

    # Now search for the Python libraries.
    execute_process(COMMAND "${PYTHON_EXECUTABLE}" "-c"
        "from distutils import sysconfig as s; import sys; import struct;
print('.'.join(str(v) for v in sys.version_info));
print(s.PREFIX);
print(s.get_python_inc(plat_specific=True));
print(s.get_python_lib(plat_specific=True));
print(s.get_config_var('SO'));
print(hasattr(sys, 'gettotalrefcount')+0);
print(struct.calcsize('@P'));
"
        RESULT_VARIABLE _PYTHON_SUCCESS
        OUTPUT_VARIABLE _PYTHON_VALUES
        ERROR_VARIABLE _PYTHON_ERROR_VALUE
        OUTPUT_STRIP_TRAILING_WHITESPACE)

    if(NOT _PYTHON_SUCCESS MATCHES 0)
        message(FATAL_ERROR
            "Python config failure:\n${_PYTHON_ERROR_VALUE}")
        set(PYTHON_FOUND FALSE CACHE BOOL "Python found." FORCE)
        return()
    endif()

    # Convert the process output into a list
    string(REGEX REPLACE ";" "\\\\;" _PYTHON_VALUES ${_PYTHON_VALUES})
    string(REGEX REPLACE "\n" ";" _PYTHON_VALUES ${_PYTHON_VALUES})
    list(GET _PYTHON_VALUES 0 _PYTHON_VERSION_LIST)
    list(GET _PYTHON_VALUES 1 PYTHON_PREFIX)
    list(GET _PYTHON_VALUES 2 PYTHON_INCLUDE_DIR)
    list(GET _PYTHON_VALUES 3 PYTHON_SITE_PACKAGES)
    list(GET _PYTHON_VALUES 4 PYTHON_MODULE_EXTENSION)
    list(GET _PYTHON_VALUES 5 PYTHON_IS_DEBUG)
    list(GET _PYTHON_VALUES 6 PYTHON_SIZEOF_VOID_P)

    # Make sure the Python has the same pointer-size as the chosen compiler.
    if(NOT ${PYTHON_SIZEOF_VOID_P} MATCHES ${CMAKE_SIZEOF_VOID_P})
        math(EXPR _PYTHON_BITS "${PYTHON_SIZEOF_VOID_P} * 8")
        math(EXPR _CMAKE_BITS "${CMAKE_SIZEOF_VOID_P} * 8")
        message(FATAL_ERROR
            "Python config failure: Python is ${_PYTHON_BITS}-bit, "
            "chosen compiler is ${_CMAKE_BITS}-bit")
        set(PYTHON_FOUND FALSE CACHE BOOL "Python found." FORCE)
        return()
    endif()

    # The built-in FindPython didn't always give the version numbers
    string(REGEX REPLACE "\\." ";" _PYTHON_VERSION_LIST ${_PYTHON_VERSION_LIST})
    list(GET _PYTHON_VERSION_LIST 0 PYTHON_VERSION_MAJOR)
    list(GET _PYTHON_VERSION_LIST 1 PYTHON_VERSION_MINOR)
    list(GET _PYTHON_VERSION_LIST 2 PYTHON_VERSION_PATCH)

    # Make sure all directory separators are '/'
    string(REGEX REPLACE "\\\\" "/" PYTHON_PREFIX ${PYTHON_PREFIX})
    string(REGEX REPLACE "\\\\" "/" PYTHON_INCLUDE_DIR ${PYTHON_INCLUDE_DIR})
    string(REGEX REPLACE "\\\\" "/" PYTHON_SITE_PACKAGES ${PYTHON_SITE_PACKAGES})

    if(${PYTHON_SIZEOF_VOID_P} MATCHES 8)
        set(_PYTHON_LIBS_SEARCH "${PYTHON_PREFIX}/lib64" "${PYTHON_PREFIX}/lib")
    else()
        set(_PYTHON_LIBS_SEARCH "${PYTHON_PREFIX}/lib")
    endif()
    # Probably this needs to be more involved. It would be nice if the config
    # information the python interpreter itself gave us were more complete.
    find_library(PYTHON_LIBRARY
        NAMES "python${PYTHON_VERSION_MAJOR}.${PYTHON_VERSION_MINOR}"
        PATHS ${_PYTHON_LIBS_SEARCH}
        NO_SYSTEM_ENVIRONMENT_PATH)

    MARK_AS_ADVANCED(
        PYTHON_LIBRARY
        PYTHON_INCLUDE_DIR
    )

    if(PYTHON_FOUND)
        set(PYTHON_PREFIX "${PYTHON_PREFIX}"
            CACHE PATH "Python prefix directory." FORCE)
        set(PYTHON_INCLUDE_DIRS "${PYTHON_INCLUDE_DIR}"
            CACHE PATH "Python include directories." FORCE)
        set(PYTHON_LIBRARIES "${PYTHON_LIBRARY}"
            CACHE FILEPATH "Python library files." FORCE)
        set(PYTHON_DEBUG_LIBRARIES "${PYTHON_DEBUG_LIBRARY}"
            CACHE FILEPATH "Python debug library files." FORCE)
        set(PYTHON_MODULE_EXTENSION "${PYTHON_MODULE_EXTENSION}"
            CACHE STRING "The Python module extension." FORCE)
        set(PYTHON_VERSION_MAJOR "${PYTHON_VERSION_MAJOR}"
            CACHE STRING "The Python major version number." FORCE)
        set(PYTHON_VERSION_MINOR "${PYTHON_VERSION_MINOR}"
            CACHE STRING "The Python minor version number." FORCE)
        set(PYTHON_VERSION_PATCH "${PYTHON_VERSION_PATCH}"
            CACHE STRING "The Python path version number." FORCE)

        message(STATUS "Found Python.")
        message(STATUS "+    Python include dirs: ${PYTHON_INCLUDE_DIRS}")
    endif()

endfunction(find_python)
