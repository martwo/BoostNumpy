# Find cmake script to search for the numpy Python package.
function(find_numpy)
    set(NUMPY_FOUND FALSE CACHE BOOL "NumPy found." FORCE)

    if(PYTHON_FOUND)
        execute_process(COMMAND ${PYTHON_EXECUTABLE} -c "import numpy"
            RESULT_VARIABLE _NUMPY_FOUND)

        if(_NUMPY_FOUND EQUAL 0)
            set(NUMPY_FOUND TRUE CACHE BOOL "NumPy found." FORCE)
        endif(_NUMPY_FOUND EQUAL 0)

        if(NUMPY_FOUND)
            message(STATUS "Numpy found.")
            execute_process(COMMAND ${PYTHON_EXECUTABLE} -c
                "import numpy; print(numpy.get_include())"
                OUTPUT_VARIABLE _NUMPY_INCLUDE_DIR
                OUTPUT_STRIP_TRAILING_WHITESPACE)

            # Look in some other places, too. This should make it
            # work on OS X, where the headers are in SDKs within XCode.app,
            # but python reports them as being available at /.
            set(NUMPY_INCLUDE_DIR_CANDIDATES ${_NUMPY_INCLUDE_DIR})
            foreach(prefix ${CMAKE_PREFIX_PATH})
                list(APPEND NUMPY_INCLUDE_DIR_CANDIDATES ${prefix}/${_NUMPY_INCLUDE_DIR})
                list(APPEND NUMPY_INCLUDE_DIR_CANDIDATES ${prefix}/../${_NUMPY_INCLUDE_DIR})
            endforeach(prefix ${CMAKE_PREFIX_PATH})
            foreach(prefix ${CMAKE_FRAMEWORK_PATH})
                list(APPEND NUMPY_INCLUDE_DIR_CANDIDATES ${prefix}/${_NUMPY_INCLUDE_DIR})
                list(APPEND NUMPY_INCLUDE_DIR_CANDIDATES ${prefix}/../../../${_NUMPY_INCLUDE_DIR})
            endforeach(prefix ${CMAKE_FRAMEWORK_PATH})

            find_path(NUMPY_INCLUDE_DIR
                NAMES numpy/ndarrayobject.h
                PATHS ${NUMPY_INCLUDE_DIR_CANDIDATES})

            set(NUMPY_INCLUDE_DIRS "${NUMPY_INCLUDE_DIR}"
                CACHE PATH "The numpy include directories." FORCE)
            message(STATUS "+    include paths: ${NUMPY_INCLUDE_DIRS}")

        endif(NUMPY_FOUND)
    endif(PYTHON_FOUND)

endfunction(find_numpy)
