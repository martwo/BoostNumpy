
# Assume that BoostNumpy is installed inside the I3_PORTS tree.
set(BOOSTNUMPY_PREFIX ${I3_PORTS})

# Glob all BoostNumpy include directories available and take the one with the
# highest version.
file(GLOB _BOOSTNUMPY_INCLUDE_DIR ${BOOSTNUMPY_PREFIX}/include/BoostNumpy-*.*.*)
list(SORT _BOOSTNUMPY_INCLUDE_DIR)
list(REVERSE _BOOSTNUMPY_INCLUDE_DIR)
list(GET _BOOSTNUMPY_INCLUDE_DIR 0 BOOSTNUMPY_INCLUDE_DIR)

if("${BOOSTNUMPY_INCLUDE_DIR}" STREQUAL "")
    colormsg(WHITE "")
    colormsg(HICYAN "boostnumpy")
    colormsg(CYAN "- Error: Not found in I3_PORTS '${I3_PORTS}'!")
    set(BOOSTNUMPY_FOUND FALSE
        CACHE BOOL "boostnumpy found successfully." FORCE)
else("${BOOSTNUMPY_INCLUDE_DIR}" STREQUAL "")
    # Get the version of the detected BoostNumpy tool.
    string(REGEX MATCH "BoostNumpy-([0-9]+)\\.([0-9]+)\\.?([0-9]*)"
        _BOOSTNUMPY_VERSION_STRING
        ${BOOSTNUMPY_INCLUDE_DIR})
    string(SUBSTRING ${_BOOSTNUMPY_VERSION_STRING} 11 -1 BOOSTNUMPY_VERSION_STRING)
    set(BOOSTNUMPY_VERSION_STRING ${BOOSTNUMPY_VERSION_STRING}
        CACHE STRING "The version of the detected BoostNumpy tool as a string." FORCE)

    set(BOOSTNUMPY_LIB_DIR ${BOOSTNUMPY_PREFIX}/lib/BoostNumpy-${BOOSTNUMPY_VERSION_STRING})
    set(BOOSTNUMPY_DOCS_DIR ${BOOSTNUMPY_PREFIX}/share/doc/BoostNumpy-${BOOSTNUMPY_VERSION_STRING}
        CACHE PATH "The path to the documentation direcroty of BoostNumpy." FORCE)
    set(BOOSTNUMPY_LIBRARIES boostnumpy)

    tooldef(boostnumpy
        ${BOOSTNUMPY_INCLUDE_DIR}
        boost/numpy/numpy.hpp
        ${BOOSTNUMPY_LIB_DIR}
        NONE # The bin direcroty is n/a, placeholder
        ${BOOSTNUMPY_LIBRARIES}
    )

    if(BOOSTNUMPY_FOUND)
        message(STATUS "+ Detected version '${BOOSTNUMPY_VERSION_STRING}'")
    endif(BOOSTNUMPY_FOUND)
endif("${BOOSTNUMPY_INCLUDE_DIR}" STREQUAL "")
