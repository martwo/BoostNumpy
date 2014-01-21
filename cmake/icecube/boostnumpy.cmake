
# Assume that BoostNumpy is installed inside the I3_PORTS tree.
file(GLOB BOOSTNUMPY_INCLUDE_DIR RELATIVE ${I3_PORTS} ${I3_PORTS}/include/BoostNumpy-*.*.*)
file(GLOB BOOSTNUMPY_LIB_DIR RELATIVE ${I3_PORTS} ${I3_PORTS}/lib/BoostNumpy-*.*.*)

if("${BOOSTNUMPY_INCLUDE_DIR}" STREQUAL "" OR "${BOOSTNUMPY_LIB_DIR}" STREQUAL "")
    colormsg(WHITE "")
    colormsg(HICYAN "boostnumpy")
    colormsg(CYAN "- Error: Not found in I3_PORTS '${I3_PORTS}'!")
    set(BOOSTNUMPY_FOUND FALSE
        CACHE BOOL "boostnumpy found successfully." FORCE)
else("${BOOSTNUMPY_INCLUDE_DIR}" STREQUAL "" OR "${BOOSTNUMPY_LIB_DIR}" STREQUAL "")
    # Get the version of the detected BoostNumpy tool.
    string(REGEX MATCH "([0-9]+)\\.([0-9]+)\\.?([0-9]*)"
        BOOSTNUMPY_VERSION_STRING
        ${BOOSTNUMPY_INCLUDE_DIR})
    set(BOOSTNUMPY_VERSION_STRING ${BOOSTNUMPY_VERSION_STRING}
        CACHE STRING "The version of the detected BoostNumpy tool as a string." FORCE)

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
endif("${BOOSTNUMPY_INCLUDE_DIR}" STREQUAL "" OR "${BOOSTNUMPY_LIB_DIR}" STREQUAL "")
