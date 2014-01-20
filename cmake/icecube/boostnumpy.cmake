colormsg("")
colormsg(HICYAN "BoostNumpy")

# Assume that BoostNumpy is installed inside the I3_PORTS tree.
set(BOOSTNUMPY_PORTSVERSION "1.0.0")
set(BOOSTNUMPY_VERSION_STRING ${BOOSTNUMPY_PORTSVERSION}
    CACHE STRING "The BoostNumpy version." FORCE)

set(BOOSTNUMPY_LIB_DIR ${I3_PORTS}/lib/BoostNumpy-${BOOSTNUMPY_VERSION_STRING}
    CACHE PATH "The library directory of BoostNumpy." FORCE)
set(BOOSTNUMPY_LIBRARIES boostnumpy
    CACHE STRING "The library names of BoostNumpy." FORCE)
set(BOOSTNUMPY_INCLUDE_DIRS ${I3_PORTS}/include/BoostNumpy-${BOOSTNUMPY_VERSION_STRING}
    CACHE PATH "The include directories of BoostNumpy." FORCE)

if(NOT EXISTS "${BOOSTNUMPY_LIBRARYDIR}")

    set(BOOSTNUMPY_FOUND FALSE
        CACHE BOOL "BoostNumpy found successfully" FORCE)
    message(STATUS "Error configuring BoostNumpy: Directory ${BOOSTNUMPY_LIBRARYDIR} does not exist.\n")

else(NOT EXISTS "${BOOSTNUMPY_LIBRARYDIR}")

    set(BOOSTNUMPY_FOUND TRUE
        CACHE BOOL "BoostNumpy found successfully" FORCE)

    message(STATUS "+ BoostNumpy found:")
    message(STATUS "+       LIBRARIES: ${BOOSTNUMPY_LIBRARIES}")
    message(STATUS "+         LIB_DIR: ${BOOSTNUMPY_LIB_DIR}")
    message(STATUS "+    INCLUDE_DIRS: ${BOOSTNUMPY_INCLUDE_DIRS}")

endif(NOT EXISTS "${BOOSTNUMPY_LIBRARYDIR}")
