
# Assume that BoostNumpy is installed inside the I3_PORTS tree.
file(GLOB BOOSTNUMPY_INCLUDE_DIR ${I3_PORTS}/include/BoostNumpy-*.*.*)
file(GLOB BOOSTNUMPY_LIB_DIR ${I3_PORTS}/lib/BoostNumpy-*.*.*)

# TODO: Add Check if glob was successfull, extract also the version.

#set(BOOSTNUMPY_PORTSVERSION "1.0.0")
#set(BOOSTNUMPY_VERSION_STRING ${BOOSTNUMPY_PORTSVERSION}
#    CACHE STRING "The BoostNumpy version." FORCE)

#set(BOOSTNUMPY_INCLUDE_DIR ${I3_PORTS}/include/BoostNumpy-${BOOSTNUMPY_VERSION_STRING})
#set(BOOSTNUMPY_LIB_DIR ${I3_PORTS}/lib/BoostNumpy-${BOOSTNUMPY_VERSION_STRING})
set(BOOSTNUMPY_LIBRARIES boostnumpy)

tooldef(boostnumpy
    ${BOOSTNUMPY_INCLUDE_DIR}
    boost/numpy/numpy.hpp
    ${BOOSTNUMPY_LIB_DIR}
    NONE # The bin direcroty is n/a, placeholder
    ${BOOSTNUMPY_LIBRARIES}
)
