find_package(PythonInterp REQUIRED)
find_package(PythonLibs ${PYTHON_VERSION_STRING} REQUIRED)

set(PYTHON_FOUND TRUE CACHE BOOL "Python found." FORCE)

message(STATUS "+    Python include dirs: ${PYTHON_INCLUDE_DIRS}")
