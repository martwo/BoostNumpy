message(STATUS "Entering 'config.cmake'")

add_definitions(-fPIC)

#
# libraries everybody links to
#
if (${CMAKE_SYSTEM_NAME} STREQUAL "FreeBSD")
    # FreeBSD keeps libdl stuff in libc
    link_libraries(m stdc++)
else (${CMAKE_SYSTEM_NAME} STREQUAL "FreeBSD")
    link_libraries(m dl stdc++)
endif (${CMAKE_SYSTEM_NAME} STREQUAL "FreeBSD")

message(STATUS "Leaving 'config.cmake'")