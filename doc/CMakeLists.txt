if(SPHINX_FOUND)

    add_custom_target(html
        ${SPHINX_BUILD}
            -b html
            -c ${CMAKE_CURRENT_SOURCE_DIR}
            -d .doctrees
            ${CMAKE_CURRENT_SOURCE_DIR}
            html

        WORKING_DIRECTORY ${PROJECT_BINARY_DIR}
        COMMENT "Generating HTML Documentation"
    )

    add_custom_command(TARGET html
        PRE_BUILD
        COMMAND mkdir -p ${PROJECT_BINARY_DIR}/html
    )

    install(DIRECTORY ${PROJECT_BINARY_DIR}/html
        DESTINATION ${CMAKE_INSTALL_PREFIX}/share/doc/libboost_numpy
        OPTIONAL
    )

else()
    message(STATUS "- Sphinx not found, cannot add doc-html target.")
endif()
