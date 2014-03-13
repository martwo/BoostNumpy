if(SPHINX_FOUND)

    file(GLOB BOOST_NUMPY_DOC_DEPS conf.py *.rst)

    message(STATUS "BOOST_NUMPY_DOC_DEPS=${BOOST_NUMPY_DOC_DEPS}")

    add_custom_target(doc-html
        ${SPHINX_BUILD}
            -b html
            -c ${CMAKE_CURRENT_SOURCE_DIR}
            -d .doctrees
            ${CMAKE_CURRENT_SOURCE_DIR}
            _html

        DEPENDS ${BOOST_NUMPY_DOC_DEPS}
        WORKING_DIRECTORY ${PROJECT_BINARY_DIR}
        COMMENT "Generating HTML Documentation"
    )

    set_property(TARGET doc-html PROPERTY FOLDER "doc")

    install(DIRECTORY ${PROJECT_BINARY_DIR}/html
        DESTINATION share/doc/libboost_numpy
        OPTIONAL
    )

else()
    message(STATUS "- Sphinx not found, cannot add doc-html target.")
endif()
