#
# $Id$
#
# Copyright (C)
# 2014 - $Date$
#     Martin Wolf <boostnumpy@martin-wolf.org>
#
# This file implements cmake configuration for generating the BoostNumpy
# documentation.
#
# This file is distributed under the Boost Software License,
# Version 1.0. (See accompanying file LICENSE_1_0.txt or copy at
# http://www.boost.org/LICENSE_1_0.txt).
#
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

    add_custom_target(html
        echo "WARNING: Sphinx is not installed. No documentation can be generated!"

        WORKING_DIRECTORY ${PROJECT_BINARY_DIR}
        COMMENT "Generating HTML Documentation"
    )

    message(STATUS "- Sphinx not found, \"make html\" will do nothing at all.")

endif()
