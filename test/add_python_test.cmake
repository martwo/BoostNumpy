function(add_python_test _NAME _PYSCRIPT)

    set(${PROJECT_NAME}_PYTEST_MODULE_LIST
        ${${PROJECT_NAME}_PYTEST_MODULE_LIST}
        ${_NAME}_module PARENT_SCOPE)
    set(${PROJECT_NAME}_PYTEST_PYSCRIPT_LIST
        ${${PROJECT_NAME}_PYTEST_PYSCRIPT_LIST}
        ${${PROJECT_NAME}_TEST_SOURCE_DIR}/${_PYSCRIPT} PARENT_SCOPE)

    # Make a test of the module using the python source file in the test
    # directory.
    add_test(${_NAME} ${${PROJECT_NAME}_RUN_TEST_SCRIPT} ${${PROJECT_NAME}_TEST_SOURCE_DIR}/${_PYSCRIPT})

    # Set the regex to use to recognize a failure since `python testfoo.py`
    # does not seem to return non-zero with a test failure.
    set_property(TEST ${_NAME} PROPERTY FAIL_REGULAR_EXPRESSION "ERROR\\:")

endfunction(add_python_test)
