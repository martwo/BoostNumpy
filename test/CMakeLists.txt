
set(${PROJECT_NAME}_RUN_TEST_SCRIPT
    run_test.sh)
configure_file(${${PROJECT_NAME}_RUN_TEST_SCRIPT}.in ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${${PROJECT_NAME}_RUN_TEST_SCRIPT} @ONLY)

set(${PROJECT_NAME}_TEST_SOURCE_DIR
    ${PROJECT_SOURCE_DIR}/test)
set(${PROJECT_NAME}_TEST_COMMAND
    ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${${PROJECT_NAME}_RUN_TEST_SCRIPT})

list(APPEND ${PROJECT_NAME}_PYMOD_ADD_LIB_LIST
    boost_numpy
    ${BOOST_LIBRARIES})

function(add_python_test _NAME _PYSCRIPT)

    # Make a test of the module using the python source file in the test
    # directory.
    add_test(${_NAME} ${${PROJECT_NAME}_TEST_COMMAND} ${${PROJECT_NAME}_TEST_SOURCE_DIR}/${_PYSCRIPT})

    # Set the regex to use to recognize a failure since `python testfoo.py`
    # does not seem to return non-zero with a test failure.
    set_property(TEST ${_NAME} PROPERTY FAIL_REGULAR_EXPRESSION "ERROR\\:")

endfunction(add_python_test)

add_python_module(dtype_test_module "${${PROJECT_NAME}_PYMOD_ADD_LIB_LIST}" dtype_test_module.cpp)
add_python_test(dtype_test dtype_test.py)

add_python_module(ndarray_test_module "${${PROJECT_NAME}_PYMOD_ADD_LIB_LIST}" ndarray_test_module.cpp)
add_python_test(ndarray_test ndarray_test.py)

add_python_module(indexing_test_module "${${PROJECT_NAME}_PYMOD_ADD_LIB_LIST}" indexing_test_module.cpp)
add_python_test(indexing_test indexing_test.py)

add_python_module(dstream_test_module "${${PROJECT_NAME}_PYMOD_ADD_LIB_LIST}" dstream_test_module.cpp)
add_python_test(dstream_test dstream_test.py)
