.. highlight:: c

.. _BoostNumpy_include:

Including BoostNumpy into your project
======================================

In order to use boost::numpy in your C++ project, you need to include the main
header file::

    #include <boost/numpy.hpp>

This main header includes the basic header files like
``<boost/numpy/ndarray.hpp>``. Header files for additional BoostNumpy libraries
such like *dstream* need to be included separately.

.. note::

    It is also sufficent to include only the main header file of the particular
    BoostNumpy sub-library.

Initialization of the numpy Python module
-----------------------------------------

Before boost::numpy can be used within our C++ project, the C-API of the numpy
Python module needs to be initialized. This can be done using the
``boost::numpy::initialize()`` function.

.. tip::

    The call to this function should probably be the first statement after the
    ``BOOST_PYTHON_MODULE()`` macro statement of your project.
