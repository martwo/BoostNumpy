.. highlight:: c

.. _BoostNumpy_include:

Including BoostNumpy into your project
======================================

In order to use boost::numpy in your C++ project, you need to include the main
header file::

    #include <boost/numpy.hpp>

This main header file includes BoostNumpy header files that extend boost::python
functionalities and hence need to be included before the ``<boost/python.hpp>``
header file is included. After including those header files, the
``<boost/python.hpp>`` header file is included. Furthermore the BoostNumpy
header file ``<boost/numpy/ndarray.hpp>`` is included for the convenience of the
user. Header files for additional BoostNumpy libraries like *dstream* need to be
included separately.

.. note::

    It is also sufficent to include only the main header file of the particular
    BoostNumpy sub-library.

Including only the pre-<boost/python.hpp> headers
-------------------------------------------------

In cases where it is necessary to include only the BoostNumpy header files that
need to be included before the ``<boost/python.hpp>`` header file is included,
one can do so by including the
``<boost/numpy/detail/pre_boost_python_hpp_includes.hpp>`` header file.

Initialization of the numpy Python module
-----------------------------------------

Before boost::numpy can be used within our C++ project, the C-API of the numpy
Python module needs to be initialized. This can be done using the
``boost::numpy::initialize()`` function. The call to this function should
probably be the first statement after the ``BOOST_PYTHON_MODULE()`` macro
statement of your project.
