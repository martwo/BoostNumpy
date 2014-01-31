.. highlight:: c

.. _BoostNumpy_ndarray:

The ndarray class
=================

Include statement::

    #include <boost/numpy/ndarray.hpp>

The ``boost::numpy::ndarray`` class is derived from the
``boost::python::object`` class and handles a ``PyArray_Type`` Python object,
i.e. a numpy ndarray object.