.. highlight:: c

.. _BoostNumpy_ndarray:

The ndarray class
=================

The ``boost::numpy::ndarray`` class is derived from the
``boost::python::object`` class and manages a Python object of type
``PyArray_Type``., i.e. a numpy ndarray object.

Array Flags
-----------

A numpy array has certain flags set that describe the array and its data. All
possible flags are defined through the ``bn::ndarray::flags`` enum type.
Possible flag values are:

    * ``NONE``
    * ``C_CONTIGUOUS``
    * ``F_CONTIGUOUS``
    * ``V_CONTIGUOUS`` (i.e. ``C_CONTIGUOUS | F_CONTIGUOUS``)
    * ``ALIGNED``
    * ``OWNDATA``
    * ``WRITEABLE``
    * ``UPDATEIFCOPY``
    * ``BEHAVED`` (i.e. ``ALIGNED | WRITEABLE``)
    * ``CARRAY`` (i.e. ``C_CONTIGUOUS | ALIGNED | WRITEABLE``)
    * ``CARRAY_RO`` (i.e. ``C_CONTIGUOUS | ALIGNED``)
    * ``FARRAY`` (i.e. ``F_CONTIGUOUS | ALIGNED | WRITEABLE``)
    * ``FARRAY_RO`` (i.e. ``F_CONTIGUOUS | ALIGNED``)
    * ``DEFAULT`` (i.e. ``CARRAY``)
    * ``UPDATE_ALL`` (i.e. ``C_CONTIGUOUS | F_CONTIGUOUS | ALIGNED``)
    * ``FORCECAST``
    * ``ENSURECOPY``
    * ``ENSUREARRAY``
    * ``NOTSWAPPED``
    * ``BEHAVED_NS`` (i.e. ``ALIGNED | WRITEABLE | NOTSWAPPED``)

All the flag values above have the same names (without the ``NPY_ARRAY_``
prefix) as they do in the numpy C-API. See [#numpy_c_api_array_flags]_ for a
complete description of all these flags.

Retrieving information about an array
-------------------------------------

The ndarray class provides several methods for retrieving information about a
numpy array:

``bp::object get_base() const``

    Returns the base object of the array.

``bn::dtype get_dtype() const``

    Returns the data type object [#dtype_class]_ of the array.

``bn::ndarray::flags get_flags() const``

    Returns the flags of the array.

References
----------

.. [#numpy_c_api_array_flags] http://docs.scipy.org/doc/numpy/reference/c-api.array.html#array-flags

.. [#dtype_class] See :ref:`BoostNumpy_dtype` for a full description of the
                  dtype class.
