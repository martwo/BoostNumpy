.. highlight:: c

.. _BoostNumpy_dstream_exposing:

Exposing
========

C++ functions and C++ member functions can be exposed to Python as generalized
universal functions (GUFs) using ``boost::python`` and
``boost::numpy::dstream``.


.. _BoostNumpy_dstream_exposing_cpp_functions:

C++ functions
-------------

This section describes how to expose a C++ function to Python as a GUF.

Assuming the to-be-exposed C++ function is defines as::

    double area(double width, double height)
    {
        return width * height;
    }

The function ``area`` can be exposed to Python as GUF using ``boost::python``
and the ``def`` function of ``boost::numpy::dstream`` ::

    #include <boost/python.hpp>
    #include <boost/numpy.hpp>
    #include <boost/numpy/dstream.hpp>

    namespace bp = boost::python;
    namespace bn = boost::numpy;

    BOOST_PYTHON_MODULE( my_py_mod )
    {
        bn::initialize();

        bn::dstream::def(“area”, &area, (bp::args(“width”), "height") );
    }

A Python script calling the ``area`` function could then look like so:

.. code-block:: python

    from my_py_mod import area

    areas = area([1,2,3], [4,5,6])


.. _BoostNumpy_dstream_exposing_cpp_member_functions:

C++ member functions
--------------------

Similar to C++ functions, also C++ member functions can be exposed to Python as
GUFs.

Assuming the to-be-exposed C++ member function is defines as::

    class A
    {
      public:
        double area(double width, double height) const
        {
            return width * height;
        }
    };

The member function ``A::area`` can be exposed to Python as GUF using
``boost::python`` and the ``method`` function of ``boost::numpy::dstream``::

    #include <boost/shared_ptr.hpp>
    #include <boost/python.hpp>
    #include <boost/numpy.hpp>
    #include <boost/numpy/dstream.hpp>

    namespace bp = boost::python;
    namespace bn = boost::numpy;

    BOOST_PYTHON_MODULE( my_py_mod )
    {
        bn::initialize();

        bp::class_<A, boost::shared_ptr<A> >(“A”)
            .def(bn::dstream::method(“area”, &A::area, (bp::args(“width”), "height")))
        ;
    }

A Python script calling the ``area`` member function of class ``A`` could then
look like so:

.. code-block:: python

    from my_py_mod import A

    a = A()
    areas = a.area([1,2,3], [4,5,6])


.. _BoostNumpy_dstream_exposing_static_cpp_class_functions:

Static C++ class functions
--------------------------

Assuming the to-be-exposed C++ function is a static class function defined
like::

    class A
    {
      public:
        static double area(double width, double height)
        {
            return width * height;
        }
    };

Such a static function can also be exposed to Python as GUF by using the
``boost::numpy::dstream::staticmethod`` function instead of the
``boost::numpy::dstream::method`` function in analogy to the example shown in
:ref:`BoostNumpy_dstream_exposing_cpp_member_functions`.
