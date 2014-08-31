BoostNumpy
==========

BoostNumpy is an extension of boost::python to handle numpy arrays in C++ code.
It introduces the boost::numpy::ndarray class derived from boost::python::object
to manage PyArrayType objects, i.e. numpy arrays.

This project is based on an implemenation by Jim Bosch & co [1].
The major development of BoostNumpy is the dstream, a.k.a. data stream,
sub-library for vectorizing (scalar) C++ functions. It implements the
Generalized Universal Functions approach described by the numpy community [2].
BoostNumpy uses meta-programming (MPL) to achieve the vectorization of a C++
function with only one line of code.

Example::

    #include <boost/python.hpp>
    #include <boost/numpy/dstream.hpp>

    namespace bp = boost::python;
    namespace bn = boost::numpy;

    double square(double v) { return v*v; }

    BOOST_PYTHON_MODULE(my_py_module)
    {
        bn::initialize();

        bn::dstream::def("square", &square, bp::arg("v"), "Calculates the square of v.");
    }

The square function in Python will accept a numpy array as input and will return
a numpy array as output::

    import numpy as np
    import my_py_module

    arr = np.array([1, 2, 3])
    out = my_py_module.square(arr)

The C++ function will be called for every entry in the
given input numpy array. Furthermore, the Python function will have an addition
optional argument named "out=None" in order to pass an already existing output
numpy array to the function.

Of course, this works also for C++ class member functions using

    bp::class_<...>(...).def(bn::dstream::method(...));

An addition feature is multi-threading. By adding the

    bn::dstream::allow_threads()

option to the bn::dstream::def function, an addition optional argument named
"nthreads=1" is available to the Python function. So calculations can be
distributed over several CPU cores.


[1] https://github.com/ndarray/Boost.NumPy
[2] http://docs.scipy.org/doc/numpy/reference/c-api.generalized-ufuncs.html

Dependencies
------------

BoostNumpy depends on the following libraries:

- cmake (>= 2.8.3)
- boost (including boost::python) (>= 1.38)
- python (>= 2.6)
- numpy (>= 1.6)

Installation
------------

An automated compilation and installation procedure is provided via cmake.
In order to create a build directory with the neccessary cmake and make files,
execute the ``configure`` script within the BoostNumpy source root directory::

    ./configure --prefix </path/to/the/installation/location>

The location of the final installation can be specified via the ``--prefix``
option. If this option is not specified, it will be installed inside the ./build
directory.

After success, change to the build directory::

    cd build

Start the compilation process::

    make

Run all the tests::

    make test

Build the documentation (if Sphinx is installed)::

    make html

After that, install BoostNumpy (and the documentation) by typing::

    make install
