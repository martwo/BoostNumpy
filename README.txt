Dependencies
============

BoostNumpy depends on the following libraries:

- boost (including boost::python) (>= 1.38)
- python (>= 2.6)
- numpy (>= 1.6)

Installation
============

An automated compilation and installation procedure is provided via cmake.
In order to create a build directory with the neccessary cmake and make files,
execute the ``configure`` script within the BoostNumpy source root directory::

    ./configure --prefix </path/to/the/installation/location>

The location of the final installation can be specified via the ``--prefix``
option. If this option is not specified, it will be installed inside the build
directory.

After success, change to the build directory::

    cd build

Start the compilation process::

    make

After that, install BoostNumpy by typing::

    make install

IceCube specific cmake build support
====================================

BoostNumpy provides a cmake tool detection script specific for the IceCube cmake
build system. It is located in the sub-directory ``cmake/icecube`` and is named
``boostnumpy.cmake``. The file ``cmake/icecube/README.txt`` gives further
instructions how to tweak the IceCube cmake build system to support BoostNumpy.
