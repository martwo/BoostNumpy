How to tweak the IceCube cmake project to support BoostNumpy (a workable suggestion)
====================================================================================

Rational
--------
Some BoostNumpy C++ header files (header-only implementation) need to be included to a C++
source code file (e.g. .cxx file) before the <boost/python.hpp> header file is
included for the first time. But the IceCube build cmake machinary automatically (via the
``-include <FILE>`` compiler argument) includes the ``I3.h`` header file at the
beginning of all IceCube source code files. The I3.h header includes the
<boost/python.hpp> header file. This makes it (almost) impossible for the IceCube user to
include specific header files before <boost/python.hpp> is included. The solution
is to include these specific BoostNumpy header files inside I3.h when python bindings
are going to be created (i.e. for libraries defined via ``i3_add_pybindings``).

Step-by-Step
------------
1. Copy the directory cmake/icecube/BoostNumpy of BoostNumpy into the
   $I3_SRC/cmake/tool-patches/common/ directory.
   This directory contains all needed BoostNumpy header files, that have to be
   included before boost/python.hpp is included.

2. Just before boost/python.hpp is included (around line 58), insert the
   following line to $I3_SRC/cmake/I3.h.in::

        #include <BoostNumpy/detail/pre_boost_python_hpp_includes.hpp>

   This includes some basic boost header files and the header files located in
   the (in 1.) copied BoostNumpy directory.

3. Copy the file cmake/boostnumpy.cmake of BoostNumpy into the
   $I3_SRC/cmake/tools/ directory. This allows the user to insert ``boostnumpy``
   to the USE_TOOLS list.

   .. note:: This cmake tool detection script assumes, that BoostNumpy is
             installed in ``$I3_PORTS``.
