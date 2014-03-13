.. highlight:: c

.. _BoostNumpy_dstream_threading:

Multi-Threading
===============

The dstream library provides multi-threading abilities for GUFs. When threading
is enabled for a GUF, it gets the extra optional argument ``nthreads``
defaulting to ``1``. In cases where ``nthreads`` is set to greater than ``1``,
different elements of the loop dimensions are computed in different threads.

In order to allow for threading of an exposed function, the optional argument
``ds::allow_threads()`` needs to be passed to the ``ds::def``, ``ds::method``,
or ``ds::staticmethod`` functions::

    bn::dstream::def(“area”, &area, (bp::args(“width”), "height"), ds::allow_threads() );

The minimum number of tasks (i.e. performed operations) per thread can be
specified using the ``ds::min_thread_size<N>()`` threading option instead of the
``ds::allow_threads()`` threading option above. The minimum number of task per
thread ``N`` should be a trade-off between the time for one fundamental
operation (on the core dimensions) and the time required to create an additional
thread. The default value (i.e. for ``ds::allow_threads()``) is 64.

By default all functions are exposed with the threading option
``ds::no_threads()``.
