/**
 * $Id$
 *
 * Copyright (C)
 * 2013 - $Date$
 *     Martin Wolf <boostnumpy@martin-wolf.org>
 *
 * \file    boost/numpy/dstream/detail/callable_call.hpp
 * \version $Revision$
 * \date    $Date$
 * \author  Martin Wolf <boostnumpy@martin-wolf.org>
 *
 * \brief This file defines the implementation of the call function of the data
 *        stream callable class template.
 *
 *        This file is distributed under the Boost Software License,
 *        Version 1.0. (See accompanying file LICENSE_1_0.txt or copy at
 *        http://www.boost.org/LICENSE_1_0.txt).
 */
#if !defined(BOOST_PP_IS_ITERATING)

#ifndef BOOST_NUMPY_DSTREAM_DETAIL_CALLABLE_CALL_HPP_INCLUDED
#define BOOST_NUMPY_DSTREAM_DETAIL_CALLABLE_CALL_HPP_INCLUDED

#include <boost/preprocessor/iterate.hpp>
#include <boost/preprocessor/repetition/enum_binary_params.hpp>
#include <boost/preprocessor/repetition/enum_params.hpp>
#include <boost/preprocessor/repetition/enum_trailing_params.hpp>

#include <boost/mpl/bitor.hpp>
#include <boost/thread.hpp>

#include <boost/python/object_fwd.hpp>
#include <boost/python/tuple.hpp>

#include <boost/numpy/limits.hpp>
#include <boost/numpy/detail/iter.hpp>
#include <boost/numpy/detail/pygil.hpp>
#include <boost/numpy/dstream/array_definition.hpp>
#include <boost/numpy/dstream/detail/input_array_service.hpp>
#include <boost/numpy/dstream/detail/output_array_service.hpp>
#include <boost/numpy/dstream/detail/loop_service.hpp>

namespace boost {
namespace numpy {
namespace dstream {
namespace detail {

template <unsigned OutArity>
struct construct_result;

template <>
struct construct_result<0>
{
    python::object
    apply()
    {
        return python::object();
    }
};

template <>
struct construct_result<1>
{
    template <class OutArrService>
    python::object
    apply(OutArrService const & out_arr_service)
    {
        return static_cast<python::object>(out_arr_service.get_arr());
    }
};

#define BOOST_PP_ITERATION_PARAMS_1 \
    (4, (2, BOOST_NUMPY_LIMIT_OUTPUT_ARITY, <boost/numpy/dstream/detail/callable_call.hpp>, 1))
#include BOOST_PP_ITERATE()

template <unsigned OutArity, unsigned InArity>
struct callable_call_outin_arity;

#define BOOST_PP_ITERATION_PARAMS_1 \
    (4, (0, BOOST_NUMPY_LIMIT_OUTPUT_ARITY, <boost/numpy/dstream/detail/callable_call.hpp>, 2))
#include BOOST_PP_ITERATE()

template <
      class FTypes
    , class FCaller
    , class MappingDefinition
    , class WiringModel
    , class ThreadAbility
>
struct callable_call
{
    typedef callable_call_outin_arity<
                  MappingDefinition::out::arity
                , MappingDefinition::in::arity
            >
            callable_call_outin_arity_t;

    typedef typename callable_call_outin_arity_t::template impl<
                  FTypes
                , FCaller
                , MappingDefinition
                , WiringModel
                , ThreadAbility
            >::type
            type;
};

}// namespace detail
}// namespace dstream
}// namespace numpy
}// namespace boost

#endif // !BOOST_NUMPY_DSTREAM_DETAIL_CALLABLE_CALL_HPP_INCLUDED
#else

#if BOOST_PP_ITERATION_DEPTH() == 1

#if BOOST_PP_ITERATION_FLAGS() == 1

#define OUT_ARITY BOOST_PP_ITERATION()

template <>
struct construct_result<OUT_ARITY>
{
    template <BOOST_PP_ENUM_PARAMS_Z(1, OUT_ARITY, class OutArrService)>
    python::object
    apply(BOOST_PP_ENUM_BINARY_PARAMS_Z(1, OUT_ARITY, OutArrService, const & out_arr_service))
    {
        // Construct a tuple with all output arrays.
        return static_cast<python::object>(python::make_tuple(
                   BOOST_PP_ENUM_BINARY_PARAMS_Z(1, OUT_ARITY, out_arr_service, .get_arr() BOOST_PP_INTERCEPT)
               ));
    }
};

#undef OUT_ARITY

#elif BOOST_PP_ITERATION_FLAGS() == 2

// Loop over the InArity.
#define BOOST_PP_ITERATION_PARAMS_2 \
    (3, (1, BOOST_NUMPY_LIMIT_INPUT_ARITY, <boost/numpy/dstream/detail/callable_call.hpp>))
#include BOOST_PP_ITERATE()

#endif // BOOST_PP_ITERATION_FLAGS

#elif BOOST_PP_ITERATION_DEPTH() == 2

#define OUT_ARITY BOOST_PP_RELATIVE_ITERATION(1)
#define IN_ARITY BOOST_PP_ITERATION()

#define BOOST_NUMPY_DSTREAM_DETAIL_CALLABLE_CALL__in_arr_def(z, n, data) \
    typedef array_definition< typename MappingDefinition::in::BOOST_PP_CAT(core_shape_t,n), typename WiringModel::api::template in_arr_value_type<n>::type> BOOST_PP_CAT(in_arr_def,n);

#define BOOST_NUMPY_DSTREAM_DETAIL_CALLABLE_CALL__out_arr_def(z, n, data) \
    typedef array_definition< typename MappingDefinition::out::BOOST_PP_CAT(core_shape_t,n), typename WiringModel::api::template out_arr_value_type<n>::type> BOOST_PP_CAT(out_arr_def,n);

#define BOOST_NUMPY_DSTREAM_DETAIL_CALLABLE_CALL__in_arr_dtype(z, n, data) \
    numpy::dtype const BOOST_PP_CAT(in_arr_dtype,n) = numpy::dtype::get_builtin< BOOST_PP_CAT(in_arr_def,n)::value_type >();

#define BOOST_NUMPY_DSTREAM_DETAIL_CALLABLE_CALL__in_arr_service(z, n, data)   \
    numpy::dstream::detail::input_array_service<BOOST_PP_CAT(in_arr_def,n)>    \
    BOOST_PP_CAT(in_arr_service,n)(                                            \
          numpy::from_object(                                                  \
              BOOST_PP_CAT(in_obj,n)                                           \
            , BOOST_PP_CAT(in_arr_dtype,n)                                     \
            , numpy::ndarray::ALIGNED                                          \
          )                                                                    \
    );

#define BOOST_NUMPY_DSTREAM_DETAIL_CALLABLE_CALL__in_arr_iter_op_flags(z, n, data) \
    numpy::detail::iter_operand_flags_t BOOST_PP_CAT(in_arr_iter_op_flags,n) = \
        WiringModel::api::template in_arr_iter_operand_flags<n>::type::value;

#define BOOST_NUMPY_DSTREAM_DETAIL_CALLABLE_CALL__in_arr_iter_op(z, n, data)   \
    numpy::detail::iter_operand BOOST_PP_CAT(in_arr_iter_op,n)(                \
          BOOST_PP_CAT(in_arr_service,n).get_arr()                             \
        , BOOST_PP_CAT(in_arr_iter_op_flags,n)                                 \
        , BOOST_PP_CAT(in_arr_service,n).get_arr_bcr_data()                    \
    );

#define BOOST_NUMPY_DSTREAM_DETAIL_CALLABLE_CALL__out_obj(z, n, data)          \
    python::object BOOST_PP_CAT(out_obj,n) = (out_obj == python::object() ? python::object() : (MappingDefinition::out::arity == 1 ? out_obj : out_obj[0]));

#define BOOST_NUMPY_DSTREAM_DETAIL_CALLABLE_CALL__out_arr_service(z, n, data)  \
    numpy::dstream::detail::output_array_service<loop_service_t, BOOST_PP_CAT(out_arr_def,n)> BOOST_PP_CAT(out_arr_service,n)(loop_service, BOOST_PP_CAT(out_obj,n));

#define BOOST_NUMPY_DSTREAM_DETAIL_CALLABLE_CALL__out_arr_iter_op_flags(z, n, data) \
    numpy::detail::iter_operand_flags_t BOOST_PP_CAT(out_arr_iter_op_flags,n) = \
        WiringModel::api::template out_arr_iter_operand_flags<n>::type::value;

#define BOOST_NUMPY_DSTREAM_DETAIL_CALLABLE_CALL__out_arr_iter_op(z, n, data)  \
    numpy::detail::iter_operand BOOST_PP_CAT(out_arr_iter_op,n)(               \
          BOOST_PP_CAT(out_arr_service,n).get_arr()                            \
        , BOOST_PP_CAT(out_arr_iter_op_flags,n)                                \
        , BOOST_PP_CAT(out_arr_service,n).get_arr_bcr_data()                   \
    );

#define BOOST_NUMPY_DSTREAM_DETAIL_CALLABLE_CALL__arr_core_shapes(z, n, arr_service) \
    core_shapes.push_back( BOOST_PP_CAT(arr_service,n).get_arr_core_shape() );

template <>
struct callable_call_outin_arity<OUT_ARITY, IN_ARITY>
{
    template <
          class FTypes
        , class FCaller
        , class MappingDefinition
        , class WiringModel
        , class ThreadAbility
    >
    struct impl
    {
        typedef impl<FTypes, FCaller, MappingDefinition, WiringModel, ThreadAbility>
                type;

        static
        python::object
        call(
              FCaller const & f_caller
            , typename FTypes::class_type & self
            , BOOST_PP_ENUM_PARAMS(IN_ARITY, python::object const & in_obj)
            , python::object & out_obj
            , unsigned nthreads
        )
        {
            numpy::detail::PyGIL pygil;

            // Construct array_definition types for all input arrays.
            BOOST_PP_REPEAT(IN_ARITY, BOOST_NUMPY_DSTREAM_DETAIL_CALLABLE_CALL__in_arr_def, ~)

            // Construct dtype objects for all input arrays.
            BOOST_PP_REPEAT(IN_ARITY, BOOST_NUMPY_DSTREAM_DETAIL_CALLABLE_CALL__in_arr_dtype, ~)

            // Construct input_array_service objects for all input arrays.
            BOOST_PP_REPEAT(IN_ARITY, BOOST_NUMPY_DSTREAM_DETAIL_CALLABLE_CALL__in_arr_service, ~)

            // Construct a loop service that determines the number of loop
            // dimensions.
            typedef numpy::dstream::detail::loop_service_arity<IN_ARITY>::loop_service<
                    BOOST_PP_ENUM_PARAMS_Z(1, IN_ARITY, in_arr_def)
                    >
                    loop_service_t;
            loop_service_t loop_service(BOOST_PP_ENUM_PARAMS_Z(1, IN_ARITY, in_arr_service));

            // Construct numpy::detail::iter_operand_flags_t objects for all
            // input arrays.
            BOOST_PP_REPEAT(IN_ARITY, BOOST_NUMPY_DSTREAM_DETAIL_CALLABLE_CALL__in_arr_iter_op_flags, ~)

            // Construct numpy::detail::iter_operand objects for all input
            // arrays.
            BOOST_PP_REPEAT(IN_ARITY, BOOST_NUMPY_DSTREAM_DETAIL_CALLABLE_CALL__in_arr_iter_op, ~)

            // Construct out_obj boos::python objects holding the individual
            // provided output objects.
            BOOST_PP_REPEAT(OUT_ARITY, BOOST_NUMPY_DSTREAM_DETAIL_CALLABLE_CALL__out_obj, ~)

            // Construct array_definition types for all output arrays.
            BOOST_PP_REPEAT(OUT_ARITY, BOOST_NUMPY_DSTREAM_DETAIL_CALLABLE_CALL__out_arr_def, ~)

            // Construct output_array_service objects for all output arrays.
            BOOST_PP_REPEAT(OUT_ARITY, BOOST_NUMPY_DSTREAM_DETAIL_CALLABLE_CALL__out_arr_service, ~)

            // Construct iter_operand_flags_t objects for all output arrays.
            BOOST_PP_REPEAT(OUT_ARITY, BOOST_NUMPY_DSTREAM_DETAIL_CALLABLE_CALL__out_arr_iter_op_flags, ~)

            // Construct iter_operand objects for all output arrays.
            BOOST_PP_REPEAT(OUT_ARITY, BOOST_NUMPY_DSTREAM_DETAIL_CALLABLE_CALL__out_arr_iter_op, ~)

            // Define the iterator flags. In order to allow multi-threading
            // the following flags MUST be set:
            //     EXTERNAL_LOOP, RANGED, BUFFERED, DELAY_BUFALLOC
            // So we set these flags by default for all mapping models and add
            // additional mapping model specific flags.
            numpy::detail::iter_flags_t iter_flags = boost::mpl::bitor_<
                  numpy::detail::iter::flags::EXTERNAL_LOOP
                , numpy::detail::iter::flags::RANGED
                , numpy::detail::iter::flags::BUFFERED
                , numpy::detail::iter::flags::DELAY_BUFALLOC
                >::type::value;
            iter_flags |= WiringModel::api::iter_flags::type::value;

            // Define other iterator properties.
            numpy::order_t order = WiringModel::api::order;
            numpy::casting_t casting = WiringModel::api::casting;
            intptr_t buffersize = WiringModel::api::buffersize;

            // Finally, create the iterator object.
            numpy::detail::iter iter(
                  iter_flags
                , order
                , casting
                , loop_service.get_loop_nd()
                , loop_service.get_loop_shape_data()
                , buffersize
                BOOST_PP_ENUM_TRAILING_PARAMS_Z(1, OUT_ARITY, out_arr_iter_op)
                BOOST_PP_ENUM_TRAILING_PARAMS_Z(1, IN_ARITY, in_arr_iter_op)
            );

            // Determine how many iterations/tasks each thread needs to perform.
            // This is done by rounding up loop_size / nthreads. This
            // assumes, that loop_size > 0 and nthreads > 0.
            intptr_t const loop_size = loop_service.get_loop_size();
            if(! (loop_size > 0))
            {
                PyErr_SetString(PyExc_RuntimeError,
                    "The total number of iteration must be >= 1. "
                    "Note: This is obviously an internal error! Sorry!");
                python::throw_error_already_set();
            }
            if(! (nthreads > 0))
            {
                PyErr_SetString(PyExc_ValueError,
                    "The number of threads must be >= 1.");
                python::throw_error_already_set();
            }

            intptr_t const n_tasks_per_thread =
                std::min(
                      loop_size
                    , std::max(
                          1 + ((loop_size - 1) / nthreads)
                        , intptr_t(ThreadAbility::min_n_tasks_per_thread_t::value)
                      )
                );
            nthreads = 1 + ((loop_size - 1) / n_tasks_per_thread);

            // Calculate task size. The task size is the product of the shape
            // values of the broadcasted iteration array without the loop
            // dimension.
            intptr_t const iter_size = iter.get_iter_size();
            intptr_t const task_size = iter_size / loop_size;

            intptr_t const parallel_iter_size = n_tasks_per_thread * task_size;

            std::cout << "Launching " << nthreads << " threads with "
                      << n_tasks_per_thread << " tasks per thread "
                      << "and a task size of " << task_size << "."
                      << std::endl;

            // Make nthreads - 1 copies of the iter object and store it inside
            // the vector. This will call NpyIter_Copy for each made copy.
            std::vector<numpy::detail::iter> iter_vec(nthreads - 1, iter);

            // Initialize the iterators for their specific range of iteration.
            iter.init_ranged_iteration(0, parallel_iter_size);
            std::vector<boost::numpy::detail::iter>::iterator it;
            std::vector<boost::numpy::detail::iter>::iterator iter_vec_end = iter_vec.end();
            intptr_t istart = parallel_iter_size;
            for(it=iter_vec.begin(); it!=iter_vec_end; ++it)
            {
                boost::numpy::detail::iter & iter_ = *it;
                iter_.init_ranged_iteration(istart, std::min(istart + parallel_iter_size, iter_size));
                istart += parallel_iter_size;
            }

            // Construct core_shapes vector holding the lengths of all
            // dimensions.
            std::vector< std::vector<intptr_t> > core_shapes;
            BOOST_PP_REPEAT(OUT_ARITY, BOOST_NUMPY_DSTREAM_DETAIL_CALLABLE_CALL__arr_core_shapes, out_arr_service)
            BOOST_PP_REPEAT(IN_ARITY,  BOOST_NUMPY_DSTREAM_DETAIL_CALLABLE_CALL__arr_core_shapes, in_arr_service)

            // Create an error flag that indicates if an error occurred for any
            // thread.
            // Note: We are just interested in the fact wether any thread
            //       failed. So we let write all threads to the same flag
            //       variable WITHOUT a mutex.
            bool thread_error_flag = false;

            // Release the Python GIL to allow threads.
            // Note: No Python API calls are allowed after that and before
            //       pygil.acquire() is called again.
            pygil.release();

            // Create threads for the second, third, fourth, etc. iterator and
            // add them to a boost::thread_group object.
            boost::thread_group threads;
            try {
                for(it = iter_vec.begin(); it!=iter_vec_end; ++it)
                {
                    boost::thread *t = new boost::thread(&WiringModel::iterate
                        , boost::ref(self)
                        , boost::cref(f_caller)
                        , boost::ref(*it)
                        , boost::cref(core_shapes)
                        , boost::ref(thread_error_flag)
                    );
                    threads.add_thread(t);
                }
            }
            catch(boost::thread_resource_error)
            {
                // TODO: Is pygil.acquire(); needed here???
                PyErr_SetString(PyExc_RuntimeError,
                    "At least one thread could not be launched due to a "
                    "resource error.");
                boost::python::throw_error_already_set();
            }

            // Do the iteration for the first iterator.
            WiringModel::iterate(self, f_caller, iter, core_shapes, thread_error_flag);

            // Join all the threads.
            threads.join_all();

            // Acquire the python GIL again.
            pygil.acquire();

            // Check if any thread failed.
            if(thread_error_flag)
            {
                PyErr_SetString(PyExc_RuntimeError,
                    "At least one thread failed during execution.");
                boost::python::throw_error_already_set();
            }

            return construct_result<MappingDefinition::out::arity>::apply(BOOST_PP_ENUM_PARAMS_Z(1, OUT_ARITY, out_arr_service));
        }
    };
};

#undef BOOST_NUMPY_DSTREAM_DETAIL_CALLABLE_CALL__arr_core_shapes
#undef BOOST_NUMPY_DSTREAM_DETAIL_CALLABLE_CALL__out_arr_iter_op
#undef BOOST_NUMPY_DSTREAM_DETAIL_CALLABLE_CALL__out_arr_iter_op_flags
#undef BOOST_NUMPY_DSTREAM_DETAIL_CALLABLE_CALL__out_arr_service
#undef BOOST_NUMPY_DSTREAM_DETAIL_CALLABLE_CALL__out_obj
#undef BOOST_NUMPY_DSTREAM_DETAIL_CALLABLE_CALL__in_arr_iter_op
#undef BOOST_NUMPY_DSTREAM_DETAIL_CALLABLE_CALL__in_arr_iter_op_flags
#undef BOOST_NUMPY_DSTREAM_DETAIL_CALLABLE_CALL__in_arr_service
#undef BOOST_NUMPY_DSTREAM_DETAIL_CALLABLE_CALL__in_arr_dtype
#undef BOOST_NUMPY_DSTREAM_DETAIL_CALLABLE_CALL__out_arr_def
#undef BOOST_NUMPY_DSTREAM_DETAIL_CALLABLE_CALL__in_arr_def

#undef IN_ARITY
#undef OUT_ARITY

#endif // BOOST_PP_ITERATION_DEPTH

#endif // BOOST_PP_IS_ITERATING
