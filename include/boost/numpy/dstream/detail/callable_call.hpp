/**
 * $Id$
 *
 * Copyright (C)
 * 2013
 *     Martin Wolf <martin.wolf@fysik.su.se>
 *
 * \file    boost/numpy/dstream/detail/callable_call.hpp
 * \version $Revision$
 * \date    $Date$
 * \author  Martin Wolf <martin.wolf@fysik.su.se>
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

#include <boost/python/object_fwd.hpp>

namespace boost {
namespace numpy {
namespace dstream {
namespace detail {

template <unsigned N>
struct callable_call_arity;

#define BOOST_PP_ITERATION_PARAMS_1                                            \
    (3, (1, BOOST_NUMPY_LIMIT_INPUT_ARITY, <boost/numpy/dstream/detail/callable_call.hpp>))
#include BOOST_PP_ITERATE()

}/*namespace detail*/
}/*namespace dstream*/
}/*namespace numpy*/
}/*namespace boost*/

#endif // !BOOST_NUMPY_DSTREAM_DETAIL_CALLABLE_CALL_HPP_INCLUDED
#else

#define N BOOST_PP_ITERATION()

#define BOOST_NUMPY_DSTREAM_DETAIL_CALLABLE_CALL__in_arr(z, n, data) \
    ndarray BOOST_PP_CAT(in_arr_,n) = boost::numpy::dstream::utils::bpobj_to_ndarray<typename MappingModel::BOOST_PP_CAT(in_arr_dshape_,n)>::apply(BOOST_PP_CAT(in_obj_,n));

#define BOOST_NUMPY_DSTREAM_DETAIL_CALLABLE_CALL__in_arr_n0(z, n, data) \
    intptr_t BOOST_PP_CAT(in_arr_n0_,n) = (BOOST_PP_CAT(in_arr_,n).get_nd() == 0 ? 0 : BOOST_PP_CAT(in_arr_,n).shape(0));

#define BOOST_NUMPY_DSTREAM_DETAIL_CALLABLE_CALL__in_arr_bcr(z, n, data) \
    std::vector<int> BOOST_PP_CAT(in_arr_bcr_,n);

#define BOOST_NUMPY_DSTREAM_DETAIL_CALLABLE_CALL__set_in_arr_bcr(z, n, data) \
    MappingModel::template set_in_arr_bcr<n>(BOOST_PP_CAT(in_arr_bcr_,n), BOOST_PP_CAT(in_arr_,n));

#define BOOST_NUMPY_DSTREAM_DETAIL_CALLABLE_CALL__in_op(z, n, data) \
    boost::numpy::detail::iter_operand BOOST_PP_CAT(in_op_,n) (BOOST_PP_CAT(in_arr_,n), MappingModel::template in_arr_iter_operand_flags<n>::value, &BOOST_PP_CAT(in_arr_bcr_,n).front());

template <>
struct callable_call_arity<N>
{
    template <
          bool has_void_return
        , class Class
        , class MappingModel
        , class WiringModel
        , class OutArrTransform
        , class ThreadAbility
    >
    struct callable_call;

    //--------------------------------------------------------------------------
    // Partial specialization for non-void-return C++ functions/member
    // functions.
    template <class Class, class MappingModel, class WiringModel, class OutArrTransform, class ThreadAbility>
    struct callable_call<false, Class, MappingModel, WiringModel, OutArrTransform, ThreadAbility>
    {
        static
        python::object
        call(
              WiringModel const & wiring_model
            , Class & self
            , BOOST_PP_ENUM_PARAMS(N, python::object const & in_obj_)
            , python::object & out_obj
            , int nthreads
        )
        {
            boost::numpy::detail::PyGIL pygil;

            //------------------------------------------------------------------
            // Construct the boost::numpy::ndarray objects for all the input
            // arrays, based on their required data shapes.
            BOOST_PP_REPEAT(N, BOOST_NUMPY_DSTREAM_DETAIL_CALLABLE_CALL__in_arr, ~)

            //------------------------------------------------------------------
            // Get the shape of the output array based on the used mapping
            // model.
            // Note: By definition of the data stream concept, the length of the
            //       first dimension of the output array is the maximum of the
            //       lengths of all the individual input arrays with a minimum
            //       of 1.
            bool is_out_arr_provided = (out_obj != python::object() ? true : false);

            BOOST_PP_REPEAT(N, BOOST_NUMPY_DSTREAM_DETAIL_CALLABLE_CALL__in_arr_n0, ~)

            intptr_t const n_axis_1_elements = boost::numpy::detail::max(intptr_t(1), BOOST_PP_ENUM_PARAMS(N, in_arr_n0_));

            std::vector<intptr_t> out_arr_shape(1, n_axis_1_elements);
            boost::numpy::dstream::utils::complete_shape<typename MappingModel::out_arr_dshape, MappingModel::out_arr_dshape::nd::value>::apply(out_arr_shape);

            dtype out_arr_dtype = dtype::get_builtin<typename MappingModel::out_arr_dshape::value_type>();
            ndarray out_arr = (is_out_arr_provided
                ? from_object(out_obj, out_arr_dtype, out_arr_shape.size(), ndarray::ALIGNED | ndarray::WRITEABLE)
                : boost::numpy::zeros(out_arr_shape, out_arr_dtype));

            if(! out_arr.has_shape(out_arr_shape))
            {
                PyErr_SetString(PyExc_ValueError,
                    "The shape of the output array does not fit the "
                    "requirements!");
                python::throw_error_already_set();
            }

            //------------------------------------------------------------------
            // Create the iterator for iterating over the input arrays and the
            // output array based on the used mapping model.
            //-- Create the iter_operand object for the output array.
            std::vector<int> out_arr_bcr;
            MappingModel::set_out_arr_bcr(out_arr_bcr);
            boost::numpy::detail::iter_operand out_op(out_arr, MappingModel::out_arr_iter_operand_flags, &out_arr_bcr.front());

            //-- Create the iter_operand objects for all the input arrays.
            BOOST_PP_REPEAT(N, BOOST_NUMPY_DSTREAM_DETAIL_CALLABLE_CALL__in_arr_bcr, ~)
            BOOST_PP_REPEAT(N, BOOST_NUMPY_DSTREAM_DETAIL_CALLABLE_CALL__set_in_arr_bcr, ~)
            BOOST_PP_REPEAT(N, BOOST_NUMPY_DSTREAM_DETAIL_CALLABLE_CALL__in_op, ~)

            //-- Get the itershape for the used mapping model.
            std::vector<intptr_t> itershape;
            MappingModel::set_itershape(itershape, n_axis_1_elements);

            //-- Define the iterator flags. In order to allow multi-threading
            //   the following flags MUST be set:
            //       EXTERNAL_LOOP, RANGED, BUFFERED, DELAY_BUFALLOC
            //   So we set these flags by default for all mapping models and add
            //   additional mapping model specific flags.
            boost::numpy::detail::iter_flags_t iter_flags =
                  boost::numpy::detail::iter::EXTERNAL_LOOP
                | boost::numpy::detail::iter::RANGED
                | boost::numpy::detail::iter::BUFFERED
                | boost::numpy::detail::iter::DELAY_BUFALLOC
                ;
            iter_flags |= MappingModel::iter_flags;

            //-- Finally, create the iterator object.
            boost::numpy::detail::iter iter(
                iter_flags,
                MappingModel::order,
                MappingModel::casting,
                MappingModel::n_iter_axes,
                &itershape.front(),
                MappingModel::buffersize,
                out_op,
                BOOST_PP_ENUM_PARAMS(N, in_op_));

            //------------------------------------------------------------------
            // Determine how many iterations/tasks each thread needs to perform.
            // This is done by rounding up n_axis_1_elements / nthreads. This
            // assumes, that n_axis_1_elements > 0 and nthreads > 0.
            if(! (nthreads > 0))
            {
                PyErr_SetString(PyExc_ValueError,
                    "The number of threads must be >= 1.");
                python::throw_error_already_set();
            }

            intptr_t n_tasks_per_thread =
                std::min(
                      n_axis_1_elements
                    , std::max(
                          1 + ((n_axis_1_elements - 1) / nthreads)
                        , intptr_t(ThreadAbility::min_n_tasks_per_thread_t::value)
                      )
                );
            nthreads = 1 + ((n_axis_1_elements - 1) / n_tasks_per_thread);

            // Calculate task size. The task size is the product of the shape
            // values of the broadcasted iteration array without the first
            // dimension.
            intptr_t const task_size = iter.get_iter_size() / n_axis_1_elements;

            intptr_t const parallel_iter_size = n_tasks_per_thread * task_size;

            std::cout << "Launching " << nthreads << " threads with " << n_tasks_per_thread << " tasks per thread and a task size of " << task_size << "." << std::endl;

            // Make nthreads - 1 copies of the iter object and store it inside
            // the vector. This will call NpyIter_Copy for each made copy.
            std::vector<boost::numpy::detail::iter> iter_vec(nthreads - 1, iter);

            // Initialize the iterators for their specific range of iteration.
            intptr_t const iter_size = iter.get_iter_size();
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
                    boost::thread *t = new boost::thread(&WiringModel::call
                        , boost::ref(self)
                        , boost::cref(wiring_model.GetConfig())
                        , boost::ref(*it)
                        , boost::ref(thread_error_flag)
                    );
                    threads.add_thread(t);
                }
            }
            catch(boost::thread_resource_error)
            {
                // TODO: Is pygil.acquire(); needed here???
                PyErr_SetString(PyExc_RuntimeError,
                    "At least one thread could not be launched due to a resource "
                    "error.");
                boost::python::throw_error_already_set();
            }

            // Do the iteration for the first iterator.
            WiringModel::call(self, wiring_model.GetConfig(), iter, thread_error_flag);

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

            if(is_out_arr_provided) {
                return boost::python::object();
            }

            //------------------------------------------------------------------
            // Transform the output array based on the configured output array
            // transformation.
            OutArrTransform::apply(out_arr, BOOST_PP_ENUM_PARAMS(N, in_arr_));

            return out_arr;
        }
    };

    //--------------------------------------------------------------------------
    // Partial specialization for void-return C++ functions/member
    // functions.
    template <class Class, class MappingModel, class WiringModel, class OutArrTransform, class ThreadAbility>
    struct callable_call<true, Class, MappingModel, WiringModel, OutArrTransform, ThreadAbility>
    {
        static
        python::object
        call(
              WiringModel const & wiring_model
            , Class & self
            , BOOST_PP_ENUM_PARAMS(N, python::object const & in_obj_)
            , int nthreads
        )
        {
            boost::numpy::detail::PyGIL pygil;

            //------------------------------------------------------------------
            // Construct the boost::numpy::ndarray objects for all the input
            // arrays, based on their required data shapes.
            BOOST_PP_REPEAT(N, BOOST_NUMPY_DSTREAM_DETAIL_CALLABLE_CALL__in_arr, ~)

            //------------------------------------------------------------------
            // Get the length of the data stream.
            // Note: By definition of the data stream concept, the length of the
            //       first dimension of the output array is the maximum of the
            //       lengths of all the individual input arrays with a minimum
            //       of 1.
            BOOST_PP_REPEAT(N, BOOST_NUMPY_DSTREAM_DETAIL_CALLABLE_CALL__in_arr_n0, ~)

            intptr_t const n_axis_1_elements = boost::numpy::detail::max(intptr_t(1), BOOST_PP_ENUM_PARAMS(N, in_arr_n0_));

            //------------------------------------------------------------------
            // Create the iterator for iterating over the input arrays based on
            // the used mapping model.
            //-- Create the iter_operand objects for all the input arrays.
            BOOST_PP_REPEAT(N, BOOST_NUMPY_DSTREAM_DETAIL_CALLABLE_CALL__in_arr_bcr, ~)
            BOOST_PP_REPEAT(N, BOOST_NUMPY_DSTREAM_DETAIL_CALLABLE_CALL__set_in_arr_bcr, ~)
            BOOST_PP_REPEAT(N, BOOST_NUMPY_DSTREAM_DETAIL_CALLABLE_CALL__in_op, ~)

            //-- Get the itershape for the used mapping model.
            std::vector<intptr_t> itershape;
            MappingModel::set_itershape(itershape, n_axis_1_elements);

            //-- Define the iterator flags. In order to allow multi-threading
            //   the following flags MUST be set:
            //       EXTERNAL_LOOP, RANGED, BUFFERED, DELAY_BUFALLOC
            //   So we set these flags by default for all mapping models and add
            //   additional mapping model specific flags.
            boost::numpy::detail::iter_flags_t iter_flags =
                  boost::numpy::detail::iter::EXTERNAL_LOOP
                | boost::numpy::detail::iter::RANGED
                | boost::numpy::detail::iter::BUFFERED
                | boost::numpy::detail::iter::DELAY_BUFALLOC
                ;
            iter_flags |= MappingModel::iter_flags;

            //-- Finally, create the iterator object.
            boost::numpy::detail::iter iter(
                  iter_flags
                , MappingModel::order
                , MappingModel::casting
                , MappingModel::n_iter_axes
                , &itershape.front()
                , MappingModel::buffersize
                , BOOST_PP_ENUM_PARAMS(N, in_op_));

            //------------------------------------------------------------------
            // Determine how many iterations/tasks each thread needs to perform.
            // This is done by rounding up n_axis_1_elements / nthreads. This
            // assumes, that n_axis_1_elements > 0 and nthreads > 0.
            if(! (nthreads > 0))
            {
                PyErr_SetString(PyExc_ValueError,
                    "The number of threads must be >= 1.");
                python::throw_error_already_set();
            }

            intptr_t n_tasks_per_thread =
                std::min(
                      n_axis_1_elements
                    , std::max(
                          1 + ((n_axis_1_elements - 1) / nthreads)
                        , intptr_t(ThreadAbility::min_n_tasks_per_thread_t::value)
                      )
                );
            nthreads = 1 + ((n_axis_1_elements - 1) / n_tasks_per_thread);

            // Calculate task size. The task size is the product of the shape
            // values of the broadcasted iteration array without the first
            // dimension.
            intptr_t const task_size = iter.get_iter_size() / n_axis_1_elements;

            intptr_t const parallel_iter_size = n_tasks_per_thread * task_size;

            std::cout << "Launching " << nthreads << " threads with " << n_tasks_per_thread << " tasks per thread and a task size of " << task_size << "." << std::endl;

            // Make nthreads - 1 copies of the iter object and store it inside
            // the vector. This will call NpyIter_Copy for each made copy.
            std::vector<boost::numpy::detail::iter> iter_vec(nthreads - 1, iter);

            // Initialize the iterators for their specific range of iteration.
            intptr_t const iter_size = iter.get_iter_size();
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
                    boost::thread *t = new boost::thread(&WiringModel::call
                        , boost::ref(self)
                        , boost::cref(wiring_model.GetConfig())
                        , boost::ref(*it)
                        , boost::ref(thread_error_flag)
                    );
                    threads.add_thread(t);
                }
            }
            catch(boost::thread_resource_error)
            {
                // TODO: Is pygil.acquire(); needed here???
                PyErr_SetString(PyExc_RuntimeError,
                    "At least one thread could not be launched due to a resource "
                    "error.");
                boost::python::throw_error_already_set();
            }

            // Do the iteration for the first iterator.
            WiringModel::call(self, wiring_model.GetConfig(), iter, thread_error_flag);

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

            return boost::python::object();
        }
    };
};

#undef BOOST_NUMPY_DSTREAM_DETAIL_CALLABLE_CALL__in_op
#undef BOOST_NUMPY_DSTREAM_DETAIL_CALLABLE_CALL__set_in_arr_bcr
#undef BOOST_NUMPY_DSTREAM_DETAIL_CALLABLE_CALL__in_arr_bcr
#undef BOOST_NUMPY_DSTREAM_DETAIL_CALLABLE_CALL__in_arr_n0
#undef BOOST_NUMPY_DSTREAM_DETAIL_CALLABLE_CALL__in_arr

#undef N

#endif // BOOST_PP_IS_ITERATING
