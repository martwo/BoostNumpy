/**
 * $Id$
 *
 * Copyright (C)
 * 2013 - $Date$
 *     Martin Wolf <boostnumpy@martin-wolf.org>
 *
 * \file    boost/numpy/detail/iter.cpp
 * \version $Revision$
 * \date    $Date$
 * \author  Martin Wolf <boostnumpy@martin-wolf.org>
 *
 * \brief This file implements the boost::numpy::detail::iter class.
 *
 *        This file is distributed under the Boost Software License,
 *        Version 1.0. (See accompanying file LICENSE_1_0.txt or copy at
 *        http://www.boost.org/LICENSE_1_0.txt).
 */
#define BOOST_NUMPY_INTERNAL_IMPL
#include <boost/numpy/internal_impl.hpp>
#include <boost/numpy/numpy_c_api.hpp>

#include <iostream>

#include <boost/preprocessor/iteration/iterate.hpp>

#include <boost/numpy/limits.hpp>
#include <boost/numpy/detail/iter.hpp>

#define print_c_array(NAME, SIZE) \
    std::cout << #NAME << ": [";\
    for(int i=0; i<SIZE; ++i)\
    {\
        if(i) std::cout << ", ";\
        std::cout << NAME[i];\
    }\
    std::cout << "]" << std::endl;

namespace boost {
namespace numpy {
namespace detail {

//______________________________________________________________________________
// Generate the constructor definitions for the different input arities.
#define BOOST_PP_ITERATION_PARAMS_1 (4, (1, BOOST_NUMPY_LIMIT_INPUT_AND_OUTPUT_ARITY, <boost/numpy/detail/iter.hpp>, 2))
#include BOOST_PP_ITERATE()

//______________________________________________________________________________
iter::
~iter()
{
    if(NpyIter_Deallocate(npyiter_) != NPY_SUCCEED)
    {
        PyErr_SetString(PyExc_RuntimeError,
            "The NpyIter iterator object could not be deallocated!");
        python::throw_error_already_set();
    }
}

//______________________________________________________________________________
iter::
iter(iter const & it)
{
    npyiter_ = NpyIter_Copy(it.npyiter_);
    if(npyiter_ == NULL)
    {
        python::throw_error_already_set();
    }
}

//______________________________________________________________________________
iter&
iter::
operator=(iter const & rhs)
{
    if(NpyIter_Deallocate(npyiter_) != NPY_SUCCEED)
    {
        PyErr_SetString(PyExc_RuntimeError,
            "The NpyIter iterator object could not be deallocated!");
        python::throw_error_already_set();
    }

    npyiter_ = NpyIter_Copy(rhs.npyiter_);
    if(npyiter_ == NULL)
    {
        python::throw_error_already_set();
    }

    return *this;
}

//______________________________________________________________________________
void
iter::
gather_iteration_pointers()
{
    iter_next_func_ = NpyIter_GetIterNext(npyiter_, NULL);
    if(iter_next_func_ == NULL) {
        python::throw_error_already_set();
    }

    data_ptr_array_ptr_          = NpyIter_GetDataPtrArray(npyiter_);
    inner_loop_stride_array_ptr_ = NpyIter_GetInnerStrideArray(npyiter_);
    inner_loop_size_ptr_         = NpyIter_GetInnerLoopSizePtr(npyiter_);
}

//______________________________________________________________________________
void
iter::
init_full_iteration()
{
    if(NpyIter_Reset(npyiter_, NULL) != NPY_SUCCEED) {
        python::throw_error_already_set();
    }

    gather_iteration_pointers();
}

//______________________________________________________________________________
void
iter::
init_ranged_iteration(intptr_t istart, intptr_t iend)
{
    if(NpyIter_ResetToIterIndexRange(npyiter_, istart, iend, NULL) != NPY_SUCCEED) {
        python::throw_error_already_set();
    }

    gather_iteration_pointers();
}

//______________________________________________________________________________
intptr_t
iter::
get_iter_size() const
{
    return NpyIter_GetIterSize(npyiter_);
}

//______________________________________________________________________________
int
iter::
get_nop() const
{
    return NpyIter_GetNOp(npyiter_);
}

}// namespace detail
}// namespace numpy
}// namespace boost
