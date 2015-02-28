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

namespace boost {
namespace numpy {
namespace detail {

//______________________________________________________________________________
// Generate the constructor definitions for the different input arities.
#define BOOST_PP_ITERATION_PARAMS_1 \
    (4, (1, BOOST_NUMPY_LIMIT_INPUT_AND_OUTPUT_ARITY, <boost/numpy/detail/iter.hpp>, 2))
#include BOOST_PP_ITERATE()

//______________________________________________________________________________
iter::
~iter()
{
    assert(npyiter_);
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
    //std::cout << "detail::iter: Copy constructor BEGIN."<<std::endl;
    npyiter_ = NpyIter_Copy(it.npyiter_);
    if(npyiter_ == NULL)
    {
        python::throw_error_already_set();
    }
    gather_iteration_pointers();
    //std::cout << "detail::iter: Copy constructor END."<<std::endl;
}

//______________________________________________________________________________
iter&
iter::
operator=(iter const & rhs)
{
    //std::cout << "detail::iter: Assignemnt operator."<<std::endl;
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
    gather_iteration_pointers();

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
    descr_ptr_array_ptr_         = NpyIter_GetDescrArray(npyiter_);
    char errmsg[2048];
    char * errmsgptr = &errmsg[0];
    get_multi_index_func_ = NpyIter_GetGetMultiIndex(npyiter_, &errmsgptr);
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
bool
iter::
iteration_needs_api()
{
    return NpyIter_IterationNeedsAPI(npyiter_);
}

//______________________________________________________________________________
intptr_t
iter::
get_iter_index() const
{
    return NpyIter_GetIterIndex(npyiter_);
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

//______________________________________________________________________________
int
iter::
get_ndim() const
{
    return NpyIter_GetNDim(npyiter_);
}

//______________________________________________________________________________
std::vector<intptr_t>
iter::
get_iteration_axis_strides(int axis)
{
    int const nop = get_nop();
    intptr_t * axis_strides_carray = NpyIter_GetAxisStrideArray(npyiter_, axis);
    std::vector<intptr_t> axis_strides(nop, 0);
    for(int i=0; i<nop; ++i)
    {
        axis_strides[i] = axis_strides_carray[i];
    }
    return axis_strides;
}

//______________________________________________________________________________
std::vector<intptr_t>
iter::
get_inner_loop_fixed_strides()
{
    std::vector<intptr_t> strides(get_nop(), 0);
    NpyIter_GetInnerFixedStrideArray(npyiter_, &strides.front());
    return strides;
}

//______________________________________________________________________________
ndarray
iter::
get_operand(size_t op_idx)
{
    PyArrayObject** op_carray = NpyIter_GetOperandArray(npyiter_);
    return ndarray(python::detail::borrowed_reference(op_carray[op_idx]));
}

//______________________________________________________________________________
void
iter::
jump_to(std::vector<intptr_t> const & indices)
{
    if(NpyIter_GotoMultiIndex(npyiter_, (npy_intp*)const_cast<intptr_t*>(&indices[0])) == NPY_FAIL)
    {
        python::throw_error_already_set();
    }
}

//______________________________________________________________________________
void
iter::
jump_to_iter_index(intptr_t iteridx)
{
    if(NpyIter_GotoIterIndex(npyiter_, iteridx) == NPY_FAIL)
    {
        python::throw_error_already_set();
    }
}

//______________________________________________________________________________
bool
iter::
reset(bool throws)
{
    if(throws)
    {
        if(NpyIter_Reset(npyiter_, NULL) == NPY_FAIL)
        {
            python::throw_error_already_set();
            return false;
        }
    }
    else
    {
        char errmsg[2048];
        char * errmsgptr = &errmsg[0];
        if(NpyIter_Reset(npyiter_, &errmsgptr) == NPY_FAIL)
        {
            return false;
        }
    }
    return true;
}

void
iter::
get_multi_index_vector(std::vector<intptr_t> & multindex) const
{
    if(get_multi_index_func_ == NULL)
    {
        PyErr_SetString(PyExc_RuntimeError,
            "The NpyIter iterator object does not carry a multi-index, which "
            "is required by the get_multi_index_vector method!");
        python::throw_error_already_set();
    }

    if(multindex.size() < get_ndim())
    {
        std::stringstream ss;
        ss << "The size of the multi-index vector must be at least "
           << get_ndim() << "!";
        PyErr_SetString(PyExc_RuntimeError, ss.str().c_str());
        python::throw_error_already_set();
    }

    get_multi_index_func_(npyiter_, &multindex[0]);
}

std::vector<intptr_t>
iter::
get_multi_index_vector() const
{
    std::vector<intptr_t> multindex(get_ndim());
    get_multi_index_vector(multindex);
    return multindex;
}

}// namespace detail
}// namespace numpy
}// namespace boost
