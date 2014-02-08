/**
 * $Id$
 *
 * Copyright (C)
 * 2013 - $Date$
 *     Martin Wolf <boostnumpy@martin-wolf.org>
 *
 * @file    boost/numpy/detail/iter.hpp
 * @version $Revision$
 * @date    $Date$
 * @author  Martin Wolf <boostnumpy@martin-wolf.org>
 *
 * @brief This file defines the boost::numpy::detail::iter class providing
 *        a multi-operand iterator for boost::numpy::ndarray objects. It
 *        uses the numpy NpyIter C-API object for the iteration through the
 *        arrays.
 *
 *        The reason why it is put into the detail namespace is, that the
 *        numpy NpyIter C-API object is not exposed to Python and can be used
 *        only internally.
 *
 *        This file is distributed under the Boost Software License,
 *        Version 1.0. (See accompanying file LICENSE_1_0.txt or copy at
 *        http://www.boost.org/LICENSE_1_0.txt).
 */
#if !BOOST_PP_IS_ITERATING
#ifndef BOOST_NUMPY_DETAIL_ITER_HPP_INCLUDED
#define BOOST_NUMPY_DETAIL_ITER_HPP_INCLUDED

#include <boost/python.hpp>

#include <boost/preprocessor/cat.hpp>
#include <boost/preprocessor/arithmetic/add.hpp>
#include <boost/preprocessor/iteration/iterate.hpp>
#include <boost/preprocessor/punctuation/comma.hpp>
#include <boost/preprocessor/repetition/enum_params.hpp>
#include <boost/preprocessor/repetition/repeat.hpp>

#include <boost/numpy/numpy_c_api.hpp>
#include <boost/numpy/ndarray.hpp>
#include <boost/numpy/types.hpp>

namespace boost {
namespace numpy {
namespace detail {

//==============================================================================
typedef npy_uint32 iter_operand_flags_t;

struct iter_operand
{
    enum flags_t
    {
          NONE         = 0x0
        , READWRITE    = NPY_ITER_READWRITE
        , READONLY     = NPY_ITER_READONLY
        , WRITEONLY    = NPY_ITER_WRITEONLY
        , COPY         = NPY_ITER_COPY
        , UPDATEIFCOPY = NPY_ITER_UPDATEIFCOPY
        , NBO          = NPY_ITER_NBO
        , ALIGNED      = NPY_ITER_ALIGNED
        , CONTIG       = NPY_ITER_CONTIG
        , ALLOCATE     = NPY_ITER_ALLOCATE
        , NO_SUBTYPE   = NPY_ITER_NO_SUBTYPE
        , NO_BROADCAST = NPY_ITER_NO_BROADCAST
        #if NPY_FEATURE_VERSION >= 0x00000007
        , ARRAYMASK     = NPY_ITER_ARRAYMASK
        , WRITEMASKED   = NPY_ITER_WRITEMASKED
        #endif
    };

    ndarray const &      ndarray_;
    iter_operand_flags_t flags_;
    int *                broadcasting_rules_;

    iter_operand(ndarray const & arr, iter_operand_flags_t f, int * bcr)
      : ndarray_(arr),
        flags_(f),
        broadcasting_rules_(bcr)
    {}
};

inline
iter_operand_flags_t
operator|(iter_operand::flags_t a, iter_operand::flags_t b)
{
    return (iter_operand_flags_t(a) | iter_operand_flags_t(b));
}

inline
iter_operand_flags_t
operator&(iter_operand::flags_t a, iter_operand::flags_t b)
{
    return (iter_operand_flags_t(a) & iter_operand_flags_t(b));
}

//==============================================================================
typedef npy_uint32 iter_flags_t;

class iter
{
  public:
    enum flags_t
    {
          NONE                = 0x0
        , C_INDEX             = NPY_ITER_C_INDEX
        , F_INDEX             = NPY_ITER_F_INDEX
        , MULTI_INDEX         = NPY_ITER_MULTI_INDEX
        , EXTERNAL_LOOP       = NPY_ITER_EXTERNAL_LOOP
        , DONT_NEGATE_STRIDES = NPY_ITER_DONT_NEGATE_STRIDES
        , COMMON_DTYPE        = NPY_ITER_COMMON_DTYPE
        , REFS_OK             = NPY_ITER_REFS_OK
        , ZEROSIZE_OK         = NPY_ITER_ZEROSIZE_OK
        , REDUCE_OK           = NPY_ITER_REDUCE_OK
        , RANGED              = NPY_ITER_RANGED
        , BUFFERED            = NPY_ITER_BUFFERED
        , GROWINNER           = NPY_ITER_GROWINNER
        , DELAY_BUFALLOC      = NPY_ITER_DELAY_BUFALLOC
    };

    //__________________________________________________________________________
    /**
     * \brief Constructs a multi operand iterator for 1+N operands.
     *
     * \param n_iter_axes The number of axes that will be iterated.
     */
    #define BOOST_PP_ITERATION_PARAMS_1 (4, (1, BOOST_NUMPY_LIMIT_INPUT_AND_OUTPUT_ARITY, <boost/numpy/detail/iter.hpp>, 1))
    #include BOOST_PP_ITERATE()

    //__________________________________________________________________________
    /**
     * \brief The destructor deallocates the internal numpy iterator object.
     */
    ~iter();

    //__________________________________________________________________________
    /**
     * \brief The copy constructor. It will call NpyIter_Copy to make a new
     *     copy of the iterator python object.
     */
    iter(iter const & it);

    //__________________________________________________________________________
    /**
     * \brief The assignment operator. It will deallocate the current iterator
     *     object and will make a copy using NpyIter_Copy of the right-hand-side
     *     iterator object, and storing that copied iterator python object.
     */
    iter& operator=(iter const & rhs);

    //__________________________________________________________________________
    /**
     * \brief Initializes the iteration by calling the NpyIter_Reset function.
     *     The iteration will be over the full itersize.
     *     It also gets the nextiter function pointer, the pointer to the data
     *     pointer array, the pointer to the inner loop stride array, and the
     *     pointer to the inner loop size.
     */
    void
    init_full_iteration();

    //__________________________________________________________________________
    /**
     * \brief Initializes the iteration by calling the
     *     NpyIter_ResetToIterIndexRange function. This is used to implement
     *     threading, where each thread iterates only over the range
     *     [istart, iend).
     *     It also gets the nextiter function pointer, the pointer to the data
     *     pointer array, the pointer to the inner loop stride array, and the
     *     pointer to the inner loop size.
     */
    void
    init_ranged_iteration(intptr_t istart, intptr_t iend);

    //__________________________________________________________________________
    /**
     * Does the next iteration. It returns ``true`` if there is still an
     * iteration left and ``false`` otherwise.
     */
    inline
    bool
    next()
    {
        return this->iter_next_func_(npyiter_);
    }

    //__________________________________________________________________________
    /**
     * \brief Returns the pointer to the data of the i-th operand for the
     *     current iteration. If the EXTERNAL_LOOP flag was set for the iterator
     *     each pointer points to the first element of the inner most loop.
     *     Note: The index 0 denotes the output operand, and 1 the first input
     *     operand, 2 the second input operand and so forth.
     */
    inline
    char*
    get_data(int i)
    {
        return data_ptr_array_ptr_[i];
    }

    //__________________________________________________________________________
    /**
     * \brief Returns the size of the inner loop, i.e. how many array elements
     *     are treated with one iteration of the inner most loop.
     */
    inline
    intptr_t
    get_inner_loop_size() const
    {
        return *inner_loop_size_ptr_;
    }

    //__________________________________________________________________________
    /**
     * \brief Returns the total number of elements that needs to be iterated
     *     over.
     */
    intptr_t
    get_iter_size() const;

    //__________________________________________________________________________
    /**
     * \brief Returns the number of operands the iterator handles.
     */
    int
    get_nop() const;

    //__________________________________________________________________________
    /**
     * \brief Returns the stride for the i'th iterator operand.
     */
    inline
    intptr_t
    get_stride(size_t i) const
    {
        return inner_loop_stride_array_ptr_[i];
    }

    //__________________________________________________________________________
    /**
     * \brief Adds n[I]*inner_loop_stride_array_ptr_[I] to the data pointers of
     *     the iterator operands, where n is a pointer to an array of integers
     *     of size nop specifying how many elements to jump in each operand
     *     array.
     */
    inline
    void
    add_strides_to_data_ptrs(int const nop, int const *n)
    {
        for(int iop=0; iop<nop; ++iop)
        {
            data_ptr_array_ptr_[iop] += n[iop] * inner_loop_stride_array_ptr_[iop];
        }
    }

  protected:
    /**
     * \brief Gathers the different pointers used for iteration, i.e. the
     *     nextiter function pointer, the pointer to the data
     *     pointer array, the pointer to the inner loop stride array, and the
     *     pointer to the inner loop size.
     */
    void
    gather_iteration_pointers();

    //__________________________________________________________________________
    /**
     * \brief The pointer to the NpyIter PyObject holding the C iterator.
     */
    NpyIter* npyiter_;

    /**
     * \brief Stores the pointer to iteration function.
     */
    NpyIter_IterNextFunc* iter_next_func_;

    /**
     * \brief The pointer to the array of pointers pointing to the operand data
     *     for the particular iteration.
     */
    char** data_ptr_array_ptr_;

    /**
     * \brief The pointer to the inner loop stride array. The i-th array element
     *     holds the inner loop stride for the i-th operand.
     */
    npy_intp* inner_loop_stride_array_ptr_;

    /**
     * \brief The pointer to the integer variable holding the size of the inner
     *     loop for the current iteration.
     */
    npy_intp* inner_loop_size_ptr_;
};

inline
iter_flags_t
operator|(iter::flags_t a, iter::flags_t b)
{
    return (iter_flags_t(a) | iter_flags_t(b));
}

inline
iter_flags_t
operator&(iter::flags_t a, iter::flags_t b)
{
    return (iter_flags_t(a) & iter_flags_t(b));
}

}// namespace detail
}// namespace numpy
}// namespace boost

#endif // !BOOST_NUMPY_DETAIL_ITER_HPP_INCLUDED

//==============================================================================
// Constructor decleration for N operands.
#elif BOOST_PP_ITERATION_FLAGS() == 1

#define N BOOST_PP_ITERATION()

iter(
      iter_flags_t iter_flags
    , order_t      order
    , casting_t    casting
    , int          n_iter_axes
    , intptr_t *   itershape
    , intptr_t     buffersize
    , BOOST_PP_ENUM_PARAMS(N, iter_operand const & op_)
);

#undef N

//==============================================================================
// Constructor definition for N operands.
#elif BOOST_PP_ITERATION_FLAGS() == 2

#define N BOOST_PP_ITERATION()

#define BOOST_NUMPY_DETAIL_ITER__op_ndarray_ptr(z, n, data)                    \
    BOOST_PP_COMMA_IF(n) reinterpret_cast<PyArrayObject*>(BOOST_PP_CAT(op_,n).ndarray_.ptr())

#define BOOST_NUMPY_DETAIL_ITER__op_flags(z, n, data)                          \
    BOOST_PP_COMMA_IF(n) BOOST_PP_CAT(op_,n).flags_

#define BOOST_NUMPY_DETAIL_ITER__op_dtype_ptr(z, n, data)                      \
    BOOST_PP_COMMA_IF(n) reinterpret_cast<PyArray_Descr*>(BOOST_PP_CAT(op_,n).ndarray_.get_dtype().ptr())

#define BOOST_NUMPY_DETAIL_ITER__op_bcr(z, n, data)                            \
    BOOST_PP_COMMA_IF(n) BOOST_PP_CAT(op_,n).broadcasting_rules_

#define BOOST_NUMPY_DETAIL_ITER__print_op_bcr(z, n, data) \
    print_c_array(BOOST_PP_CAT(op_,n).broadcasting_rules_, n_iter_axes);

iter::
iter(
      iter_flags_t iter_flags
    , order_t      order
    , casting_t    casting
    , int          n_iter_axes
    , intptr_t *   itershape
    , intptr_t     buffersize
    , BOOST_PP_ENUM_PARAMS(N, iter_operand const & op_)
)
{
    npy_intp nop = N;

    PyArrayObject* op[] = {
        BOOST_PP_REPEAT(N, BOOST_NUMPY_DETAIL_ITER__op_ndarray_ptr, ~)
    };

    npy_uint32 op_flags[] = {
        BOOST_PP_REPEAT(N, BOOST_NUMPY_DETAIL_ITER__op_flags, ~)
    };

    PyArray_Descr* op_dtypes[] = {
        BOOST_PP_REPEAT(N, BOOST_NUMPY_DETAIL_ITER__op_dtype_ptr, ~)
    };

    int* op_axes[] = {
        BOOST_PP_REPEAT(N, BOOST_NUMPY_DETAIL_ITER__op_bcr, ~)
    };

    print_c_array(itershape, n_iter_axes);
    BOOST_PP_REPEAT(N, BOOST_NUMPY_DETAIL_ITER__print_op_bcr, ~)

    npyiter_ = NpyIter_AdvancedNew(
          nop
        , op
        , iter_flags
        , NPY_ORDER(order)
        , NPY_CASTING(casting)
        , op_flags
        , op_dtypes
        , n_iter_axes
        , op_axes
        , itershape
        , buffersize
    );
    if(npyiter_ == NULL)
    {
        python::throw_error_already_set();
    }
}

#undef BOOST_NUMPY_DETAIL_ITER__print_op_bcr

#undef BOOST_NUMPY_DETAIL_ITER__op_bcr
#undef BOOST_NUMPY_DETAIL_ITER__op_dtype_ptr
#undef BOOST_NUMPY_DETAIL_ITER__op_flags
#undef BOOST_NUMPY_DETAIL_ITER__op_ndarray_ptr

#undef N

#endif // !BOOST_PP_IS_ITERATING
