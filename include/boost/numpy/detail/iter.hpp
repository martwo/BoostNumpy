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
#define BOOST_NUMPY_DETAIL_ITER_HPP_INCLUDED 1

#include <limits>
#include <sstream>
#include <string>

#include <boost/python.hpp>

#include <boost/preprocessor/cat.hpp>
#include <boost/preprocessor/arithmetic/add.hpp>
#include <boost/preprocessor/iteration/iterate.hpp>
#include <boost/preprocessor/punctuation/comma.hpp>
#include <boost/preprocessor/repetition/enum_params.hpp>
#include <boost/preprocessor/repetition/repeat.hpp>
#include <boost/preprocessor/stringize.hpp>

#include <boost/numpy/limits.hpp>
#include <boost/numpy/numpy_c_api.hpp>
#include <boost/numpy/detail/logging.hpp>
#include <boost/numpy/ndarray.hpp>
#include <boost/numpy/types.hpp>

namespace boost {
namespace numpy {
namespace detail {

template <typename ArrElementT>
std::string
c_array_to_string(ArrElementT * arr, size_t size)
{
    std::stringstream ss;
    ss << "[";
    for(size_t i=0; i<size; ++i)
    {
        if(i) ss << ", ";
        ss << arr[i];
    }
    ss << "]";
    return ss.str();
}

//==============================================================================
typedef npy_uint32 iter_operand_flags_t;

struct iter_operand
{
    struct flags
    {
        typedef boost::mpl::integral_c<iter_operand_flags_t, 0>                     NONE;
        typedef boost::mpl::integral_c<iter_operand_flags_t, NPY_ITER_READWRITE>    READWRITE;
        typedef boost::mpl::integral_c<iter_operand_flags_t, NPY_ITER_READONLY>     READONLY;
        typedef boost::mpl::integral_c<iter_operand_flags_t, NPY_ITER_WRITEONLY>    WRITEONLY;
        typedef boost::mpl::integral_c<iter_operand_flags_t, NPY_ITER_COPY>         COPY;
        typedef boost::mpl::integral_c<iter_operand_flags_t, NPY_ITER_UPDATEIFCOPY> UPDATEIFCOPY;
        typedef boost::mpl::integral_c<iter_operand_flags_t, NPY_ITER_NBO>          NBO;
        typedef boost::mpl::integral_c<iter_operand_flags_t, NPY_ITER_ALIGNED>      ALIGNED;
        typedef boost::mpl::integral_c<iter_operand_flags_t, NPY_ITER_CONTIG>       CONTIG;
        typedef boost::mpl::integral_c<iter_operand_flags_t, NPY_ITER_ALLOCATE>     ALLOCATE;
        typedef boost::mpl::integral_c<iter_operand_flags_t, NPY_ITER_NO_SUBTYPE>   NO_SUBTYPE;
        typedef boost::mpl::integral_c<iter_operand_flags_t, NPY_ITER_NO_BROADCAST> NO_BROADCAST;
        #if NPY_FEATURE_VERSION >= 0x00000007
        typedef boost::mpl::integral_c<iter_operand_flags_t, NPY_ITER_ARRAYMASK>    ARRAYMASK;
        typedef boost::mpl::integral_c<iter_operand_flags_t, NPY_ITER_WRITEMASKED>  WRITEMASKED;
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

//==============================================================================
typedef npy_uint32 iter_flags_t;

class iter
{
  public:
    struct flags
    {
        typedef boost::mpl::integral_c<iter_flags_t, 0>                            NONE;
        typedef boost::mpl::integral_c<iter_flags_t, NPY_ITER_C_INDEX>             C_INDEX;
        typedef boost::mpl::integral_c<iter_flags_t, NPY_ITER_F_INDEX>             F_INDEX;
        typedef boost::mpl::integral_c<iter_flags_t, NPY_ITER_MULTI_INDEX>         MULTI_INDEX;
        typedef boost::mpl::integral_c<iter_flags_t, NPY_ITER_EXTERNAL_LOOP>       EXTERNAL_LOOP;
        typedef boost::mpl::integral_c<iter_flags_t, NPY_ITER_DONT_NEGATE_STRIDES> DONT_NEGATE_STRIDES;
        typedef boost::mpl::integral_c<iter_flags_t, NPY_ITER_COMMON_DTYPE>        COMMON_DTYPE;
        typedef boost::mpl::integral_c<iter_flags_t, NPY_ITER_REFS_OK>             REFS_OK;
        typedef boost::mpl::integral_c<iter_flags_t, NPY_ITER_ZEROSIZE_OK>         ZEROSIZE_OK;
        typedef boost::mpl::integral_c<iter_flags_t, NPY_ITER_REDUCE_OK>           REDUCE_OK;
        typedef boost::mpl::integral_c<iter_flags_t, NPY_ITER_RANGED>              RANGED;
        typedef boost::mpl::integral_c<iter_flags_t, NPY_ITER_BUFFERED>            BUFFERED;
        typedef boost::mpl::integral_c<iter_flags_t, NPY_ITER_GROWINNER>           GROWINNER;
        typedef boost::mpl::integral_c<iter_flags_t, NPY_ITER_DELAY_BUFALLOC>      DELAY_BUFALLOC;
    };

    //__________________________________________________________________________
    /**
     * \brief Constructs a multi operand iterator for 1+N operands.
     *
     * \param n_iter_axes The number of axes that will be iterated.
     */
    #define BOOST_PP_ITERATION_PARAMS_1 \
        (4, (1, BOOST_NUMPY_LIMIT_INPUT_AND_OUTPUT_ARITY, <boost/numpy/detail/iter.hpp>, 1))
    #include BOOST_PP_ITERATE()

    //__________________________________________________________________________
    /**
     * \brief The destructor deallocates the internal numpy iterator object.
     */
    virtual
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
     * \brief Checks if the iteration needs the Python API. If true, the Python
     *        global interpreter lock (GIL) may not released during the
     *        iteration.
     * \internal Calls the NpyIter_IterationNeedsAPI numpy function.
     */
    bool
    iteration_needs_api();

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
     * \brief Returns the size in bytes of an item (i.e. an element) of the
     *     iterator's i'th operand array.
     */
    inline
    intptr_t
    get_item_size(size_t i)
    {
        return descr_ptr_array_ptr_[i]->elsize;
    }

    //__________________________________________________________________________
    /**
     * \brief Returns the iteration index of the iterator, which is an index
     *        matching the iteration order of the iterator.
     *
     * \note The iterator must be created with the C_INDEX or F_INDEX iterator
     *       flag set.
     */
    intptr_t
    get_iter_index() const;

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
     * \brief Returns the inner loop stride (in bytes) of the i'th iterator
     *     operand.
     */
    inline
    intptr_t
    get_inner_loop_stride(size_t i) const
    {
        return inner_loop_stride_array_ptr_[i];
    }

    //__________________________________________________________________________
    /**
     * \brief Moves the data pointers to the location pointed by the specified
     *     indices.
     *     It throws an exception if there was an error.
     */
    void
    jump_to(std::vector<intptr_t> const & indices);

    //__________________________________________________________________________
    /**
     * \brief Moves the data pointers to the location pointed by the specified
     *     iteration index. (See also the method get_iter_index())
     *     It throws an exception if there was an error.
     */
    void
    jump_to_iter_index(intptr_t iteridx);

    //__________________________________________________________________________
    /**
     * \brief Adds inner_loop_stride_array_ptr_[I] to the data pointer of the
     *     iterator's I'th operand. It does this for all operands.
     */
    inline
    void
    add_inner_loop_strides_to_data_ptrs()
    {
        int const nop = get_nop();
        for(int iop=0; iop<nop; ++iop)
        {
            data_ptr_array_ptr_[iop] += inner_loop_stride_array_ptr_[iop];
        }
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

    //__________________________________________________________________________
    /**
     * \brief Retrieves the stides of the specified iteration axis for all
     *        iterator operands.
     * \internal Calls NpyIter_GetAxisStrideArray for the specified axis.
     */
    std::vector<intptr_t>
    get_iteration_axis_strides(int axis);

    //__________________________________________________________________________
    /**
     * \brief Retrieves the fixed inner loop stide for each iterator operand.
     *        If the inner loop stide for an operand is not fixed, it will have
     *        the value std::numeric_limits<intptr_t>::max().
     * \internal Calls NpyIter_GetInnerFixedStrideArray.
     */
    std::vector<intptr_t>
    get_inner_loop_fixed_strides();

    //__________________________________________________________________________
    /**
     * \brief Retrieves the fixed inner loop stride of the specified iterator
     *        operand. If the inner loop stide is not fixed, it will have
     *        the value std::numeric_limits<intptr_t>::max().
     */
    intptr_t
    get_inner_loop_fixed_stride(size_t op_idx)
    {
        return get_inner_loop_fixed_strides()[op_idx];
    }

    //__________________________________________________________________________
    /**
     * \brief Checks if the inner loop stride of the specified iterator operand
     *        is fixed during the entire iteration.
     */
    bool
    is_inner_loop_stride_fixed(size_t op_idx)
    {
        return (! (get_inner_loop_fixed_stride(op_idx) == std::numeric_limits<intptr_t>::max()));
    }

    //__________________________________________________________________________
    /**
     * \brief Returns the ndarray object of the specified iterator operand.
     * \note The returned array is owned by the iterator, thus this object
     *       should get out of scope before the iter object itself gets
     *       destroyed. Otherwise bad things could happen.
     * TODO: The ndarray operand objects should be created by this iter class
     *       directly after the initialization of the iteration. So this
     *       method just returns a reference to these ndarray objects.
     */
    ndarray
    get_operand(size_t op_idx);

    //__________________________________________________________________________
    /**
     * \brief Resets the iterator to its initial state, i.e. to the beginning
     *        of the iteration range. Returns ``true`` after success and
     *       ``false`` otherwise. If the parameter ``throws`` is set to ``true``
     *       this function will throw an exception on failure. Otherwise it will
     *       not and it can be used even without holding the Python GIL.
     */
    bool
    reset(bool throws=true);

    //--------------------------------------------------------------------------
    // Here starts the private section, but we keep it public, so applications
    // don't have to call accessor methods, which slows things down.

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

    /**
     * \brief The pointer to the nop PyArray_Descr pointers pointing to the
     *     description objects for all iterator operands.
     */
    PyArray_Descr** descr_ptr_array_ptr_;
};

}// namespace detail
}// namespace numpy
}// namespace boost

#endif // !BOOST_NUMPY_DETAIL_ITER_HPP_INCLUDED

//==============================================================================
// Constructor decleration for N operands.

#else
// BOOST_PP_IS_ITERATING is true

#if BOOST_PP_ITERATION_FLAGS() == 1

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

#else
#if BOOST_PP_ITERATION_FLAGS() == 2

#define N BOOST_PP_ITERATION()

#define BOOST_NUMPY_DETAIL_ITER__op_ndarray_ptr(z, n, data)                    \
    BOOST_PP_COMMA_IF(n) reinterpret_cast<PyArrayObject*>(BOOST_PP_CAT(op_,n).ndarray_.ptr())

#define BOOST_NUMPY_DETAIL_ITER__op_flags(z, n, data)                          \
    BOOST_PP_COMMA_IF(n) BOOST_PP_CAT(op_,n).flags_

#define BOOST_NUMPY_DETAIL_ITER__op_dtype_ptr(z, n, data)                      \
    BOOST_PP_COMMA_IF(n) reinterpret_cast<PyArray_Descr*>(BOOST_PP_CAT(op_,n).ndarray_.get_dtype().ptr())

#define BOOST_NUMPY_DETAIL_ITER__op_bcr(z, n, data)                            \
    BOOST_PP_COMMA_IF(n) BOOST_PP_CAT(op_,n).broadcasting_rules_

#define BOOST_NUMPY_DETAIL_ITER__log_op_bcr(z, n, data) \
    BOOST_NUMPY_LOG( BOOST_PP_STRINGIZE(BOOST_PP_CAT(op_,n)) << ".broadcasting_rules_: " << c_array_to_string(BOOST_PP_CAT(op_,n).broadcasting_rules_, n_iter_axes) )

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

    BOOST_NUMPY_LOG("itershape: " << c_array_to_string(itershape, n_iter_axes))
    BOOST_PP_REPEAT(N, BOOST_NUMPY_DETAIL_ITER__log_op_bcr, ~)

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

#undef BOOST_NUMPY_DETAIL_ITER__log_op_bcr
#undef BOOST_NUMPY_DETAIL_ITER__op_bcr
#undef BOOST_NUMPY_DETAIL_ITER__op_dtype_ptr
#undef BOOST_NUMPY_DETAIL_ITER__op_flags
#undef BOOST_NUMPY_DETAIL_ITER__op_ndarray_ptr

#undef N

#endif // BOOST_PP_ITERATION_FLAGS() == 2
#endif // BOOST_PP_ITERATION_FLAGS() == 1
#endif // !BOOST_PP_IS_ITERATING
