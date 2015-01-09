/**
 * $Id$
 *
 * Copyright (C)
 * 2014 - $Date$
 *     Martin Wolf <boostnumpy@martin-wolf.org>
 *
 * @file    boost/numpy/iterators/detail/iter_iterator.hpp
 * @version $Revision$
 * @date    $Date$
 * @author  Martin Wolf <boostnumpy@martin-wolf.org>
 *
 * @brief This file defines the
 *        boost::numpy::iterators::detail::iter_iterator
 *        template providing the base for all BoostNumpy C++ style iterators
 *        using the boost::numpy::detail::iter class iterating over a single
 *        ndarray.
 *        The value type of the operand is specified via a value type traits
 *        class, which also provides the appropriate dereferencing procedure.
 *
 *        This file is distributed under the Boost Software License,
 *        Version 1.0. (See accompanying file LICENSE_1_0.txt or copy at
 *        http://www.boost.org/LICENSE_1_0.txt).
 */
#ifndef BOOST_NUMPY_ITERATORS_DETAIL_ITER_ITERATOR_HPP_INCLUDED
#define BOOST_NUMPY_ITERATORS_DETAIL_ITER_ITERATOR_HPP_INCLUDED 1

#include <boost/iterator/iterator_facade.hpp>

#include <boost/numpy/ndarray.hpp>
#include <boost/numpy/detail/iter.hpp>
#include <boost/numpy/dstream.hpp>

namespace boost {
namespace numpy {
namespace iterators {
namespace detail {

struct iter_iterator_type
{};

template <class Derived, class CategoryOrTraversal, typename ValueTypeTraits>
class iter_iterator
  : public boost::iterator_facade<
        Derived
      , typename ValueTypeTraits::value_type
      , CategoryOrTraversal
      , typename ValueTypeTraits::value_ref_type
      //, DifferenceType
    >
    , public iter_iterator_type
{
  public:
    typedef iter_iterator<Derived, CategoryOrTraversal, ValueTypeTraits>
            type_t;
    typedef typename boost::iterator_facade<Derived, typename ValueTypeTraits::value_type, CategoryOrTraversal, typename ValueTypeTraits::value_ref_type>::difference_type
            difference_type;

    static
    boost::shared_ptr<boost::numpy::detail::iter>
    construct_iter(
          iter_iterator_type & iter_base
        , boost::numpy::detail::iter_flags_t special_iter_flags
        , ndarray & arr)
    {
        type_t & thisiter = *static_cast<type_t *>(&iter_base);

        // Define a scalar core shape for the iterator operand.
        typedef dstream::mapping::detail::core_shape<0>::shape<>
                scalar_core_shape_t;
        typedef dstream::array_definition<scalar_core_shape_t, typename ValueTypeTraits::value_type>
                arr_def;

        // Define the type of the loop service.
        typedef dstream::detail::loop_service_arity<1>::loop_service<arr_def>
                loop_service_t;

        // Create input_array_service object for each array.
        dstream::detail::input_array_service<arr_def> in_arr_service(arr);

        // Create the loop service object.
        loop_service_t loop_service(in_arr_service);

        boost::numpy::detail::iter_operand_flags_t in_arr_op_flags = boost::numpy::detail::iter_operand::flags::NONE::value;
        in_arr_op_flags |= thisiter.arr_access_flags_ & boost::numpy::detail::iter_operand::flags::READONLY::value;
        in_arr_op_flags |= thisiter.arr_access_flags_ & boost::numpy::detail::iter_operand::flags::WRITEONLY::value;
        in_arr_op_flags |= thisiter.arr_access_flags_ & boost::numpy::detail::iter_operand::flags::READWRITE::value;

        boost::numpy::detail::iter_operand in_arr_iter_op(in_arr_service.get_arr(), in_arr_op_flags, in_arr_service.get_arr_bcr_data());

        // Define the iterator flags.
        boost::numpy::detail::iter_flags_t iter_flags =
            boost::numpy::detail::iter::flags::REFS_OK::value
          | boost::numpy::detail::iter::flags::DONT_NEGATE_STRIDES::value;
        iter_flags |= special_iter_flags;

        // Create the multi iterator object.
        boost::shared_ptr<boost::numpy::detail::iter> it(new boost::numpy::detail::iter(
            iter_flags
          , KEEPORDER
          , NO_CASTING
          , loop_service.get_loop_nd()
          , loop_service.get_loop_shape_data()
          , 0 // Use default buffersize.
          , in_arr_iter_op
        ));
        it->init_full_iteration();
        return it;
    }

    // Default constructor.
    iter_iterator()
      : is_end_point_(true)
      , arr_access_flags_(boost::numpy::detail::iter_operand::flags::READONLY::value)
    {}

    // Explicit constructor.
    explicit iter_iterator(
        ndarray & arr
      , boost::numpy::detail::iter_operand_flags_t arr_access_flags
    )
      : is_end_point_(false)
      , arr_access_flags_(arr_access_flags)
    {
        iter_ptr_ = Derived::construct_iter(*this, arr);
        vtt_ = ValueTypeTraits(arr);
    }

    // In case a const array is given, the READONLY flag for the array set
    // automatically.
    explicit iter_iterator(
        ndarray const & arr
      , boost::numpy::detail::iter_operand_flags_t arr_access_flags
    )
      : is_end_point_(false)
      , arr_access_flags_(arr_access_flags | boost::numpy::detail::iter_operand::flags::READONLY::value)
    {
        iter_ptr_ = Derived::construct_iter(*this, const_cast<ndarray &>(arr));
        vtt_ = ValueTypeTraits(const_cast<ndarray &>(arr));
    }

    // Copy constructor.
    iter_iterator(type_t const & other)
      : is_end_point_(other.is_end_point_)
      , arr_access_flags_(other.arr_access_flags_)
      , vtt_(other.vtt_)
    {
        if(other.iter_ptr_.get()) {
            iter_ptr_ = boost::shared_ptr<boost::numpy::detail::iter>(new boost::numpy::detail::iter(*other.iter_ptr_));
        }
    }

    // Creates an interator that points to the first element.
    Derived begin() const
    {
        Derived it(*static_cast<Derived*>(const_cast<type_t*>(this)));
        it.reset();
        return it;
    }

    // Creates an iterator that points to the element after the last element.
    Derived end() const
    {
        Derived it(*static_cast<Derived*>(const_cast<type_t*>(this)));
        it.is_end_point_ = true;
        return it;
    }

    void
    increment()
    {
        if(is_end())
        {
            reset();
        }
        else if(! iter_ptr_->next())
        {
            // We reached the end of the iteration. So we need to put this
            // iterator into the END state, wich is (by definition) indicated
            // through the is_end_point_ member variable set to ``true``.
            // Note: We still keep the iterator object, in case the user wants
            //       to reset the iterator and start iterating from the
            //       beginning.
            is_end_point_ = true;
        }
    }

    bool
    equal(type_t const & other) const
    {
        //std::cout << "iter_iterator: equal" << std::endl;
        if(is_end_point_ && other.is_end_point_)
        {
            return true;
        }
        // Check if one of the two iterators is the END state.
        if(is_end_point_ || other.is_end_point_)
        {
            return false;
        }
        // If the data pointers point to the same address, we are equal.
        return (iter_ptr_->get_data(0) == other.iter_ptr_->get_data(0));
    }

    bool
    reset(bool throws=true)
    {
        //std::cout << "iter_iterator: reset" << std::endl;
        is_end_point_ = false;
        return iter_ptr_->reset(throws);
    }

    bool
    is_end() const
    {
        return is_end_point_;
    }

    boost::numpy::detail::iter &
    get_detail_iter()
    {
        return *iter_ptr_;
    }

    typename ValueTypeTraits::value_ref_type
    dereference()
    {
        typename ValueTypeTraits::value_ptr_type value_ptr;
        ValueTypeTraits::dereference(
            vtt_
          , value_ptr
          , iter_ptr_->data_ptr_array_ptr_[0]
        );

        return *value_ptr;
    }

  protected:
    boost::shared_ptr<boost::numpy::detail::iter> iter_ptr_;

    bool is_end_point_;

    // Stores if the array is readonly, writeonly or readwrite'able.
    boost::numpy::detail::iter_operand_flags_t arr_access_flags_;

    // Define object vtt_ of the ValueTypeTraits class
    // (using the default constructor).
    ValueTypeTraits vtt_;
};

}// namespace detail
}// namespace iterators
}// namespace numpy
}// namespace boost

#endif // !BOOST_NUMPY_ITERATORS_DETAIL_ITER_ITERATOR_HPP_INCLUDED
