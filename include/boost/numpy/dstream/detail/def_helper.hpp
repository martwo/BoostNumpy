/**
 * $Id$
 *
 * Copyright (C)
 * 2013
 *     Martin Wolf <martin.wolf@fysik.su.se>
 *
 * @file    boost/numpy/dstream/detail/def_helper.hpp
 * @version $Revision$
 * @date    $Date$
 * @author  Martin Wolf <martin.wolf@fysik.su.se>
 *
 * @brief This file defines a helper mechanism to extract parameters for the def
 *        and classdef function templates.
 *        It uses the tuple_extract code from boost/python/detail/def_helper.hpp
 *        and defines a boost::numpy specific def_helper template.
 *
 *        This file is distributed under the Boost Software License,
 *        Version 1.0. (See accompanying file LICENSE_1_0.txt or copy at
 *        http://www.boost.org/LICENSE_1_0.txt).
 */
#ifndef BOOST_NUMPY_DSTREAM_DETAIL_DEF_HELPER_HPP_INCLUDED
#define BOOST_NUMPY_DSTREAM_DETAIL_DEF_HELPER_HPP_INCLUDED

#include <boost/type_traits/is_base_of.hpp>

#include <boost/python/detail/mpl_lambda.hpp>
#include <boost/python/detail/def_helper.hpp>

#include <boost/numpy/mpl/unspecified.hpp>
#include <boost/numpy/dstream/mapping.hpp>
#include <boost/numpy/dstream/wiring.hpp>
#include <boost/numpy/dstream/out_arr_transform.hpp>
#include <boost/numpy/dstream/threading.hpp>

namespace boost {
namespace numpy {
namespace dstream {
namespace detail {

template <
      class DefaultMappingModelSelector
    , class DefaultWiringModelSelector
    , class DefaultOutArrTransformSelector
    , class DefaultThreadAbilitySelector
    , class T1 = numpy::mpl::unspecified
    , class T2 = numpy::mpl::unspecified
    , class T3 = numpy::mpl::unspecified
    , class T4 = numpy::mpl::unspecified
    , class T5 = numpy::mpl::unspecified
>
struct def_helper;


template <typename T>
class is_reference_to_mapping_model_selector
{
  public:
    typedef boost::is_base_of<mapping::mapping_model_selector_type, typename boost::remove_reference<T>::type >
            type;
    BOOST_PYTHON_MPL_LAMBDA_SUPPORT(1, is_reference_to_mapping_model_selector, (T))
};

template <typename T>
class is_reference_to_wiring_model_selector
{
  public:
    typedef boost::is_base_of<wiring::wiring_model_selector_type, typename boost::remove_reference<T>::type >
            type;
    BOOST_PYTHON_MPL_LAMBDA_SUPPORT(1, is_reference_to_wiring_model_selector, (T))
};

template <typename T>
class is_reference_to_out_arr_transform_selector
{
  public:
    typedef boost::is_base_of<out_arr_transforms::out_arr_transform_selector_type, typename boost::remove_reference<T>::type >
            type;
    BOOST_PYTHON_MPL_LAMBDA_SUPPORT(1, is_reference_to_out_arr_transform_selector, (T))
};

template <typename T>
class is_reference_to_thread_ability_selector
{
  public:
    typedef boost::is_base_of<threading::thread_ability_selector_type, typename boost::remove_reference<T>::type >
            type;
    BOOST_PYTHON_MPL_LAMBDA_SUPPORT(1, is_reference_to_thread_ability_selector, (T))
};



template <class Tuple>
struct mapping_model_selector_extract
  : python::detail::tuple_extract
        < Tuple
        , is_reference_to_mapping_model_selector< boost::mpl::_1 >
        >
{};

template <class Tuple>
struct wiring_model_selector_extract
  : python::detail::tuple_extract
        < Tuple
        , is_reference_to_wiring_model_selector< boost::mpl::_1 >
        >
{};

template <class Tuple>
struct out_arr_transform_selector_extract
  : python::detail::tuple_extract
        < Tuple
        , is_reference_to_out_arr_transform_selector< boost::mpl::_1 >
        >
{};

template <class Tuple>
struct thread_ability_selector_extract
  : python::detail::tuple_extract
        < Tuple
        , is_reference_to_thread_ability_selector< boost::mpl::_1 >
        >
{};

template <
      class DefaultMappingModelSelector
    , class DefaultWiringModelSelector
    , class DefaultOutArrTransformSelector
    , class DefaultThreadAbilitySelector
    , class T1
    , class T2
    , class T3
    , class T4
    , class T5
>
struct def_helper
{
    typedef boost::tuples::tuple<
          T1 const&
        , T2 const&
        , T3 const&
        , T4 const&
        , T5 const&
        , DefaultMappingModelSelector
        , DefaultWiringModelSelector
        , DefaultOutArrTransformSelector
        , DefaultThreadAbilitySelector
        , char const*
        > all_t;


    def_helper(T1 const& a1) : m_all(a1,m_nil,m_nil,m_nil,m_nil) {}
    def_helper(T1 const& a1, T2 const& a2) : m_all(a1,a2,m_nil,m_nil,m_nil) {}
    def_helper(T1 const& a1, T2 const& a2, T3 const& a3) : m_all(a1,a2,a3,m_nil,m_nil) {}
    def_helper(T1 const& a1, T2 const& a2, T3 const& a3, T4 const& a4) : m_all(a1,a2,a3,a4,m_nil) {}
    def_helper(T1 const& a1, T2 const& a2, T3 const& a3, T4 const& a4, T5 const& a5) : m_all(a1,a2,a3,a4,a5) {}

    // Extractor functions which pull the appropriate value out of the tuple.
    char const*
    get_doc() const
    {
        return python::detail::doc_extract<all_t>::extract(m_all);
    }

    typename mapping_model_selector_extract<all_t>::result_type
    get_mapping_model_selector() const
    {
        return mapping_model_selector_extract<all_t>::extract(m_all);
    }

    typename wiring_model_selector_extract<all_t>::result_type
    get_wiring_model_selector() const
    {
        return wiring_model_selector_extract<all_t>::extract(m_all);
    }

    typename out_arr_transform_selector_extract<all_t>::result_type
    get_out_arr_transform_selector() const
    {
        return out_arr_transform_selector_extract<all_t>::extract(m_all);
    }

    typename thread_ability_selector_extract<all_t>::result_type
    get_thread_ability_selector() const
    {
        return thread_ability_selector_extract<all_t>::extract(m_all);
    }

  private:
    all_t m_all;
    // For filling in numpy::mpl::unspecified slots.
    numpy::mpl::unspecified m_nil;
};

}/*namespace detail*/
}/*namespace dstream*/
}/*namespace numpy*/
}/*namespace boost*/

#endif // BOOST_NUMPY_DSTREAM_DETAIL_DEF_HELPER_HPP_INCLUDED
