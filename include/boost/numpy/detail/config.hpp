/**
 * $Id$
 *
 * Copyright (C)
 * 2013
 *     Martin Wolf <martin.wolf@fysik.su.se>
 *
 * \file    boost/numpy/detail/config.hpp
 * \version $Revision$
 * \date    $Date$
 * \author  Martin Wolf <martin.wolf@fysik.su.se>
 *
 * \brief This file defines templates for configuration purposes, that can be
 *        used by wiring models. The concept is based on boost::python::arg.
 *
 *        This file is distributed under the Boost Software License,
 *        Version 1.0. (See accompanying file LICENSE_1_0.txt or copy at
 *        http://www.boost.org/LICENSE_1_0.txt).
 */
#ifndef BOOST_NUMPY_DETAIL_CONFIG_HPP_INCLUDED
#define BOOST_NUMPY_DETAIL_CONFIG_HPP_INCLUDED

#include <cstddef>
#include <vector>

#include <boost/config.hpp>
#include <boost/assert.hpp>
#include <boost/function.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/version.hpp>
#include <boost/preprocessor/repetition/enum_params.hpp>
#include <boost/type_traits/is_member_function_pointer.hpp>
#include <boost/type_traits/is_pointer.hpp>

#include <boost/numpy/limits.hpp>

namespace boost {
namespace numpy {
namespace detail {

typedef enum {
    NONE                       = 0,
    IS_POINTER                 = 1,
    IS_MEMBER_FUNCTION_POINTER = 2
} setting_flags_t;

template <class T>
struct setting_storage
{
    explicit setting_storage(T const & value_)
      : value(value_)
    {}

    T value;
};

template <std::size_t nsettings = 0>
struct settings;

template <>
struct settings<0>
{
    BOOST_STATIC_CONSTANT(std::size_t, size = 0);
};

typedef detail::settings<1> cfg;

struct setting_t
{
    boost::shared_ptr<void> ptr;
    int flags;

    //__________________________________________________________________________
    setting_t()
      : flags(0)
    {}

    //__________________________________________________________________________
    setting_t(boost::shared_ptr<void> const & ptr_, int flags_)
      : ptr(ptr_),
        flags(flags_)
    {}

    //__________________________________________________________________________
    /**
     * \brief Returns the setting struct defined by the user at position i.
     */
    template <class T>
    T const &
    get_data() const
    {
        return (*reinterpret_cast<setting_storage<T> const *>(ptr.get())).value;
    }
};

template <std::size_t nsettings>
struct settings_base
{
    BOOST_STATIC_CONSTANT(std::size_t, size = nsettings);

    // The elements member stores a list of setting_t objects, pointing to
    // objects of type setting_storage<T>.
    std::vector< setting_t > elements;

    settings_base()
      : elements(std::vector< setting_t >(nsettings))
    {}

    settings<nsettings+1>
    operator,(cfg const &s) const;
};

template <std::size_t nsettings>
struct settings : settings_base<nsettings>
{
    typedef settings<nsettings> type;

    settings()
      : settings_base<nsettings>()
    {}
};

template <>
struct settings<1> : settings_base<1>
{
    typedef settings<1> type;

    settings()
      : settings_base<1>()
    {}

    template <class T>
    cfg& operator=(T const & value)
    {
        int flags = 0;
        flags |= (boost::is_pointer<T>::value ? int(IS_POINTER) : 0);
        flags |= (boost::is_member_function_pointer<T>::value ? int(IS_MEMBER_FUNCTION_POINTER) : 0);

        elements[0] = setting_t(boost::shared_ptr<void>(new setting_storage<T>(value)), flags);

        return *this;
    }
};

template <std::size_t nsettings>
inline
settings<nsettings+1>
settings_base<nsettings>::operator,(cfg const &s) const
{
    settings<nsettings> const& l = *static_cast<settings<nsettings> const*>(this);
    settings<nsettings+1> res;
    std::copy(l.elements.begin(), l.elements.end(), res.elements.begin());
    res.elements[nsettings] = s.elements[0];
    return res;
}

//==============================================================================
/**
 * \brief The config template provides a base template for all possible
 *     configurations needed by the different models.
 */
template <class Settings>
struct config
{
    typedef config<Settings> type;
    typedef Settings settings_t;

    settings_t settings_;

    //__________________________________________________________________________
    config(settings_t const & settings)
      : settings_(settings)
    {}

    //__________________________________________________________________________
    /**
     * \brief Returns the setting_t struct defined by the user at position i.
     */
    setting_t const &
    get_setting(std::size_t i) const
    {
        BOOST_ASSERT(i < settings_t::size);
        return settings_.elements[i];
    }
};

}/*namespace detail*/
}/*namespace numpy*/
}/*namespace boost*/

#endif//! BOOST_NUMPY_DETAIL_CONFIG_HPP_INCLUDED
