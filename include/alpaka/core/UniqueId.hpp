/**
* \file
* Copyright 2015 Benjamin Worpitz
*
* This file is part of alpaka.
*
* alpaka is free software: you can redistribute it and/or modify
* it under the terms of the GNU Lesser General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* alpaka is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU Lesser General Public License for more details.
*
* You should have received a copy of the GNU Lesser General Public License
* along with alpaka.
* If not, see <http://www.gnu.org/licenses/>.
*/

#pragma once

#include <boost/predef.h>           // workarounds

#include <cstddef>                  // std::size_t

namespace alpaka
{
    namespace core
    {
        namespace detail
        {
            //#############################################################################
            // Based on the code from Filip Roséen at http://b.atch.se/posts/constexpr-counter/
            //#############################################################################
            /*template<
                std::size_t N>
            struct flag;

            //-----------------------------------------------------------------------------
            //
            //-----------------------------------------------------------------------------
            template<
                std::size_t N>
            constexpr int adl_flag(flag<N>);*/

            //#############################################################################
            //
            //#############################################################################
            template<
                std::size_t N>
            struct flag
            {
                friend constexpr std::size_t adl_flag(flag<N>);
            };

            //#############################################################################
            //
            //#############################################################################
            template<
                std::size_t N>
            struct writer
            {
                //-----------------------------------------------------------------------------
                //
                //-----------------------------------------------------------------------------
                friend constexpr std::size_t adl_flag(flag<N>)
                {
                    return N;
                }

                static constexpr std::size_t value = N;
            };

#ifdef BOOST_COMP_MSVC
            //-----------------------------------------------------------------------------
            //! The matcher.
            //-----------------------------------------------------------------------------
            template<
                std::size_t N,
                class = char[noexcept(adl_flag(flag<N>{})) ? +1u : -1u]>
            auto constexpr reader(std::size_t, flag<N>)
            -> std::size_t
            {
              return N;
            }
#else
            //-----------------------------------------------------------------------------
            //! The matcher.
            //-----------------------------------------------------------------------------
            template<
                std::size_t N,
                std::size_t = adl_flag(flag<N>{})>
            auto constexpr reader(
                std::size_t,
                flag<N>)
            -> std::size_t
            {
                return N;
            }
#endif
            //-----------------------------------------------------------------------------
            //! The searcher.
            //-----------------------------------------------------------------------------
            template<
                std::size_t N>
            auto constexpr reader(
                float,
                flag<N>,
                std::size_t R = reader(std::size_t{0u}, flag<N-1u>{}))
            -> std::size_t
            {
                return R;
            }
            //-----------------------------------------------------------------------------
            //! Reader base case.
            //-----------------------------------------------------------------------------
            std::size_t constexpr reader(float, flag<0u>)
            {
                return 0u;
            }

#ifdef BOOST_COMP_MSVC
            //-----------------------------------------------------------------------------
            //! \return An unique compile time ID.
            //-----------------------------------------------------------------------------
            template<
                std::size_t N = 1u,
                std::size_t C = reader(std::size_t{0u}, flag<64u>{})>
            auto constexpr uniqueId(
                std::size_t R = writer<C + N>::value)
            -> std::size_t
            {
              return R;
            }
#else
            //-----------------------------------------------------------------------------
            //! \return An unique compile time ID.
            //-----------------------------------------------------------------------------
            template<
                std::size_t N = 1u>
            auto constexpr uniqueId(
                std::size_t R = writer<reader(std::size_t{0u}, flag<64u>{}) + N>::value)
            -> std::size_t
            {
                return R;
            }
#endif
        }
    }
}