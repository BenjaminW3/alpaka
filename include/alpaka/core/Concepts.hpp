/* Copyright 2019 Benjamin Worpitz
 *
 * This file is part of alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#include <type_traits>

namespace alpaka
{
    namespace concepts
    {
        //#############################################################################
        //! Tag used in class inheritance hierarchies that describes that a specific concept (TConcept) is implemented.
        template<
            typename TConcept>
        struct Implements
        {
        };

        //#############################################################################
        //! Tag used in class inheritance hierarchies that describes that a specific concept (TConcept)
        //! is implemented by the given base class (TBase).
        template<
            typename TConcept,
            typename TBase>
        struct ImplementsViaBase : public Implements<TConcept>
        {
        };

        //#############################################################################
        //! Checks whether the concept is implemented by the given class
        template<
            typename TConcept,
            typename TType>
        struct ImplementsConcept {
            static auto implements(Implements<TConcept>&) -> std::true_type;
            static auto implements(...) -> std::false_type;

            static constexpr auto value = decltype(implements(std::declval<TType&>()))::value;
        };

        //#############################################################################
        //! Checks whether the concept is implemented by the given class via inheritance
        template<
            typename TConcept,
            typename TType>
        struct ImplementsConceptViaBase {
            template<typename TBase>
            static auto implements(ImplementsViaBase<TConcept, TBase>&) -> std::true_type;
            static auto implements(...) -> std::false_type;

            static constexpr auto value = decltype(implements(std::declval<TType&>()))::value;
        };

        namespace detail
        {
            //#############################################################################
            //! Returns the type that implements the given concept in the inheritance hierarchy.
            template<
                typename TConcept,
                typename TType,
                typename Sfinae = void>
            struct GetImplementation;

            //#############################################################################
            //! Base case for types that do not inherit from "ImplementsViaBase<TConcept, ...>" is the type itself.
            template<
                typename TConcept,
                typename TType>
            struct GetImplementation<
                TConcept,
                TType,
                std::enable_if_t<!ImplementsConcept<TConcept, TType>::value>>
            {
                using Type = TType;

                static decltype(auto) getImplementation(TType const & type)
                {
                    return type;
                }
            };

            //#############################################################################
            //! For types that inherit from "ImplementsViaBase<TConcept, ...>" it finds the base class (TBase) which implements the concept.
            template<
                typename TConcept,
                typename TType>
            struct GetImplementation<
                TConcept,
                TType,
                std::enable_if_t<ImplementsConceptViaBase<TConcept, TType>::value>>
            {
                template<
                    typename TBase>
                static auto implementer(ImplementsViaBase<TConcept, TBase>&) -> TBase;

                using Type = decltype(implementer(std::declval<TType&>()));

                static_assert(std::is_base_of<Type, TType>::value, "The type implementing the concept has to be a publicly accessible base class!");

                static decltype(auto) getImplementation(TType const & type)
                {
                    return static_cast<Type const &>(type);
                }
            };
        }

        //#############################################################################
        //! Returns the type that implements the given concept in the inheritance hierarchy.
        template<
            typename TConcept,
            typename TType>
        static auto getImplementation(TType const & type) -> typename detail::GetImplementation<TConcept, TType>::Type const &
        {
            return detail::GetImplementation<TConcept, TType>::getImplementation(type);
        };

        template<
            typename TConcept,
            typename TType>
        using ImplementationType = typename detail::GetImplementation<TConcept, TType>::Type;
    }
}
