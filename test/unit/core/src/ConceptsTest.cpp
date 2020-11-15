/* Copyright 2019 Axel Huebl, Benjamin Worpitz
 *
 * This file is part of alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#include <alpaka/core/Concepts.hpp>

#include <catch2/catch.hpp>

#include <type_traits>

struct ConceptExample{};
struct ConceptNonMatchingExample{};

struct ImplementerNotTagged
{
};

struct ImplementerNotTaggedButNonMatchingTagged
    : public alpaka::concepts::ImplementsViaBase<ConceptNonMatchingExample, ImplementerNotTaggedButNonMatchingTagged>
{
};

struct ImplementerTagged
    : public alpaka::concepts::ImplementsViaBase<ConceptExample, ImplementerTagged>
{
};

struct ImplementerTaggedButAlsoNonMatchingTagged
    : public alpaka::concepts::ImplementsViaBase<ConceptNonMatchingExample, ImplementerTaggedButAlsoNonMatchingTagged>
    , public alpaka::concepts::ImplementsViaBase<ConceptExample, ImplementerTaggedButAlsoNonMatchingTagged>
{
};

struct ImplementerWithTaggedBase
    : public ImplementerTagged
{
};

struct ImplementerWithTaggedBaseAlsoNonMatchingTagged
    : public ImplementerTaggedButAlsoNonMatchingTagged
{
};

struct ImplementerTaggedToBase
    : public ImplementerNotTagged
    , public alpaka::concepts::ImplementsViaBase<ConceptExample, ImplementerNotTagged>
{
};

struct ImplementerTaggedToBaseAlsoNonMatchingTagged
    : public ImplementerNotTaggedButNonMatchingTagged
    , public alpaka::concepts::ImplementsViaBase<ConceptExample, ImplementerNotTaggedButNonMatchingTagged>
{
};

struct ImplementerNonMatchingTaggedTaggedToBase
    : public ImplementerNotTagged
    , public alpaka::concepts::ImplementsViaBase<ConceptNonMatchingExample, ImplementerTaggedToBaseAlsoNonMatchingTagged>
    , public alpaka::concepts::ImplementsViaBase<ConceptExample, ImplementerNotTagged>
{
};

//-----------------------------------------------------------------------------
TEST_CASE("ImplementerNotTagged", "[meta]")
{
    using ImplementationType = alpaka::concepts::ImplementationType<ConceptExample, ImplementerNotTagged>;

    static_assert(
        std::is_same<
            ImplementerNotTagged,
            ImplementationType
        >::value,
        "alpaka::meta::ImplementationType failed!");

    const ImplementerNotTagged type;
    const auto& implementation = alpaka::concepts::getImplementation<ConceptExample>(type);
    REQUIRE(&type == &implementation);
}

//-----------------------------------------------------------------------------
TEST_CASE("ImplementerNotTaggedButNonMatchingTagged", "[meta]")
{
    using ImplementationType = alpaka::concepts::ImplementationType<ConceptExample, ImplementerNotTaggedButNonMatchingTagged>;

    static_assert(
        std::is_same<
            ImplementerNotTaggedButNonMatchingTagged,
            ImplementationType
        >::value,
        "alpaka::meta::ImplementationType failed!");

    const ImplementerNotTaggedButNonMatchingTagged type;
    const auto& implementation = alpaka::concepts::getImplementation<ConceptExample>(type);
    REQUIRE(&type == &implementation);
}

//-----------------------------------------------------------------------------
TEST_CASE("ImplementerTagged", "[meta]")
{
    using ImplementationType = alpaka::concepts::ImplementationType<ConceptExample, ImplementerTagged>;

    static_assert(
        std::is_same<
            ImplementerTagged,
            ImplementationType
        >::value,
        "alpaka::meta::ImplementationType failed!");

    const ImplementerTagged type;
    const auto& implementation = alpaka::concepts::getImplementation<ConceptExample>(type);
    REQUIRE(&type == &implementation);
}

//-----------------------------------------------------------------------------
TEST_CASE("ImplementerTaggedButAlsoNonMatchingTagged", "[meta]")
{
    using ImplementationType = alpaka::concepts::ImplementationType<ConceptExample, ImplementerTaggedButAlsoNonMatchingTagged>;

    static_assert(
        std::is_same<
            ImplementerTaggedButAlsoNonMatchingTagged,
            ImplementationType
        >::value,
        "alpaka::meta::ImplementationType failed!");

    const ImplementerTaggedButAlsoNonMatchingTagged type;
    const auto& implementation = alpaka::concepts::getImplementation<ConceptExample>(type);
    REQUIRE(&type == &implementation);
}

//-----------------------------------------------------------------------------
TEST_CASE("ImplementerWithTaggedBase", "[meta]")
{
    using ImplementationType = alpaka::concepts::ImplementationType<ConceptExample, ImplementerWithTaggedBase>;

    static_assert(
        std::is_same<
            ImplementerTagged,
            ImplementationType
        >::value,
        "alpaka::meta::ImplementationType failed!");

    const ImplementerWithTaggedBase type;
    const auto& implementation = alpaka::concepts::getImplementation<ConceptExample>(type);
    REQUIRE(&type == &implementation);
}

//-----------------------------------------------------------------------------
TEST_CASE("ImplementerWithTaggedBaseAlsoNonMatchingTagged", "[meta]")
{
    using ImplementationType = alpaka::concepts::ImplementationType<ConceptExample, ImplementerWithTaggedBaseAlsoNonMatchingTagged>;

    static_assert(
        std::is_same<
            ImplementerTaggedButAlsoNonMatchingTagged,
            ImplementationType
        >::value,
        "alpaka::meta::ImplementationType failed!");

    const ImplementerWithTaggedBaseAlsoNonMatchingTagged type;
    const auto& implementation = alpaka::concepts::getImplementation<ConceptExample>(type);
    REQUIRE(&type == &implementation);
}

//-----------------------------------------------------------------------------
TEST_CASE("ImplementerTaggedToBase", "[meta]")
{
    using ImplementationType = alpaka::concepts::ImplementationType<ConceptExample, ImplementerTaggedToBase>;

    static_assert(
        std::is_same<
            ImplementerNotTagged,
            ImplementationType
        >::value,
        "alpaka::meta::ImplementationType failed!");

    const ImplementerTaggedToBase type;
    const auto& implementation = alpaka::concepts::getImplementation<ConceptExample>(type);
    REQUIRE(&type == &implementation);
}

//-----------------------------------------------------------------------------
TEST_CASE("ImplementerTaggedToBaseAlsoNonMatchingTagged", "[meta]")
{
    using ImplementationType = alpaka::concepts::ImplementationType<ConceptExample, ImplementerTaggedToBaseAlsoNonMatchingTagged>;

    static_assert(
        std::is_same<
            ImplementerNotTaggedButNonMatchingTagged,
            ImplementationType
        >::value,
        "alpaka::meta::ImplementationType failed!");

    const ImplementerTaggedToBaseAlsoNonMatchingTagged type;
    const auto& implementation = alpaka::concepts::getImplementation<ConceptExample>(type);
    REQUIRE(&type == &implementation);
}

//-----------------------------------------------------------------------------
TEST_CASE("ImplementerNonMatchingTaggedTaggedToBase", "[meta]")
{
    using ImplementationType = alpaka::concepts::ImplementationType<ConceptExample, ImplementerNonMatchingTaggedTaggedToBase>;

    static_assert(
        std::is_same<
            ImplementerNotTagged,
            ImplementationType
        >::value,
        "alpaka::meta::ImplementationType failed!");

    const ImplementerNonMatchingTaggedTaggedToBase type;
    const auto& implementation = alpaka::concepts::getImplementation<ConceptExample>(type);
    REQUIRE(&type == &implementation);
}
