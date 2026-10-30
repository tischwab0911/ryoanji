/*
 * Cornerstone octree
 *
 * Copyright (c) 2024 CSCS, ETH Zurich
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: MIT License
 */

/*! @file
 * @brief Test generic SFC functionality
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#include <random>

#include "gtest/gtest.h"
#include "cstone/sfc/sfc.hpp"

using namespace cstone;

TEST(SFC, commonNodePrefix)
{
    using T       = double;
    using KeyType = unsigned;

    Box<T> box(0, 1);

    {
        auto key = commonNodePrefix<HilbertKey<KeyType>>(Vec3<T>{0.7, 0.2, 0.2}, Vec3<T>{0.01, 0.01, 0.01}, box);
        EXPECT_EQ(key, 0000176134);
    }
    {
        auto key = commonNodePrefix<HilbertKey<KeyType>>(Vec3<T>{0.2393, 0.3272, 0.29372},
                                                         Vec3<T>{0.0012, 0.0011, 0.00098}, box);
        EXPECT_EQ(key, 0000104322);
    }
}

TEST(SFC, center)
{
    using T       = double;
    using KeyType = unsigned;

    Box<T> box(-1, 1);

    {
        // The exact center belongs to octant farthest from the origin
        T x           = 0.0;
        KeyType probe = sfc3D<HilbertKey<KeyType>>(x, x, x, box);
        KeyType ref   = sfc3D<HilbertKey<KeyType>>(1.0, 1.0, 1.0, box);
        EXPECT_EQ(octalDigit(probe, 1), octalDigit(ref, 1));
    }
    {
        // Center - epsilon should be in the octant closest to the origin
        T x           = -1e-40;
        KeyType probe = sfc3D<HilbertKey<KeyType>>(x, x, x, box);
        KeyType ref   = sfc3D<HilbertKey<KeyType>>(-1.0, -1.0, -1.0, box);
        EXPECT_EQ(octalDigit(probe, 1), octalDigit(ref, 1));
    }
}
/*! @brief the cell of a key contains its point also in boxes whose xmin * m is not an integer (fork patch)
 *
 * A box fitted to a bunch (OPAL-X) has arbitrary bounds. The first point is the particle of an 8-rank SwissFEL run
 * whose key was one cell too low in y with upstream's floor(y * m) - ymin * m (0.56 cells outside its key's box),
 * which left a cell failing its MAC out of the locally essential tree.
 */
TEST(SFC, keyCellContainsPointInFittedBox)
{
    using T       = double;
    using KeyType = uint64_t;
    using SK      = HilbertKey<KeyType>;

    Box<T> box(-0.0011882485223581088, 0.0011884225944410061, -0.0011882157500425073, 0.001188368630643466,
               -0.012426126792806139, 0.045320500145225831);
    const auto axesBits = box.getBoxDimBits(maxTreeLevel<KeyType>{});
    ASSERT_EQ(axesBits[0], 16u);
    ASSERT_EQ(axesBits[2], 21u);

    auto contained = [&](T x, T y, T z)
    {
        KeyType key       = KeyType(sfc3D<SK>(x, y, z, box));
        auto [c, s]       = centerAndSize<KeyType>(sfcIBox(SK(key), maxTreeLevel<KeyType>{}, axesBits), box);
        // relative rounding of the cell bounds, far below one cell (2^-16 of the box along x and y)
        const T tolerance = 1e-12 * box.maxExtent();
        return std::abs(x - c[0]) <= s[0] + tolerance && std::abs(y - c[1]) <= s[1] + tolerance &&
               std::abs(z - c[2]) <= s[2] + tolerance;
    };

    EXPECT_TRUE(contained(-2.3742214883045176e-05, 9.6842915978489733e-08, 0.010289945748438454));

    std::mt19937_64                   rng(42);
    std::uniform_real_distribution<T> ux(box.xmin(), box.xmax()), uy(box.ymin(), box.ymax()),
        uz(box.zmin(), box.zmax());
    int outside = 0;
    for (int i = 0; i < 100000; ++i)
    {
        outside += !contained(ux(rng), uy(rng), uz(rng));
    }
    EXPECT_EQ(outside, 0);
}
