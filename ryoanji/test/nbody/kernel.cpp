/*
 * Ryoanji N-body solver
 *
 * Copyright (c) 2024 CSCS, ETH Zurich
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: MIT License
 */

/*! @file
 * @brief Compare and test different multipole approximations
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#include "gtest/gtest.h"

#include "cstone/focus/source_center.hpp"

#include "dataset.hpp"
#include "ryoanji/nbody/cartesian_qpole.hpp"
#include "ryoanji/nbody/kernel.hpp"

using namespace ryoanji;

//! @brief Tests direct particle-to-particle gravity interactions with mass softening
TEST(Gravity, P2PmsoftBase)
{
    using T = double;

    cstone::Vec3<T> target{1, 1, 1};
    T               h = std::sqrt(3) / 2 - 0.001;

    cstone::Vec3<T> source1{2, 2, 2}, source2{-2, -2, -2};

    cstone::Vec4<T> acc{0, 0, 0, 0};
    acc = P2P(acc, target, source1, 1.0, h, h);
    acc = P2P(acc, target, source2, 1.0, h, h);

    // h too small to trigger softening, so results should match the non-softened numbers
    EXPECT_NEAR(acc[0], -0.76980035891950138, 1e-10);
    EXPECT_NEAR(acc[1], 0.17106674642655587, 1e-10);
    EXPECT_NEAR(acc[2], 0.17106674642655587, 1e-10);
    EXPECT_NEAR(acc[3], 0.17106674642655587, 1e-10);
}

//! @brief Tests direct particle-to-particle gravity interactions with mass softening
TEST(Gravity, P2PmsoftH)
{
    using T = double;

    cstone::Vec3<T> target{1, 1, 1};
    T               h = std::sqrt(3) / 2 + 0.001;

    cstone::Vec3<T> source1{2, 2, 2}, source2{-2, -2, -2};

    cstone::Vec4<T> acc{0, 0, 0, 0};
    acc = P2P(acc, target, source1, 1.0, h, h);
    acc = P2P(acc, target, source2, 1.0, h, h);

    EXPECT_NEAR(acc[0], -0.7678049688481372, 1e-10);
    EXPECT_NEAR(acc[1], 0.1704016164027678, 1e-10);
    EXPECT_NEAR(acc[2], 0.1704016164027678, 1e-10);
    EXPECT_NEAR(acc[3], 0.1704016164027678, 1e-10);
}

//! @brief The traversal code relies on P2P self interaction being zero
TEST(Gravity, P2PselfInteraction)
{
    using T = double;

    Vec3<T> pos_i{0, 0, 0};
    T       h = 0.1;
    Vec4<T> acc{0, 0, 0, 0};

    auto self = P2P(acc, pos_i, pos_i, 1.0, h, h);
    EXPECT_EQ(self, acc);
}

namespace
{
HeavisideFrame<double> makeFrame(double gammaRef, bool netForceChannel)
{
    HeavisideFrame<double> frame;
    frame.gammaRef        = gammaRef;
    frame.betaRef         = std::sqrt(1.0 - 1.0 / (gammaRef * gammaRef));
    frame.wRef            = std::acosh(gammaRef);
    frame.netForceChannel = netForceChannel;
    return frame;
}
} // namespace

//! @brief The Heaviside kernel reproduces the lab-frame field of a uniformly moving charge
TEST(Gravity, P2PHeavisideMatchesUniformMotionField)
{
    using T = double;

    const T gammaRef = 2.0;
    const T gammaJ   = 3.5;
    const T betaJ    = std::sqrt(1.0 - 1.0 / (gammaJ * gammaJ));
    const T m        = 1.3;

    // lab displacement target - source at equal lab time
    const Vec3<T> R{0.3, -0.2, 0.5};
    const T       den = std::pow(R[0] * R[0] + R[1] * R[1] + gammaJ * gammaJ * R[2] * R[2], 1.5);
    const Vec3<T> Elab{m * gammaJ * R[0] / den, m * gammaJ * R[1] / den, m * gammaJ * R[2] / den};

    for (bool net : {false, true})
    {
        auto frame = makeFrame(gammaRef, net);
        T    s, chi;
        frame.sourceFactors(std::atanh(betaJ) - frame.wRef, s, chi);
        EXPECT_NEAR(s, gammaJ / gammaRef, 1e-14);

        // tree frame: z stretched by gammaRef
        Vec3<T>           source{0.1, 0.2, 0.3};
        Vec3<T>           target{source[0] + R[0], source[1] + R[1], source[2] + gammaRef * R[2]};
        util::array<T, 5> acc{0, 0, 0, 0, 0};
        acc = P2PHeaviside(acc, target, source, m, s, chi, T(0), T(0));

        // P2P sign convention: acc = -field; lab E_perp = gammaRef E~_perp, E_z = E~_z
        EXPECT_NEAR(-gammaRef * acc[0], Elab[0], 1e-12);
        EXPECT_NEAR(-gammaRef * acc[1], Elab[1], 1e-12);
        EXPECT_NEAR(-acc[2], Elab[2], 1e-12);

        T weight = net ? 1.0 - frame.betaRef * betaJ : betaJ;
        EXPECT_NEAR(-gammaRef * acc[3], weight * Elab[0], 1e-12);
        EXPECT_NEAR(-gammaRef * acc[4], weight * Elab[1], 1e-12);
    }
}

//! @brief A source at the frame velocity (s = 1) has the Coulomb field of P2P, and no self interaction
TEST(Gravity, P2PHeavisideUnitStretchEqualsP2P)
{
    using T = double;

    auto frame = makeFrame(5.0, false);
    T    s, chi;
    frame.sourceFactors(0.0, s, chi);
    EXPECT_NEAR(s, 1.0, 1e-15);
    EXPECT_NEAR(chi, frame.betaRef, 1e-15);

    Vec3<T> target{1, 1, 1}, source{2, 2.5, -1};
    T       h = 0.1;

    Vec4<T>           acc4{0, 0, 0, 0};
    util::array<T, 5> acc5{0, 0, 0, 0, 0};
    acc4 = P2P(acc4, target, source, 0.7, h, h);
    acc5 = P2PHeaviside(acc5, target, source, 0.7, s, chi, h, h);
    for (int k = 0; k < 3; ++k)
    {
        EXPECT_NEAR(acc5[k], acc4[k + 1], 1e-15);
    }

    util::array<T, 5> zero{0, 0, 0, 0, 0};
    EXPECT_EQ(P2PHeaviside(zero, target, target, 0.7, s, chi, h, h), zero);
}
