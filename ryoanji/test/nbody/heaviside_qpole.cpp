/*
 * Ryoanji N-body solver
 *
 * Copyright (c) 2026 CSCS, ETH Zurich
 *               2026 PSI, Villigen
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: MIT License
 */

/*! @file
 * @brief Multipoles of sources with per-particle velocities (HeavisideQpole) against direct sums
 */

#include <cmath>
#include <random>
#include <vector>

#include "gtest/gtest.h"

#include "ryoanji/nbody/cartesian_qpole.hpp"
#include "ryoanji/nbody/heaviside_qpole.hpp"

using namespace ryoanji;

namespace
{
using T = double;

HeavisideFrame<T> frameAt(T gammaRef, bool net)
{
    HeavisideFrame<T> f;
    f.gammaRef        = gammaRef;
    f.wRef            = std::acosh(gammaRef);
    f.betaRef         = std::tanh(f.wRef);
    f.netForceChannel = net;
    return f;
}

//! @brief a unit cell of sources with rapidity w^ = w0 + spread * (0.7 z + 0.3 u), u uniform
struct Cell
{
    std::vector<T> x, y, z, m, w;
    Vec3<T>        center{0, 0, 0};

    Cell(int n, T w0, T spread, unsigned seed = 5)
    {
        std::mt19937_64                   rng(seed);
        std::uniform_real_distribution<T> U(-0.5, 0.5);
        for (int i = 0; i < n; ++i)
        {
            x.push_back(U(rng));
            y.push_back(U(rng));
            z.push_back(U(rng));
            m.push_back(1.0);
            w.push_back(w0 + spread * (0.7 * z.back() + 0.3 * U(rng)));
            for (int d = 0; d < 3; ++d)
            {
                center[d] += (d == 0 ? x.back() : d == 1 ? y.back() : z.back()) / n;
            }
        }
    }

    HeavisideQpole<T> raw(int b, int e, Vec3<T> c) const
    {
        auto M = heavisideZero<T>();
        P2MHeaviside_add<1>(x.data(), y.data(), z.data(), m.data(), w.data(), b, e, c, M);
        return M;
    }

    util::array<T, 5> direct(const HeavisideFrame<T>& frame, Vec3<T> target) const
    {
        util::array<T, 5> acc{0, 0, 0, 0, 0};
        for (size_t i = 0; i < x.size(); ++i)
        {
            T s, chi;
            frame.sourceFactors(w[i], s, chi);
            acc = P2PHeaviside<true>(acc, target, Vec3<T>{x[i], y[i], z[i]}, m[i], s, chi, T(0), T(0));
        }
        return acc;
    }
};

//! @brief relative error of E (0..2) and C (3..4), both relative to |E|
std::array<T, 2> relativeError(const util::array<T, 5>& a, const util::array<T, 5>& ref)
{
    T e = 0, c = 0, n = 0;
    for (int k = 0; k < 3; ++k)
    {
        e += (a[k] - ref[k]) * (a[k] - ref[k]);
        n += ref[k] * ref[k];
    }
    for (int k = 3; k < 5; ++k)
    {
        c += (a[k] - ref[k]) * (a[k] - ref[k]);
    }
    return {std::sqrt(e / n), std::sqrt(c / n)};
}

} // namespace

//! @brief the upsweep translation is exact: two children shifted to the parent equal a direct P2M
TEST(HeavisideQpole, M2MEqualsP2M)
{
    Cell cell(300, 0.3, 0.8);
    auto parent = cell.raw(0, 300, cell.center);
    auto merged = heavisideZero<T>();
    for (int part = 0; part < 2; ++part)
    {
        int     b = part * 150, e = b + 150;
        Vec3<T> c{0, 0, 0};
        for (int i = b; i < e; ++i)
        {
            c[0] += cell.x[i] / 150;
            c[1] += cell.y[i] / 150;
            c[2] += cell.z[i] / 150;
        }
        heavisideM2M(merged, Vec3<T>{cell.center[0] - c[0], cell.center[1] - c[1], cell.center[2] - c[2]},
                     cell.raw(b, e, c));
    }
    for (int k = 0; k < 24; ++k)
    {
        EXPECT_NEAR(parent[k], merged[k], 1e-12 * (1 + std::abs(parent[k])));
    }
}

//! @brief without velocity spread the spatial part is ryoanji's Laplace M2P in the cell's own frame
TEST(HeavisideQpole, ZeroSpreadIsTheOwnFrameLaplaceMultipole)
{
    auto frame = frameAt(2.0, false);
    Cell cell(300, 0.4, 0.0);
    auto M = cell.raw(0, 300, cell.center);
    heavisideFinalize(M, frame, true);
    const T s = M[Hqi::s];

    std::vector<T> zs(cell.z.size());
    for (size_t i = 0; i < zs.size(); ++i)
    {
        zs[i] = cell.center[2] + s * (cell.z[i] - cell.center[2]);
    }
    CartesianMDQpole<T> q;
    P2M(cell.x.data(), cell.y.data(), zs.data(), cell.m.data(), 0, 300,
        Vec4<T>{cell.center[0], cell.center[1], cell.center[2], 0}, q);

    Vec3<T> target{cell.center[0] + 3, cell.center[1] - 1, cell.center[2] + 2};
    Vec4<T> a{0, 0, 0, 0};
    a = M2P(a, Vec3<T>{target[0], target[1], cell.center[2] + s * (target[2] - cell.center[2])}, cell.center, q);
    util::array<T, 5> h{0, 0, 0, 0, 0};
    h = M2PHeaviside(h, target, cell.center, M);
    EXPECT_NEAR(h[0], s * a[1], 1e-13 * std::abs(h[0]));
    EXPECT_NEAR(h[1], s * a[2], 1e-13 * std::abs(h[0]));
    EXPECT_NEAR(h[2], a[3], 1e-13 * std::abs(h[0]));
}

/*! @brief the velocity-spread error does not fall off with distance at the cell's mean velocity, the
 *         rapidity moments remove it to third order in the spread
 */
TEST(HeavisideQpole, RapidityMomentsRemoveTheDistanceIndependentError)
{
    for (bool net : {false, true})
    {
        auto frame = frameAt(2.0, net);
        Cell cell(400, 0.3, 0.8);
        auto withMoments = cell.raw(0, 400, cell.center), atMean = withMoments;
        heavisideFinalize(withMoments, frame, true);
        heavisideFinalize(atMean, frame, false);
        for (T D : {64.0, 1024.0})
        {
            Vec3<T> target{cell.center[0] + 0.6 * D, cell.center[1] + 0.48 * D, cell.center[2] + 0.64 * D};
            auto    ref = cell.direct(frame, target);
            util::array<T, 5> a{0, 0, 0, 0, 0}, b{0, 0, 0, 0, 0};
            a = M2PHeaviside(a, target, cell.center, withMoments);
            b = M2PHeaviside(b, target, cell.center, atMean);
            auto ea = relativeError(a, ref), eb = relativeError(b, ref);
            EXPECT_GT(eb[0], 3e-3);       // cell rest frame: percent level at any distance
            EXPECT_LT(ea[0], 1e-3);       // measured 4.9e-4
            EXPECT_LT(ea[1], 1e-3);
            EXPECT_LT(ea[0], 0.1 * eb[0]);
        }
        // third order: the distance-independent error falls ~8x per halving of the spread (a wrong
        // second-order term would give 4x); measured 6.6x and 7.5x. The errors (1e-8 ... 1e-10) are below the
        // single-precision inverseSquareRoot of the traversal, so the multipole is evaluated in full precision.
        Vec3<T> far{cell.center[0] + 16384, cell.center[1], cell.center[2]};
        T       previous = 0;
        for (T spread : {0.1, 0.05, 0.025})
        {
            Cell c2(400, 0.3, spread);
            auto M = c2.raw(0, 400, c2.center);
            heavisideFinalize(M, frame, true);
            util::array<T, 5> a{0, 0, 0, 0, 0};
            a       = M2PHeaviside<true>(a, far, c2.center, M);
            T error = relativeError(a, c2.direct(frame, far))[0];
            std::printf("[HV] point-like cell far away, spread %.3f: error %.3e (ratio %.2f)\n", spread, error,
                        previous > 0 ? previous / error : 0.0);
            if (previous > 0) { EXPECT_GT(previous / error, 5.5); }
            previous = error;
        }
    }
}

//! @brief MAC scale: kappa^2 for a narrow cell, "always open" beyond thetaV, 1 for an empty cell
TEST(HeavisideQpole, MacScale)
{
    auto frame = frameAt(2.0, false);
    Cell narrow(100, 0.0, 0.1), wide(100, 0.0, 2.0);
    auto Mn = narrow.raw(0, 100, narrow.center), Mw = wide.raw(0, 100, wide.center);
    heavisideFinalize(Mn, frame, true);
    heavisideFinalize(Mw, frame, true);
    T scale = heavisideMacScale2(Mn, frame, T(0.25), true);
    EXPECT_GT(scale, 1.0);
    EXPECT_LT(scale, 1.2);
    EXPECT_LT(heavisideMacScale2(Mw, frame, T(0.25), true), 0.0);
    EXPECT_GT(heavisideMacScale2(Mw, frame, T(0.25), false), 0.0);  // no rapidity MAC without moments
    auto empty = heavisideZero<T>();
    heavisideFinalize(empty, frame, true);
    EXPECT_EQ(heavisideMacScale2(empty, frame, T(0.25), true), 1.0);
}

//! @brief cstone's per-cell rapidity summary combines exactly and gives the multipole's MAC scale
TEST(HeavisideQpole, VelocitySummaryMatchesMultipoleMacScale)
{
    for (T gammaRef : {1.05, 2.0, 30.0})
    {
        for (T w0 : {-0.6, 0.0, 0.3})
        {
            for (T spread : {0.05, 0.4, 1.5})
            {
                auto  frame = frameAt(gammaRef, gammaRef > 1.2);
                Cell  cell(200, w0, spread);
                for (size_t i = 0; i < cell.m.size(); i += 3)
                {
                    cell.m[i] = -0.5; // signed charges (images): |q| weights
                }

                auto whole = cstone::emptyVelocitySummary<T>();
                auto lo = whole, hi = whole;
                for (int i = 0; i < 200; ++i)
                {
                    cstone::addVelocitySource(whole, cell.m[i], cell.w[i]);
                    cstone::addVelocitySource(i < 120 ? lo : hi, cell.m[i], cell.w[i]);
                }
                auto combined = cstone::combineVelocitySummary(lo, hi);
                for (int k = 0; k < 4; ++k)
                {
                    EXPECT_NEAR(combined[k], whole[k], 1e-12 * (1 + std::abs(whole[k])));
                }

                auto M = cell.raw(0, 200, cell.center);
                heavisideFinalize(M, frame, true);
                for (bool moments : {true, false})
                {
                    auto p = velocityMac(frame, T(0.25), moments);
                    EXPECT_NEAR(cstone::velocityMacScale2(whole, p), heavisideMacScale2(M, frame, T(0.25), moments),
                                1e-12);
                }
            }
        }
    }
}

//! @brief kappa = max over the cell's rapidity range of max(s, 1/s), s = cosh(wRef + w^) / gammaRef (brute force)
TEST(HeavisideQpole, VelocityMacKappaIsTheLargestStretch)
{
    for (T gammaRef : {1.0, 1.3, 4.0, 100.0})
    {
        T    wRef = std::acosh(gammaRef);
        auto p    = velocityMac(frameAt(gammaRef, false), T(10), true); // no rapidity criterion in this range
        for (T wMin : {-2.5, -0.9, -0.1, 0.2})
        {
            for (T width : {0.0, 0.05, 0.7, 2.0})
            {
                T wMax = wMin + width, wc = wMin + 0.4 * width;
                T kappa = 1;
                for (int k = 0; k <= 20000; ++k)
                {
                    T s   = std::cosh(wRef + wMin + width * k / 20000.0) / gammaRef;
                    kappa = std::max(kappa, std::max(s, 1 / s));
                }
                EXPECT_NEAR(cstone::velocityMacScale2(wMin, wMax, wc, p), kappa * kappa, 1e-6 * kappa * kappa)
                    << "gammaRef " << gammaRef << " range [" << wMin << ", " << wMax << "]";
            }
        }
    }
}

//! @brief empty cells keep radius 0, cells beyond thetaV are always opened, others are scaled by kappa^2
TEST(HeavisideQpole, ScaleVelocityMac)
{
    auto p     = velocityMac(frameAt(3.0, true), T(0.25), true);
    auto empty = cstone::emptyVelocitySummary<T>();
    EXPECT_EQ(cstone::scaleVelocityMac(T(0), empty, p), T(0));
    EXPECT_EQ(cstone::scaleVelocityMac(T(2), empty, p), T(2));

    auto narrow = empty, wide = empty;
    cstone::addVelocitySource(narrow, T(1), T(0.1));
    cstone::addVelocitySource(narrow, T(1), T(-0.1));
    cstone::addVelocitySource(wide, T(1), T(0.5));
    cstone::addVelocitySource(wide, T(-1), T(-0.5));
    EXPECT_EQ(cstone::scaleVelocityMac(T(0), narrow, p), T(0));
    EXPECT_NEAR(cstone::scaleVelocityMac(T(2), narrow, p), 2 * cstone::velocityMacScale2(narrow, p), 1e-15);
    EXPECT_GT(cstone::velocityMacScale2(narrow, p), 1.0);
    EXPECT_EQ(cstone::scaleVelocityMac(T(2), wide, p), cstone::alwaysOpenMac<T>());
}
