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
 * @brief Two-node Gauss multipoles of sources with per-particle velocities (Gauss2Qpole) against
 *        direct sums and the rapidity Taylor multipole (HeavisideQpole)
 */

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

#include "gtest/gtest.h"

#include "ryoanji/nbody/gauss2_qpole.hpp"
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

//! @brief sources in a unit cube (tree coordinates) with rapidities w^ relative to the frame
struct Sources
{
    std::vector<T> x, y, z, m, w;

    std::size_t size() const { return x.size(); }

    Vec3<T> absCentre() const
    {
        Vec3<T> c{0, 0, 0};
        T       a = 0;
        for (std::size_t i = 0; i < size(); ++i)
        {
            T aq = std::abs(m[i]);
            c[0] += aq * x[i];
            c[1] += aq * y[i];
            c[2] += aq * z[i];
            a += aq;
        }
        return c * (T(1) / a);
    }

    util::array<T, 5> direct(const HeavisideFrame<T>& frame, Vec3<T> target) const
    {
        util::array<T, 5> acc{0, 0, 0, 0, 0};
        for (std::size_t i = 0; i < size(); ++i)
        {
            T s, chi;
            frame.sourceFactors(w[i], s, chi);
            acc = P2PHeaviside<true>(acc, target, Vec3<T>{x[i], y[i], z[i]}, m[i], s, chi, T(0), T(0));
        }
        return acc;
    }

    Gauss2Qpole<T> gauss2(const HeavisideFrame<T>& frame, Vec3<T> c, bool moments = true) const
    {
        auto M = P2MGauss2<T, T, T>(x.data(), y.data(), z.data(), m.data(), w.data(), 0, size(), c);
        gauss2Finalize(M, frame, moments);
        return M;
    }

    HeavisideQpole<T> taylor(const HeavisideFrame<T>& frame, Vec3<T> c, bool moments = true) const
    {
        auto M = heavisideZero<T>();
        P2MHeaviside_add<1>(x.data(), y.data(), z.data(), m.data(), w.data(), 0, size(), c, M);
        heavisideFinalize(M, frame, moments);
        return M;
    }
};

/*! @brief cell contents of the earlier single-cell assessment (gamma per source, unit cube)
 *
 * chirp: gamma linear in z; split: head and tail at gamma_c +- dg/2; mixed: the same two energies
 * randomly mixed; uniform: flat uncorrelated spread.
 */
Sources makeCell(int n, T gammaRef, T gc, T dg, const char* content, bool image, unsigned seed)
{
    std::mt19937_64                   rng(seed);
    std::uniform_real_distribution<T> U(-0.5, 0.5);
    std::normal_distribution<T>       N01(0, 1);
    const T                           wRef = std::acosh(gammaRef);
    Sources                           s;
    for (int i = 0; i < n; ++i)
    {
        T xi = U(rng), yi = U(rng), zi = U(rng);
        T g  = gc;
        if (std::string(content) == "chirp") { g = gc + dg * zi * (1 + 0.01 * N01(rng)); }
        else if (std::string(content) == "split") { g = zi > 0 ? gc + dg / 2 : gc - dg / 2; }
        else if (std::string(content) == "mixed") { g = U(rng) > 0 ? gc + dg / 2 : gc - dg / 2; }
        else if (std::string(content) == "uniform") { g = gc + dg * U(rng); }
        g      = std::max(g, T(1) + 1e-12);
        T w    = std::acosh(g);
        T q    = T(1) / n;
        if (image) // mirror charge: -q, moving backwards
        {
            q = -q;
            w = -w;
        }
        s.x.push_back(xi);
        s.y.push_back(yi);
        s.z.push_back(zi);
        s.m.push_back(q);
        s.w.push_back(w - wRef);
    }
    return s;
}

//! @brief relative error of E (0..2) and C (3..4), each relative to its own reference norm
T relativeError(const std::vector<util::array<T, 5>>& a, const std::vector<util::array<T, 5>>& ref)
{
    T e = 0, ne = 0, c = 0, nc = 0;
    for (std::size_t t = 0; t < a.size(); ++t)
    {
        for (int k = 0; k < 3; ++k)
        {
            e += (a[t][k] - ref[t][k]) * (a[t][k] - ref[t][k]);
            ne += ref[t][k] * ref[t][k];
        }
        for (int k = 3; k < 5; ++k)
        {
            c += (a[t][k] - ref[t][k]) * (a[t][k] - ref[t][k]);
            nc += ref[t][k] * ref[t][k];
        }
    }
    return std::max(std::sqrt(e / ne), std::sqrt(c / nc));
}

//! @brief targets at distance 1 / theta (tree frame) in random directions from @p c
std::vector<Vec3<T>> targetsAround(Vec3<T> c, T distance, int n, unsigned seed)
{
    std::mt19937_64             rng(seed);
    std::normal_distribution<T> N01(0, 1);
    std::vector<Vec3<T>>        t;
    for (int i = 0; i < n; ++i)
    {
        Vec3<T> d{N01(rng), N01(rng), N01(rng)};
        d = d * (distance / std::sqrt(norm2(d)));
        t.push_back(c + d);
    }
    return t;
}

} // namespace

//! @brief the two nodes reproduce the |q|-moments of order 0..3 and lie inside the rapidity range
TEST(Gauss2Qpole, NodesReproduceMomentsToThirdOrder)
{
    std::mt19937_64                   rng(11);
    std::uniform_real_distribution<T> U(0, 1);
    std::exponential_distribution<T>  E(3.0);
    for (int dist = 0; dist < 4; ++dist)
    {
        std::vector<T> t, mu;
        for (int i = 0; i < 500; ++i)
        {
            T v = dist == 0 ? U(rng) : dist == 1 ? E(rng) : dist == 2 ? -E(rng) : (i == 0 ? 5.0 : 0.01 * U(rng));
            t.push_back(v);
            mu.push_back(0.5 + U(rng));
        }
        T mu0 = 0, mean = 0;
        for (int i = 0; i < 500; ++i) { mu0 += mu[i], mean += mu[i] * t[i]; }
        mean /= mu0;
        T a2 = 0, a3 = 0;
        for (int i = 0; i < 500; ++i)
        {
            t[i] -= mean;
            a2 += mu[i] * t[i] * t[i];
            a3 += mu[i] * t[i] * t[i] * t[i];
        }
        auto n = gauss2Nodes(mu0, a2, a3, 1e-16);
        ASSERT_FALSE(n.single);
        EXPECT_GT(n.t1, *std::min_element(t.begin(), t.end()));
        EXPECT_LT(n.t2, *std::max_element(t.begin(), t.end()));
        EXPECT_LT(n.t1, n.t2);
        const T W1 = mu0 * n.t2 / (n.t2 - n.t1), W2 = -mu0 * n.t1 / (n.t2 - n.t1);
        EXPECT_GT(W1, 0);
        EXPECT_GT(W2, 0);
        for (int k = 0; k <= 3; ++k)
        {
            T exact = 0, scale = 0;
            for (int i = 0; i < 500; ++i)
            {
                exact += mu[i] * std::pow(t[i], k);
                scale += mu[i] * std::pow(std::abs(t[i]), k);
            }
            EXPECT_NEAR(W1 * std::pow(n.t1, k) + W2 * std::pow(n.t2, k), exact, 1e-12 * scale)
                << "distribution " << dist << " order " << k;
        }
    }
    EXPECT_TRUE(gauss2Nodes(T(1), T(1e-20), T(0), T(1e-16)).single); // no spread: one node
    EXPECT_TRUE(gauss2Nodes(T(0), T(0), T(0), T(1e-16)).single);     // empty cell
}

//! @brief the two-pass P2M equals the moments computed from their definitions in long double
TEST(Gauss2Qpole, P2MMatchesDefinition)
{
    auto    s = makeCell(400, 2.0, 2.0, 1.0, "chirp", false, 3);
    for (std::size_t i = 0; i < s.size(); i += 4) { s.m[i] = -s.m[i]; } // mixed signs
    Vec3<T> c{0.01, -0.02, 0.03};
    auto    M = P2MGauss2<T, T, T>(s.x.data(), s.y.data(), s.z.data(), s.m.data(), s.w.data(), 0, s.size(), c);

    using L         = long double;
    L       am = 0, aw = 0;
    for (std::size_t i = 0; i < s.size(); ++i) { am += std::abs(L(s.m[i])), aw += std::abs(L(s.m[i])) * s.w[i]; }
    const L mean = aw / am;
    L       ref[32] = {};
    ref[G2i::absMass] = am;
    ref[G2i::absW]    = aw;
    for (std::size_t i = 0; i < s.size(); ++i)
    {
        L q = s.m[i], aq = std::abs(q), t = L(s.w[i]) - mean;
        L d[3] = {L(s.x[i]) - c[0], L(s.y[i]) - c[1], L(s.z[i]) - c[2]};
        const int pairs[6][2] = {{0, 0}, {0, 1}, {0, 2}, {1, 1}, {1, 2}, {2, 2}};
        ref[G2i::mass] += q;
        for (int k = 0; k < 3; ++k)
        {
            ref[G2i::px + k] += q * d[k];
            ref[G2i::xx + k] += q * t * d[k];
            ref[G2i::zx + k] += aq * t * d[k];
            ref[G2i::bx + k] += aq * d[k];
        }
        for (int k = 0; k < 6; ++k)
        {
            ref[G2i::sxx + k] += q * d[pairs[k][0]] * d[pairs[k][1]];
            ref[G2i::yxx + k] += q * t * d[pairs[k][0]] * d[pairs[k][1]];
        }
        ref[G2i::v1] += q * t;
        ref[G2i::a2] += aq * t * t;
        ref[G2i::a3] += aq * t * t * t;
    }
    for (int k = 0; k < 32; ++k)
    {
        EXPECT_NEAR(M[k], double(ref[k]), 1e-13 * (1 + std::abs(double(ref[k])))) << "entry " << k;
    }
}

//! @brief the upsweep translation is exact: two children with their own centres and mean rapidities,
//!        shifted to the parent, equal a direct P2M of the parent
TEST(Gauss2Qpole, M2MEqualsP2M)
{
    auto s = makeCell(300, 3.0, 2.0, 1.5, "chirp", false, 5);
    for (std::size_t i = 0; i < s.size(); i += 3) { s.m[i] = -0.5 * s.m[i]; } // mixed signs
    // children: tail and head (different centres and, with the chirp, different mean rapidities)
    Sources a, b;
    for (std::size_t i = 0; i < s.size(); ++i)
    {
        Sources& c = s.z[i] < 0 ? a : b;
        c.x.push_back(s.x[i]), c.y.push_back(s.y[i]), c.z.push_back(s.z[i]);
        c.m.push_back(s.m[i]), c.w.push_back(s.w[i]);
    }
    const Vec3<T> cp = s.absCentre(), ca = a.absCentre(), cb = b.absCentre();
    auto parent = P2MGauss2<T, T, T>(s.x.data(), s.y.data(), s.z.data(), s.m.data(), s.w.data(), 0, s.size(), cp);
    auto Ma     = P2MGauss2<T, T, T>(a.x.data(), a.y.data(), a.z.data(), a.m.data(), a.w.data(), 0, a.size(), ca);
    auto Mb     = P2MGauss2<T, T, T>(b.x.data(), b.y.data(), b.z.data(), b.m.data(), b.w.data(), 0, b.size(), cb);
    ASSERT_GT(std::abs(gauss2Mean(Ma) - gauss2Mean(Mb)), 0.1);

    const T wMean = gauss2Mean(Ma[G2i::absMass] + Mb[G2i::absMass], Ma[G2i::absW] + Mb[G2i::absW]);
    auto    M     = gauss2Zero<T>();
    gauss2M2M(M, cp - ca, wMean, Ma);
    gauss2M2M(M, cp - cb, wMean, Mb);
    for (int k = 0; k < 32; ++k)
    {
        EXPECT_NEAR(M[k], parent[k], 1e-12 * (1 + std::abs(parent[k]))) << "entry " << k;
    }
}

//! @brief a cell of two populations at two rapidities: the nodes are the two rapidities, the node
//!        monopoles the populations' charges and the node centres their centroids
TEST(Gauss2Qpole, TwoPopulationsGiveTheirCentroids)
{
    auto    frame = frameAt(3.0, true);
    auto    s     = makeCell(1000, 3.0, 3.0, 1.0, "split", false, 37);
    Vec3<T> c     = s.absCentre() + Vec3<T>{0.01, 0.02, -0.03}; // any expansion centre
    auto    M = P2MGauss2<T, T, T>(s.x.data(), s.y.data(), s.z.data(), s.m.data(), s.w.data(), 0, s.size(), c);
    gauss2Finalize(M, frame, true);
    Vec3<T> centroid[2] = {{0, 0, 0}, {0, 0, 0}};
    T       charge[2]   = {0, 0};
    const T wLow        = *std::min_element(s.w.begin(), s.w.end());
    for (std::size_t i = 0; i < s.size(); ++i)
    {
        int nu = s.w[i] > wLow + 1e-9 ? 1 : 0;
        centroid[nu] += Vec3<T>{s.x[i], s.y[i], s.z[i]} * s.m[i];
        charge[nu] += s.m[i];
    }
    for (int nu = 0; nu < 2; ++nu)
    {
        const int base = nu * G2n::size;
        EXPECT_NEAR(M[base + G2n::mass], charge[nu], 1e-12);
        for (int k = 0; k < 3; ++k)
        {
            EXPECT_NEAR(c[k] + M[base + G2n::ex + k], centroid[nu][k] / charge[nu], 1e-12) << "node " << nu;
            EXPECT_NEAR(M[base + G2n::px + k], 0.0, 1e-12) << "node dipole about its centroid";
        }
    }
    // the node rapidities are the two population rapidities
    for (int nu = 0; nu < 2; ++nu)
    {
        const T wPop = nu == 0 ? wLow : *std::max_element(s.w.begin(), s.w.end());
        EXPECT_NEAR(M[nu * G2n::size + G2n::s], std::cosh(wPop) + frame.betaRef * std::sinh(wPop), 1e-12);
    }
}

//! @brief without velocity spread both forms are the own-frame Laplace multipole: G2 equals HV2
TEST(Gauss2Qpole, ZeroSpreadEqualsTheTaylorMultipole)
{
    for (bool net : {false, true})
    {
        auto    frame = frameAt(net ? 5.0 : 1.5, net);
        auto    s     = makeCell(300, frame.gammaRef, 3.0, 0.0, "chirp", false, 7);
        Vec3<T> c     = s.absCentre();
        auto    G     = s.gauss2(frame, c);
        auto    H     = s.taylor(frame, c);
        EXPECT_EQ(G[G2n::size + G2n::mass], 0.0); // one node
        for (auto target : targetsAround(c, 3.0, 20, 9))
        {
            util::array<T, 5> g{0, 0, 0, 0, 0}, h{0, 0, 0, 0, 0};
            g = M2PGauss2(g, target, c, G);
            h = M2PHeaviside(h, target, c, H);
            const T scale = std::sqrt(h[0] * h[0] + h[1] * h[1] + h[2] * h[2]);
            for (int k = 0; k < 5; ++k) { EXPECT_NEAR(g[k], h[k], 1e-13 * scale); }
        }
    }
}

/*! @brief the monopole is exact for kernels cubic in the rapidity: for a point-like cell (every source at
 *         the centre) the error falls as the fourth power of the spread, the Taylor multipole's as the third
 */
TEST(Gauss2Qpole, MonopoleIsFourthOrderInTheSpread)
{
    auto frame = frameAt(2.0, false);
    T    prevG = 0, prevH = 0;
    for (T spread : {0.2, 0.1, 0.05})
    {
        Sources s;
        std::mt19937_64                   rng(13);
        std::exponential_distribution<T>  E(1.0); // skewed: the third moment matters
        for (int i = 0; i < 200; ++i)
        {
            s.x.push_back(0), s.y.push_back(0), s.z.push_back(0);
            s.m.push_back(1.0 / 200);
            s.w.push_back(0.3 + spread * (E(rng) - 1.0));
        }
        Vec3<T> c{0, 0, 0};
        auto    G = s.gauss2(frame, c);
        auto    H = s.taylor(frame, c);
        std::vector<util::array<T, 5>> g, h, ref;
        for (auto target : targetsAround(c, 50.0, 40, 17))
        {
            ref.push_back(s.direct(frame, target));
            g.push_back(M2PGauss2(util::array<T, 5>{0, 0, 0, 0, 0}, target, c, G));
            h.push_back(M2PHeaviside(util::array<T, 5>{0, 0, 0, 0, 0}, target, c, H));
        }
        T eg = relativeError(g, ref), eh = relativeError(h, ref);
        std::printf("[G2] point cell, spread %.3f: G2 %.3e (ratio %.2f), Taylor %.3e (ratio %.2f)\n", spread, eg,
                    prevG > 0 ? prevG / eg : 0.0, eh, prevH > 0 ? prevH / eh : 0.0);
        if (prevG > 0)
        {
            EXPECT_GT(prevG / eg, 12.0);
            EXPECT_LT(eg, eh);
        }
        prevG = eg;
        prevH = eh;
    }
}

namespace
{
struct CellErrors
{
    T noSpread, taylor, gauss;
};

//! @brief errors at distance 1 / theta: spatial quadrupole alone (sources at the mean), HV2 and G2
CellErrors cellErrors(T gc, T dg, const char* content, bool image, T theta = 0.5)
{
    auto    frame = frameAt(gc, true);
    auto    s     = makeCell(2000, gc, gc, dg, content, image, 21);
    Vec3<T> c     = s.absCentre();
    T       am = 0, aw = 0;
    for (std::size_t i = 0; i < s.size(); ++i) { am += std::abs(s.m[i]), aw += std::abs(s.m[i]) * s.w[i]; }
    Sources flat = s;
    std::fill(flat.w.begin(), flat.w.end(), aw / am);

    auto G = s.gauss2(frame, c);
    auto H = s.taylor(frame, c);
    auto F = flat.gauss2(frame, c);
    std::vector<util::array<T, 5>> g, h, f, ref, refFlat;
    for (auto target : targetsAround(c, 1.0 / theta, 300, 23))
    {
        ref.push_back(s.direct(frame, target));
        refFlat.push_back(flat.direct(frame, target));
        g.push_back(M2PGauss2(util::array<T, 5>{0, 0, 0, 0, 0}, target, c, G));
        h.push_back(M2PHeaviside(util::array<T, 5>{0, 0, 0, 0, 0}, target, c, H));
        f.push_back(M2PGauss2(util::array<T, 5>{0, 0, 0, 0, 0}, target, c, F));
    }
    return {relativeError(f, refFlat), relativeError(h, ref), relativeError(g, ref)};
}
} // namespace

/*! @brief single cells of the earlier assessment (2000 sources, targets at 1 / theta = 2 cell edges):
 *         the two-node multipole stays near the error of the spatial quadrupole alone
 *
 * Measured (G2 / no spread): gamma_c 10: chirp 1.32, split 0.78; 3: chirp 1.44, split 0.85, mixed 0.97;
 * 1.8: chirp 1.86, split 1.17; near rest (1.1 and 1.02, small spread) chirp 2.17 and 2.12, where the
 * Taylor multipole is better (1.56, 1.04); images as their particles. Taylor up to 6.45 (1.8 split).
 */
TEST(Gauss2Qpole, SingleCellErrorStaysNearTheSpatialError)
{
    struct Case
    {
        T           gc, dg;
        const char* content;
        bool        image;
        bool        beatsTaylor;
    };
    for (Case k : {Case{10, 1, "chirp", false, false}, Case{10, 1, "split", false, true},
                   Case{3, 1, "chirp", false, true}, Case{3, 1, "split", false, true}, Case{3, 1, "mixed", false, true},
                   Case{1.8, 1, "chirp", false, true}, Case{1.8, 1, "split", false, true},
                   Case{1.1, 0.2, "chirp", false, false}, Case{1.02, 0.04, "chirp", false, false},
                   Case{3, 1, "chirp", true, true}, Case{1.8, 1, "split", true, true}})
    {
        auto e = cellErrors(k.gc, k.dg, k.content, k.image);
        std::printf("[G2] gamma_c %5.2f dg %4.2f %-7s%s: no spread %.3e  Taylor %.3e (%.2fx)  Gauss %.3e (%.2fx)\n",
                    k.gc, k.dg, k.content, k.image ? " image" : "      ", e.noSpread, e.taylor,
                    e.taylor / e.noSpread, e.gauss, e.gauss / e.noSpread);
        EXPECT_LT(e.gauss, 2.5 * e.noSpread) << k.gc << " " << k.content;
        if (k.beatsTaylor) { EXPECT_LT(e.gauss, e.taylor) << k.gc << " " << k.content; }
    }
}

//! @brief raw moments stored in single precision (MIXED): central moments keep cells far from the frame
//!        rapidity accurate (measured float vs double <= 2.0e-5 at 2 kappa cell edges)
TEST(Gauss2Qpole, FloatStorageIsStable)
{
    auto frame = frameAt(30.0, true);
    for (T w0 : {-4.0, 0.0, 2.0})
    {
        auto s = makeCell(1000, frame.gammaRef, 3.0, 0.0, "chirp", false, 29);
        for (std::size_t i = 0; i < s.size(); ++i) { s.w[i] = w0 + 0.01 * s.z[i] + 0.002 * std::sin(37.0 * s.x[i]); }
        Vec3<T> c  = s.absCentre();
        auto    Md = P2MGauss2<T, T, T>(s.x.data(), s.y.data(), s.z.data(), s.m.data(), s.w.data(), 0, s.size(), c);
        auto    Mf = P2MGauss2<T, T, float>(s.x.data(), s.y.data(), s.z.data(), s.m.data(), s.w.data(), 0, s.size(),
                                            c);
        auto    nf = gauss2Nodes(T(Mf[G2i::absMass]), T(Mf[G2i::a2]), T(Mf[G2i::a3]),
                                 T(std::numeric_limits<float>::epsilon()));
        T       wm = gauss2Mean(Mf);
        gauss2Finalize(Md, frame, true);
        gauss2Finalize(Mf, frame, true);
        std::vector<util::array<T, 5>> d, f, ref;
        // outside the region that the velocity MAC excludes: 2 kappa cell edges
        const T sNode = std::cosh(w0) + frame.betaRef * std::sinh(w0);
        for (auto target : targetsAround(c, 2.0 * std::max(sNode, T(1) / sNode), 100, 31))
        {
            ref.push_back(s.direct(frame, target));
            d.push_back(M2PGauss2(util::array<T, 5>{0, 0, 0, 0, 0}, target, c, Md));
            f.push_back(M2PGauss2(util::array<T, 5>{0, 0, 0, 0, 0}, target, c, Mf));
        }
        T wmin = *std::min_element(s.w.begin(), s.w.end()), wmax = *std::max_element(s.w.begin(), s.w.end());
        std::printf("[G2] float storage w0 %.1f: single %d, nodes [%.6f, %.6f] in [%.6f, %.6f]; float vs double "
                    "%.3e, double vs direct %.3e\n",
                    w0, int(nf.single), wm + nf.t1, wm + nf.t2, wmin, wmax, relativeError(f, d),
                    relativeError(d, ref));
        EXPECT_FALSE(nf.single);
        EXPECT_GT(wm + nf.t1, wmin);
        EXPECT_LT(wm + nf.t2, wmax);
        EXPECT_LT(relativeError(f, d), 1e-4);
        EXPECT_LT(relativeError(d, ref), 1e-2);
    }
}
