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
 * @brief Multipoles of sources moving with different velocities (Heaviside kernel, see kernel.hpp)
 *
 * Every source moves uniformly along z with its own rapidity w; in tree coordinates (z stretched by
 * gammaRef) its field is E~ = q s R / (rho^2 + s^2 zeta^2)^(3/2) with s = cosh(w) / gammaRef, plus the
 * transverse second channel chi E~_perp. A cell's far field is the Taylor expansion of the sum in
 * the source displacement d and the rapidity deviation dw = w - w_c up to total order 2:
 *
 *   sum q K(R - d, w_c + dw) ~ M0 K - P.grad K + 1/2 S:grad grad K + V1 dK/dw - X.grad dK/dw
 *                              + 1/2 V2 d2K/dw2
 *
 * with P = sum q d, S = sum q d d^T, V1 = sum q dw, X = sum q d dw, V2 = sum q dw^2. The spatial part
 * at fixed w_c is the Laplace multipole in the cell's own rest frame (z scaled by s_c); the rapidity
 * derivatives have closed forms. The velocity-spread error does not decrease with distance, which is
 * why the moments V1, X, V2 are needed and why the acceptance criterion also limits the rapidity
 * range of a cell (heavisideMacScale2).
 *
 * Two forms share one array:
 *  - raw (upsweep): moments about the expansion centre in space and about w^ = 0 (the frame
 *    rapidity) in rapidity, plus the rapidity range; translates exactly with heavisideM2M,
 *  - evaluation (after heavisideFinalize): moments re-centred at w_c, the own-frame traceless
 *    quadrupole and the per-cell kinematic factors used by M2PHeaviside.
 */

#pragma once

#include <cmath>

#include "cstone/traversal/velocity_mac.hpp"
#include "kernel.hpp"
#include "types.h"

namespace ryoanji
{

template<class T>
using HeavisideQpole = util::array<T, 24>;

//! @brief HeavisideQpole index names; entries with two names change meaning in heavisideFinalize
struct Hqi
{
    enum IndexNames
    {
        mass    = 0,  //!< sum q
        absMass = 1,  //!< raw: sum |q|;                   evaluation: w^_c
        px      = 2,  //!< raw: P = sum q d (tree frame);   evaluation: P' = (px, py, s pz)
        py      = 3,
        pz      = 4,
        sxx     = 5,  //!< raw: S = sum q d d^T;           evaluation: 3 S' - tr(S') of the own frame
        sxy     = 6,
        sxz     = 7,
        syy     = 8,
        syz     = 9,
        szz     = 10,
        absW    = 11, //!< raw: sum |q| w^;                evaluation: 1 - beta_c^2
        v1      = 12, //!< raw: sum q w^;                  evaluation: V1 = sum q dw
        xx      = 13, //!< raw: sum q d w^;                evaluation: X = sum q d dw
        xy      = 14,
        xz      = 15,
        v2      = 16, //!< raw: sum q w^2;                 evaluation: V2 = sum q dw^2
        wMax    = 17, //!< largest w^ of the cell (kNoRapidity if empty)
        wNegMin = 18, //!< minus the smallest w^ of the cell
        s       = 19, //!< evaluation: s_c = gamma(w_c) / gammaRef
        beta    = 20, //!< evaluation: beta_c = tanh(w_c)
        chi     = 21, //!< evaluation: second-channel weight and its first two rapidity derivatives
        dchi    = 22,
        ddchi   = 23,
    };
};

//! @brief rapidity-range sentinel of an empty cell (combined with max)
constexpr float kNoRapidity = -1.0e30f;

//! @brief parameters of the velocity-aware MAC in cstone for a tree frame
template<class Tf>
HOST_DEVICE_FUN cstone::VelocityMac<Tf> velocityMac(const HeavisideFrame<Tf>& frame, Tf thetaV, bool velocityMoments)
{
    cstone::VelocityMac<Tf> p;
    p.gammaRef          = frame.gammaRef;
    p.betaRef           = frame.betaRef;
    p.wRef              = frame.wRef;
    p.thetaV            = thetaV;
    p.rapidityCriterion = velocityMoments;
    p.active            = true;
    return p;
}

//! @brief raw multipole of an empty cell
template<class T>
HOST_DEVICE_FUN HeavisideQpole<T> heavisideZero()
{
    HeavisideQpole<T> M;
    M             = T(0);
    M[Hqi::wMax]    = T(kNoRapidity);
    M[Hqi::wNegMin] = T(kNoRapidity);
    return M;
}

//! @brief combine entry @p i of two raw multipoles about the same centres (sum, max for the range)
template<class T>
HOST_DEVICE_FUN T heavisideCombine(int i, T a, T b)
{
    return (i == Hqi::wMax || i == Hqi::wNegMin) ? (a > b ? a : b) : a + b;
}

/*! @brief add sources [begin, end) (stride) to the raw multipole @p M about @p center
 *
 * @p w holds the source rapidities relative to the frame (w^).
 */
template<int stride, class T1, class T2, class T3, class Tc>
HOST_DEVICE_FUN void P2MHeaviside_add(const T1* x, const T1* y, const T1* z, const T2* m, const T2* w,
                                      LocalIndex begin, LocalIndex end, const Tc& center, HeavisideQpole<T3>& M)
{
    for (LocalIndex i = begin; i < end; i += stride)
    {
        T1 dx = x[i] - center[0];
        T1 dy = y[i] - center[1];
        T1 dz = z[i] - center[2];
        T1 q  = m[i];
        T1 aq = q < T1(0) ? -q : q;
        T1 wi = w[i];

        M[Hqi::mass] += q;
        M[Hqi::absMass] += aq;
        M[Hqi::px] += q * dx;
        M[Hqi::py] += q * dy;
        M[Hqi::pz] += q * dz;
        M[Hqi::sxx] += q * dx * dx;
        M[Hqi::sxy] += q * dx * dy;
        M[Hqi::sxz] += q * dx * dz;
        M[Hqi::syy] += q * dy * dy;
        M[Hqi::syz] += q * dy * dz;
        M[Hqi::szz] += q * dz * dz;
        M[Hqi::absW] += aq * wi;
        M[Hqi::v1] += q * wi;
        M[Hqi::xx] += q * dx * wi;
        M[Hqi::xy] += q * dy * wi;
        M[Hqi::xz] += q * dz * wi;
        M[Hqi::v2] += q * wi * wi;
        M[Hqi::wMax]    = wi > M[Hqi::wMax] ? T3(wi) : M[Hqi::wMax];
        M[Hqi::wNegMin] = -wi > M[Hqi::wNegMin] ? T3(-wi) : M[Hqi::wNegMin];
    }
}

/*! @brief add the raw multipole @p child, expanded about a centre at @p dX from the composite centre
 *
 * @p dX = composite centre - child centre (the convention of addQuadrupole). Exact: the moments are
 * polynomial in the displacement d = d_child - dX and do not depend on the reference rapidity.
 */
template<class T, class Tc>
HOST_DEVICE_FUN void heavisideM2M(HeavisideQpole<T>& M, Vec3<Tc> dX, const HeavisideQpole<T>& child)
{
    const Tc  ex = -dX[0], ey = -dX[1], ez = -dX[2];  // child centre - composite centre
    const Tc  m  = child[Hqi::mass];
    const Tc  px = child[Hqi::px], py = child[Hqi::py], pz = child[Hqi::pz];
    const Tc  v1 = child[Hqi::v1];

    M[Hqi::mass] += child[Hqi::mass];
    M[Hqi::absMass] += child[Hqi::absMass];
    M[Hqi::px] += px + m * ex;
    M[Hqi::py] += py + m * ey;
    M[Hqi::pz] += pz + m * ez;
    M[Hqi::sxx] += child[Hqi::sxx] + 2 * ex * px + m * ex * ex;
    M[Hqi::sxy] += child[Hqi::sxy] + ex * py + ey * px + m * ex * ey;
    M[Hqi::sxz] += child[Hqi::sxz] + ex * pz + ez * px + m * ex * ez;
    M[Hqi::syy] += child[Hqi::syy] + 2 * ey * py + m * ey * ey;
    M[Hqi::syz] += child[Hqi::syz] + ey * pz + ez * py + m * ey * ez;
    M[Hqi::szz] += child[Hqi::szz] + 2 * ez * pz + m * ez * ez;
    M[Hqi::absW] += child[Hqi::absW];
    M[Hqi::v1] += v1;
    M[Hqi::xx] += child[Hqi::xx] + ex * v1;
    M[Hqi::xy] += child[Hqi::xy] + ey * v1;
    M[Hqi::xz] += child[Hqi::xz] + ez * v1;
    M[Hqi::v2] += child[Hqi::v2];
    M[Hqi::wMax]    = child[Hqi::wMax] > M[Hqi::wMax] ? child[Hqi::wMax] : M[Hqi::wMax];
    M[Hqi::wNegMin] = child[Hqi::wNegMin] > M[Hqi::wNegMin] ? child[Hqi::wNegMin] : M[Hqi::wNegMin];
}

//! @brief stretch, velocity and second-channel weight (with its first two rapidity derivatives) at w^
template<class Tf>
struct OwnFrameKinematics
{
    Tf s, beta, omb2, chi, dchi, ddchi;
};

//! @brief kinematic factors of sources at rapidity @p wHat relative to the frame
template<class Tf>
HOST_DEVICE_FUN OwnFrameKinematics<Tf> ownFrameKinematics(const HeavisideFrame<Tf>& frame, Tf wHat)
{
    OwnFrameKinematics<Tf> k;
    const Tf ch = std::cosh(wHat);
    k.s         = ch + frame.betaRef * std::sinh(wHat);
    k.beta      = std::tanh(frame.wRef + wHat);
    k.omb2      = Tf(1) / (frame.gammaRef * frame.gammaRef * k.s * k.s); // 1 - beta^2 = 1 / gamma^2
    if (frame.netForceChannel) // chi = 1 - betaRef beta
    {
        k.chi   = ch / (frame.gammaRef * frame.gammaRef * k.s);
        k.dchi  = -frame.betaRef * k.omb2;
        k.ddchi = 2 * frame.betaRef * k.beta * k.omb2;
    }
    else // chi = beta
    {
        k.chi   = k.beta;
        k.dchi  = k.omb2;
        k.ddchi = -2 * k.beta * k.omb2;
    }
    return k;
}

/*! @brief own-frame (z scaled by @p s) traceless second moments 3 S' - tr(S') of tree-frame moments
 *
 * @p S and @p Q: xx, xy, xz, yy, yz, zz
 */
template<class Tf>
HOST_DEVICE_FUN util::array<Tf, 6> ownFrameQuadrupole(Tf s, const util::array<Tf, 6>& S)
{
    const Tf sxx = S[0], sxy = S[1], sxz = s * S[2];
    const Tf syy = S[3], syz = s * S[4], szz = s * s * S[5];
    const Tf tr  = sxx + syy + szz;
    return {3 * sxx - tr, 3 * sxy, 3 * sxz, 3 * syy - tr, 3 * syz, 3 * szz - tr};
}

/*! @brief raw -> evaluation form of a cell multipole
 *
 * Without @p velocityMoments (test only) every source is put at the cell's mean rapidity: the
 * "cell rest frame" approximation, whose error does not decrease with distance.
 */
template<class T, class Tf>
HOST_DEVICE_FUN void heavisideFinalize(HeavisideQpole<T>& M, const HeavisideFrame<Tf>& frame, bool velocityMoments)
{
    const Tf absMass = M[Hqi::absMass];
    if (!(absMass > Tf(0)))
    {
        M        = T(0);
        M[Hqi::s] = T(1);
        return;
    }
    const Tf wc   = Tf(M[Hqi::absW]) / absMass;
    const auto k  = ownFrameKinematics(frame, wc);
    const Tf s    = k.s;

    const Tf m = M[Hqi::mass];
    const Tf px = M[Hqi::px], py = M[Hqi::py], pz = M[Hqi::pz];
    const Tf v1 = M[Hqi::v1];
    // moments re-centred at w_c
    const Tf V1 = v1 - wc * m;
    const Tf Xx = Tf(M[Hqi::xx]) - wc * px, Xy = Tf(M[Hqi::xy]) - wc * py, Xz = Tf(M[Hqi::xz]) - wc * pz;
    const Tf V2 = Tf(M[Hqi::v2]) - 2 * wc * v1 + wc * wc * m;
    // own-frame (z scaled by s) second moments, traceless form 3 S' - tr S'
    const auto Q = ownFrameQuadrupole(s, util::array<Tf, 6>{Tf(M[Hqi::sxx]), Tf(M[Hqi::sxy]), Tf(M[Hqi::sxz]),
                                                            Tf(M[Hqi::syy]), Tf(M[Hqi::syz]), Tf(M[Hqi::szz])});

    M[Hqi::absMass] = wc;
    M[Hqi::pz]      = s * pz;
    M[Hqi::sxx]     = Q[0];
    M[Hqi::sxy]     = Q[1];
    M[Hqi::sxz]     = Q[2];
    M[Hqi::syy]     = Q[3];
    M[Hqi::syz]     = Q[4];
    M[Hqi::szz]     = Q[5];
    M[Hqi::absW]    = k.omb2;
    M[Hqi::v1]      = velocityMoments ? V1 : Tf(0);
    M[Hqi::xx]      = velocityMoments ? Xx : Tf(0);
    M[Hqi::xy]      = velocityMoments ? Xy : Tf(0);
    M[Hqi::xz]      = velocityMoments ? Xz : Tf(0);
    M[Hqi::v2]      = velocityMoments ? V2 : Tf(0);
    M[Hqi::s]       = s;
    M[Hqi::beta]    = k.beta;
    M[Hqi::chi]     = k.chi;
    M[Hqi::dchi]    = k.dchi;
    M[Hqi::ddchi]   = k.ddchi;
}

/*! @brief factor for the squared MAC radius of a cell in evaluation form, or a negative value if the
 *         cell must always be opened
 *
 * The criterion itself is cstone::velocityMacScale2 (kappa^2 from the rapidity range, "always open"
 * beyond thetaV), which cstone applies to the MAC radii it uses for the locally essential tree and which
 * the traversal reads. This overload evaluates it from a multipole for tests.
 */
template<class T, class Tf>
HOST_DEVICE_FUN Tf heavisideMacScale2(const HeavisideQpole<T>& M, const HeavisideFrame<Tf>& frame, Tf thetaV,
                                      bool velocityMoments)
{
    if (M[Hqi::wMax] <= T(kNoRapidity)) { return Tf(1); }
    const Tf wmax = M[Hqi::wMax], wmin = -Tf(M[Hqi::wNegMin]), wc = M[Hqi::absMass];
    return cstone::velocityMacScale2(wmin, wmax, wc, velocityMac(frame, thetaV, velocityMoments));
}

//! @brief Laplace field of an own-frame multipole and the powers of the own-frame distance it used
template<class Ta>
struct OwnFrameField
{
    Ta ax, ay, az; //!< Laplace field in own-frame coordinates, acceleration convention (as M2P)
    Ta zp, r2, rinv2, rinv3, rinv5;
};

/*! @brief Laplace monopole, dipole and quadrupole of a multipole in the own frame of its sources
 *
 * @param rx, ry, rz  target - expansion centre in tree coordinates
 * @param s           stretch of the sources relative to the tree frame (own-frame z' = s z)
 * @param mass        monopole
 * @param P           own-frame dipole (px, py, s pz)
 * @param Q           own-frame traceless quadrupole 3 S' - tr(S'): xx, xy, xz, yy, yz, zz
 *
 * Ryoanji's dipole-quadrupole M2P (CartesianMDQpole) evaluated at R' = (rx, ry, s rz). The tree-frame
 * field E~ of the sources is (s ax, s ay, az). @p Exact: full-precision 1 / sqrt (see inverseSquareRootExact),
 * for tests of the truncation error below the single-precision inverseSquareRoot.
 */
template<bool Exact = false, class Ta, class Tm>
HOST_DEVICE_FUN DEVICE_INLINE OwnFrameField<Ta> ownFrameMDQ(Ta rx, Ta ry, Ta rz, Ta s, Tm mass, const Tm* P,
                                                            const Tm* Q)
{
    OwnFrameField<Ta> f;
    f.zp     = s * rz; // own-frame z
    const Ta zp = f.zp;

    f.r2     = rx * rx + ry * ry + zp * zp;
    const Ta rinv = inverseSquareRootOf<Exact>(f.r2);
    f.rinv2  = rinv * rinv;
    f.rinv3  = rinv * f.rinv2;
    f.rinv5  = f.rinv3 * f.rinv2;

    const Ta Qrx = rx * Q[0] + ry * Q[1] + zp * Q[2];
    const Ta Qry = rx * Q[1] + ry * Q[3] + zp * Q[4];
    const Ta Qrz = rx * Q[2] + ry * Q[4] + zp * Q[5];
    const Ta rQr = rx * Qrx + ry * Qry + zp * Qrz;
    const Ta rQrAndMonopole = (Ta(-2.5) * rQr * f.rinv5 - Ta(mass) * rinv) * f.rinv2;
    const Ta PR  = rx * P[0] + ry * P[1] + zp * P[2];
    f.ax = f.rinv5 * Qrx + rQrAndMonopole * rx + f.rinv5 * (P[0] * f.r2 - Ta(3) * PR * rx);
    f.ay = f.rinv5 * Qry + rQrAndMonopole * ry + f.rinv5 * (P[1] * f.r2 - Ta(3) * PR * ry);
    f.az = f.rinv5 * Qrz + rQrAndMonopole * zp + f.rinv5 * (P[2] * f.r2 - Ta(3) * PR * zp);
    return f;
}

/*! @brief apply a cell multipole in evaluation form to a target, see P2PHeaviside for the outputs
 *
 * @return @p acc + contribution, (E_x, E_y, E_z, C_x, C_y) in the sign convention of P2P (-field)
 * @tparam Exact  full-precision 1 / sqrt, as ownFrameMDQ
 */
template<bool Exact = false, class Ta, class Tc, class Tmp>
HOST_DEVICE_FUN DEVICE_INLINE util::array<Ta, 5> M2PHeaviside(util::array<Ta, 5> acc, const Vec3<Tc>& target,
                                                              const Vec3<Tc>& center,
                                                              const HeavisideQpole<Tmp>& M)
{
    const Ta s  = M[Hqi::s];
    const Ta rx = target[0] - center[0];
    const Ta ry = target[1] - center[1];
    const Ta rz = target[2] - center[2];

    // Laplace monopole, dipole and quadrupole in the own frame (acceleration convention, as M2P)
    const auto L   = ownFrameMDQ<Exact>(rx, ry, rz, s, M[Hqi::mass], &M[Hqi::px], &M[Hqi::sxx]);
    const Ta zp    = L.zp, r2 = L.r2, rinv2 = L.rinv2, rinv3 = L.rinv3, rinv5 = L.rinv5;
    const Ta az    = L.az;
    // tree frame: E~_perp = s E_L,perp, E~_z = E_L,z
    const Ta sax = s * L.ax, say = s * L.ay;

    // rapidity terms on the unit monopole field E0 = s R / r'^3 (field convention)
    const Ta beta = M[Hqi::beta];
    const Ta omb2 = M[Hqi::absW];
    const Ta V1   = M[Hqi::v1];
    const Ta V2   = M[Hqi::v2];
    const Ta Xx = M[Hqi::xx], Xy = M[Hqi::xy], Xz = M[Hqi::xz];

    const Ta e0  = s * rinv3;
    const Ta E0x = e0 * rx, E0y = e0 * ry, E0z = e0 * rz;
    const Ta c2  = zp * zp * rinv2;
    const Ta f   = Ta(1) - Ta(3) * c2;
    const Ta A2  = omb2 * f + beta * beta * f * f - Ta(6) * beta * beta * c2 * (Ta(1) - c2);

    // X.grad of E0 and of f; sum_k Lambda_k^2 R_k X_k with Lambda = (1, 1, s)
    const Ta XLR     = Xx * rx + Xy * ry + s * s * Xz * rz;
    const Ta gE      = Ta(3) * XLR * rinv5 * s;
    const Ta XgE0x   = s * Xx * rinv3 - gE * rx;
    const Ta XgE0y   = s * Xy * rinv3 - gE * ry;
    const Ta XgE0z   = s * Xz * rinv3 - gE * rz;
    const Ta Xgradf  = Ta(-6) * s * s * rz * (Xz * r2 - rz * XLR) * rinv2 * rinv2;

    const Ta h1E  = beta * f;
    const Ta mono = V1 * h1E + Ta(0.5) * V2 * A2 - beta * Xgradf;
    const Ta fEx  = mono * E0x - h1E * XgE0x;
    const Ta fEy  = mono * E0y - h1E * XgE0y;
    const Ta fEz  = mono * E0z - h1E * XgE0z;

    const Ta chi   = M[Hqi::chi];
    const Ta h1C   = M[Hqi::dchi] + chi * beta * f;
    const Ta h2C   = M[Hqi::ddchi] + Ta(2) * M[Hqi::dchi] * beta * f + chi * A2;
    const Ta monoC = V1 * h1C + Ta(0.5) * V2 * h2C - chi * beta * Xgradf;
    const Ta fCx   = monoC * E0x - h1C * XgE0x;
    const Ta fCy   = monoC * E0y - h1C * XgE0y;

    acc[0] += sax - fEx;
    acc[1] += say - fEy;
    acc[2] += az - fEz;
    acc[3] += chi * sax - fCx;
    acc[4] += chi * say - fCy;
    return acc;
}

} // namespace ryoanji
