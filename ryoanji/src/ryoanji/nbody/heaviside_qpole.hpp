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
    const Tf ch   = std::cosh(wc);
    const Tf s    = ch + frame.betaRef * std::sinh(wc);
    const Tf beta = std::tanh(frame.wRef + wc);
    const Tf omb2 = Tf(1) / (frame.gammaRef * frame.gammaRef * s * s); // 1 - beta^2 = 1 / gamma^2
    Tf       chi, dchi, ddchi;
    if (frame.netForceChannel) // chi = 1 - betaRef beta
    {
        chi   = ch / (frame.gammaRef * frame.gammaRef * s);
        dchi  = -frame.betaRef * omb2;
        ddchi = 2 * frame.betaRef * beta * omb2;
    }
    else // chi = beta
    {
        chi   = beta;
        dchi  = omb2;
        ddchi = -2 * beta * omb2;
    }

    const Tf m = M[Hqi::mass];
    const Tf px = M[Hqi::px], py = M[Hqi::py], pz = M[Hqi::pz];
    const Tf v1 = M[Hqi::v1];
    // moments re-centred at w_c
    const Tf V1 = v1 - wc * m;
    const Tf Xx = Tf(M[Hqi::xx]) - wc * px, Xy = Tf(M[Hqi::xy]) - wc * py, Xz = Tf(M[Hqi::xz]) - wc * pz;
    const Tf V2 = Tf(M[Hqi::v2]) - 2 * wc * v1 + wc * wc * m;
    // own-frame (z scaled by s) second moments, traceless form 3 S' - tr S'
    const Tf sxx = M[Hqi::sxx], sxy = M[Hqi::sxy], sxz = s * Tf(M[Hqi::sxz]);
    const Tf syy = M[Hqi::syy], syz = s * Tf(M[Hqi::syz]), szz = s * s * Tf(M[Hqi::szz]);
    const Tf tr  = sxx + syy + szz;

    M[Hqi::absMass] = wc;
    M[Hqi::pz]      = s * pz;
    M[Hqi::sxx]     = 3 * sxx - tr;
    M[Hqi::sxy]     = 3 * sxy;
    M[Hqi::sxz]     = 3 * sxz;
    M[Hqi::syy]     = 3 * syy - tr;
    M[Hqi::syz]     = 3 * syz;
    M[Hqi::szz]     = 3 * szz - tr;
    M[Hqi::absW]    = omb2;
    M[Hqi::v1]      = velocityMoments ? V1 : Tf(0);
    M[Hqi::xx]      = velocityMoments ? Xx : Tf(0);
    M[Hqi::xy]      = velocityMoments ? Xy : Tf(0);
    M[Hqi::xz]      = velocityMoments ? Xz : Tf(0);
    M[Hqi::v2]      = velocityMoments ? V2 : Tf(0);
    M[Hqi::s]       = s;
    M[Hqi::beta]    = beta;
    M[Hqi::chi]     = chi;
    M[Hqi::dchi]    = dchi;
    M[Hqi::ddchi]   = ddchi;
}

/*! @brief factor for the squared MAC radius of a cell in evaluation form, or a negative value if the
 *         cell must always be opened
 *
 * The spatial expansion converges in the metric of each source's own rest frame. Scaling z by
 * s = gamma / gammaRef changes lengths by at most max(s, 1/s), so a tree-frame radius times
 * kappa = max(s_max, 1/s_min) over the cell's rapidity range is conservative. The rapidity
 * expansion converges for |dw| < pi/2 (kernel and weights are analytic in |Im w| < pi/2 at any
 * energy); cells with max |dw| / (pi/2) > thetaV are opened.
 */
template<class T, class Tf>
HOST_DEVICE_FUN Tf heavisideMacScale2(const HeavisideQpole<T>& M, const HeavisideFrame<Tf>& frame, Tf thetaV,
                                      bool velocityMoments)
{
    if (M[Hqi::wMax] <= T(kNoRapidity)) { return Tf(1); }
    const Tf wmax = M[Hqi::wMax], wmin = -Tf(M[Hqi::wNegMin]), wc = M[Hqi::absMass];
    const Tf halfRange = (wmax - wc > wc - wmin) ? wmax - wc : wc - wmin;
    if (velocityMoments && halfRange > thetaV * Tf(1.5707963267948966)) { return Tf(-1); }

    auto stretch = [&frame](Tf wHat) { return std::cosh(wHat) + frame.betaRef * std::sinh(wHat); };
    const Tf sa = stretch(wmin), sb = stretch(wmax);
    const Tf sMax = sa > sb ? sa : sb;
    // s is smallest where the absolute rapidity w = wRef + w^ is closest to 0
    const bool atRest = frame.wRef + wmin <= Tf(0) && frame.wRef + wmax >= Tf(0);
    const Tf   sMin   = atRest ? Tf(1) / frame.gammaRef : (sa < sb ? sa : sb);
    const Tf   kappa  = (sMax > Tf(1) / sMin) ? sMax : Tf(1) / sMin;
    return kappa > Tf(1) ? kappa * kappa : Tf(1);
}

/*! @brief apply a cell multipole in evaluation form to a target, see P2PHeaviside for the outputs
 *
 * @return @p acc + contribution, (E_x, E_y, E_z, C_x, C_y) in the sign convention of P2P (-field)
 */
template<class Ta, class Tc, class Tmp>
HOST_DEVICE_FUN DEVICE_INLINE util::array<Ta, 5> M2PHeaviside(util::array<Ta, 5> acc, const Vec3<Tc>& target,
                                                              const Vec3<Tc>& center,
                                                              const HeavisideQpole<Tmp>& M)
{
    const Ta s  = M[Hqi::s];
    const Ta rx = target[0] - center[0];
    const Ta ry = target[1] - center[1];
    const Ta rz = target[2] - center[2];
    const Ta zp = s * rz; // own-frame z

    const Ta r2    = rx * rx + ry * ry + zp * zp;
    const Ta rinv  = inverseSquareRoot(r2);
    const Ta rinv2 = rinv * rinv;
    const Ta rinv3 = rinv * rinv2;
    const Ta rinv5 = rinv3 * rinv2;

    // Laplace monopole, dipole and quadrupole in the own frame (acceleration convention, as M2P)
    const Ta mass = M[Hqi::mass];
    const Ta Qrx  = rx * M[Hqi::sxx] + ry * M[Hqi::sxy] + zp * M[Hqi::sxz];
    const Ta Qry  = rx * M[Hqi::sxy] + ry * M[Hqi::syy] + zp * M[Hqi::syz];
    const Ta Qrz  = rx * M[Hqi::sxz] + ry * M[Hqi::syz] + zp * M[Hqi::szz];
    const Ta rQr  = rx * Qrx + ry * Qry + zp * Qrz;
    const Ta rQrAndMonopole = (Ta(-2.5) * rQr * rinv5 - mass * rinv) * rinv2;
    const Ta PR   = rx * M[Hqi::px] + ry * M[Hqi::py] + zp * M[Hqi::pz];
    const Ta ax   = rinv5 * Qrx + rQrAndMonopole * rx + rinv5 * (M[Hqi::px] * r2 - Ta(3) * PR * rx);
    const Ta ay   = rinv5 * Qry + rQrAndMonopole * ry + rinv5 * (M[Hqi::py] * r2 - Ta(3) * PR * ry);
    const Ta az   = rinv5 * Qrz + rQrAndMonopole * zp + rinv5 * (M[Hqi::pz] * r2 - Ta(3) * PR * zp);
    // tree frame: E~_perp = s E_L,perp, E~_z = E_L,z
    const Ta sax = s * ax, say = s * ay;

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
