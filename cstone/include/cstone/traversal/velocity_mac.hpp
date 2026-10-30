/*
 * Cornerstone octree
 *
 * Copyright (c) 2026 CSCS, ETH Zurich
 *               2026 PSI, Villigen
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: MIT License
 */

/*! @file
 * @brief Velocity-aware multipole acceptance criterion for sources with per-particle velocities
 *
 * Sources that move along z with their own longitudinal rapidity w are placed in one tree frame
 * (Lorentz factor gammaRef, z stretched by gammaRef) and carry w^ = w - wRef. A cell's multipole
 * converges in the rest-frame metric of each of its sources, and its rapidity expansion converges
 * for |w - w_c| < pi/2. Both conditions depend only on the cell's rapidity range and |q|-weighted
 * mean rapidity, so they are folded into the squared MAC radius of the cell: multiplied by kappa^2,
 * or set to "always open". Applying the scale here, before markMacs and halo discovery, makes the
 * locally essential tree and the traversal use the same radius for every cell.
 */

#pragma once

#include <cmath>

#include "cstone/cuda/annotation.hpp"
#include "cstone/tree/definitions.h"
#include "cstone/util/array.hpp"

namespace cstone
{

//! @brief tree frame and velocity opening parameter of the velocity-aware MAC
template<class T>
struct VelocityMac
{
    T    gammaRef = 1;
    T    betaRef  = 0;
    T    wRef     = 0;
    //! rapidity criterion: cells with max |w - w_c| > thetaV * pi / 2 are always opened
    T    thetaV = 0.25;
    //! false: kappa only, no rapidity criterion (tests of the expansion without rapidity moments)
    bool rapidityCriterion = true;
    //! false: the geometric MAC is used unchanged
    bool active = false;
};

/*! @brief per-cell rapidity summary {sum |m| w^, sum |m|, max w^, -min w^}
 *
 * Sums and maxima, so a parent's summary follows exactly from its children's.
 */
template<class T>
using VelocitySummary = util::array<T, 4>;

//! @brief range entries of an empty cell
constexpr float kNoVelocity = -1.0e30f;

template<class T>
HOST_DEVICE_FUN constexpr VelocitySummary<T> emptyVelocitySummary()
{
    return {T(0), T(0), T(kNoVelocity), T(kNoVelocity)};
}

template<class T>
HOST_DEVICE_FUN void addVelocitySource(VelocitySummary<T>& s, T m, T wHat)
{
    T aq = std::abs(m);
    s[0] += aq * wHat;
    s[1] += aq;
    s[2] = wHat > s[2] ? wHat : s[2];
    s[3] = -wHat > s[3] ? -wHat : s[3];
}

template<class T>
HOST_DEVICE_FUN VelocitySummary<T> combineVelocitySummary(const VelocitySummary<T>& a, const VelocitySummary<T>& b)
{
    return {a[0] + b[0], a[1] + b[1], a[2] > b[2] ? a[2] : b[2], a[3] > b[3] ? a[3] : b[3]};
}

//! @brief combine the summaries of 8 children for use in tree upsweeps
template<class T>
struct CombineVelocitySummary
{
    HOST_DEVICE_FUN
    VelocitySummary<T> operator()(TreeNodeIndex /*nodeIdx*/, TreeNodeIndex child, const VelocitySummary<T>* s)
    {
        VelocitySummary<T> r = emptyVelocitySummary<T>();
        for (TreeNodeIndex i = child; i < child + 8; ++i)
        {
            r = combineVelocitySummary(r, s[i]);
        }
        return r;
    }
};

/*! @brief squared MAC radius meaning "always open" (larger than any squared distance)
 *
 * Positive: markMacs and the traversals compare against |macSq|, a cell with radius 0 is empty.
 */
template<class T>
HOST_DEVICE_FUN constexpr T alwaysOpenMac()
{
    if constexpr (sizeof(T) == sizeof(float)) { return T(3.0e38f); }
    else { return T(1.0e300); }
}

/*! @brief factor for the squared MAC radius of a cell with rapidity range [wMin, wMax] relative to the
 *         frame and |q|-weighted mean wc, or a negative value if the cell must always be opened
 *
 * The spatial expansion converges in the metric of each source's own rest frame. Scaling z by
 * s = gamma / gammaRef changes lengths by at most max(s, 1/s), so a tree-frame radius times
 * kappa = max(s_max, 1/s_min) over the cell's rapidity range is conservative. The rapidity
 * expansion converges for |dw| < pi/2 (kernel and weights are analytic in |Im w| < pi/2 at any
 * energy); cells with max |dw| / (pi/2) > thetaV are opened.
 */
template<class Tf>
HOST_DEVICE_FUN Tf velocityMacScale2(Tf wMin, Tf wMax, Tf wc, const VelocityMac<Tf>& p)
{
    const Tf halfRange = (wMax - wc > wc - wMin) ? wMax - wc : wc - wMin;
    if (p.rapidityCriterion && halfRange > p.thetaV * Tf(1.5707963267948966)) { return Tf(-1); }

    auto stretch = [&p](Tf wHat) { return std::cosh(wHat) + p.betaRef * std::sinh(wHat); };
    const Tf sa = stretch(wMin), sb = stretch(wMax);
    const Tf sMax = sa > sb ? sa : sb;
    // s is smallest where the absolute rapidity w = wRef + w^ is closest to 0
    const bool atRest = p.wRef + wMin <= Tf(0) && p.wRef + wMax >= Tf(0);
    const Tf   sMin   = atRest ? Tf(1) / p.gammaRef : (sa < sb ? sa : sb);
    const Tf   kappa  = (sMax > Tf(1) / sMin) ? sMax : Tf(1) / sMin;
    return kappa > Tf(1) ? kappa * kappa : Tf(1);
}

//! @brief velocityMacScale2 of a cell summary; 1 for an empty cell
template<class Tf>
HOST_DEVICE_FUN Tf velocityMacScale2(const VelocitySummary<Tf>& s, const VelocityMac<Tf>& p)
{
    if (s[2] <= Tf(kNoVelocity) || !(s[1] > Tf(0))) { return Tf(1); }
    return velocityMacScale2(-s[3], s[2], s[0] / s[1], p);
}

//! @brief scale a squared MAC radius (0 for empty cells) with the velocity criterion of its cell
template<class Tf>
HOST_DEVICE_FUN Tf scaleVelocityMac(Tf macSq, const VelocitySummary<Tf>& s, const VelocityMac<Tf>& p)
{
    if (macSq == Tf(0)) { return macSq; }
    Tf scale = velocityMacScale2(s, p);
    return scale < Tf(0) ? alwaysOpenMac<Tf>() : macSq * scale;
}

} // namespace cstone
