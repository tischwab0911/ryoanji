/*
 * Ryoanji N-body solver
 *
 * Copyright (c) 2024 CSCS, ETH Zurich
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: MIT License
 */

#pragma once

/*! @file
 * @brief  Particle-2-particle (P2P) kernel
 *
 * @author Rio Yokota <rioyokota@gsic.titech.ac.jp>
 */

#include <cmath>

#include "types.h"

namespace ryoanji
{

HOST_DEVICE_FUN HOST_DEVICE_INLINE float inverseSquareRoot(float x)
{
#if defined(__HIP_DEVICE_COMPILE__) || defined(__CUDA_ARCH__)
    return rsqrtf(x);
#else
    return 1.0f / std::sqrt(x);
#endif
}

HOST_DEVICE_FUN HOST_DEVICE_INLINE double inverseSquareRoot(double x)
{
#if defined(__HIP_DEVICE_COMPILE__) || defined(__CUDA_ARCH__)
    return rsqrt(x);
#else
    return 1.0 / std::sqrt(x);
#endif
}

/*! @brief interaction between two particles
 *
 * @param acc     acceleration to add to
 * @param pos_i
 * @param pos_j
 * @param m_j
 * @param h_i
 * @param h_j
 * @return        input acceleration plus contribution from this call
 */
template<class Ta, class Tc, class Th, class Tm>
HOST_DEVICE_FUN DEVICE_INLINE Vec4<Ta> P2P(Vec4<Ta> acc, const Vec3<Tc>& pos_i, const Vec3<Tc>& pos_j, Tm m_j, Th h_i,
                                           Th h_j)
{
    Vec3<Tc> dX = pos_j - pos_i;
    Tc       R2 = norm2(dX);

    Th h_ij  = h_i + h_j;
    Th h_ij2 = h_ij * h_ij;
    Tc R2eff = (R2 < h_ij2) ? h_ij2 : R2;

    Tc invR   = inverseSquareRoot(R2eff);
    Tc invR2  = invR * invR;
    Tc invR3m = m_j * invR * invR2;

    acc[0] -= invR3m * R2;
    acc[1] += dX[0] * invR3m;
    acc[2] += dX[1] * invR3m;
    acc[3] += dX[2] * invR3m;

    return acc;
}

/*! @brief Frame of a uniformly moving source population, for the Heaviside kernel
 *
 * Coordinates are lab coordinates at equal lab time with z stretched by gammaRef about some origin
 * (the "tree frame"). A source moving along z with rapidity w = atanh(beta_z) is described by its
 * rapidity relative to the frame, wHat = w - wRef (wRef = acosh(gammaRef) with the sign of the
 * mean motion), which keeps s - 1 and 1 - betaRef * beta accurate in single precision.
 *
 * The second field channel is weighted by beta (netForceChannel = false) or by 1 - betaRef * beta
 * (netForceChannel = true). The latter avoids cancellation in E_perp - c beta_t B for nearly
 * co-moving targets at high gamma.
 */
template<class T>
struct HeavisideFrame
{
    T    gammaRef = 1;
    T    betaRef  = 0;
    T    wRef     = 0;
    bool netForceChannel = false;

    //! @brief relative stretch s = gamma / gammaRef and second-channel weight of a source
    HOST_DEVICE_FUN void sourceFactors(T wHat, T& s, T& chi) const
    {
        T ch = std::cosh(wHat);
        s    = ch + betaRef * std::sinh(wHat);
        chi  = netForceChannel ? ch / (gammaRef * gammaRef * s) : std::tanh(wRef + wHat);
    }
};

/*! @brief interaction of a target with a uniformly moving source (Heaviside field)
 *
 * @param acc     (E_x, E_y, E_z, C_x, C_y) to add to, in the sign convention of P2P (unit charge
 *                at the source attracts), tree-frame components
 * @param pos_i   target position (tree frame)
 * @param pos_j   source position (tree frame)
 * @param m_j     source charge
 * @param s_j     source stretch gamma_j / gammaRef
 * @param chi_j   source weight of the second (transverse) channel C
 * @param h_i     target softening
 * @param h_j     source softening
 *
 * The field of a charge moving uniformly along z is the Coulomb field of its rest frame, boosted:
 * in tree coordinates E = m s R / (rho^2 + s^2 zeta^2)^(3/2); C = chi E_perp. Softening clamps the
 * source's own rest-frame distance rho^2 + s^2 zeta^2 at (h_i + h_j)^2. With s = 1 the field equals
 * that of P2P.
 */
template<class Ta, class Tc, class Th, class Tm>
HOST_DEVICE_FUN DEVICE_INLINE util::array<Ta, 5> P2PHeaviside(util::array<Ta, 5> acc, const Vec3<Tc>& pos_i,
                                                              const Vec3<Tc>& pos_j, Tm m_j, Tm s_j, Tm chi_j,
                                                              Th h_i, Th h_j)
{
    Vec3<Tc> dX  = pos_j - pos_i;
    Tc       dz  = s_j * dX[2];
    Tc       D2  = dX[0] * dX[0] + dX[1] * dX[1] + dz * dz;

    Th h_ij  = h_i + h_j;
    Th h_ij2 = h_ij * h_ij;
    Tc D2eff = (D2 < h_ij2) ? h_ij2 : D2;

    Tc invD   = inverseSquareRoot(D2eff);
    Tc invD3m = m_j * s_j * invD * invD * invD;

    acc[0] += dX[0] * invD3m;
    acc[1] += dX[1] * invD3m;
    acc[2] += dX[2] * invD3m;
    acc[3] += chi_j * dX[0] * invD3m;
    acc[4] += chi_j * dX[1] * invD3m;

    return acc;
}

} // namespace ryoanji
