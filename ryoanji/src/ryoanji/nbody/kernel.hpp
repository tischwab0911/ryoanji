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

#include "types.h"

namespace ryoanji
{

HOST_DEVICE_FUN HOST_DEVICE_INLINE float inverseSquareRoot(float x)
{
#ifdef __CUDA_ARCH__
    // inline PTX using rsqrt.approx.ftz.f32 which flushes to zero and is much faster than rsqrtf when not using
    // fast-math
    float y;
    asm("rsqrt.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
#elif defined(__HIP_DEVICE_COMPILE__)
    return rsqrtf(x);
#else
    return 1.0f / std::sqrt(x);
#endif
}

HOST_DEVICE_FUN HOST_DEVICE_INLINE double inverseSquareRoot(double x)
{
    // perform a single-precision reciprocal square root, accumulated error should still be orders of magnitude below
    // Barnes-Hut approximation error
#ifdef __CUDA_ARCH__
    // inline PTX using rsqrt.approx.ftz.f32 which flushes to zero and is much faster than rsqrtf when not using
    // fast-math
    float xf = float(x), yf;
    asm("rsqrt.approx.ftz.f32 %0, %1;" : "=f"(yf) : "f"(xf));
    return double(yf);
#elif defined(__HIP_DEVICE_COMPILE__)
    return double(rsqrtf(float(x)));
#else
    return double(1.0f / std::sqrt(float(x)));
#endif
}

/*! @brief 1 / sqrt(x) to the precision of the type, for direct sums that serve as accuracy references
 *
 * inverseSquareRoot evaluates in single precision also for double arguments (fast, and far below the error of the
 * multipole approximation); a direct sum is the reference that tree errors are measured against, so it keeps the
 * full precision of its type. (IPPL/OPAL-X fork addition.)
 */
HOST_DEVICE_FUN HOST_DEVICE_INLINE float inverseSquareRootExact(float x)
{
#if defined(__HIP_DEVICE_COMPILE__) || defined(__CUDA_ARCH__)
    return rsqrtf(x);
#else
    return 1.0f / std::sqrt(x);
#endif
}

HOST_DEVICE_FUN HOST_DEVICE_INLINE double inverseSquareRootExact(double x)
{
#if defined(__HIP_DEVICE_COMPILE__) || defined(__CUDA_ARCH__)
    return rsqrt(x);
#else
    return 1.0 / std::sqrt(x);
#endif
}

//! @brief inverseSquareRoot, or inverseSquareRootExact for direct sums (@p Exact)
template<bool Exact, class T>
HOST_DEVICE_FUN HOST_DEVICE_INLINE T inverseSquareRootOf(T x)
{
    if constexpr (Exact) { return inverseSquareRootExact(x); }
    else { return inverseSquareRoot(x); }
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
 *
 * @tparam Exact  full-precision 1 / sqrt (direct sums), else the fast single-precision one (tree traversal)
 */
template<bool Exact = false, class Ta, class Tc, class Th, class Tm>
HOST_DEVICE_FUN DEVICE_INLINE Vec4<Ta> P2P(Vec4<Ta> acc, const Vec3<Tc>& pos_i, const Vec3<Tc>& pos_j, Tm m_j, Th h_i,
                                           Th h_j)
{
    Vec3<Tc> dX = pos_j - pos_i;
    Tc       R2 = norm2(dX);

    Th h_ij  = h_i + h_j;
    Th h_ij2 = h_ij * h_ij;
    Tc R2eff = (R2 < h_ij2) ? h_ij2 : R2;

    Tc invR   = inverseSquareRootOf<Exact>(R2eff);
    Tc invR2  = invR * invR;
    Tc invR3m = m_j * invR * invR2;

    acc[0] -= invR3m * R2;
    acc[1] += dX[0] * invR3m;
    acc[2] += dX[1] * invR3m;
    acc[3] += dX[2] * invR3m;

    return acc;
}

} // namespace ryoanji
