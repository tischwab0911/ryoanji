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
 * @brief Two-node Gauss multipoles of sources moving with different velocities (Heaviside kernel)
 *
 * Alternative to the rapidity Taylor expansion of heaviside_qpole.hpp. The rapidity dependence of every
 * source's field is interpolated linearly between two nodes w_1 < w_2, the 2-point Gauss rule of the
 * cell's |q|-weighted rapidity distribution (quadrature method of moments):
 *
 *   sum_j q_j K(R - d_j, w_j) ~ sum_nu sum_j q_j L_nu(w_j) K(R - d_j, w_nu),  L_1 = (w - w_2) / (w_1 - w_2)
 *
 * Each node is a source population with one rapidity, i.e. a Laplace multipole in its own rest frame
 * (z scaled by s_nu): monopole, dipole and quadrupole about the node centre c_nu = c + <d> + b t_nu, the
 * position that the |q|-weighted linear regression of the source offsets d on the rapidity t predicts at
 * the node (b = sum |q| d t / sum |q| t^2). For a cell of two populations at two rapidities these are the
 * populations' centroids. For a cell of one charge sign the monopole is exact for kernels cubic in w; the
 * spatial terms are exact to first order in the rapidity offset. The nodes lie inside the cell's rapidity
 * range, so cstone's velocity MAC (kappa from the range, rapidity criterion) applies to every node.
 *
 * Two forms share one array:
 *  - raw (upsweep): moments about the expansion centre c in space and about the cell's own |q|-weighted
 *    mean rapidity m = absW / absMass in rapidity (central moments: stable in single precision);
 *    translated exactly by gauss2M2M,
 *  - evaluation (after gauss2Finalize): per node the monopole, own-frame dipole and traceless
 *    quadrupole about the node centre, stretch, second-channel weight and node centre - c, applied by
 *    M2PGauss2.
 */

#pragma once

#include <cmath>

#include "heaviside_qpole.hpp"

namespace ryoanji
{

template<class T>
using Gauss2Qpole = util::array<T, 32>;

//! @brief Gauss2Qpole index names of the raw form (d: offset from the centre c, t = w^ - m)
struct G2i
{
    enum IndexNames
    {
        mass    = 0,  //!< sum q
        absMass = 1,  //!< sum |q|
        absW    = 2,  //!< sum |q| w^ (mean m = absW / absMass)
        px      = 3,  //!< P = sum q d
        py      = 4,
        pz      = 5,
        sxx     = 6,  //!< S = sum q d d^T
        sxy     = 7,
        sxz     = 8,
        syy     = 9,
        syz     = 10,
        szz     = 11,
        v1      = 12, //!< sum q t
        xx      = 13, //!< X = sum q d t
        xy      = 14,
        xz      = 15,
        yxx     = 16, //!< Y = sum q d d^T t
        yxy     = 17,
        yxz     = 18,
        yyy     = 19,
        yyz     = 20,
        yzz     = 21,
        a2      = 22, //!< sum |q| t^2
        a3      = 23, //!< sum |q| t^3
        zx      = 24, //!< Z = sum |q| d t (regression of the position on the rapidity)
        zy      = 25,
        zz      = 26,
        bx      = 27, //!< B = sum |q| d (zero if c is the |q|-weighted centre)
        by      = 28,
        bz      = 29,
        numRaw  = 30,
    };
};

//! @brief offsets within node nu of the evaluation form, which starts at nu * G2n::size
struct G2n
{
    enum IndexNames
    {
        mass = 0,  //!< node monopole W_nu
        px   = 1,  //!< own-frame dipole about the node centre (px, py, s pz)
        qxx  = 4,  //!< own-frame traceless quadrupole 3 S' - tr S': xx, xy, xz, yy, yz, zz
        s    = 10, //!< stretch s_nu = gamma(w_nu) / gammaRef
        chi  = 11, //!< second-channel weight at w_nu
        ex   = 12, //!< node centre - c (tree frame)
        size = 15,
    };
};

template<class T>
HOST_DEVICE_FUN Gauss2Qpole<T> gauss2Zero()
{
    Gauss2Qpole<T> M;
    M = T(0);
    return M;
}

//! @brief |q|-weighted mean rapidity about which the raw rapidity moments are taken
template<class T>
HOST_DEVICE_FUN T gauss2Mean(T absMass, T absW)
{
    return absMass > T(0) ? absW / absMass : T(0);
}

template<class T>
HOST_DEVICE_FUN T gauss2Mean(const Gauss2Qpole<T>& M)
{
    return gauss2Mean(M[G2i::absMass], M[G2i::absW]);
}

//! @brief first P2M pass over sources [begin, end) (stride): sum |q| and sum |q| w^
template<int stride, class T2, class T3>
HOST_DEVICE_FUN void P2MGauss2_sums(const T2* m, const T2* w, LocalIndex begin, LocalIndex end, T3& absMass, T3& absW)
{
    for (LocalIndex i = begin; i < end; i += stride)
    {
        T3 aq = m[i] < T2(0) ? T3(-m[i]) : T3(m[i]);
        absMass += aq;
        absW += aq * T3(w[i]);
    }
}

/*! @brief second P2M pass: add the moments of sources [begin, end) (stride) about @p center and the mean
 *         rapidity @p wMean (absMass and absW are not touched)
 */
template<int stride, class T1, class T2, class T3, class Tc>
HOST_DEVICE_FUN void P2MGauss2_add(const T1* x, const T1* y, const T1* z, const T2* m, const T2* w,
                                   LocalIndex begin, LocalIndex end, const Tc& center, T3 wMean, Gauss2Qpole<T3>& M)
{
    for (LocalIndex i = begin; i < end; i += stride)
    {
        T1 dx  = x[i] - center[0];
        T1 dy  = y[i] - center[1];
        T1 dz  = z[i] - center[2];
        T1 q   = m[i];
        T1 aq  = q < T1(0) ? -q : q;
        T1 t   = T1(T3(w[i]) - wMean);
        T1 qt  = q * t;
        T1 aqt = aq * t;

        M[G2i::mass] += q;
        M[G2i::px] += q * dx;
        M[G2i::py] += q * dy;
        M[G2i::pz] += q * dz;
        M[G2i::sxx] += q * dx * dx;
        M[G2i::sxy] += q * dx * dy;
        M[G2i::sxz] += q * dx * dz;
        M[G2i::syy] += q * dy * dy;
        M[G2i::syz] += q * dy * dz;
        M[G2i::szz] += q * dz * dz;
        M[G2i::v1] += qt;
        M[G2i::xx] += qt * dx;
        M[G2i::xy] += qt * dy;
        M[G2i::xz] += qt * dz;
        M[G2i::yxx] += qt * dx * dx;
        M[G2i::yxy] += qt * dx * dy;
        M[G2i::yxz] += qt * dx * dz;
        M[G2i::yyy] += qt * dy * dy;
        M[G2i::yyz] += qt * dy * dz;
        M[G2i::yzz] += qt * dz * dz;
        M[G2i::a2] += aqt * t;
        M[G2i::a3] += aqt * t * t;
        M[G2i::zx] += aqt * dx;
        M[G2i::zy] += aqt * dy;
        M[G2i::zz] += aqt * dz;
        M[G2i::bx] += aq * dx;
        M[G2i::by] += aq * dy;
        M[G2i::bz] += aq * dz;
    }
}

//! @brief raw multipole of sources [begin, end) about @p center (both P2M passes, host and tests)
template<class T1, class T2, class T3, class Tc>
HOST_DEVICE_FUN Gauss2Qpole<T3> P2MGauss2(const T1* x, const T1* y, const T1* z, const T2* m, const T2* w,
                                          LocalIndex begin, LocalIndex end, const Tc& center)
{
    Gauss2Qpole<T3> M = gauss2Zero<T3>();
    P2MGauss2_sums<1>(m, w, begin, end, M[G2i::absMass], M[G2i::absW]);
    P2MGauss2_add<1>(x, y, z, m, w, begin, end, center, gauss2Mean(M), M);
    return M;
}

/*! @brief add the raw multipole @p child, re-expanded about the composite centre and mean rapidity
 *
 * @p dX = composite centre - child centre (the convention of heavisideM2M), @p wMean = the composite
 * cell's mean rapidity, gauss2Mean of its final absMass and absW. Exact: with e = child centre - composite
 * centre and delta = child mean - wMean, every moment is a polynomial in d + e and t + delta; the
 * |q|-weighted moments use sum |q| t = 0 of the child (parallel combination of central moments).
 */
template<class T, class Tc>
HOST_DEVICE_FUN void gauss2M2M(Gauss2Qpole<T>& M, Vec3<Tc> dX, T wMean, const Gauss2Qpole<T>& child)
{
    const T  ex = -dX[0], ey = -dX[1], ez = -dX[2];
    const T  dl = gauss2Mean(child) - wMean;
    const T  m  = child[G2i::mass], am = child[G2i::absMass];
    const T  px = child[G2i::px], py = child[G2i::py], pz = child[G2i::pz];
    const T  v1 = child[G2i::v1];
    const T  Xx = child[G2i::xx], Xy = child[G2i::xy], Xz = child[G2i::xz];
    const T  Bx = child[G2i::bx], By = child[G2i::by], Bz = child[G2i::bz];
    const T  a2 = child[G2i::a2];

    // spatial moments about the composite centre
    const T sxx = child[G2i::sxx] + 2 * ex * px + m * ex * ex;
    const T sxy = child[G2i::sxy] + ex * py + ey * px + m * ex * ey;
    const T sxz = child[G2i::sxz] + ex * pz + ez * px + m * ex * ez;
    const T syy = child[G2i::syy] + 2 * ey * py + m * ey * ey;
    const T syz = child[G2i::syz] + ey * pz + ez * py + m * ey * ez;
    const T szz = child[G2i::szz] + 2 * ez * pz + m * ez * ez;

    M[G2i::mass] += m;
    M[G2i::absMass] += am;
    M[G2i::absW] += child[G2i::absW];
    M[G2i::px] += px + m * ex;
    M[G2i::py] += py + m * ey;
    M[G2i::pz] += pz + m * ez;
    M[G2i::sxx] += sxx;
    M[G2i::sxy] += sxy;
    M[G2i::sxz] += sxz;
    M[G2i::syy] += syy;
    M[G2i::syz] += syz;
    M[G2i::szz] += szz;
    // v1' = v1 + delta m, X' = X + delta P + e v1 + delta e m
    M[G2i::v1] += v1 + dl * m;
    M[G2i::xx] += Xx + dl * px + ex * v1 + dl * ex * m;
    M[G2i::xy] += Xy + dl * py + ey * v1 + dl * ey * m;
    M[G2i::xz] += Xz + dl * pz + ez * v1 + dl * ez * m;
    // Y' = Y + X e^T + e X^T + v1 e e^T + delta S'
    M[G2i::yxx] += child[G2i::yxx] + 2 * ex * Xx + v1 * ex * ex + dl * sxx;
    M[G2i::yxy] += child[G2i::yxy] + ex * Xy + ey * Xx + v1 * ex * ey + dl * sxy;
    M[G2i::yxz] += child[G2i::yxz] + ex * Xz + ez * Xx + v1 * ex * ez + dl * sxz;
    M[G2i::yyy] += child[G2i::yyy] + 2 * ey * Xy + v1 * ey * ey + dl * syy;
    M[G2i::yyz] += child[G2i::yyz] + ey * Xz + ez * Xy + v1 * ey * ez + dl * syz;
    M[G2i::yzz] += child[G2i::yzz] + 2 * ez * Xz + v1 * ez * ez + dl * szz;
    // |q|-weighted: a2' = a2 + delta^2 A, a3' = a3 + 3 delta a2 + delta^3 A, Z' = Z + delta B + delta e A,
    // B' = B + A e
    M[G2i::a2] += a2 + dl * dl * am;
    M[G2i::a3] += child[G2i::a3] + 3 * dl * a2 + dl * dl * dl * am;
    M[G2i::zx] += child[G2i::zx] + dl * Bx + dl * ex * am;
    M[G2i::zy] += child[G2i::zy] + dl * By + dl * ey * am;
    M[G2i::zz] += child[G2i::zz] + dl * Bz + dl * ez * am;
    M[G2i::bx] += Bx + am * ex;
    M[G2i::by] += By + am * ey;
    M[G2i::bz] += Bz + am * ez;
}

//! @brief machine epsilon of the multipole storage type (std::numeric_limits is host-only without
//!        --expt-relaxed-constexpr)
template<class T>
HOST_DEVICE_FUN constexpr T storageEpsilon()
{
    static_assert(sizeof(T) == sizeof(float) || sizeof(T) == sizeof(double));
    if constexpr (sizeof(T) == sizeof(float)) { return T(1.1920928955078125e-7); }
    else { return T(2.220446049250313e-16); }
}

//! @brief the two Gauss nodes t_1 < t_2 relative to the mean, or a single node at the mean
template<class Tf>
struct Gauss2Nodes
{
    Tf   t1 = 0, t2 = 0;
    bool single = true;
};

/*! @brief 2-point Gauss rule of a positive measure with mass @p mu0 and central moments @p a2, @p a3
 *
 * The nodes are the roots of the orthogonal polynomial t^2 - (k3 / s2) t - s2 (s2 = a2 / mu0,
 * k3 = a3 / mu0), t = a -+ sqrt(a^2 + s2) with a = k3 / (2 s2), taken in the form without cancellation
 * (t1 t2 = -s2). They lie inside the support of the measure. A spread s2 <= @p eps (the storage
 * precision) gives a single node: the second-order rapidity term it drops is below that precision.
 */
template<class Tf>
HOST_DEVICE_FUN Gauss2Nodes<Tf> gauss2Nodes(Tf mu0, Tf a2, Tf a3, Tf eps)
{
    Gauss2Nodes<Tf> n;
    if (!(mu0 > Tf(0))) { return n; }
    const Tf s2 = a2 / mu0;
    if (!(s2 > eps)) { return n; }
    const Tf a = a3 / (2 * mu0 * s2);
    const Tf r = std::sqrt(a * a + s2);
    if (a >= Tf(0))
    {
        n.t2 = a + r;
        n.t1 = -s2 / n.t2;
    }
    else
    {
        n.t1 = a - r;
        n.t2 = -s2 / n.t1;
    }
    n.single = false;
    return n;
}

/*! @brief raw -> evaluation form of a cell multipole
 *
 * Splits the moments between the two nodes with the Lagrange weights L_nu, which are linear in t:
 * W_1 = (v1 - t2 mass) / (t1 - t2), P_1 = (X - t2 P) / (t1 - t2), S_1 = (Y - t2 S) / (t1 - t2), and
 * node 2 with t1 and t2 exchanged; moves each node's moments to its centre c + B / A + Z t_nu / a2 and
 * converts them to the node's own frame. Without @p velocityMoments (test only) the whole cell is one
 * node at its mean rapidity, as heavisideFinalize without moments.
 */
template<class T, class Tf>
HOST_DEVICE_FUN void gauss2Finalize(Gauss2Qpole<T>& M, const HeavisideFrame<Tf>& frame, bool velocityMoments)
{
    const Tf absMass = M[G2i::absMass];
    if (!(absMass > Tf(0)))
    {
        M                     = T(0);
        M[G2n::s]             = T(1);
        M[G2n::size + G2n::s] = T(1);
        return;
    }
    const Tf wMean = gauss2Mean(M); // the mean the raw moments are central about (storage precision)
    const Tf eps   = storageEpsilon<T>();
    const Tf a2    = M[G2i::a2];
    const auto nodes = velocityMoments ? gauss2Nodes(absMass, a2, Tf(M[G2i::a3]), eps) : Gauss2Nodes<Tf>{};

    const Tf mass = M[G2i::mass];
    const Tf P[3] = {Tf(M[G2i::px]), Tf(M[G2i::py]), Tf(M[G2i::pz])};
    const Tf S[6] = {Tf(M[G2i::sxx]), Tf(M[G2i::sxy]), Tf(M[G2i::sxz]),
                     Tf(M[G2i::syy]), Tf(M[G2i::syz]), Tf(M[G2i::szz])};

    // node moments about c: [nu][0] monopole, [1..3] dipole, [4..9] second moments (tree frame); node centres
    Tf node[2][10];
    Tf e[2][3]  = {{0, 0, 0}, {0, 0, 0}};
    Tf tNode[2] = {nodes.t1, nodes.t2};
    if (nodes.single)
    {
        node[0][0] = mass;
        for (int k = 0; k < 3; ++k) { node[0][1 + k] = P[k]; }
        for (int k = 0; k < 6; ++k) { node[0][4 + k] = S[k]; }
        for (int k = 0; k < 10; ++k) { node[1][k] = Tf(0); }
    }
    else
    {
        const Tf X[3] = {Tf(M[G2i::xx]), Tf(M[G2i::xy]), Tf(M[G2i::xz])};
        const Tf Y[6] = {Tf(M[G2i::yxx]), Tf(M[G2i::yxy]), Tf(M[G2i::yxz]),
                         Tf(M[G2i::yyy]), Tf(M[G2i::yyz]), Tf(M[G2i::yzz])};
        const Tf Z[3] = {Tf(M[G2i::zx]), Tf(M[G2i::zy]), Tf(M[G2i::zz])};
        const Tf B[3] = {Tf(M[G2i::bx]), Tf(M[G2i::by]), Tf(M[G2i::bz])};
        const Tf v1   = M[G2i::v1];
        for (int nu = 0; nu < 2; ++nu)
        {
            const Tf other = tNode[1 - nu];
            const Tf inv   = Tf(1) / (tNode[nu] - other);
            node[nu][0]    = (v1 - other * mass) * inv;
            for (int k = 0; k < 3; ++k) { node[nu][1 + k] = (X[k] - other * P[k]) * inv; }
            for (int k = 0; k < 6; ++k) { node[nu][4 + k] = (Y[k] - other * S[k]) * inv; }
            // regression of the |q|-weighted position on the rapidity, at the node
            for (int k = 0; k < 3; ++k) { e[nu][k] = B[k] / absMass + Z[k] / a2 * tNode[nu]; }
        }
        // node moments about the node centre: P' = P - W e, S' = S - P e^T - e P^T + W e e^T
        const int pairs[6][2] = {{0, 0}, {0, 1}, {0, 2}, {1, 1}, {1, 2}, {2, 2}};
        for (int nu = 0; nu < 2; ++nu)
        {
            const Tf W     = node[nu][0];
            const Tf Pc[3] = {node[nu][1], node[nu][2], node[nu][3]};
            for (int k = 0; k < 6; ++k)
            {
                const int i = pairs[k][0], j = pairs[k][1];
                node[nu][4 + k] += -Pc[i] * e[nu][j] - e[nu][i] * Pc[j] + W * e[nu][i] * e[nu][j];
            }
            for (int k = 0; k < 3; ++k) { node[nu][1 + k] -= W * e[nu][k]; }
        }
    }

    for (int nu = 0; nu < 2; ++nu)
    {
        const int  base = nu * G2n::size;
        const auto k    = ownFrameKinematics(frame, wMean + tNode[nu]);
        const auto Q    = ownFrameQuadrupole(k.s, util::array<Tf, 6>{node[nu][4], node[nu][5], node[nu][6],
                                                                     node[nu][7], node[nu][8], node[nu][9]});
        M[base + G2n::mass]   = node[nu][0];
        M[base + G2n::px]     = node[nu][1];
        M[base + G2n::px + 1] = node[nu][2];
        M[base + G2n::px + 2] = k.s * node[nu][3];
        for (int j = 0; j < 6; ++j) { M[base + G2n::qxx + j] = Q[j]; }
        M[base + G2n::s]   = k.s;
        M[base + G2n::chi] = k.chi;
        for (int j = 0; j < 3; ++j) { M[base + G2n::ex + j] = e[nu][j]; }
    }
    M[2 * G2n::size]     = T(0);
    M[2 * G2n::size + 1] = T(0);
}

/*! @brief apply a cell multipole in evaluation form to a target, see P2PHeaviside for the outputs
 *
 * @return @p acc + contribution, (E_x, E_y, E_z, C_x, C_y) in the sign convention of P2P (-field)
 * @tparam Exact  full-precision 1 / sqrt, as ownFrameMDQ
 */
template<bool Exact = false, class Ta, class Tc, class Tmp>
HOST_DEVICE_FUN DEVICE_INLINE util::array<Ta, 5> M2PGauss2(util::array<Ta, 5> acc, const Vec3<Tc>& target,
                                                           const Vec3<Tc>& center, const Gauss2Qpole<Tmp>& M)
{
    for (int nu = 0; nu < 2; ++nu)
    {
        const int base = nu * G2n::size;
        const Ta  rx   = target[0] - center[0] - M[base + G2n::ex];
        const Ta  ry   = target[1] - center[1] - M[base + G2n::ex + 1];
        const Ta  rz   = target[2] - center[2] - M[base + G2n::ex + 2];
        const Ta  s    = M[base + G2n::s];
        const Ta  chi  = M[base + G2n::chi];
        const auto L   = ownFrameMDQ<Exact>(rx, ry, rz, s, M[base + G2n::mass], &M[base + G2n::px], &M[base + G2n::qxx]);
        // tree frame: E~_perp = s E_L,perp, E~_z = E_L,z
        const Ta sax = s * L.ax, say = s * L.ay;
        acc[0] += sax;
        acc[1] += say;
        acc[2] += L.az;
        acc[3] += chi * sax;
        acc[4] += chi * say;
    }
    return acc;
}

} // namespace ryoanji
