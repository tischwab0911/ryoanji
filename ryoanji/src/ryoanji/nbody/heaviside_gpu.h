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
 * @brief GPU upsweep and traversal for sources with per-particle velocities (HeavisideQpole)
 */

#pragma once

#include "cstone/traversal/groups.hpp"
#include "cstone/util/array.hpp"
#include "heaviside_qpole.hpp"

namespace ryoanji
{

//! @brief raw leaf multipoles from the sources (x, y, z, m, w^) of each leaf
template<class Tc, class Tm, class Tf, class T>
extern void computeLeafMultipolesHeaviside(const Tc* x, const Tc* y, const Tc* z, const Tm* m, const Tm* w,
                                           const TreeNodeIndex* leafToInternal, TreeNodeIndex numLeaves,
                                           const LocalIndex* layout, const Vec4<Tf>* centers,
                                           HeavisideQpole<T>* multipoles);

//! @brief raw multipoles of the internal cells [firstCell, lastCell) from their children
template<class Tf, class T>
extern void upsweepMultipolesHeaviside(TreeNodeIndex firstCell, TreeNodeIndex lastCell,
                                       const TreeNodeIndex* childOffsets, const Vec4<Tf>* centers,
                                       HeavisideQpole<T>* multipoles);

/*! @brief raw -> evaluation form of all @p numNodes multipoles, and the MAC spheres of the traversal
 *
 * @p macOut[i] = centers[i] with the squared MAC radius scaled by heavisideMacScale2 (or the largest
 * value, meaning always open); empty cells keep radius 0.
 */
template<class Tf, class T>
extern void finalizeHeaviside(TreeNodeIndex numNodes, const Vec4<Tf>* centers, HeavisideQpole<T>* multipoles,
                              HeavisideFrame<Tf> frame, Tf thetaV, bool velocityMoments, Vec4<Tf>* macOut);

/*! @brief Barnes-Hut traversal with the Heaviside kernel, sources = targets in SFC order
 *
 * Outputs (ex, ey, ez, cx, cy) are added to with factor G, in the convention of P2PHeaviside.
 */
template<class Tc, class Th, class Tm, class Ta, class Tf, class T>
extern void traverseHeaviside(cstone::GroupView grp, int initNodeIdx, const Tc* x, const Tc* y, const Tc* z,
                              const Tm* m, const Th* h, const Tm* w, HeavisideFrame<Tf> frame,
                              const TreeNodeIndex* childOffsets, const TreeNodeIndex* internalToLeaf,
                              const LocalIndex* layout, const Vec4<Tf>* sourceCenter,
                              const HeavisideQpole<T>* multipoles, Tc G, Ta* ex, Ta* ey, Ta* ez, Ta* cx, Ta* cy,
                              int* gmPool);

/*! @brief statistics of the last traverseHeaviside: sumP2P, maxP2P, sumM2P, maxM2P, maxStack and the
 *         number of remote leaves applied as multipoles although they failed the MAC (incomplete LET)
 */
util::array<uint64_t, 6> readHeavisideStats();

//! @brief target groups that overflowed the cell queue and were split and traversed again in the last traverseHeaviside
uint64_t readHeavisideRetries();

} // namespace ryoanji
