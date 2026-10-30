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

#include <stdexcept>
#include <vector>

#include "cstone/traversal/groups.hpp"
#include "cstone/util/array.hpp"
#include "gauss2_qpole.hpp"
#include "heaviside_qpole.hpp"

namespace ryoanji
{

/*! @brief raw leaf multipoles from the sources (x, y, z, m, w^) of each leaf
 *
 * Each function below exists for both velocity multipoles, HeavisideQpole (rapidity Taylor expansion) and
 * Gauss2Qpole (two-node Gauss interpolation), selected by the type of @p multipoles.
 */
template<class Tc, class Tm, class Tf, class T>
extern void computeLeafMultipolesHeaviside(const Tc* x, const Tc* y, const Tc* z, const Tm* m, const Tm* w,
                                           const TreeNodeIndex* leafToInternal, TreeNodeIndex numLeaves,
                                           const LocalIndex* layout, const Vec4<Tf>* centers,
                                           HeavisideQpole<T>* multipoles);
template<class Tc, class Tm, class Tf, class T>
extern void computeLeafMultipolesHeaviside(const Tc* x, const Tc* y, const Tc* z, const Tm* m, const Tm* w,
                                           const TreeNodeIndex* leafToInternal, TreeNodeIndex numLeaves,
                                           const LocalIndex* layout, const Vec4<Tf>* centers,
                                           Gauss2Qpole<T>* multipoles);

//! @brief raw multipoles of the internal cells [firstCell, lastCell) from their children
template<class Tf, class T>
extern void upsweepMultipolesHeaviside(TreeNodeIndex firstCell, TreeNodeIndex lastCell,
                                       const TreeNodeIndex* childOffsets, const Vec4<Tf>* centers,
                                       HeavisideQpole<T>* multipoles);
template<class Tf, class T>
extern void upsweepMultipolesHeaviside(TreeNodeIndex firstCell, TreeNodeIndex lastCell,
                                       const TreeNodeIndex* childOffsets, const Vec4<Tf>* centers,
                                       Gauss2Qpole<T>* multipoles);

/*! @brief raw -> evaluation form of all @p numNodes multipoles
 *
 * The MAC radii are not touched: cstone scales them with the velocity criterion (setVelocityMac) before it
 * builds the locally essential tree, and the traversal reads cstone's expansion centers directly.
 */
template<class Tf, class T>
extern void finalizeHeaviside(TreeNodeIndex numNodes, HeavisideQpole<T>* multipoles, HeavisideFrame<Tf> frame,
                              bool velocityMoments);
template<class Tf, class T>
extern void finalizeHeaviside(TreeNodeIndex numNodes, Gauss2Qpole<T>* multipoles, HeavisideFrame<Tf> frame,
                              bool velocityMoments);

/*! @brief Barnes-Hut traversal with the Heaviside kernel, sources = targets in SFC order
 *
 * Outputs (ex, ey, ez, cx, cy) are added to with factor G, in the convention of P2PHeaviside. A target
 * group whose traversal overflows the cell queue, or meets a remote leaf (no particles in the locally
 * essential tree) that fails its MAC, writes nothing and is split and traversed again. A single target
 * that meets such a leaf throws: the locally essential tree was built with other radii.
 */
template<class Tc, class Th, class Tm, class Ta, class Tf, class MType>
extern void traverseHeaviside(cstone::GroupView grp, int initNodeIdx, const Tc* x, const Tc* y, const Tc* z,
                              const Tm* m, const Th* h, const Tm* w, HeavisideFrame<Tf> frame,
                              const TreeNodeIndex* childOffsets, const TreeNodeIndex* internalToLeaf,
                              const LocalIndex* layout, const Vec4<Tf>* sourceCenter, const MType* multipoles, Tc G,
                              Ta* ex, Ta* ey, Ta* ez, Ta* cx, Ta* cy, int* gmPool);

/*! @brief statistics of the last traverseHeaviside: sumP2P, maxP2P, sumM2P, maxM2P, maxStack and the
 *         number of group traversals aborted (and split) because a remote leaf failed the MAC
 */
util::array<uint64_t, 6> readHeavisideStats();

//! @brief target groups that were split and traversed again in the last traverseHeaviside (overflow or remote leaf)
uint64_t readHeavisideRetries();

//! @brief thrown by traverseHeaviside when a single target meets a remote leaf that fails its MAC
struct HeavisideLetIncomplete : public std::runtime_error
{
    using std::runtime_error::runtime_error;
};

//! @brief (target particle index, failing cell index) of the unresolved single targets of the last throw (at most 8)
std::vector<util::array<unsigned, 2>> readHeavisideUnresolved();

} // namespace ryoanji
