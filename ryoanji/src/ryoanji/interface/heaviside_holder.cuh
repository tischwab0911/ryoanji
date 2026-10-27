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
 * @brief Barnes-Hut field of sources with per-particle velocities (Heaviside kernel)
 *
 * Same sequence as MultipoleHolder (groups, upsweep, traversal) with HeavisideQpole multipoles: every
 * source carries its rapidity relative to the frame, targets receive E~ and the second channel C
 * (see P2PHeaviside). The MAC spheres of the traversal are cstone's expansion centres with radii
 * scaled by heavisideMacScale2. cstone's locally essential tree is built with the unscaled radii, so
 * on several ranks the traversal can meet remote leaves that fail the scaled MAC; they are applied
 * as multipoles and counted (readStats()[5]).
 */

#pragma once

#include <memory>

#include "cstone/focus/octree_focus_mpi.hpp"
#include "cstone/traversal/groups.hpp"
#include "ryoanji/nbody/heaviside_qpole.hpp"
#include "ryoanji/nbody/types.h"

namespace ryoanji
{

//! @brief Heaviside-kernel parameters of a solve
template<class Tf>
struct HeavisideParams
{
    HeavisideFrame<Tf> frame;
    //! rapidity MAC: cells with max |w - w_c| > thetaV * pi / 2 are opened
    Tf thetaV = 0.25;
    //! false (tests only): all sources of a cell at its mean rapidity, no rapidity moments or MAC
    bool velocityMoments = true;
};

template<class Tc, class Th, class Tm, class Ta, class Tf, class KeyType>
class HeavisideHolder
{
public:
    using MType = HeavisideQpole<Tm>;

    HeavisideHolder();
    ~HeavisideHolder();

    cstone::GroupView computeSpatialGroups(LocalIndex first, LocalIndex last, const Tc* x, const Tc* y, const Tc* z,
                                           const cstone::FocusedOctree<KeyType, Tf, cstone::execution::Gpu>& focusTree,
                                           const cstone::LocalIndex* layout, const cstone::Box<Tc>& box);

    /*! @brief multipoles in evaluation form and the MAC spheres of the next compute()
     *
     * @param[in] x, y, z, m, w   sources in SFC order as referenced by layout, w = rapidity - wRef
     */
    void upsweep(const Tc* x, const Tc* y, const Tc* z, const Tm* m, const Tm* w,
                 cstone::OctreeView<const KeyType> gOctree,
                 const cstone::FocusedOctree<KeyType, Tf, cstone::execution::Gpu>& focusTree,
                 const cstone::LocalIndex* layout, const HeavisideParams<Tf>& params);

    //! @brief add G times (E~, C) of all sources (= targets) to ex, ey, ez, cx, cy
    void compute(cstone::GroupView grp, const Tc* x, const Tc* y, const Tc* z, const Tm* m, const Th* h, const Tm* w,
                 Tc G, Ta* ex, Ta* ey, Ta* ez, Ta* cx, Ta* cy);

    //! @brief sumP2P, maxP2P, sumM2P, maxM2P, maxStack, remote leaves failing the MAC
    util::array<uint64_t, 6> readStats() const;

    //! @brief target groups split and traversed again after overflowing the traversal stack in the last compute
    uint64_t retriedGroups() const;

    const MType* deviceMultipoles() const;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace ryoanji
