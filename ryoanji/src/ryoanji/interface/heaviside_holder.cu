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
 * @brief Barnes-Hut field of sources with per-particle velocities, see heaviside_holder.cuh
 */

#include "cstone/cuda/cuda_utils.cuh"
#include "cstone/traversal/groups_gpu.h"
#include "cstone/util/reallocate.hpp"
#include "ryoanji/nbody/heaviside_gpu.h"
#include "ryoanji/nbody/traversal_gpu.h"
#include "heaviside_holder.cuh"

namespace ryoanji
{
using cstone::GroupView;

template<class Tc, class Th, class Tm, class Ta, class Tf, class KeyType>
class HeavisideHolder<Tc, Th, Tm, Ta, Tf, KeyType>::Impl
{
public:
    GroupView computeSpatialGroups(LocalIndex first, LocalIndex last, const Tc* x, const Tc* y, const Tc* z,
                                   const cstone::FocusedOctree<KeyType, Tf, cstone::execution::Gpu>& focusTree,
                                   const cstone::LocalIndex* layout, const cstone::Box<Tc>& box)
    {
        auto  d_leaves  = focusTree.treeLeavesAcc();
        float tolFactor = 2.0f;
        cstone::computeGroupSplits(cstone::execution::gpuDefaultStream, first, last, x, y, z,
                                   static_cast<const Th*>(nullptr), d_leaves.data(), d_leaves.size() - 1, layout, box,
                                   bhMaxTargetSize(), tolFactor, traversalStack_, groups_.data);

        groups_.firstBody  = first;
        groups_.lastBody   = last;
        groups_.numGroups  = groups_.data.size() - 1;
        groups_.groupStart = rawPtr(groups_.data);
        groups_.groupEnd   = rawPtr(groups_.data) + 1;
        return groups_.view();
    }

    void upsweep(const Tc* x, const Tc* y, const Tc* z, const Tm* m, const Tm* w,
                 cstone::OctreeView<const KeyType> gOctree,
                 const cstone::FocusedOctree<KeyType, Tf, cstone::execution::Gpu>& focusTree,
                 const cstone::LocalIndex* layout, const HeavisideParams<Tf>& params)
    {
        octree_ = focusTree.octreeViewAcc();
        double growthRate = 1.01;
        reallocateDestructive(multipoles_, octree_.numNodes, growthRate);
        reallocateDestructive(macSpheres_, octree_.numNodes, growthRate);

        auto globalCenters = focusTree.globalExpansionCenters();
        layout_            = layout;
        const Vec4<Tf>* centers = focusTree.expansionCentersAcc().data();
        frame_             = params.frame;

        computeLeafMultipolesHeaviside(x, y, z, m, w, octree_.leafToInternal + octree_.numInternalNodes,
                                       octree_.numLeafNodes, layout_, centers, rawPtr(multipoles_));

        auto upsweepGpu = [](auto levelRange, auto childOffsets, auto M, auto centers)
        {
            int numLevels = levelRange.size() - 2;
            for (int level = numLevels - 1; level >= 0; level--)
            {
                int numCellsLevel = levelRange[level + 1] - levelRange[level];
                if (numCellsLevel)
                {
                    upsweepMultipolesHeaviside(levelRange[level], levelRange[level + 1], childOffsets, centers, M);
                }
            }
        };

        upsweepGpu(octree_.levelRangeSpan(), octree_.childOffsets, rawPtr(multipoles_), centers);

        std::span d_multipoleSpan{rawPtr(multipoles_), size_t(octree_.numNodes)};
        focusTree.globalExchange(gOctree, d_multipoleSpan, std::span<MType>{}, traversalStack_, upsweepGpu,
                                 globalCenters.data());
        focusTree.peerExchange(d_multipoleSpan, static_cast<int>(cstone::P2pTags::focusPeerCenters) + 1,
                               traversalStack_);

        //! second upsweep with leaf data from peer and global ranks in place
        upsweepGpu(octree_.levelRangeSpan(), octree_.childOffsets, rawPtr(multipoles_), centers);

        finalizeHeaviside(octree_.numNodes, centers, rawPtr(multipoles_), params.frame, params.thetaV,
                          params.velocityMoments, rawPtr(macSpheres_));
    }

    void compute(GroupView grp, const Tc* x, const Tc* y, const Tc* z, const Tm* m, const Th* h, const Tm* w, Tc G,
                 Ta* ex, Ta* ey, Ta* ez, Ta* cx, Ta* cy)
    {
        reallocate(traversalStack_, stackSize(grp.numGroups), 1.01);
        traverseHeaviside(grp, 1, x, y, z, m, h, w, frame_, octree_.childOffsets, octree_.internalToLeaf, layout_,
                          rawPtr(macSpheres_), rawPtr(multipoles_), G, ex, ey, ez, cx, cy,
                          (int*)rawPtr(traversalStack_));
    }

    util::array<uint64_t, 6> readStats() const { return readHeavisideStats(); }

    uint64_t retriedGroups() const { return readHeavisideRetries(); }

    const MType* deviceMultipoles() const { return rawPtr(multipoles_); }

private:
    cstone::OctreeView<const KeyType> octree_;
    const LocalIndex*                 layout_ = nullptr;
    HeavisideFrame<Tf>                frame_;
    cstone::DeviceVector<MType>       multipoles_;
    cstone::DeviceVector<Vec4<Tf>>    macSpheres_;

    cstone::GroupData<cstone::execution::Gpu> groups_;
    cstone::DeviceVector<LocalIndex>          traversalStack_;
};

template<class Tc, class Th, class Tm, class Ta, class Tf, class KeyType>
HeavisideHolder<Tc, Th, Tm, Ta, Tf, KeyType>::HeavisideHolder()
    : impl_(new Impl())
{
}

template<class Tc, class Th, class Tm, class Ta, class Tf, class KeyType>
HeavisideHolder<Tc, Th, Tm, Ta, Tf, KeyType>::~HeavisideHolder() = default;

template<class Tc, class Th, class Tm, class Ta, class Tf, class KeyType>
GroupView HeavisideHolder<Tc, Th, Tm, Ta, Tf, KeyType>::computeSpatialGroups(
    LocalIndex first, LocalIndex last, const Tc* x, const Tc* y, const Tc* z,
    const cstone::FocusedOctree<KeyType, Tf, cstone::execution::Gpu>& focusTree, const LocalIndex* layout,
    const cstone::Box<Tc>& box)
{
    return impl_->computeSpatialGroups(first, last, x, y, z, focusTree, layout, box);
}

template<class Tc, class Th, class Tm, class Ta, class Tf, class KeyType>
void HeavisideHolder<Tc, Th, Tm, Ta, Tf, KeyType>::upsweep(
    const Tc* x, const Tc* y, const Tc* z, const Tm* m, const Tm* w, cstone::OctreeView<const KeyType> gOctree,
    const cstone::FocusedOctree<KeyType, Tf, cstone::execution::Gpu>& focusTree, const LocalIndex* layout,
    const HeavisideParams<Tf>& params)
{
    impl_->upsweep(x, y, z, m, w, gOctree, focusTree, layout, params);
}

template<class Tc, class Th, class Tm, class Ta, class Tf, class KeyType>
void HeavisideHolder<Tc, Th, Tm, Ta, Tf, KeyType>::compute(GroupView grp, const Tc* x, const Tc* y, const Tc* z,
                                                           const Tm* m, const Th* h, const Tm* w, Tc G, Ta* ex,
                                                           Ta* ey, Ta* ez, Ta* cx, Ta* cy)
{
    impl_->compute(grp, x, y, z, m, h, w, G, ex, ey, ez, cx, cy);
}

template<class Tc, class Th, class Tm, class Ta, class Tf, class KeyType>
util::array<uint64_t, 6> HeavisideHolder<Tc, Th, Tm, Ta, Tf, KeyType>::readStats() const
{
    return impl_->readStats();
}

template<class Tc, class Th, class Tm, class Ta, class Tf, class KeyType>
uint64_t HeavisideHolder<Tc, Th, Tm, Ta, Tf, KeyType>::retriedGroups() const
{
    return impl_->retriedGroups();
}

template<class Tc, class Th, class Tm, class Ta, class Tf, class KeyType>
const typename HeavisideHolder<Tc, Th, Tm, Ta, Tf, KeyType>::MType*
HeavisideHolder<Tc, Th, Tm, Ta, Tf, KeyType>::deviceMultipoles() const
{
    return impl_->deviceMultipoles();
}

template class HeavisideHolder<double, double, double, double, double, uint64_t>;
template class HeavisideHolder<double, float, float, float, double, uint64_t>;
template class HeavisideHolder<float, float, float, float, float, uint64_t>;

} // namespace ryoanji
