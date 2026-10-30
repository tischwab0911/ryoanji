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

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

#include "cstone/cuda/cuda_utils.cuh"
#include "cstone/sfc/box.hpp"
#include "cstone/sfc/sfc.hpp"
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
        box_            = box;
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
        // The traversal applies cstone's MAC radii; they must carry the velocity criterion, or the locally
        // essential tree and the accepted multipoles do not match the expansion's convergence region.
        if (!focusTree.velocityMacApplied())
        {
            throw std::runtime_error("ryoanji::HeavisideHolder: the focus tree MAC radii are not velocity-aware; "
                                     "synchronize with Domain::setVelocityMac and syncGrav(..., m, w, ...)");
        }

        octree_    = focusTree.octreeViewAcc();
        leaves_    = focusTree.treeLeavesAcc();
        layout_    = layout;
        centers_   = focusTree.expansionCentersAcc().data();
        frame_     = params.frame;
        expansion_ = params.expansion;

        if (expansion_ == VelocityExpansion::Gauss2)
        {
            upsweepImpl(multipolesGauss2_, x, y, z, m, w, gOctree, focusTree, params);
        }
        else { upsweepImpl(multipoles_, x, y, z, m, w, gOctree, focusTree, params); }
    }

    void compute(GroupView grp, const Tc* x, const Tc* y, const Tc* z, const Tm* m, const Th* h, const Tm* w, Tc G,
                 Ta* ex, Ta* ey, Ta* ez, Ta* cx, Ta* cy)
    {
        reallocate(traversalStack_, stackSize(grp.numGroups), 1.01);
        auto traverse = [&](const auto* multipoles)
        {
            traverseHeaviside(grp, 1, x, y, z, m, h, w, frame_, octree_.childOffsets, octree_.internalToLeaf, layout_,
                              centers_, multipoles, G, ex, ey, ez, cx, cy, (int*)rawPtr(traversalStack_));
        };
        try
        {
            if (expansion_ == VelocityExpansion::Gauss2) { traverse(rawPtr(multipolesGauss2_)); }
            else { traverse(rawPtr(multipoles_)); }
        }
        catch (const HeavisideLetIncomplete& e)
        {
            throw std::runtime_error(std::string(e.what()) + diagnoseUnresolved(x, y, z));
        }
    }

    /*! @brief geometry of the unresolved single targets of the last traversal
     *
     * For each (target, cell): whether the target lies in the box of its focus leaf (the SFC key of its position),
     * its distance to the cell's expansion centre, the distance of that leaf box (what cstone's markMacs
     * evaluates) and the cell's MAC radius. A leaf box closer than the radius means cstone should have marked
     * the cell; a leaf box farther than the radius with the target inside it cannot occur.
     */
    std::string diagnoseUnresolved(const Tc* x, const Tc* y, const Tc* z) const
    {
        using SfcKey  = cstone::SfcKind<KeyType>;
        const auto axesBits = box_.getBoxDimBits(cstone::maxTreeLevel<KeyType>{});
        std::vector<KeyType> leaves(leaves_.size());
        checkGpuErrors(cudaMemcpy(leaves.data(), leaves_.data(), leaves.size() * sizeof(KeyType), cudaMemcpyDeviceToHost));

        auto boxDistance = [](const Vec3<Tf>& p, const Vec3<Tf>& center, const Vec3<Tf>& size)
        {
            Tf d2 = 0;
            for (int k = 0; k < 3; ++k)
            {
                Tf d = std::max(std::abs(p[k] - center[k]) - size[k], Tf(0));
                d2 += d * d;
            }
            return std::sqrt(d2);
        };
        auto tf  = [](const auto& v) { return Vec3<Tf>{Tf(v[0]), Tf(v[1]), Tf(v[2])}; };
        auto vec = [](const Vec3<Tf>& v)
        {
            char buf[96];
            std::snprintf(buf, sizeof(buf), "(%.17g, %.17g, %.17g)", double(v[0]), double(v[1]), double(v[2]));
            return std::string(buf);
        };

        std::string out = "\n  box " + vec(tf(Vec3<Tc>{box_.xmin(), box_.ymin(), box_.zmin()})) + " - " +
                          vec(tf(Vec3<Tc>{box_.xmax(), box_.ymax(), box_.zmax()})) + ", axes bits " +
                          std::to_string(axesBits[0]) + " " + std::to_string(axesBits[1]) + " " +
                          std::to_string(axesBits[2]);
        for (auto [target, cell] : readHeavisideUnresolved())
        {
            Tc       p[3];
            Vec4<Tf> c;
            KeyType  prefix;
            checkGpuErrors(cudaMemcpy(&p[0], x + target, sizeof(Tc), cudaMemcpyDeviceToHost));
            checkGpuErrors(cudaMemcpy(&p[1], y + target, sizeof(Tc), cudaMemcpyDeviceToHost));
            checkGpuErrors(cudaMemcpy(&p[2], z + target, sizeof(Tc), cudaMemcpyDeviceToHost));
            checkGpuErrors(cudaMemcpy(&c, centers_ + cell, sizeof(c), cudaMemcpyDeviceToHost));
            checkGpuErrors(cudaMemcpy(&prefix, octree_.prefixes + cell, sizeof(KeyType), cudaMemcpyDeviceToHost));
            const Vec3<Tf> pos{Tf(p[0]), Tf(p[1]), Tf(p[2])};

            // the cell
            const KeyType  cellStart = cstone::decodePlaceholderBit(prefix);
            const unsigned cellLevel = cstone::decodePrefixLength(prefix) / 3;
            auto [cellCenter, cellSize] =
                cstone::centerAndSize<KeyType>(cstone::sfcIBox(cstone::sfcKey(cellStart), cellLevel, axesBits), box_);

            // the focus leaf of the target's key
            const KeyType key  = KeyType(cstone::sfc3D<SfcKey>(p[0], p[1], p[2], box_));
            const auto    it   = std::upper_bound(leaves.begin(), leaves.end(), key);
            const size_t  leaf = size_t(std::max<std::ptrdiff_t>(it - leaves.begin() - 1, 0));
            const bool    keyInTree = leaf + 1 < leaves.size();
            Vec3<Tf>      leafCenter{0, 0, 0}, leafSize{0, 0, 0};
            if (keyInTree)
            {
                auto [lc, ls] = cstone::centerAndSize<KeyType>(
                    cstone::sfcIBox(cstone::sfcKey(leaves[leaf]), cstone::sfcKey(leaves[leaf + 1]), axesBits), box_);
                leafCenter = tf(lc), leafSize = tf(ls);
            }
            const Tf outside = boxDistance(pos, leafCenter, leafSize);
            const Tf rMac    = std::sqrt(std::abs(c[3]));
            const Tf dTarget = std::sqrt(norm2(pos - makeVec3(c)));
            const Tf dLeaf   = boxDistance(makeVec3(c), leafCenter, leafSize);

            char buf[2048];
            std::snprintf(buf, sizeof(buf),
                          "\n  target %u at %s, key %llu in focus leaf %zu [%llu, %llu)%s, %.3g outside its box;"
                          "\n    cell %u level %u key %llu, cell box centre %s half size %s, expansion centre %s,"
                          "\n    MAC radius %.17g, target distance %.17g (%.3g of the radius), leaf box distance %.17g"
                          " (%.3g of the radius): %s",
                          target, vec(pos).c_str(), (unsigned long long)key, leaf,
                          (unsigned long long)(keyInTree ? leaves[leaf] : 0),
                          (unsigned long long)(keyInTree ? leaves[leaf + 1] : 0), keyInTree ? "" : " (key outside tree)",
                          double(outside), cell, cellLevel, (unsigned long long)cellStart, vec(tf(cellCenter)).c_str(),
                          vec(tf(cellSize)).c_str(), vec(makeVec3(c)).c_str(), double(rMac), double(dTarget),
                          double(dTarget / rMac - 1), double(dLeaf), double(dLeaf / rMac - 1),
                          dLeaf < rMac ? "the leaf box fails the MAC, cstone should have marked the cell"
                                       : (outside > 0 ? "the target lies outside its leaf box"
                                                      : "inconsistent: target in its leaf box, box passes, point fails"));
            out += buf;
        }
        return out;
    }

    util::array<uint64_t, 6> readStats() const { return readHeavisideStats(); }

    uint64_t retriedGroups() const { return readHeavisideRetries(); }

    const MType* deviceMultipoles() const { return rawPtr(multipoles_); }

private:
    //! @brief P2M, upsweep, exchange of remote multipoles, second upsweep and finalize for one multipole type
    template<class MT>
    void upsweepImpl(cstone::DeviceVector<MT>& multipoles, const Tc* x, const Tc* y, const Tc* z, const Tm* m,
                     const Tm* w, cstone::OctreeView<const KeyType> gOctree,
                     const cstone::FocusedOctree<KeyType, Tf, cstone::execution::Gpu>& focusTree,
                     const HeavisideParams<Tf>& params)
    {
        double growthRate = 1.01;
        reallocateDestructive(multipoles, octree_.numNodes, growthRate);

        auto            globalCenters = focusTree.globalExpansionCenters();
        const Vec4<Tf>* centers       = centers_;

        computeLeafMultipolesHeaviside(x, y, z, m, w, octree_.leafToInternal + octree_.numInternalNodes,
                                       octree_.numLeafNodes, layout_, centers, rawPtr(multipoles));

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

        upsweepGpu(octree_.levelRangeSpan(), octree_.childOffsets, rawPtr(multipoles), centers);

        std::span d_multipoleSpan{rawPtr(multipoles), size_t(octree_.numNodes)};
        focusTree.globalExchange(gOctree, d_multipoleSpan, std::span<MT>{}, traversalStack_, upsweepGpu,
                                 globalCenters.data());
        focusTree.peerExchange(d_multipoleSpan, static_cast<int>(cstone::P2pTags::focusPeerCenters) + 1,
                               traversalStack_);

        //! second upsweep with leaf data from peer and global ranks in place
        upsweepGpu(octree_.levelRangeSpan(), octree_.childOffsets, rawPtr(multipoles), centers);

        finalizeHeaviside(octree_.numNodes, rawPtr(multipoles), params.frame, params.velocityMoments);
    }

    cstone::OctreeView<const KeyType> octree_;
    std::span<const KeyType>          leaves_;
    cstone::Box<Tc>                   box_{0, 1};
    const LocalIndex*                 layout_ = nullptr;
    HeavisideFrame<Tf>                frame_;
    //! expansion centers and velocity-scaled MAC radii of the focus tree (cstone)
    const Vec4<Tf>*                   centers_ = nullptr;
    VelocityExpansion                 expansion_ = VelocityExpansion::Taylor;
    cstone::DeviceVector<MType>       multipoles_;
    cstone::DeviceVector<Gauss2Qpole<Tm>> multipolesGauss2_;

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
