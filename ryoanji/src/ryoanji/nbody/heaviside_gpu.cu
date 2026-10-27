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
 * @brief GPU upsweep and Barnes-Hut traversal for sources with per-particle velocities
 *
 * The traversal follows traversal_gpu.cu (breadth-first, warp-aware, 64 targets per warp) with the
 * Heaviside kernels: each source carries its rapidity relative to the frame, targets accumulate
 * five values (E~ and the second channel C), and multipoles are HeavisideQpole in evaluation form.
 * It is a separate translation unit so that traversal_gpu.cu stays identical to upstream.
 */

#include "cstone/cuda/device_vector.h"
#include "cstone/cuda/gpu_config.cuh"
#include "cstone/primitives/math.hpp"
#include "cstone/primitives/warpscan.cuh"

#include "heaviside_gpu.h"
#include "traversal_gpu.h"

namespace ryoanji
{

using cstone::ballotSync;
using cstone::GpuConfig;
using cstone::imin;
using cstone::inclusiveScanInt;
using cstone::inclusiveSegscanInt;
using cstone::shflDownSync;
using cstone::shflSync;
using cstone::shflUpSync;
using cstone::shflXorSync;
using cstone::spreadSeg8;
using cstone::streamCompact;
using cstone::syncWarp;
using cstone::warpMax;
using cstone::warpMin;

namespace
{

constexpr int kUpsweepThreads = 256;

struct HTravConfig
{
    static constexpr int memPerWarp = 512 * GpuConfig::warpSize;
    static_assert((memPerWarp & (memPerWarp - 1)) == 0);
    static constexpr int numThreads    = 256;
    static constexpr int numWarpsPerSm = 20;
    inline static int maxNumActiveBlocks =
        GpuConfig::smCount * (HTravConfig::numWarpsPerSm / (HTravConfig::numThreads / GpuConfig::warpSize));
    static constexpr int targetSize = 64;
    static constexpr int nwt        = targetSize / GpuConfig::warpSize;
};

//! @brief squared MAC radius meaning "always open" (larger than any distance)
template<class T>
__host__ __device__ constexpr T alwaysOpen()
{
    if constexpr (sizeof(T) == sizeof(float)) { return T(3.0e38f); }
    else { return T(1.0e300); }
}

__device__ __forceinline__ int ringAddrH(const int i) { return i & (HTravConfig::memPerWarp - 1); }

template<class T>
__device__ __forceinline__ bool applyMACH(Vec3<T> sourceCenter, T MAC, Vec3<T> targetCenter, Vec3<T> targetSize)
{
    Vec3<T> dX = abs(targetCenter - sourceCenter) - targetSize;
    dX += abs(dX);
    dX *= T(0.5);
    return norm2(dX) < MAC;
}

} // namespace

// ------------------------------------------------------------------------------------------------ upsweep

template<int TPL, class Tc, class Tm, class Tf, class T>
__global__ void leafMultipolesHeavisideKernel(const Tc* x, const Tc* y, const Tc* z, const Tm* m, const Tm* w,
                                              const TreeNodeIndex* leafToInternal, TreeNodeIndex numLeaves,
                                              const LocalIndex* layout, const Vec4<Tf>* centers,
                                              HeavisideQpole<T>* multipoles)
{
    TreeNodeIndex tid     = blockIdx.x * blockDim.x + threadIdx.x;
    TreeNodeIndex leafIdx = tid / TPL;
    TreeNodeIndex internalIdx = 0;

    HeavisideQpole<T> mp = heavisideZero<T>();
    if (leafIdx < numLeaves)
    {
        internalIdx = leafToInternal[leafIdx];
        P2MHeaviside_add<TPL>(x, y, z, m, w, layout[leafIdx] + threadIdx.x % TPL, layout[leafIdx + 1],
                              centers[internalIdx], mp);
    }

#pragma unroll
    for (int offset = 1; offset < TPL; offset *= 2)
    {
#pragma unroll
        for (int mi = 0; mi < int(mp.size()); ++mi)
        {
            mp[mi] = heavisideCombine(mi, mp[mi], cstone::shflDownSync(mp[mi], offset));
        }
    }

    if (tid % TPL == 0 && leafIdx < numLeaves) { multipoles[internalIdx] = mp; }
}

template<class Tc, class Tm, class Tf, class T>
void computeLeafMultipolesHeaviside(const Tc* x, const Tc* y, const Tc* z, const Tm* m, const Tm* w,
                                    const TreeNodeIndex* leafToInternal, TreeNodeIndex numLeaves,
                                    const LocalIndex* layout, const Vec4<Tf>* centers, HeavisideQpole<T>* multipoles)
{
    constexpr int threadsPerLeaf = 8;
    int           numBlocks      = cstone::iceil(threadsPerLeaf * numLeaves, kUpsweepThreads);
    if (numBlocks)
    {
        leafMultipolesHeavisideKernel<threadsPerLeaf><<<numBlocks, kUpsweepThreads>>>(
            x, y, z, m, w, leafToInternal, numLeaves, layout, centers, multipoles);
    }
}

template<class Tf, class T>
__global__ void upsweepHeavisideKernel(TreeNodeIndex firstCell, TreeNodeIndex lastCell,
                                       const TreeNodeIndex* childOffsets, const Vec4<Tf>* centers,
                                       HeavisideQpole<T>* multipoles)
{
    TreeNodeIndex tid     = blockIdx.x * blockDim.x + threadIdx.x;
    const int     cellIdx = tid / 8 + firstCell;

    TreeNodeIndex firstChild = 0;
    if (cellIdx < lastCell) { firstChild = childOffsets[cellIdx]; }

    HeavisideQpole<T> Mout = heavisideZero<T>();
    if (firstChild) // zero for leaves
    {
        int  child = firstChild + threadIdx.x % 8;
        auto dX    = makeVec3(centers[cellIdx] - centers[child]);
        heavisideM2M(Mout, dX, multipoles[child]);
    }

#pragma unroll
    for (int offset = 1; offset < 8; offset *= 2)
    {
#pragma unroll
        for (int mi = 0; mi < int(Mout.size()); ++mi)
        {
            Mout[mi] = heavisideCombine(mi, Mout[mi], cstone::shflDownSync(Mout[mi], offset));
        }
    }

    if (firstChild && threadIdx.x % 8 == 0) { multipoles[cellIdx] = Mout; }
}

template<class Tf, class T>
void upsweepMultipolesHeaviside(TreeNodeIndex firstCell, TreeNodeIndex lastCell, const TreeNodeIndex* childOffsets,
                                const Vec4<Tf>* centers, HeavisideQpole<T>* multipoles)
{
    if (lastCell > firstCell)
    {
        upsweepHeavisideKernel<<<cstone::iceil(8 * (lastCell - firstCell), kUpsweepThreads), kUpsweepThreads>>>(
            firstCell, lastCell, childOffsets, centers, multipoles);
    }
}

template<class Tf, class T>
__global__ void finalizeHeavisideKernel(TreeNodeIndex numNodes, const Vec4<Tf>* centers,
                                        HeavisideQpole<T>* multipoles, HeavisideFrame<Tf> frame, Tf thetaV,
                                        bool velocityMoments, Vec4<Tf>* macOut)
{
    TreeNodeIndex i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numNodes) { return; }
    HeavisideQpole<T> M = multipoles[i];
    heavisideFinalize(M, frame, velocityMoments);
    multipoles[i] = M;

    Vec4<Tf> c   = centers[i];
    Tf       mac = c[3];
    if (mac != Tf(0))
    {
        Tf scale = heavisideMacScale2(M, frame, thetaV, velocityMoments);
        mac      = scale < Tf(0) ? alwaysOpen<Tf>() : mac * scale;
    }
    macOut[i] = {c[0], c[1], c[2], mac};
}

template<class Tf, class T>
void finalizeHeaviside(TreeNodeIndex numNodes, const Vec4<Tf>* centers, HeavisideQpole<T>* multipoles,
                       HeavisideFrame<Tf> frame, Tf thetaV, bool velocityMoments, Vec4<Tf>* macOut)
{
    if (numNodes > 0)
    {
        finalizeHeavisideKernel<<<cstone::iceil(numNodes, kUpsweepThreads), kUpsweepThreads>>>(
            numNodes, centers, multipoles, frame, thetaV, velocityMoments, macOut);
    }
}

// ------------------------------------------------------------------------------------------------ traversal

//! @brief M2P with GpuConfig::warpSize multipoles (one per lane in @p cellIdx) on the warp's targets
template<class Ta, class Tc, class Tf, class T>
__device__ void approxAccHeaviside(util::array<Ta, 5> acc_i[HTravConfig::nwt], const Vec4<Tc> pos_i[HTravConfig::nwt],
                                   const int cellIdx, const Vec4<Tf>* __restrict__ srcCenter,
                                   const HeavisideQpole<T>* __restrict__ multipoles, volatile int* warpSpace)
{
    constexpr int termSize = HeavisideQpole<T>{}.size();
    static_assert(termSize <= GpuConfig::warpSize, "multipole size too large for shared-mem warpSpace");

    auto* sm_Multipole      = reinterpret_cast<volatile T*>(warpSpace);
    auto* __restrict__ gm_M = reinterpret_cast<const T*>(multipoles);

    const int laneIdx = threadIdx.x & (GpuConfig::warpSize - 1);

    for (int j = 0; j < GpuConfig::warpSize; j++)
    {
        int currentCell = shflSync(cellIdx, j);
        if (currentCell < 0) { continue; }

        Vec3<Tf> pos_j = makeVec3(srcCenter[currentCell]);

        if (laneIdx < termSize) { sm_Multipole[laneIdx] = gm_M[currentCell * termSize + laneIdx]; }
        syncWarp();

#pragma unroll
        for (int k = 0; k < HTravConfig::nwt; k++)
        {
            acc_i[k] = M2PHeaviside(acc_i[k], makeVec3(pos_i[k]), pos_j, *(HeavisideQpole<T>*)sm_Multipole);
        }
        syncWarp();
    }
}

/*! @brief P2P of GpuConfig::warpSize sources (one per lane) with the warp's targets
 *
 * @p sourceBody: x, y, z, m; @p sChi: stretch and second-channel weight of the lane's source
 */
template<class Tc, class Th, class Tm, class Ta>
__device__ void directAccHeaviside(Vec4<Tc> sourceBody, Th hSource, util::array<Tm, 2> sChi,
                                   util::array<Ta, 5> acc_i[HTravConfig::nwt], const Vec4<Tc> pos_i[HTravConfig::nwt])
{
    for (int j = 0; j < GpuConfig::warpSize; j++)
    {
        Vec3<Tc> pos_j{shflSync(sourceBody[0], j), shflSync(sourceBody[1], j), shflSync(sourceBody[2], j)};
        Tc       m_j   = shflSync(sourceBody[3], j);
        Th       h_j   = shflSync(hSource, j);
        Tm       s_j   = shflSync(sChi[0], j);
        Tm       chi_j = shflSync(sChi[1], j);

#pragma unroll
        for (int k = 0; k < HTravConfig::nwt; k++)
        {
            acc_i[k] = P2PHeaviside(acc_i[k], makeVec3(pos_i[k]), pos_j, Tm(m_j), s_j, chi_j, Th(pos_i[k][3]), h_j);
        }
    }
}

template<class Tm, class Tf>
__device__ __forceinline__ util::array<Tm, 2> sourceFactorsOf(const HeavisideFrame<Tf>& frame, Tm wHat)
{
    Tf s, chi;
    frame.sourceFactors(Tf(wHat), s, chi);
    return {Tm(s), Tm(chi)};
}

template<class Tc, class Th, class Tm, class Ta, class Tf, class T>
__device__ util::array<unsigned, 4>
traverseWarpHeaviside(util::array<Ta, 5>* acc_i, const Vec4<Tc> pos_i[HTravConfig::nwt], const Vec3<Tf> targetCenter,
                      const Vec3<Tf> targetSize, const Tc* __restrict__ x, const Tc* __restrict__ y,
                      const Tc* __restrict__ z, const Tm* __restrict__ m, const Th* __restrict__ h,
                      const Tm* __restrict__ w, const HeavisideFrame<Tf>& frame,
                      const TreeNodeIndex* __restrict__ childOffsets, const TreeNodeIndex* __restrict__ internalToLeaf,
                      const LocalIndex* __restrict__ layout, const Vec4<Tf>* __restrict__ sourceCenter,
                      const HeavisideQpole<T>* __restrict__ multipoles, int initNodeIdx, volatile int* tempQueue,
                      int* cellQueue)
{
    const int laneIdx = threadIdx.x & (GpuConfig::warpSize - 1);

    unsigned p2pCounter = 0, m2pCounter = 0, maxStack = 0, remoteFailed = 0;

    int approxQueue;
    int bodyQueue;

    if (laneIdx == 0) { cellQueue[0] = initNodeIdx; }

    int numSources   = 1;
    int newSources   = 0;
    int oldSources   = 0;
    int sourceOffset = 0;
    int apxFillLevel = 0;
    int bdyFillLevel = 0;

    auto loadBody = [&](int bodyIdx, Vec4<Tc>& body, Th& hs, util::array<Tm, 2>& sChi)
    {
        body = {x[bodyIdx], y[bodyIdx], z[bodyIdx], Tc(m[bodyIdx])};
        hs   = h[bodyIdx];
        sChi = sourceFactorsOf(frame, w[bodyIdx]);
    };

    while (numSources > 0)
    {
        int sourceIdx   = sourceOffset + laneIdx;
        int sourceQueue = 0;
        if (laneIdx < GpuConfig::warpSize / 8) { sourceQueue = cellQueue[ringAddrH(oldSources + sourceIdx)]; }
        sourceQueue         = spreadSeg8(sourceQueue);
        sourceIdx           = shflSync(sourceIdx, laneIdx >> 3);
        const bool isSource = sourceIdx < numSources;
        if (!isSource) { sourceQueue = 0; }

        const Vec4<Tf> MAC = sourceCenter[sourceQueue];
        const Vec3<Tf> curSrcCenter{MAC[0], MAC[1], MAC[2]};
        const int      childBegin = childOffsets[sourceQueue];
        const bool     isNode     = childBegin;
        const bool     isClose    = applyMACH(curSrcCenter, MAC[3], targetCenter, targetSize);
        const bool     isDirect   = isClose && !isNode && isSource;
        const int      leafIdx    = (isDirect) ? internalToLeaf[sourceQueue] : 0;

        const bool isSplit      = isNode && isClose && isSource;
        const int  numChildLane = cstone::exclusiveScanBool(isSplit);
        const int  numChildWarp = cstone::reduceBool(isSplit);
        sourceOffset += imin(GpuConfig::warpSize / 8, numSources - sourceOffset);
        int childIdx = oldSources + numSources + newSources + numChildLane;
        if (isSplit) { cellQueue[ringAddrH(childIdx)] = childBegin; }
        newSources += numChildWarp;

        const unsigned stackUsed = newSources + numSources - sourceOffset;
        maxStack                 = max(stackUsed, maxStack);
        if (stackUsed > HTravConfig::memPerWarp) { return {0xFFFFFFFF, 0xFFFFFFFF, maxStack, 0}; }

        // A remote leaf (no local sources, non-zero MAC radius) can only be approximated. If it fails
        // the MAC, the locally essential tree lacks its particles: count it.
        const bool isRemote    = layout[leafIdx + 1] == layout[leafIdx] && MAC[3] != Tf(0);
        const bool isApprox    = (!isClose || (!isNode && isRemote)) && isSource;
        remoteFailed += cstone::reduceBool(isClose && !isNode && isRemote && isSource);
        int        numKeepWarp = streamCompact(&sourceQueue, isApprox, tempQueue);
        const int  apxTopUp    = shflUpSync(sourceQueue, apxFillLevel);
        approxQueue            = (laneIdx < apxFillLevel) ? approxQueue : apxTopUp;
        apxFillLevel += numKeepWarp;

        if (apxFillLevel >= GpuConfig::warpSize)
        {
            approxAccHeaviside(acc_i, pos_i, approxQueue, sourceCenter, multipoles, tempQueue);
            apxFillLevel -= GpuConfig::warpSize;
            approxQueue = shflDownSync(sourceQueue, numKeepWarp - apxFillLevel);
            m2pCounter += GpuConfig::warpSize;
        }

        const int firstBody     = layout[leafIdx];
        const int numBodies     = (layout[leafIdx + 1] - firstBody) & -int(isDirect);
        bool      directTodo    = numBodies;
        const int numBodiesScan = inclusiveScanInt(numBodies);
        int       numBodiesLane = numBodiesScan - numBodies;
        int       numBodiesWarp = shflSync(numBodiesScan, GpuConfig::warpSize - 1);
        int       prevBodyIdx   = 0;
        while (numBodiesWarp > 0)
        {
            tempQueue[laneIdx] = 1;
            if (directTodo && (numBodiesLane < GpuConfig::warpSize))
            {
                directTodo               = false;
                tempQueue[numBodiesLane] = -1 - firstBody;
            }
            const int bodyIdx = inclusiveSegscanInt(tempQueue[laneIdx], prevBodyIdx);
            prevBodyIdx       = shflSync(bodyIdx, GpuConfig::warpSize - 1);

            if (numBodiesWarp >= GpuConfig::warpSize)
            {
                Vec4<Tc>           body;
                Th                 hs;
                util::array<Tm, 2> sChi;
                loadBody(bodyIdx, body, hs, sChi);
                directAccHeaviside(body, hs, sChi, acc_i, pos_i);
                numBodiesWarp -= GpuConfig::warpSize;
                numBodiesLane -= GpuConfig::warpSize;
                p2pCounter += GpuConfig::warpSize;
            }
            else
            {
                int topUp = shflUpSync(bodyIdx, bdyFillLevel);
                bodyQueue = (laneIdx < bdyFillLevel) ? bodyQueue : topUp;

                bdyFillLevel += numBodiesWarp;
                if (bdyFillLevel >= GpuConfig::warpSize)
                {
                    Vec4<Tc>           body;
                    Th                 hs;
                    util::array<Tm, 2> sChi;
                    loadBody(bodyQueue, body, hs, sChi);
                    directAccHeaviside(body, hs, sChi, acc_i, pos_i);
                    bdyFillLevel -= GpuConfig::warpSize;
                    bodyQueue = shflDownSync(bodyIdx, numBodiesWarp - bdyFillLevel);
                    p2pCounter += GpuConfig::warpSize;
                }
                numBodiesWarp = 0;
            }
        }

        if (sourceOffset >= numSources)
        {
            oldSources += numSources;
            numSources   = newSources;
            sourceOffset = newSources = 0;
        }
    }

    if (apxFillLevel > 0)
    {
        approxAccHeaviside(acc_i, pos_i, laneIdx < apxFillLevel ? approxQueue : -1, sourceCenter, multipoles,
                           tempQueue);
        m2pCounter += apxFillLevel;
    }

    if (bdyFillLevel > 0)
    {
        const int          bodyIdx = laneIdx < bdyFillLevel ? bodyQueue : -1;
        Vec4<Tc>           body{Tc(0), Tc(0), Tc(0), Tc(0)};
        Th                 hs = Th(0);
        util::array<Tm, 2> sChi{Tm(1), Tm(0)};
        if (bodyIdx >= 0) { loadBody(bodyIdx, body, hs, sChi); }
        directAccHeaviside(body, hs, sChi, acc_i, pos_i);
        p2pCounter += bdyFillLevel;
    }

    return {m2pCounter, p2pCounter, maxStack, remoteFailed};
}

namespace
{
struct HStats
{
    using type = unsigned long long;
    enum IndexNames
    {
        sumP2P,
        maxP2P,
        sumM2P,
        maxM2P,
        maxStack,
        remoteFailed,
        numStats
    };
};
} // namespace

static __device__ HStats::type heavisideStats[HStats::numStats];
__device__ unsigned int        heavisideTargetCounter = 0;
//! @brief number of target groups whose traversal overflowed the cell queue in the current pass
__device__ unsigned int heavisideFailedCount = 0;

//! @brief reset the work counters of one traversal pass, and the statistics before the first pass
__global__ void resetHeavisideCounters(bool resetStats)
{
    if (resetStats)
    {
        for (int i = 0; i < HStats::numStats; ++i)
        {
            heavisideStats[i] = 0;
        }
    }
    heavisideTargetCounter = 0;
    heavisideFailedCount   = 0;
}

//! @brief target groups re-traversed (split) after overflowing the cell queue in the last traverseHeaviside
static uint64_t heavisideRetriedLast = 0;

template<class Tc, class Th, class Tm, class Ta, class Tf, class T>
__global__ __launch_bounds__(HTravConfig::numThreads) void traverseHeavisideKernel(
    cstone::GroupView grp, const int initNodeIdx, const Tc* __restrict__ x, const Tc* __restrict__ y,
    const Tc* __restrict__ z, const Tm* __restrict__ m, const Th* __restrict__ h, const Tm* __restrict__ w,
    HeavisideFrame<Tf> frame, const TreeNodeIndex* __restrict__ childOffsets,
    const TreeNodeIndex* __restrict__ internalToLeaf, const LocalIndex* __restrict__ layout,
    const Vec4<Tf>* __restrict__ sourceCenter, const HeavisideQpole<T>* __restrict__ multipoles, Tc G, Ta* ex,
    Ta* ey, Ta* ez, Ta* cx, Ta* cy, int* gmPool, LocalIndex* failedGroups, unsigned failedCapacity)
{
    const int laneIdx = threadIdx.x & (GpuConfig::warpSize - 1);
    const int warpIdx = threadIdx.x >> GpuConfig::warpSizeLog2;

    constexpr int termSize         = HeavisideQpole<T>{}.size();
    constexpr int numWarpsPerBlock = HTravConfig::numThreads / GpuConfig::warpSize;
    constexpr int mSizeRatio       = sizeof(T) / sizeof(int);
    constexpr int smSize =
        (HTravConfig::numThreads > termSize * numWarpsPerBlock) ? HTravConfig::numThreads : termSize * numWarpsPerBlock;
    __shared__ int sharedPool[smSize * mSizeRatio];

    int* tempQueue = sharedPool + GpuConfig::warpSize * warpIdx * mSizeRatio;
    int* cellQueue = gmPool + HTravConfig::memPerWarp * ((blockIdx.x * numWarpsPerBlock) + warpIdx);

    LocalIndex targetIdx = 0;

    while (true)
    {
        if (laneIdx == 0) { targetIdx = atomicAdd(&heavisideTargetCounter, 1); }
        targetIdx = shflSync(targetIdx, 0);

        if (targetIdx >= grp.numGroups) return;

        const int bodyBegin = grp.groupStart[targetIdx];
        const int bodyEnd   = grp.groupEnd[targetIdx];

        Vec4<Tc> pos_i[HTravConfig::nwt];
        for (int i = 0; i < HTravConfig::nwt; i++)
        {
            int bodyIdx = imin(bodyBegin + i * GpuConfig::warpSize + laneIdx, bodyEnd - 1);
            pos_i[i]    = {x[bodyIdx], y[bodyIdx], z[bodyIdx], Tc(h[bodyIdx])};
        }

        Vec3<Tc> Xmin = makeVec3(pos_i[0]);
        Vec3<Tc> Xmax = makeVec3(pos_i[0]);
        for (int i = 1; i < HTravConfig::nwt; i++)
        {
            Xmin = min(Xmin, makeVec3(pos_i[i]));
            Xmax = max(Xmax, makeVec3(pos_i[i]));
        }
        Xmin = {warpMin(Xmin[0]), warpMin(Xmin[1]), warpMin(Xmin[2])};
        Xmax = {warpMax(Xmax[0]), warpMax(Xmax[1]), warpMax(Xmax[2])};

        const Vec3<Tf> targetCenter = (Xmax + Xmin) * Tf(0.5);
        const Vec3<Tf> targetSize   = (Xmax - Xmin) * Tf(0.5);

        util::array<Ta, 5> acc_i[HTravConfig::nwt];
        for (int i = 0; i < HTravConfig::nwt; i++)
        {
            acc_i[i] = Ta(0);
        }

        const util::array<unsigned, 4> counts = traverseWarpHeaviside(
            acc_i, pos_i, targetCenter, targetSize, x, y, z, m, h, w, frame, childOffsets, internalToLeaf, layout,
            sourceCenter, multipoles, initNodeIdx, tempQueue, cellQueue);
        const unsigned numM2P = counts[0], numP2P = counts[1], maxStack = counts[2], remoteFailed = counts[3];
        if (numM2P == 0xFFFFFFFF)
        {
            // cell queue overflow: write nothing, the host splits the group and traverses the halves again
            if (laneIdx == 0)
            {
                unsigned slot = atomicAdd(&heavisideFailedCount, 1u);
                if (slot < failedCapacity) { failedGroups[slot] = targetIdx; }
                atomicMax(&heavisideStats[HStats::maxStack], maxStack);
            }
            continue;
        }

        if (laneIdx == 0)
        {
            int targetGroupSize = bodyEnd - bodyBegin;
            atomicMax(&heavisideStats[HStats::maxP2P], numP2P);
            atomicAdd(&heavisideStats[HStats::sumP2P], HStats::type(numP2P) * targetGroupSize);
            atomicMax(&heavisideStats[HStats::maxM2P], numM2P);
            atomicAdd(&heavisideStats[HStats::sumM2P], HStats::type(numM2P) * targetGroupSize);
            atomicMax(&heavisideStats[HStats::maxStack], maxStack);
            atomicAdd(&heavisideStats[HStats::remoteFailed], HStats::type(remoteFailed));
        }

        const int bodyIdxLane = bodyBegin + laneIdx;
        for (int i = 0; i < HTravConfig::nwt; i++)
        {
            const int bodyIdx = bodyIdxLane + i * GpuConfig::warpSize;
            if (bodyIdx < bodyEnd)
            {
                ex[bodyIdx] += G * acc_i[i][0];
                ey[bodyIdx] += G * acc_i[i][1];
                ez[bodyIdx] += G * acc_i[i][2];
                cx[bodyIdx] += G * acc_i[i][3];
                cy[bodyIdx] += G * acc_i[i][4];
            }
        }
    }
}

template<class Tc, class Th, class Tm, class Ta, class Tf, class T>
void traverseHeaviside(cstone::GroupView grp, int initNodeIdx, const Tc* x, const Tc* y, const Tc* z, const Tm* m,
                       const Th* h, const Tm* w, HeavisideFrame<Tf> frame, const TreeNodeIndex* childOffsets,
                       const TreeNodeIndex* internalToLeaf, const LocalIndex* layout, const Vec4<Tf>* sourceCenter,
                       const HeavisideQpole<T>* multipoles, Tc G, Ta* ex, Ta* ey, Ta* ez, Ta* cx, Ta* cy, int* gmPool)
{
    // Overflowing target groups write nothing and are split and traversed again, as in ryoanji::traverse.
    int       numWarpsPerBlock = HTravConfig::numThreads / GpuConfig::warpSize;
    const int poolBlocks = std::min<int>(cstone::iceil(grp.numGroups, numWarpsPerBlock), HTravConfig::maxNumActiveBlocks);
    heavisideRetriedLast = 0;

    cstone::DeviceVector<LocalIndex> failed(std::max<LocalIndex>(grp.numGroups, 1));
    cstone::DeviceVector<LocalIndex> retryStart, retryEnd;
    cstone::GroupView                pass = grp;
    resetHeavisideCounters<<<1, 1>>>(true);
    for (bool first = true; pass.numGroups > 0; first = false)
    {
        const int numBlocks = std::min<int>(cstone::iceil(pass.numGroups, numWarpsPerBlock), poolBlocks);
        if (!first) { resetHeavisideCounters<<<1, 1>>>(false); }
        traverseHeavisideKernel<<<numBlocks, HTravConfig::numThreads>>>(
            pass, initNodeIdx, x, y, z, m, h, w, frame, childOffsets, internalToLeaf, layout, sourceCenter, multipoles,
            G, ex, ey, ez, cx, cy, gmPool, rawPtr(failed), failed.size());

        unsigned numFailed;
        checkGpuErrors(cudaMemcpyFromSymbol(&numFailed, GPU_SYMBOL(heavisideFailedCount), sizeof(unsigned)));
        if (numFailed == 0) { break; }
        heavisideRetriedLast += numFailed;

        auto [next, unsplittable] = splitFailedGroups(pass, rawPtr(failed), numFailed, retryStart, retryEnd);
        if (unsplittable > 0)
        {
            HStats::type flag = 0xFFFFFFFF;  // reported as an exhausted traversal stack
            checkGpuErrors(cudaMemcpyToSymbol(GPU_SYMBOL(heavisideStats), &flag, sizeof(flag),
                                              HStats::maxP2P * sizeof(HStats::type)));
        }
        pass = next;
        failed.resize(std::max<LocalIndex>(pass.numGroups, 1));
    }
}

uint64_t readHeavisideRetries() { return heavisideRetriedLast; }

util::array<uint64_t, 6> readHeavisideStats()
{
    HStats::type stats[HStats::numStats];
    checkGpuErrors(cudaMemcpyFromSymbol(stats, GPU_SYMBOL(heavisideStats), HStats::numStats * sizeof(HStats::type)));
    return {stats[HStats::sumP2P], stats[HStats::maxP2P], stats[HStats::sumM2P],
            stats[HStats::maxM2P], stats[HStats::maxStack], stats[HStats::remoteFailed]};
}

#define HEAVISIDE_INSTANTIATE(Tc, Th, Tm, Ta, Tf, T)                                                                   \
    template void computeLeafMultipolesHeaviside(const Tc*, const Tc*, const Tc*, const Tm*, const Tm*,                \
                                                 const TreeNodeIndex*, TreeNodeIndex, const LocalIndex*,               \
                                                 const Vec4<Tf>*, HeavisideQpole<T>*);                                 \
    template void upsweepMultipolesHeaviside(TreeNodeIndex, TreeNodeIndex, const TreeNodeIndex*, const Vec4<Tf>*,      \
                                             HeavisideQpole<T>*);                                                      \
    template void finalizeHeaviside(TreeNodeIndex, const Vec4<Tf>*, HeavisideQpole<T>*, HeavisideFrame<Tf>, Tf, bool,  \
                                    Vec4<Tf>*);                                                                        \
    template void traverseHeaviside(cstone::GroupView, int, const Tc*, const Tc*, const Tc*, const Tm*, const Th*,     \
                                    const Tm*, HeavisideFrame<Tf>, const TreeNodeIndex*, const TreeNodeIndex*,         \
                                    const LocalIndex*, const Vec4<Tf>*, const HeavisideQpole<T>*, Tc, Ta*, Ta*, Ta*,   \
                                    Ta*, Ta*, int*)

HEAVISIDE_INSTANTIATE(double, double, double, double, double, double);
HEAVISIDE_INSTANTIATE(double, float, float, float, double, float);
HEAVISIDE_INSTANTIATE(float, float, float, float, float, float);

} // namespace ryoanji
