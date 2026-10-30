/*
 * Cornerstone octree
 *
 * Copyright (c) 2024 CSCS, ETH Zurich
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: MIT License
 */

/*! @file
 * @brief  Compute cell mass centers for use in focus tree
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#include "cstone/cuda/errorcheck.cuh"
#include "cstone/primitives/math.hpp"
#include "cstone/primitives/warpscan.cuh"
#include "source_center.hpp"
#include "source_center_gpu.h"

namespace cstone
{
template<int TPL, class Tc, class Th>
__global__ void computeBoundingBoxKernel(const Tc* x,
                                         const Tc* y,
                                         const Tc* z,
                                         const Th* h,
                                         const LocalIndex* layout,
                                         TreeNodeIndex firstLeaf,
                                         TreeNodeIndex lastLeaf,
                                         Th scale,
                                         Vec3<Tc>* searchCenters,
                                         Vec3<Tc>* searchSizes)
{
    TreeNodeIndex tid     = blockIdx.x * blockDim.x + threadIdx.x;
    TreeNodeIndex leafIdx = firstLeaf + tid / TPL;

    Vec3<Tc> init      = searchCenters[leafIdx];
    Vec3<Tc> commonMin = init, commonMax = init;
    if (leafIdx < lastLeaf)
    {
        auto first = layout[leafIdx];
        auto last  = layout[leafIdx + 1];

        for (LocalIndex i = first + threadIdx.x % TPL; i < last; i += TPL)
        {
            auto r = h[i] * scale;
            Vec3<Tc> p{x[i], y[i], z[i]};
            commonMin = min(commonMin, Vec3<Tc>{p[0] - r, p[1] - r, p[2] - r});
            commonMax = max(commonMax, Vec3<Tc>{p[0] + r, p[1] + r, p[2] + r});
        }
    }

#pragma unroll
    for (int offset = 1; offset < TPL; offset *= 2)
    {
#pragma unroll
        for (int d = 0; d < 3; ++d)
        {
            commonMin[d] = stl::min(commonMin[d], shflDownSync(commonMin[d], offset));
            commonMax[d] = stl::max(commonMax[d], shflDownSync(commonMax[d], offset));
        }
    }

    if (tid % TPL == 0 && leafIdx < lastLeaf)
    {
        searchCenters[leafIdx] = (commonMax + commonMin) * Tc(0.5);
        searchSizes[leafIdx]   = (commonMax - commonMin) * Tc(0.5);
    }
}

template<class Tc, class Th>
void computeBoundingBoxGpu(execution::Gpu exec,
                           const Tc* x,
                           const Tc* y,
                           const Tc* z,
                           const Th* h,
                           const LocalIndex* layout,
                           TreeNodeIndex first,
                           TreeNodeIndex last,
                           Th scale,
                           Vec3<Tc>* searchCenters,
                           Vec3<Tc>* searchSizes)
{
    constexpr int threadsPerLeaf = 8;
    unsigned numThreads          = 256;
    unsigned numBlocks           = iceil(threadsPerLeaf * (last - first), numThreads);

    if (numBlocks == 0) { return; }
    computeBoundingBoxKernel<threadsPerLeaf>
        <<<numBlocks, numThreads, 0, exec>>>(x, y, z, h, layout, first, last, scale, searchCenters, searchSizes);
}

#define COMPUTE_BOUNDING_BOX_GPU(Tc, Th)                                                                               \
    template void computeBoundingBoxGpu(execution::Gpu, const Tc* x, const Tc* y, const Tc* z, const Th* h,            \
                                        const LocalIndex* layout, TreeNodeIndex first, TreeNodeIndex last, Th scale,   \
                                        Vec3<Tc>* searchCenters, Vec3<Tc>* searchSizes);

COMPUTE_BOUNDING_BOX_GPU(double, double);
COMPUTE_BOUNDING_BOX_GPU(double, float);
COMPUTE_BOUNDING_BOX_GPU(float, float);

template<int TPL, class Tc, class Tm, class Tf>
__global__ void computeLeafSourceCenterKernel(const Tc* x,
                                              const Tc* y,
                                              const Tc* z,
                                              const Tm* m,
                                              const TreeNodeIndex* leafToInternal,
                                              TreeNodeIndex numLeaves,
                                              const LocalIndex* layout,
                                              Vec4<Tf>* centers)
{
    TreeNodeIndex tid     = blockIdx.x * blockDim.x + threadIdx.x;
    TreeNodeIndex leafIdx = tid / TPL;

    Vec4<Tf> mc_loc{0, 0, 0, 0};
    if (leafIdx < numLeaves)
    {
        auto first = layout[leafIdx];
        auto last  = layout[leafIdx + 1];

        for (LocalIndex i = first + threadIdx.x % TPL; i < last; i += TPL)
        {
            addBody(mc_loc, {x[i], y[i], z[i], m[i]});
        }
    }

#pragma unroll
    for (int offset = 1; offset < TPL; offset *= 2)
    {
        mc_loc[0] += shflDownSync(mc_loc[0], offset);
        mc_loc[1] += shflDownSync(mc_loc[1], offset);
        mc_loc[2] += shflDownSync(mc_loc[2], offset);
        mc_loc[3] += shflDownSync(mc_loc[3], offset);
    }

    if (tid % TPL == 0 && leafIdx < numLeaves)
    {
        TreeNodeIndex nodeIdx = leafToInternal[leafIdx];
        centers[nodeIdx]      = normalizeMass(mc_loc);
    }
}

template<class Tc, class Tm, class Tf>
void computeLeafSourceCenterGpu(execution::Gpu exec,
                                const Tc* x,
                                const Tc* y,
                                const Tc* z,
                                const Tm* m,
                                const TreeNodeIndex* leafToInternal,
                                TreeNodeIndex numLeaves,
                                const LocalIndex* layout,
                                Vec4<Tf>* centers)
{
    constexpr int threadsPerLeaf = 8;
    unsigned numThreads          = 256;
    unsigned numBlocks           = iceil(threadsPerLeaf * numLeaves, numThreads);

    if (numBlocks == 0) { return; }
    computeLeafSourceCenterKernel<threadsPerLeaf>
        <<<numBlocks, numThreads, 0, exec>>>(x, y, z, m, leafToInternal, numLeaves, layout, centers);
}

#define COMPUTE_LEAF_SOURCE_CENTER_GPU(Tc, Tm, Tf)                                                                     \
    template void computeLeafSourceCenterGpu(execution::Gpu, const Tc*, const Tc*, const Tc*, const Tm*,               \
                                             const TreeNodeIndex*, TreeNodeIndex, const LocalIndex*, Vec4<Tf>*);

COMPUTE_LEAF_SOURCE_CENTER_GPU(double, double, double);
COMPUTE_LEAF_SOURCE_CENTER_GPU(double, float, double);
COMPUTE_LEAF_SOURCE_CENTER_GPU(float, float, float);

template<class T>
__global__ void upsweepCentersKernel(TreeNodeIndex firstCell,
                                     TreeNodeIndex lastCell,
                                     const TreeNodeIndex* childOffsets,
                                     SourceCenterType<T>* centers)
{
    const int cellIdx = blockIdx.x * blockDim.x + threadIdx.x + firstCell;
    if (cellIdx >= lastCell) return;

    TreeNodeIndex firstChild = childOffsets[cellIdx];

    if (firstChild) { centers[cellIdx] = CombineSourceCenter<T>{}(cellIdx, firstChild, centers); }
}

template<class T>
void upsweepCentersGpu(execution::Gpu exec,
                       int numLevels,
                       const TreeNodeIndex* levelRange,
                       const TreeNodeIndex* childOffsets,
                       SourceCenterType<T>* centers)
{
    constexpr int numThreads = 256;

    for (int level = numLevels - 1; level >= 0; level--)
    {
        int numCellsLevel = levelRange[level + 1] - levelRange[level];
        int numBlocks     = (numCellsLevel - 1) / numThreads + 1;
        if (numCellsLevel)
        {
            upsweepCentersKernel<<<numBlocks, numThreads, 0, exec>>>(levelRange[level], levelRange[level + 1],
                                                                     childOffsets, centers);
        }
    }
}

template void
upsweepCentersGpu(execution::Gpu, int, const TreeNodeIndex*, const TreeNodeIndex*, SourceCenterType<float>*);
template void
upsweepCentersGpu(execution::Gpu, int, const TreeNodeIndex*, const TreeNodeIndex*, SourceCenterType<double>*);

template<class KeyType, class T>
__global__ void computeGeoCentersKernel(const KeyType* prefixes,
                                        TreeNodeIndex numNodes,
                                        Vec3<T>* centers,
                                        Vec3<T>* sizes,
                                        __grid_constant__ const Box<T> box)
{
    TreeNodeIndex i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numNodes) { return; }

    KeyType prefix   = prefixes[i];
    KeyType startKey = decodePlaceholderBit(prefix);
    unsigned level   = decodePrefixLength(prefix) / 3;
    auto nodeBox     = sfcIBox(sfcKey<KeyType>(startKey), level, box.getBoxDimBits(maxTreeLevel<KeyType>{}));
    util::tie(centers[i], sizes[i]) = centerAndSize<KeyType>(nodeBox, box);
}

template<class KeyType, class T>
void computeGeoCentersGpu(execution::Gpu exec,
                          const KeyType* prefixes,
                          TreeNodeIndex numNodes,
                          Vec3<T>* centers,
                          Vec3<T>* sizes,
                          const Box<T>& box)
{
    unsigned numThreads = 256;
    unsigned numBlocks  = iceil(numNodes, numThreads);
    computeGeoCentersKernel<<<numBlocks, numThreads, 0, exec>>>(prefixes, numNodes, centers, sizes, box);
}

#define GEO_CENTERS_GPU(KeyType, T)                                                                                    \
    template void computeGeoCentersGpu(execution::Gpu, const KeyType* prefixes, TreeNodeIndex numNodes,                \
                                       Vec3<T>* centers, Vec3<T>* sizes, const Box<T>& box)
GEO_CENTERS_GPU(uint32_t, float);
GEO_CENTERS_GPU(uint32_t, double);
GEO_CENTERS_GPU(uint64_t, float);
GEO_CENTERS_GPU(uint64_t, double);

template<class KeyType, class T>
__global__ void geoMacSpheresKernel(const KeyType* prefixes,
                                    TreeNodeIndex numNodes,
                                    SourceCenterType<T>* centers,
                                    float invTheta,
                                    __grid_constant__ const Box<T> box)
{
    TreeNodeIndex i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numNodes) { return; }
    centers[i] = computeMinMacR2(prefixes[i], invTheta, box);
}

//! @brief set @p centers to geometric node centers with Mac radius l * invTheta
template<class KeyType, class T>
void geoMacSpheresGpu(execution::Gpu exec,
                      const KeyType* prefixes,
                      TreeNodeIndex numNodes,
                      SourceCenterType<T>* centers,
                      float invTheta,
                      const Box<T>& box)
{
    unsigned numThreads = 256;
    unsigned numBlocks  = iceil(numNodes, numThreads);
    geoMacSpheresKernel<<<numBlocks, numThreads, 0, exec>>>(prefixes, numNodes, centers, invTheta, box);
}

#define GEO_MAC_SPHERES_GPU(KeyType, T)                                                                                \
    template void geoMacSpheresGpu(execution::Gpu, const KeyType* prefixes, TreeNodeIndex numNodes,                    \
                                   SourceCenterType<T>* centers, float invTheta, const Box<T>& box)
GEO_MAC_SPHERES_GPU(uint32_t, float);
GEO_MAC_SPHERES_GPU(uint32_t, double);
GEO_MAC_SPHERES_GPU(uint64_t, float);
GEO_MAC_SPHERES_GPU(uint64_t, double);

template<class KeyType, class T>
__global__ void setMacKernel(const KeyType* prefixes,
                             TreeNodeIndex numNodes,
                             Vec4<T>* macSpheres,
                             float invTheta,
                             __grid_constant__ const Box<T> box)
{
    TreeNodeIndex i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numNodes) { return; }

    Vec4<T> center   = macSpheres[i];
    T mac            = computeVecMacR2(prefixes[i], util::makeVec3(center), invTheta, box);
    macSpheres[i][3] = (center[3] != T(0)) ? mac : T(0);
}

template<class KeyType, class T>
void setMacGpu(execution::Gpu exec,
               const KeyType* prefixes,
               TreeNodeIndex numNodes,
               Vec4<T>* macSpheres,
               float invTheta,
               const Box<T>& box)
{
    unsigned numThreads = 256;
    unsigned numBlocks  = iceil(numNodes, numThreads);
    setMacKernel<<<numBlocks, numThreads, 0, exec>>>(prefixes, numNodes, macSpheres, invTheta, box);
}

#define SET_MAC_GPU(KeyType, T)                                                                                        \
    template void setMacGpu(execution::Gpu, const KeyType* prefixes, TreeNodeIndex numNodes, Vec4<T>* macSpheres,      \
                            float invTheta, const Box<T>& box)

SET_MAC_GPU(uint32_t, float);
SET_MAC_GPU(uint64_t, float);
SET_MAC_GPU(uint32_t, double);
SET_MAC_GPU(uint64_t, double);

template<int TPL, class Tm, class Tf>
__global__ void computeLeafVelocitySummaryKernel(const Tm* w,
                                                 const Tm* m,
                                                 const TreeNodeIndex* leafToInternal,
                                                 TreeNodeIndex numLeaves,
                                                 const LocalIndex* layout,
                                                 VelocitySummary<Tf>* summaries)
{
    TreeNodeIndex tid     = blockIdx.x * blockDim.x + threadIdx.x;
    TreeNodeIndex leafIdx = tid / TPL;

    VelocitySummary<Tf> s = emptyVelocitySummary<Tf>();
    if (leafIdx < numLeaves)
    {
        auto first = layout[leafIdx];
        auto last  = layout[leafIdx + 1];

        for (LocalIndex i = first + threadIdx.x % TPL; i < last; i += TPL)
        {
            addVelocitySource(s, Tf(m[i]), Tf(w[i]));
        }
    }

#pragma unroll
    for (int offset = 1; offset < TPL; offset *= 2)
    {
        VelocitySummary<Tf> o{shflDownSync(s[0], offset), shflDownSync(s[1], offset), shflDownSync(s[2], offset),
                              shflDownSync(s[3], offset)};
        s = combineVelocitySummary(s, o);
    }

    if (tid % TPL == 0 && leafIdx < numLeaves) { summaries[leafToInternal[leafIdx]] = s; }
}

template<class Tm, class Tf>
void computeLeafVelocitySummaryGpu(execution::Gpu exec,
                                   const Tm* w,
                                   const Tm* m,
                                   const TreeNodeIndex* leafToInternal,
                                   TreeNodeIndex numLeaves,
                                   const LocalIndex* layout,
                                   VelocitySummary<Tf>* summaries)
{
    constexpr int threadsPerLeaf = 8;
    unsigned numThreads          = 256;
    unsigned numBlocks           = iceil(threadsPerLeaf * numLeaves, numThreads);

    if (numBlocks == 0) { return; }
    computeLeafVelocitySummaryKernel<threadsPerLeaf>
        <<<numBlocks, numThreads, 0, exec>>>(w, m, leafToInternal, numLeaves, layout, summaries);
}

#define COMPUTE_LEAF_VELOCITY_SUMMARY_GPU(Tm, Tf)                                                                      \
    template void computeLeafVelocitySummaryGpu(execution::Gpu, const Tm*, const Tm*, const TreeNodeIndex*,           \
                                                TreeNodeIndex, const LocalIndex*, VelocitySummary<Tf>*);

COMPUTE_LEAF_VELOCITY_SUMMARY_GPU(double, double);
COMPUTE_LEAF_VELOCITY_SUMMARY_GPU(float, double);
COMPUTE_LEAF_VELOCITY_SUMMARY_GPU(float, float);

template<class T>
__global__ void upsweepVelocitySummaryKernel(TreeNodeIndex firstCell,
                                             TreeNodeIndex lastCell,
                                             const TreeNodeIndex* childOffsets,
                                             VelocitySummary<T>* summaries)
{
    const int cellIdx = blockIdx.x * blockDim.x + threadIdx.x + firstCell;
    if (cellIdx >= lastCell) return;

    TreeNodeIndex firstChild = childOffsets[cellIdx];

    if (firstChild) { summaries[cellIdx] = CombineVelocitySummary<T>{}(cellIdx, firstChild, summaries); }
}

template<class T>
void upsweepVelocitySummaryGpu(execution::Gpu exec,
                               int numLevels,
                               const TreeNodeIndex* levelRange,
                               const TreeNodeIndex* childOffsets,
                               VelocitySummary<T>* summaries)
{
    constexpr int numThreads = 256;

    for (int level = numLevels - 1; level >= 0; level--)
    {
        int numCellsLevel = levelRange[level + 1] - levelRange[level];
        int numBlocks     = (numCellsLevel - 1) / numThreads + 1;
        if (numCellsLevel)
        {
            upsweepVelocitySummaryKernel<<<numBlocks, numThreads, 0, exec>>>(levelRange[level], levelRange[level + 1],
                                                                             childOffsets, summaries);
        }
    }
}

template void
upsweepVelocitySummaryGpu(execution::Gpu, int, const TreeNodeIndex*, const TreeNodeIndex*, VelocitySummary<float>*);
template void
upsweepVelocitySummaryGpu(execution::Gpu, int, const TreeNodeIndex*, const TreeNodeIndex*, VelocitySummary<double>*);

template<class T>
__global__ void scaleMacsVelocityKernel(TreeNodeIndex numNodes,
                                        const VelocitySummary<T>* summaries,
                                        __grid_constant__ const VelocityMac<T> params,
                                        Vec4<T>* macSpheres)
{
    TreeNodeIndex i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numNodes) { return; }
    macSpheres[i][3] = scaleVelocityMac(macSpheres[i][3], summaries[i], params);
}

template<class T>
void scaleMacsVelocityGpu(execution::Gpu exec,
                          TreeNodeIndex numNodes,
                          const VelocitySummary<T>* summaries,
                          const VelocityMac<T>& params,
                          Vec4<T>* macSpheres)
{
    unsigned numThreads = 256;
    unsigned numBlocks  = iceil(numNodes, numThreads);
    if (numBlocks == 0) { return; }
    scaleMacsVelocityKernel<<<numBlocks, numThreads, 0, exec>>>(numNodes, summaries, params, macSpheres);
}

template void scaleMacsVelocityGpu(execution::Gpu, TreeNodeIndex, const VelocitySummary<float>*,
                                   const VelocityMac<float>&, Vec4<float>*);
template void scaleMacsVelocityGpu(execution::Gpu, TreeNodeIndex, const VelocitySummary<double>*,
                                   const VelocityMac<double>&, Vec4<double>*);

template<class T>
__global__ void velocityMacStatsKernel(TreeNodeIndex numNodes,
                                       const VelocitySummary<T>* summaries,
                                       __grid_constant__ const VelocityMac<T> params,
                                       unsigned long long* counts)
{
    TreeNodeIndex i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numNodes || !(summaries[i][1] > T(0))) { return; }
    T scale = velocityMacScale2(summaries[i], params);
    if (scale < T(0)) { atomicAdd(counts, 1ull); }
    if (scale > T(1)) { atomicAdd(counts + 1, 1ull); }
}

template<class T>
util::array<uint64_t, 2> velocityMacStatsGpu(execution::Gpu exec,
                                             TreeNodeIndex numNodes,
                                             const VelocitySummary<T>* summaries,
                                             const VelocityMac<T>& params)
{
    unsigned long long* d_counts;
    unsigned long long  counts[2] = {0, 0};
    checkGpuErrors(cudaMallocAsync(&d_counts, sizeof(counts), exec));
    checkGpuErrors(cudaMemcpyAsync(d_counts, counts, sizeof(counts), cudaMemcpyHostToDevice, exec));
    unsigned numThreads = 256;
    unsigned numBlocks  = iceil(numNodes, numThreads);
    if (numBlocks > 0) { velocityMacStatsKernel<<<numBlocks, numThreads, 0, exec>>>(numNodes, summaries, params, d_counts); }
    checkGpuErrors(cudaMemcpyAsync(counts, d_counts, sizeof(counts), cudaMemcpyDeviceToHost, exec));
    checkGpuErrors(cudaFreeAsync(d_counts, exec));
    checkGpuErrors(cudaStreamSynchronize(exec));
    return {counts[0], counts[1]};
}

template util::array<uint64_t, 2> velocityMacStatsGpu(execution::Gpu, TreeNodeIndex, const VelocitySummary<float>*,
                                                      const VelocityMac<float>&);
template util::array<uint64_t, 2> velocityMacStatsGpu(execution::Gpu, TreeNodeIndex, const VelocitySummary<double>*,
                                                      const VelocityMac<double>&);

template<class T>
__global__ void moveCentersKernel(const Vec3<T>* src, TreeNodeIndex numNodes, Vec4<T>* dest)
{
    TreeNodeIndex i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numNodes) { return; }
    dest[i][0] = src[i][0];
    dest[i][1] = src[i][1];
    dest[i][2] = src[i][2];
    dest[i][3] = 1.0;
}

template<class T>
void moveCenters(execution::Gpu exec, const Vec3<T>* src, TreeNodeIndex numNodes, Vec4<T>* dest)
{
    unsigned numThreads = 256;
    unsigned numBlocks  = iceil(numNodes, numThreads);
    moveCentersKernel<<<numBlocks, numThreads, 0, exec>>>(src, numNodes, dest);
}

template void moveCenters(execution::Gpu, const Vec3<double>*, TreeNodeIndex, Vec4<double>*);
template void moveCenters(execution::Gpu, const Vec3<float>*, TreeNodeIndex, Vec4<float>*);

} // namespace cstone
