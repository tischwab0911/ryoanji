/*
 * Cornerstone octree
 *
 * Copyright (c) 2024 CSCS, ETH Zurich
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: MIT License
 */

/*! @file
 * @brief Utilities for Thrust device vectors for use in .cu translation units only
 */

#pragma once

#include <thrust/device_vector.h>
#include <thrust/universal_vector.h>

#if defined(__HIPCC__)
#include <thrust/system/hip/execution_policy.h>
#else
#include <thrust/system/cuda/execution_policy.h>
#endif

#include "cuda_runtime.hpp"
#include "cstone/execution.hpp"

namespace cstone
{

template<class T, class Alloc>
T* rawPtr(thrust::device_vector<T, Alloc>& p)
{
    return thrust::raw_pointer_cast(p.data());
}

template<class T, class Alloc>
const T* rawPtr(const thrust::device_vector<T, Alloc>& p)
{
    return thrust::raw_pointer_cast(p.data());
}

template<class T, class Alloc>
T* rawPtr(thrust::universal_vector<T, Alloc>& p)
{
    return thrust::raw_pointer_cast(p.data());
}

template<class T, class Alloc>
const T* rawPtr(const thrust::universal_vector<T, Alloc>& p)
{
    return thrust::raw_pointer_cast(p.data());
}

/*! @brief thrust execution policy on the stream of @p exec
 *
 * IPPL/OPAL-X fork patch (keep on sphexa syncs): par, not sphexa #651's par_nosync. With par_nosync the
 * multi-rank domain synchronisation of IPPL's NBody solver took 1.7x as long (4 GH200, one node: SwissFEL
 * X = 64 gun section FRAME 49.9 -> 68.9 s space charge, syncGrav 20 -> 34 s; 4-rank unit tests 2x), with
 * host-staged and with GPU-aware MPI alike (work/sync-ab, 2026-10-29).
 */
inline auto thrustExecPolicy(execution::Gpu exec)
{
#if defined(__HIPCC__)
    return thrust::hip::par.on(exec);
#else
    return thrust::cuda::par.on(exec);
#endif
}

} // namespace cstone
