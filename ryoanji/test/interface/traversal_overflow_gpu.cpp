/*
 * Ryoanji N-body solver
 *
 * Copyright (c) 2026 CSCS, ETH Zurich
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: MIT License
 */

/*! @file
 * @brief A target group that overflows the per-warp traversal stack is split and traversed again
 *
 * Sources: 4M Gaussian particles. Targets: one group of 64 points on a grid filling the cloud, whose bounding box
 * overlaps every cell: no cell passes the MAC and the queue holds whole tree levels (~260k leaves), far beyond the
 * cell queue of a warp. A thin line would not do: only cells in a tube around it are opened. The traversal must
 * retry the overflowing group in pieces and return the forces of a direct sum within the Barnes-Hut accuracy.
 */

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>

#include <mpi.h>

#define USE_CUDA
#include "cstone/cuda/cuda_utils.cuh"
#include "cstone/cuda/device_vector.h"
#include "cstone/domain/domain.hpp"
#include "coord_samples/random.hpp"

#include "ryoanji/interface/multipole_holder.cuh"
#include "ryoanji/nbody/cartesian_qpole.hpp"
#include "ryoanji/nbody/kernel.hpp"

using namespace ryoanji;

template<class T, class KeyType>
static int overflowTest(int thisRank, int numRanks)
{
    using MultipoleType           = CartesianQuadrupole<T>;
    const LocalIndex numParticles = 1u << 22;
    const unsigned   bucketSize   = 16;  // ~260k leaves: a group opening everything queues ~4x the cell queue
    const float      theta        = 0.5;
    const T          G            = 1.0;
    const T          softening    = 1e-6;  // far below the spacing: the reference is the plain Coulomb sum

    cstone::Box<T>                                                 box(-1, 1, cstone::BoundaryType::fixed);
    cstone::RandomGaussianCoordinates<T, cstone::SfcKind<KeyType>> coords(numParticles, box);

    std::vector<T>       x(coords.x().begin(), coords.x().end()), y(coords.y().begin(), coords.y().end()),
        z(coords.z().begin(), coords.z().end());
    std::vector<T>       h(numParticles, softening), m(numParticles, T(1) / numParticles);
    std::vector<KeyType> h_keys(numParticles);

    cstone::Domain<KeyType, T, cstone::execution::Gpu> domain(cstone::execution::gpuDefaultStream, thisRank, numRanks,
                                                              bucketSize, bucketSize, theta, MPI_COMM_WORLD, box);
    MultipoleHolder<T, T, T, T, T, KeyType, MultipoleType> holder;

    cstone::DeviceVector<KeyType> d_keys = h_keys;
    cstone::DeviceVector<T>       d_x = x, d_y = y, d_z = z, d_h = h, d_m = m;
    cstone::DeviceVector<T>       s1, s2, s3;
    domain.syncGrav(d_keys, d_x, d_y, d_z, d_h, d_m, std::tuple{}, std::tie(s1, s2, s3));
    holder.upsweep(rawPtr(d_x), rawPtr(d_y), rawPtr(d_z), rawPtr(d_m), domain.globalTree(), domain.focusTree(),
                   domain.layout().data());

    // group 0: a 4^3 grid over [-0.9, 0.9]^3; group 1: compact
    const int      numTargets = 128;
    std::vector<T> xt(numTargets), yt(numTargets), zt(numTargets), ht(numTargets, softening), mt(numTargets, 0);
    for (int i = 0; i < 64; ++i)
    {
        xt[i] = T(-0.9) + T(0.6) * (i % 4);
        yt[i] = T(-0.9) + T(0.6) * ((i / 4) % 4);
        zt[i] = T(-0.9) + T(0.6) * (i / 16);
    }
    for (int i = 64; i < numTargets; ++i)
    {
        xt[i] = T(0.1) + T(1e-3) * (i - 64) / 63;
        yt[i] = T(0.1);
        zt[i] = T(0.1);
    }
    cstone::DeviceVector<T>          d_xt = xt, d_yt = yt, d_zt = zt, d_ht = ht, d_mt = mt;
    cstone::DeviceVector<LocalIndex> d_bounds = std::vector<LocalIndex>{0, 64, numTargets};
    cstone::GroupView grp{0, LocalIndex(numTargets), 2, rawPtr(d_bounds), rawPtr(d_bounds) + 1};

    cstone::DeviceVector<T> d_ax(numTargets, 0), d_ay(numTargets, 0), d_az(numTargets, 0);
    holder.compute(grp, rawPtr(d_xt), rawPtr(d_yt), rawPtr(d_zt), rawPtr(d_mt), rawPtr(d_ht), rawPtr(d_x),
                   rawPtr(d_y), rawPtr(d_z), rawPtr(d_m), rawPtr(d_h), G, 0, box, nullptr, rawPtr(d_ax), rawPtr(d_ay),
                   rawPtr(d_az));
    const auto     stats   = holder.readStats();
    const uint64_t retried = holder.retriedGroups();

    std::vector<T> ax = toHost(d_ax), ay = toHost(d_ay), az = toHost(d_az);

    // direct sum with the same P2P kernel
    std::vector<T> xs = toHost(d_x), ys = toHost(d_y), zs = toHost(d_z), ms = toHost(d_m), hs = toHost(d_h);
    double maxError = 0;
    for (int i = 0; i < numTargets; ++i)
    {
        Vec4<T> acc{0, 0, 0, 0};
        for (LocalIndex j = domain.startIndex(); j < domain.endIndex(); ++j)
        {
            acc = P2P(acc, Vec3<T>{xt[i], yt[i], zt[i]}, Vec3<T>{xs[j], ys[j], zs[j]}, ms[j], ht[i], hs[j]);
        }
        Vec3<T> ref{G * acc[1], G * acc[2], G * acc[3]}, probe{ax[i], ay[i], az[i]};
        maxError = std::max(maxError, double(std::sqrt(norm2(ref - probe) / norm2(ref))));
    }

    std::cout << "retried groups " << retried << ", maxP2P " << stats[1] << ", max stack " << stats[4]
              << ", max relative error " << maxError << std::endl;

    bool pass = retried > 0             // the grid group overflowed and was split
                && stats[1] != 0xFFFFFFFF // no single target overflowed
                && maxError < 1e-2;
    return pass ? EXIT_SUCCESS : EXIT_FAILURE;
}

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);
    int rank = 0, numRanks = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numRanks);

    int fail = overflowTest<double, uint64_t>(rank, numRanks);

    MPI_Finalize();
    return fail;
}
