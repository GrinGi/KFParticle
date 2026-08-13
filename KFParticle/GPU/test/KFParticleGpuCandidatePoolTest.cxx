/*
 * This file is part of KFParticle package
 * Copyright (C) 2007-2026
 *
 * KFParticle is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 */

#include "KFParticleGpuParity.h"

#include <cassert>
#include <type_traits>
#include <vector>

namespace
{
  void TestCandidatePool()
  {
    const unsigned int capacity = 4;
    std::vector<float> parameters(KFParticleGpuFitState::NumberOfParameters * capacity, -1.f);
    std::vector<float> covariances(KFParticleGpuFitState::NumberOfCovarianceElements * capacity, -1.f);
    std::vector<float> fitScalars(KFParticleGpuFitSoALayout::NumberOfFloatComponents * capacity, -1.f);
    std::vector<int> fitIntegers(KFParticleGpuFitSoALayout::NumberOfIntegerComponents * capacity, -1);
    std::vector<int> metadataIntegers(
      KFParticleGpuCandidateMetadataLayout::NumberOfIntegerComponents * capacity, -1);
    std::vector<unsigned int> metadataUnsigned(
      KFParticleGpuCandidateMetadataLayout::NumberOfUnsignedComponents * capacity, 0);

    const unsigned int daughterCapacity = 12;
    std::vector<int> daughterSourceIds(daughterCapacity, -1);
    unsigned int candidateSize = 0;
    unsigned int daughterSize = 0;
    unsigned int overflowFlags = 0;

    KFParticleGpuFitSoAView fit(&parameters[0],
                                &covariances[0],
                                &fitScalars[0],
                                &fitIntegers[0],
                                capacity,
                                capacity);
    KFParticleGpuCandidateMetadataSoAView metadata(
      &metadataIntegers[0], &metadataUnsigned[0], capacity);
    KFParticleGpuDaughterStorageView daughters(
      &daughterSourceIds[0], &daughterSize, daughterCapacity);
    KFParticleGpuCandidatePoolView pool(
      fit, metadata, daughters, &candidateSize, &overflowFlags, capacity);

    KFParticleGpuFitState state;
    state.X() = 1.f;
    state.Px() = 2.f;
    state.Chi2() = 3.f;
    state.NDF() = 4;
    assert(StoreCandidateFit(state, pool, 1));

    pool.Metadata().Pdg(1) = 3122;
    pool.Metadata().PrimaryVertexIndex(1) = -1;
    pool.Metadata().EventIndex(1) = 7;
    pool.Metadata().DaughterOffset(1) = 3;
    pool.Metadata().DaughterCount(1) = 2;
    pool.Metadata().Flags(1) = 0;
    pool.Metadata().ChannelId(1) = 17u;
    pool.Metadata().Topology(1) = KFGpuGraphTopologyCompositeTrack;
    pool.Metadata().OutputClass(1) = KFGpuGraphOutputSecondary;
    pool.Metadata().OperationStatus(1) = KFGpuCandidateOperationAccepted;
    pool.Metadata().DirectDaughterCount(1) = 2u;
    SetCandidateDirectDaughter(
      pool.Metadata(), 1u, 0u, KFGpuDirectDaughterCandidate, 0u);
    SetCandidateDirectDaughter(
      pool.Metadata(), 1u, 1u, KFGpuDirectDaughterInputTrack, 3u);
    pool.Daughters().SourceId(3) = 101;
    pool.Daughters().SourceId(4) = 205;
    candidateSize = 2;
    daughterSize = 5;

    const KFParticleGpuConstCandidatePoolView constPool = MakeConstView(pool);
    KFParticleGpuFitState loaded;
    LoadCandidateFit(constPool, 1, loaded);

    assert(loaded.X() == 1.f);
    assert(loaded.Px() == 2.f);
    assert(loaded.Chi2() == 3.f);
    assert(loaded.NDF() == 4);
    assert(constPool.Size() == 2);
    assert(constPool.Metadata().Pdg(1) == 3122);
    assert(constPool.Metadata().PrimaryVertexIndex(1) == -1);
    assert(constPool.Metadata().EventIndex(1) == 7);
    assert(constPool.Metadata().DaughterOffset(1) == 3);
    assert(constPool.Metadata().DaughterCount(1) == 2);
    assert(constPool.Metadata().Topology(1) == KFGpuGraphTopologyCompositeTrack);
    assert(constPool.Metadata().OutputClass(1) == KFGpuGraphOutputSecondary);
    assert(constPool.Metadata().OperationStatus(1) == KFGpuCandidateOperationAccepted);
    assert(constPool.Metadata().DirectDaughterCount(1) == 2u);
    assert(constPool.Metadata().DirectFirstKind(1) == KFGpuDirectDaughterCandidate);
    assert(constPool.Metadata().DirectFirstIndex(1) == 0u);
    assert(constPool.Metadata().DirectSecondKind(1) == KFGpuDirectDaughterInputTrack);
    assert(constPool.Metadata().DirectSecondIndex(1) == 3u);
    assert(constPool.Daughters().SourceId(3) == 101);
    assert(constPool.Daughters().SourceId(4) == 205);
    assert(constPool.Daughters().Size() == 5);

    KFParticleGpuParitySnapshot snapshot;
    assert(KFParticleGpuParity::BuildSnapshot(
      constPool, 1u, 91u, nullptr, false, snapshot));
    assert(snapshot.key.eventId == 91u);
    assert(snapshot.key.channelId == 17u);
    assert(snapshot.key.lineageSize == 2u);
    assert(snapshot.key.lineage[0] == 101);
    assert(snapshot.key.lineage[1] == 205);
    assert(snapshot.topology == KFGpuGraphTopologyCompositeTrack);
    assert(snapshot.outputClass == KFGpuGraphOutputSecondary);
    assert(snapshot.operationStatus == KFGpuCandidateOperationAccepted);

    // Component-major metadata keeps adjacent candidates adjacent in memory.
    assert(metadataIntegers[KFParticleGpuCandidateMetadataLayout::Pdg * capacity + 1] == 3122);
    assert(metadataUnsigned[
             KFParticleGpuCandidateMetadataLayout::DaughterOffset * capacity + 1] == 3);

    assert(pool.CanStoreCandidate(capacity - 1));
    assert(!pool.CanStoreCandidate(capacity));
    assert(pool.Daughters().CanStore(10, 2));
    assert(!pool.Daughters().CanStore(11, 2));
    assert(!StoreCandidateFit(state, pool, capacity));

    overflowFlags = CandidateCapacityExceeded | DaughterCapacityExceeded;
    assert(constPool.OverflowFlags() ==
           static_cast<unsigned int>(CandidateCapacityExceeded | DaughterCapacityExceeded));
  }
}

int main()
{
  static_assert(std::is_trivially_copyable<KFParticleGpuCandidateMetadataSoAView>::value,
                "Candidate metadata view must be trivially copyable");
  static_assert(std::is_trivially_copyable<KFParticleGpuDaughterStorageView>::value,
                "Daughter storage view must be trivially copyable");
  static_assert(std::is_trivially_copyable<KFParticleGpuCandidatePoolView>::value,
                "Candidate pool view must be trivially copyable");
  static_assert(std::is_trivially_copyable<KFParticleGpuConstCandidatePoolView>::value,
                "Const candidate pool view must be trivially copyable");

  TestCandidatePool();
  return 0;
}
