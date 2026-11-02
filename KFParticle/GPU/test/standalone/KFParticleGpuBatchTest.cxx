/* Copyright (C) 2026 GSI Helmholtzzentrum fuer Schwerionenforschung, Darmstadt
   SPDX-License-Identifier: GPL-3.0-only */

#include "KFParticleGpuTestFixture.h"

using namespace kfp_gpu_test;

TEST(KFParticleGpuHostInterfaceTest, DefaultBatchLimitsCoverMaximumGraphLineage)
{
  constexpr unsigned int events = 1024u;
  const auto limits = ::kfparticle::gpu::DefaultGpuBatchLimits(events);

  EXPECT_EQ(limits.events, events);
  EXPECT_EQ(limits.candidates, 16777216u);
  EXPECT_EQ(limits.twoDaughterTasks, limits.candidates);
  EXPECT_EQ(limits.selectedCandidates, limits.candidates);
  EXPECT_EQ(limits.daughterIds,
            limits.candidates * KFParticleGpuV0TrackLineage::MaximumSize);
}

TEST(KFParticleGpuHostInterfaceTest, BuildsAndWritesBoundedMultiEventBatch)
{
  Fixture k0Short;
  Fixture lambda;
  Fixture antiLambda;
  lambda.first.SetPDG(2212, 0u);
  lambda.last.SetPDG(2212, 0u);
  antiLambda.first.SetPDG(-2212, 1u);
  antiLambda.last.SetPDG(-2212, 1u);
  lambda.first.SetId(201, 0u);
  lambda.last.SetId(201, 0u);
  antiLambda.first.SetId(302, 1u);
  antiLambda.last.SetId(302, 1u);

  // Lambda routing deliberately scans the complete secondary-positive range:
  // the source PDG decides between the nominal proton and alternate pion mass.
  const auto k0Capacity = FullPlanCapacity(k0Short, 81u);
  const auto lambdaCapacity = FullPlanCapacity(lambda, 82u);
  const auto antiLambdaCapacity = FullPlanCapacity(antiLambda, 83u);
  const unsigned int k0Tasks = k0Capacity.taskCapacity;
  const unsigned int lambdaTasks = lambdaCapacity.taskCapacity;
  const unsigned int antiLambdaTasks = antiLambdaCapacity.taskCapacity;
  const unsigned int totalTasks = k0Tasks + lambdaTasks + antiLambdaTasks;
  const auto totalCapacity = SumCapacity(
    {k0Capacity, lambdaCapacity, antiLambdaCapacity});
  EXPECT_GT(k0Tasks, 4u);
  ::kfparticle::gpu::GpuBatchBuilder builder(
    BatchLimits(3u, 30u, 3u, totalCapacity));
  EXPECT_EQ(builder.TryAppend(1001u, k0Short.Input(81u)), ::kfparticle::gpu::GpuBatchAppendResult::Added);
  EXPECT_EQ(builder.TryAppend(1002u, lambda.Input(82u)), ::kfparticle::gpu::GpuBatchAppendResult::Added);
  EXPECT_EQ(builder.TryAppend(1003u, antiLambda.Input(83u)), ::kfparticle::gpu::GpuBatchAppendResult::Added);

  auto batch = builder.Seal();
  EXPECT_EQ(batch.EventCount(), 3u);
  EXPECT_EQ(batch.TrackCount(), 30u);
  EXPECT_EQ(batch.VertexCount(), 3u);
  EXPECT_EQ(batch.TaskCapacity(), totalTasks);
  EXPECT_EQ(batch.Capacities().candidates, totalCapacity.candidates);
  EXPECT_EQ(batch.Capacities().daughterIds, totalCapacity.daughterIds);
  EXPECT_EQ(batch.Event(0u).taskCapacity, k0Tasks);
  EXPECT_EQ(batch.Event(1u).taskCapacity, lambdaTasks);
  EXPECT_EQ(batch.Event(2u).taskCapacity, antiLambdaTasks);
  ExpectEventCapacity(batch.Event(0u), k0Capacity);
  ExpectEventCapacity(batch.Event(1u), lambdaCapacity);
  ExpectEventCapacity(batch.Event(2u), antiLambdaCapacity);
  EXPECT_GE(batch.Event(0u).capacityPlanSetupMilliseconds, 0.);
  EXPECT_EQ(batch.Event(0u).capacityPlanBuilds, 1u);
  EXPECT_EQ(batch.Event(1u).capacityPlanSetupMilliseconds, 0.);
  EXPECT_EQ(batch.Event(1u).capacityPlanBuilds, 0u);
  EXPECT_EQ(batch.Event(2u).capacityPlanSetupMilliseconds, 0.);
  EXPECT_EQ(batch.Event(2u).capacityPlanBuilds, 0u);
  EXPECT_TRUE(batch.Event(0u).validatedPublicationInput);
  EXPECT_TRUE(batch.Event(1u).validatedPublicationInput);
  EXPECT_TRUE(batch.Event(2u).validatedPublicationInput);
  EXPECT_GE(batch.Event(0u).capacityPlanningMilliseconds,
            batch.Event(0u).capacityPlanSetupMilliseconds);
  EXPECT_EQ(batch.Event(1u).sourceEventId, 1002u);
  EXPECT_EQ(batch.Event(1u).trackOffset, 10u);
  EXPECT_EQ(batch.Event(2u).vertexOffset, 2u);
  EXPECT_EQ(batch.Event(2u).descriptor.TrackSet(SecondaryNegativeFirst).Species(Proton).size, 1u);

  HostStorage serialStorage;
  std::array<KFParticleGpuEventDesc, 3u> serialEvents;
  EXPECT_EQ(batch.Write(
              serialStorage.Tracks(batch.TrackCount()),
              serialStorage.Vertices(batch.VertexCount()),
              serialEvents.data(), 1u),
            1u);

  HostStorage storage;
  std::array<KFParticleGpuEventDesc, 3u> events;
  const unsigned int packingWorkers = batch.Write(
    storage.Tracks(batch.TrackCount()), storage.Vertices(batch.VertexCount()),
    events.data(), 4u);

  EXPECT_GE(packingWorkers, 1u);
  EXPECT_LE(packingWorkers, 3u);

  EXPECT_EQ(storage.parameters, serialStorage.parameters);
  EXPECT_EQ(storage.covariances, serialStorage.covariances);
  EXPECT_EQ(storage.field, serialStorage.field);
  EXPECT_EQ(storage.chi, serialStorage.chi);
  EXPECT_EQ(storage.integers, serialStorage.integers);
  EXPECT_EQ(storage.vertexParameters, serialStorage.vertexParameters);
  EXPECT_EQ(storage.vertexCovariances, serialStorage.vertexCovariances);
  EXPECT_EQ(storage.vertexChi2, serialStorage.vertexChi2);
  EXPECT_EQ(storage.vertexIntegers, serialStorage.vertexIntegers);

  EXPECT_EQ(events[0].eventId, 81u);
  EXPECT_EQ(events[1].eventId, 82u);
  EXPECT_EQ(events[2].eventId, 83u);
  EXPECT_EQ(events[1].TrackSet(SecondaryPositiveFirst).Species(Proton).offset, 10u);
  EXPECT_EQ(events[2].TrackSet(SecondaryNegativeFirst).Species(Proton).offset, 22u);
  EXPECT_EQ(events[2].primaryVertices.offset, 2u);
  EXPECT_EQ(storage.Tracks(batch.TrackCount()).SourceId(10u), 201);
  EXPECT_EQ(storage.Tracks(batch.TrackCount()).SourceId(22u), 302);
  EXPECT_FLOAT_EQ(storage.Vertices(batch.VertexCount()).Parameter(0u, 2u), 1.f);
  const auto packedTracks = storage.Tracks(batch.TrackCount());
  for (std::size_t eventIndex = 0u; eventIndex < events.size(); ++eventIndex) {
    EXPECT_EQ(events[eventIndex].eventId, serialEvents[eventIndex].eventId);
    EXPECT_EQ(events[eventIndex].primaryVertices.offset,
              serialEvents[eventIndex].primaryVertices.offset);
    EXPECT_EQ(events[eventIndex].primaryVertices.size,
              serialEvents[eventIndex].primaryVertices.size);
    for (unsigned int set = 0u; set < NumberOfTrackSets; ++set) {
      const auto range = events[eventIndex].trackSets[set].tracks;
      for (unsigned int index = range.offset; index < range.End(); ++index) {
        const bool primary = set == PrimaryPositiveFirst || set == PrimaryNegativeFirst
                             || set == PrimaryPositiveLast || set == PrimaryNegativeLast;
        EXPECT_EQ(packedTracks.PrimaryVertexIndex(index),
                  primary ? static_cast<int>(batch.Event(eventIndex).vertexOffset) : -1);
      }
      EXPECT_EQ(events[eventIndex].trackSets[set].tracks.offset,
                serialEvents[eventIndex].trackSets[set].tracks.offset);
      EXPECT_EQ(events[eventIndex].trackSets[set].tracks.size,
                serialEvents[eventIndex].trackSets[set].tracks.size);
      for (unsigned int species = 0u; species < NumberOfTrackSpecies;
           ++species) {
        EXPECT_EQ(events[eventIndex].trackSets[set].species[species].offset,
                  serialEvents[eventIndex].trackSets[set].species[species].offset);
        EXPECT_EQ(events[eventIndex].trackSets[set].species[species].size,
                  serialEvents[eventIndex].trackSets[set].species[species].size);
      }
    }
  }
}

TEST(KFParticleGpuHostInterfaceTest, BuildsFlatEventTransactionContract)
{
  Fixture first;
  Fixture second;
  const auto firstCapacity = FullPlanCapacity(first, 101u);
  const auto secondCapacity = FullPlanCapacity(second, 102u);
  const unsigned int firstTasks = firstCapacity.taskCapacity;
  const unsigned int secondTasks = secondCapacity.taskCapacity;
  ::kfparticle::gpu::GpuBatchBuilder builder(
    BatchLimits(2u, 20u, 2u, SumCapacity({firstCapacity, secondCapacity})));
  ASSERT_EQ(builder.TryAppend(7001u, first.Input(101u)),
            ::kfparticle::gpu::GpuBatchAppendResult::Added);
  ASSERT_EQ(builder.TryAppend(7002u, second.Input(102u)),
            ::kfparticle::gpu::GpuBatchAppendResult::Added);
  auto batch = builder.Seal();

  std::array<KFParticleGpuTransactionDesc, 1u> transactions;
  std::array<KFParticleGpuDomainDesc, 2u> domains;
  std::array<KFParticleGpuWorkRange, 2u> workRanges;
  std::array<KFParticleGpuWorkStatus, 2u> workStatuses;
  std::array<KFParticleGpuDomainStatus, 2u> statuses;
  // Model reuse after a completed device batch. Preparation must replace the
  // previous execution-owned fields instead of inheriting stale channel ABI.
  transactions[0].channelCount = 99u;
  domains[0].channelStatuses = {77u, 99u};
  domains[1].channelStatuses = {176u, 99u};
  workRanges[0].secondarySourceTracks = {500u, 10u};
  workRanges[1].secondarySourceTracks = {510u, 10u};
  workRanges[0].sourceCandidates = {700u, 20u};
  workRanges[1].sourceCandidates = {720u, 20u};
  workStatuses[0].flags = KFParticleGpuWorkSourceResolved;
  workStatuses[1].flags = KFParticleGpuWorkSourceResolved;
  KFParticleGpuTransactionView view(transactions.data(),
                                    domains.data(),
                                    workRanges.data(),
                                    workStatuses.data(),
                                    statuses.data(),
                                    2u,
                                    2u,
                                    2u);
  batch.WriteTransaction(view, 17u, 9001u);

  EXPECT_EQ(transactions[0].transactionId, 9001u);
  EXPECT_EQ(transactions[0].decayPlanRevision, 17u);
  EXPECT_EQ(transactions[0].domains.size, 2u);
  EXPECT_EQ(transactions[0].channelCount, 0u);
  EXPECT_EQ(view.TimeDomains(), nullptr);
  EXPECT_EQ(view.TimeDomainCapacity(), 0u);
  EXPECT_EQ(domains[0].kind, KFParticleGpuEventDomain);
  EXPECT_EQ(domains[0].sourceId, 7001u);
  EXPECT_EQ(domains[1].inputDescriptorIndex, 1u);
  EXPECT_EQ(domains[1].sourceTracks.offset, batch.Event(1u).trackOffset);
  EXPECT_EQ(domains[0].channelStatuses.size, 0u);
  EXPECT_EQ(domains[1].channelStatuses.size, 0u);
  EXPECT_FALSE(KFParticleGpuRangesOverlap(domains[0].taskReservation,
                                          domains[1].taskReservation));
  EXPECT_EQ(workRanges[1].domainIndex, 1u);
  EXPECT_EQ(workRanges[1].family, KFParticleGpuDomainInputWork);
  EXPECT_EQ(workRanges[0].secondarySourceTracks.size, 0u);
  EXPECT_EQ(workRanges[1].secondarySourceTracks.size, 0u);
  EXPECT_EQ(workRanges[0].sourceCandidates.size, 0u);
  EXPECT_EQ(workRanges[1].sourceCandidates.size, 0u);
  EXPECT_EQ(workStatuses[0].flags, 0u);
  EXPECT_EQ(workStatuses[1].flags, 0u);
  EXPECT_EQ(KFParticleGpuValidateTransaction(view,
                                             17u,
                                             batch.TrackCount(),
                                             batch.VertexCount(),
                                             batch.EventCount(),
                                             batch.Capacities().twoDaughterTasks,
                                             batch.Capacities().candidates,
                                             batch.Capacities().daughterIds,
                                             batch.Capacities().selectedCandidates),
            KFParticleGpuTransactionValid);

  EXPECT_EQ(KFParticleGpuValidateTransaction(view,
                                             18u,
                                             batch.TrackCount(),
                                             batch.VertexCount(),
                                             batch.EventCount(),
                                             batch.Capacities().twoDaughterTasks,
                                             batch.Capacities().candidates,
                                             batch.Capacities().daughterIds,
                                             batch.Capacities().selectedCandidates),
            KFParticleGpuTransactionPlanRevisionMismatch);

  const auto routingCapacities = batch.RoutingCapacities();
  EXPECT_EQ(routingCapacities.tracks, batch.Capacities().tracks);
  EXPECT_EQ(routingCapacities.twoDaughterTasks, 0u);
  EXPECT_GT(batch.TaskCapacity(), 0u);
  EXPECT_EQ(routingCapacities.candidates, 1u);
  EXPECT_EQ(routingCapacities.daughterIds, 2u);
  EXPECT_EQ(routingCapacities.selectedCandidates, 1u);
  batch.WriteTransaction(view, 17u, 9002u, true);
  EXPECT_EQ(transactions[0].transactionId, 9002u);
  for (const auto& domain : domains) {
    EXPECT_NE(domain.flags & KFParticleGpuDomainCompactOutput, 0u);
    EXPECT_EQ(domain.candidateReservation.size, 0u);
    EXPECT_EQ(domain.daughterReservation.size, 0u);
    EXPECT_EQ(domain.selectedReservation.size, 0u);
  }
  EXPECT_EQ(KFParticleGpuValidateTransaction(view,
                                             17u,
                                             batch.TrackCount(),
                                             batch.VertexCount(),
                                             batch.EventCount(),
                                             batch.TaskCapacity(),
                                             routingCapacities.candidates,
                                             routingCapacities.daughterIds,
                                             routingCapacities.selectedCandidates),
            KFParticleGpuTransactionValid);

  domains[1].taskReservation = domains[0].taskReservation;
  EXPECT_EQ(KFParticleGpuValidateTransaction(view,
                                             17u,
                                             batch.TrackCount(),
                                             batch.VertexCount(),
                                             batch.EventCount(),
                                             batch.Capacities().twoDaughterTasks,
                                             batch.Capacities().candidates,
                                             batch.Capacities().daughterIds,
                                             batch.Capacities().selectedCandidates),
            KFParticleGpuTransactionOverlappingOutput);
}

TEST(KFParticleGpuHostInterfaceTest, BuildsExactTwoDaughterPairTiles)
{
  struct Case {
    unsigned int pairs;
    unsigned int tiles;
  };
  constexpr std::array<Case, 6u> cases{{
    {0u, 0u}, {1u, 1u}, {63u, 1u}, {64u, 1u}, {65u, 2u}, {99u, 2u}}};

  for (const Case& test : cases) {
    EXPECT_EQ(KFParticleGpuPairTileCount(test.pairs), test.tiles);
    std::array<KFParticleGpuPairTile, 2u> tiles{};
    unsigned int cursor = 0u;
    ASSERT_TRUE(KFParticleGpuAppendPairTiles(
      7u, test.pairs, tiles.data(), test.tiles, cursor));
    ASSERT_EQ(cursor, test.tiles);
    for (unsigned int tile = 0u; tile < test.tiles; ++tile) {
      EXPECT_EQ(tiles[tile].workIndex, 7u);
      EXPECT_EQ(tiles[tile].firstPair,
                tile * KFParticleGpuPairTileSize);
      EXPECT_LT(tiles[tile].firstPair, test.pairs);
    }
  }

  std::array<KFParticleGpuPairTile, 1u> insufficient{{{91u, 92u}}};
  unsigned int cursor = 0u;
  EXPECT_FALSE(KFParticleGpuAppendPairTiles(
    3u, 65u, insufficient.data(), insufficient.size(), cursor));
  EXPECT_EQ(cursor, 0u);
  EXPECT_EQ(insufficient[0].workIndex, 91u);
  EXPECT_EQ(insufficient[0].firstPair, 92u);
}

TEST(KFParticleGpuHostInterfaceTest, RejectsCrossEventTwoDaughterWork)
{
  std::array<KFParticleGpuTransactionDesc, 1u> transactions{};
  std::array<KFParticleGpuDomainDesc, 2u> domains{};
  std::array<KFParticleGpuWorkRange, 1u> workRanges{};
  std::array<KFParticleGpuWorkStatus, 1u> workStatuses{};
  std::array<KFParticleGpuDomainStatus, 2u> statuses{};

  transactions[0].decayPlanRevision = 17u;
  transactions[0].domains = {0u, 2u};
  transactions[0].workRanges = {0u, 1u};
  domains[0].inputDescriptorIndex = 0u;
  domains[0].sourceTracks = {0u, 10u};
  domains[0].workRanges = {0u, 1u};
  domains[0].statusIndex = 0u;
  domains[1].inputDescriptorIndex = 1u;
  domains[1].sourceTracks = {10u, 10u};
  domains[1].workRanges = {1u, 0u};
  domains[1].statusIndex = 1u;
  workRanges[0].domainIndex = 0u;
  workRanges[0].family = KFParticleGpuTwoDaughterWork;
  workRanges[0].sourceTracks = {0u, 2u};
  workRanges[0].secondarySourceTracks = {10u, 2u};

  KFParticleGpuTransactionView view(transactions.data(),
                                    domains.data(),
                                    workRanges.data(),
                                    workStatuses.data(),
                                    statuses.data(),
                                    domains.size(),
                                    workRanges.size(),
                                    statuses.size());
  EXPECT_EQ(KFParticleGpuValidateTransaction(
              view, 17u, 20u, 0u, 2u, 0u, 0u, 0u, 0u),
            KFParticleGpuTransactionInvalidDomain);

  workRanges[0].secondarySourceTracks = {2u, 2u};
  EXPECT_EQ(KFParticleGpuValidateTransaction(
              view, 17u, 20u, 0u, 2u, 0u, 0u, 0u, 0u),
            KFParticleGpuTransactionValid);
}

TEST(KFParticleGpuHostInterfaceTest, TransactionValidationUsesLinearOrderedRangeScans)
{
  constexpr unsigned int domains = 2048u;
  ValidationStorage storage(domains, false, 1u);
  KFParticleGpuTransactionValidationStats optimized;
  KFParticleGpuTransactionValidationStats reference;

  EXPECT_EQ(storage.Validate(false, &optimized), KFParticleGpuTransactionValid);
  EXPECT_EQ(storage.Validate(true, &reference), KFParticleGpuTransactionValid);
  EXPECT_EQ(optimized.rangeFamilies, 6u);
  EXPECT_EQ(optimized.sortedFallbackFamilies, 0u);
  EXPECT_EQ(optimized.domainPairComparisons, 0u);
  EXPECT_EQ(optimized.adjacentRangeComparisons,
            6ull * static_cast<unsigned long long>(domains - 1u));
  EXPECT_EQ(reference.domainPairComparisons,
            6ull * domains * (domains - 1ull) / 2ull);
}

TEST(KFParticleGpuHostInterfaceTest, TransactionValidationFallsBackForPermutedRanges)
{
  constexpr unsigned int domains = 257u;
  ValidationStorage storage(domains, true);
  KFParticleGpuTransactionValidationStats optimized;
  std::vector<unsigned int> sourceOffsets;
  sourceOffsets.reserve(storage.domains.size());
  for (const auto& domain : storage.domains) {
    sourceOffsets.push_back(domain.sourceTracks.offset);
  }

  EXPECT_EQ(storage.Validate(false, &optimized), storage.Validate(true));
  EXPECT_EQ(optimized.sortedFallbackFamilies, 6u);
  EXPECT_EQ(optimized.domainPairComparisons, 0u);
  EXPECT_EQ(optimized.adjacentRangeComparisons,
            6ull * static_cast<unsigned long long>(domains - 1u));
  for (std::size_t index = 0u; index < storage.domains.size(); ++index) {
    EXPECT_EQ(storage.domains[index].sourceTracks.offset, sourceOffsets[index]);
  }
}

TEST(KFParticleGpuHostInterfaceTest, TransactionValidationMatchesReferenceForEveryRangeFamily)
{
  using RangeMember = KFParticleGpuRange KFParticleGpuDomainDesc::*;
  const std::array<RangeMember, 6u> ranges{{
    &KFParticleGpuDomainDesc::sourceTracks,
    &KFParticleGpuDomainDesc::primaryVertices,
    &KFParticleGpuDomainDesc::taskReservation,
    &KFParticleGpuDomainDesc::candidateReservation,
    &KFParticleGpuDomainDesc::daughterReservation,
    &KFParticleGpuDomainDesc::selectedReservation}};

  for (const auto member : ranges) {
    ValidationStorage storage(5u, true);
    auto& first = storage.domains[0u];
    auto& second = storage.domains[2u];
    second.*member = first.*member;
    if (member == &KFParticleGpuDomainDesc::sourceTracks) {
      storage.workRanges[2u].sourceTracks = second.sourceTracks;
      storage.workRanges[2u].secondarySourceTracks = second.sourceTracks;
    }
    EXPECT_EQ(storage.Validate(false), KFParticleGpuTransactionOverlappingOutput);
    EXPECT_EQ(storage.Validate(false), storage.Validate(true));
  }

  ValidationStorage adjacent(3u);
  adjacent.domains[1u].candidateReservation =
    {adjacent.domains[0u].candidateReservation.offset
       + adjacent.domains[0u].candidateReservation.size,
     2u};
  adjacent.domains[2u].candidateReservation = {};
  EXPECT_EQ(adjacent.Validate(false), adjacent.Validate(true));
  EXPECT_EQ(adjacent.Validate(false), KFParticleGpuTransactionValid);

  ValidationStorage nested(4u);
  nested.domains[2u].daughterReservation =
    {nested.domains[0u].daughterReservation.offset + 1u, 1u};
  EXPECT_EQ(nested.Validate(false), KFParticleGpuTransactionOverlappingOutput);
  EXPECT_EQ(nested.Validate(false), nested.Validate(true));
}

TEST(KFParticleGpuHostInterfaceTest, TransactionValidationRejectsOverflowBeforeOverlapScan)
{
  ValidationStorage storage(3u);
  storage.domains[1u].sourceTracks =
    {std::numeric_limits<unsigned int>::max(), 1u};
  EXPECT_EQ(storage.Validate(false), KFParticleGpuTransactionInvalidDomain);
  EXPECT_EQ(storage.Validate(false), storage.Validate(true));
}

TEST(KFParticleGpuHostInterfaceTest, LimitsBatchPlanToRequestedCpuChannels)
{
  Fixture fixture;
  const auto prepared = ::kfparticle::gpu::GpuInputPacker::Prepare(fixture.Input(84u));
  KFParticleGpuDecayPlan requestedPlan;
  AddRequestedCpuFinderChannels(requestedPlan, {310});
  const auto requestedCapacity = KFParticleGpuCpuFinderEventCapacityPlan(
    prepared.Descriptor(), requestedPlan);
  const unsigned int requestedTasks = requestedCapacity.taskCapacity;
  const unsigned int fullTasks = FullPlanTaskCapacity(fixture, 84u);
  ASSERT_GT(requestedTasks, 0u);

  for (std::size_t index = 0u; index < requestedPlan.NumberOfTwoDaughterChannels(); ++index) {
    EXPECT_NE(requestedPlan.TwoDaughterChannel(index).motherPdg, 113);
  }

  KFParticleGpuDecayPlan cascadePlan;
  AddRequestedCpuFinderChannels(cascadePlan, {3312});
  ASSERT_EQ(cascadePlan.NumberOfTwoDaughterChannels(), 1u);
  EXPECT_EQ(cascadePlan.TwoDaughterChannel(0u).channelId,
            KFGpuChannelLambdaToProtonPiMinus);
  ASSERT_EQ(cascadePlan.NumberOfV0TrackCascadeChannels(), 1u);
  EXPECT_EQ(cascadePlan.V0TrackCascadeChannel(0u).channelId,
            KFGpuChannelXiMinusToLambdaPiMinus);

  ::kfparticle::gpu::GpuBatchBuilder builder(
    BatchLimits(1u, 10u, 1u, requestedCapacity), {310});
  ASSERT_EQ(builder.TryAppend(1005u, fixture.Input(84u)),
            ::kfparticle::gpu::GpuBatchAppendResult::Added);
  auto batch = builder.Seal();
  EXPECT_EQ(batch.TaskCapacity(), requestedTasks);
  ExpectEventCapacity(batch.Event(0u), requestedCapacity);
  ASSERT_EQ(batch.RequestedMotherPdgs().size(), 1u);
  EXPECT_EQ(batch.RequestedMotherPdgs().front(), 310);
}

TEST(KFParticleGpuHostInterfaceTest, ReusesRequestedCapacityPlanAcrossEvents)
{
  Fixture first;
  Fixture second;
  second.first.SetPDG(2212, 0u);
  second.last.SetPDG(2212, 0u);

  KFParticleGpuDecayPlan referencePlan;
  AddRequestedCpuFinderChannels(referencePlan, {3312});
  const auto firstPrepared = ::kfparticle::gpu::GpuInputPacker::Prepare(first.Input(85u));
  const auto secondPrepared = ::kfparticle::gpu::GpuInputPacker::Prepare(second.Input(86u));
  const auto firstCapacity = KFParticleGpuCpuFinderEventCapacityPlan(
    firstPrepared.Descriptor(), referencePlan);
  const auto secondCapacity = KFParticleGpuCpuFinderEventCapacityPlan(
    secondPrepared.Descriptor(), referencePlan);
  const auto totalCapacity = SumCapacity({firstCapacity, secondCapacity});

  ::kfparticle::gpu::GpuBatchBuilder builder(
    BatchLimits(2u, 20u, 2u, totalCapacity), {3312});
  ASSERT_EQ(builder.TryAppend(1006u, first.Input(85u)),
            ::kfparticle::gpu::GpuBatchAppendResult::Added);
  ASSERT_EQ(builder.TryAppend(1007u, second.Input(86u)),
            ::kfparticle::gpu::GpuBatchAppendResult::Added);
  const auto batch = builder.Seal();

  ASSERT_EQ(batch.EventCount(), 2u);
  ExpectEventCapacity(batch.Event(0u), firstCapacity);
  ExpectEventCapacity(batch.Event(1u), secondCapacity);
  EXPECT_EQ(batch.Event(0u).capacityPlanBuilds, 1u);
  EXPECT_EQ(batch.Event(1u).capacityPlanBuilds, 0u);
  EXPECT_GE(batch.Event(0u).capacityPlanSetupMilliseconds, 0.);
  EXPECT_EQ(batch.Event(1u).capacityPlanSetupMilliseconds, 0.);
}

TEST(KFParticleGpuHostInterfaceTest, DenseCapacityPlannerMatchesReferenceAndClearsState)
{
  Fixture first;
  Fixture second;
  Fixture empty;
  second.first.SetPDG(2212, 0u);
  second.last.SetPDG(2212, 0u);
  empty.first.Resize(0u);
  empty.last.Resize(0u);
  empty.chi.clear();

  const auto firstPrepared = ::kfparticle::gpu::GpuInputPacker::Prepare(first.Input(87u));
  const auto secondPrepared = ::kfparticle::gpu::GpuInputPacker::Prepare(second.Input(88u));
  const auto emptyPrepared = ::kfparticle::gpu::GpuInputPacker::Prepare(empty.Input(89u));

  KFParticleGpuDecayPlan completePlan;
  AddCompleteCpuFinderChannels(completePlan);
  KFParticleGpuCpuFinderCapacityPlanner completePlanner(completePlan);
  for (const KFParticleGpuEventDesc* descriptor :
       {&firstPrepared.Descriptor(), &emptyPrepared.Descriptor(),
        &secondPrepared.Descriptor(), &firstPrepared.Descriptor()}) {
    ExpectCapacityPlan(
      completePlanner.Plan(*descriptor),
      KFParticleGpuCpuFinderEventCapacityPlan(*descriptor, completePlan));
  }

  KFParticleGpuDecayPlan requestedPlan;
  AddRequestedCpuFinderChannels(requestedPlan, {3312});
  KFParticleGpuCpuFinderCapacityPlanner requestedPlanner(requestedPlan);
  for (const KFParticleGpuEventDesc* descriptor :
       {&secondPrepared.Descriptor(), &emptyPrepared.Descriptor(),
        &firstPrepared.Descriptor(), &secondPrepared.Descriptor()}) {
    ExpectCapacityPlan(
      requestedPlanner.Plan(*descriptor),
      KFParticleGpuCpuFinderEventCapacityPlan(*descriptor, requestedPlan));
  }

  requestedPlan.Clear();
  EXPECT_THROW(requestedPlanner.Plan(firstPrepared.Descriptor()),
               std::logic_error);
}

TEST(KFParticleGpuHostInterfaceTest, V0TaskWorkspaceIncludesDependentGenerations)
{
  for (const auto& pdgs : {std::array<int, 2>{211, -211},
                           std::array<int, 2>{2212, -211},
                           std::array<int, 2>{211, -2212}}) {
    Fixture fixture;
    fixture.first.Resize(2u);
    fixture.last.Resize(2u);
    fixture.chi.resize(2u);
    for (unsigned int index = 0u; index < 2u; ++index) {
      fixture.first.SetPDG(pdgs[index], index);
      fixture.last.SetPDG(pdgs[index], index);
    }
    const auto expected = FullPlanCapacity(fixture);
    ASSERT_GT(expected.twoDaughterCandidates, 0u);
    ASSERT_GT(expected.taskCapacity, expected.twoDaughterCandidates);
    ::kfparticle::gpu::GpuBatchBuilder builder(::kfparticle::gpu::DefaultGpuBatchLimits(1u));
    ASSERT_EQ(builder.TryAppend(1080u, fixture.Input()), ::kfparticle::gpu::GpuBatchAppendResult::Added);
    const auto batch = builder.Seal();
    EXPECT_EQ(batch.Event(0u).taskCapacity, expected.taskCapacity);
    EXPECT_EQ(batch.TaskCapacity(), expected.taskCapacity);
  }
}

TEST(KFParticleGpuHostInterfaceTest, DenseCapacityPlannerMatchesReferenceAtBoundaries)
{
  Fixture fixture;
  const auto prepared = ::kfparticle::gpu::GpuInputPacker::Prepare(fixture.Input(90u));

  KFParticleGpuDecayPlan missingParentPlan;
  KFParticleGpuV0TrackCascadeChannel missingParent;
  missingParent.channelId = 900002u;
  missingParent.parentChannelId = 900001u;
  missingParentPlan.AddV0TrackCascadeChannel(missingParent);
  KFParticleGpuCpuFinderCapacityPlanner missingParentPlanner(
    missingParentPlan);
  ExpectCapacityPlan(
    missingParentPlanner.Plan(prepared.Descriptor()),
    KFParticleGpuCpuFinderEventCapacityPlan(prepared.Descriptor(),
                                            missingParentPlan));

  KFParticleGpuDecayPlan overflowPlan;
  AddRequestedCpuFinderChannels(overflowPlan, {310});
  KFParticleGpuCpuFinderCapacityPlanner overflowPlanner(overflowPlan);
  auto overflowEvent = prepared.Descriptor();
  overflowEvent.TrackSet(SecondaryPositiveFirst).Species(Pion) =
    {0u, std::numeric_limits<unsigned int>::max()};
  overflowEvent.TrackSet(SecondaryNegativeFirst).Species(Pion) = {0u, 2u};
  EXPECT_THROW(KFParticleGpuCpuFinderEventCapacityPlan(overflowEvent,
                                                       overflowPlan),
               std::overflow_error);
  EXPECT_THROW(overflowPlanner.Plan(overflowEvent), std::overflow_error);
}

TEST(KFParticleGpuHostInterfaceTest, ReservesDependencyAwareEventDomainStorage)
{
  Fixture fixture;
  const auto prepared = ::kfparticle::gpu::GpuInputPacker::Prepare(fixture.Input(89u));
  KFParticleGpuDecayPlan plan;
  AddRequestedCpuFinderChannels(plan, {3312});
  const auto capacity = KFParticleGpuCpuFinderEventCapacityPlan(
    prepared.Descriptor(), plan);
  ASSERT_GT(capacity.twoDaughterCandidates, 0u);
  ASSERT_GT(capacity.cascadeCandidates, 0u);
  ASSERT_GT(capacity.graphCandidates, 0u);
  EXPECT_EQ(capacity.candidates,
            capacity.twoDaughterCandidates + capacity.cascadeCandidates
              + capacity.graphCandidates);
  EXPECT_GT(capacity.daughterIds,
            2u * capacity.twoDaughterCandidates);
  EXPECT_EQ(capacity.selectedCandidates, capacity.candidates);

  ::kfparticle::gpu::GpuBatchBuilder builder(
    BatchLimits(1u, 10u, 1u, capacity), {3312});
  ASSERT_EQ(builder.TryAppend(1202u, fixture.Input(89u)),
            ::kfparticle::gpu::GpuBatchAppendResult::Added);
  auto batch = builder.Seal();
  ASSERT_EQ(batch.EventCount(), 1u);
  const auto& event = batch.Event(0u);
  EXPECT_EQ(event.taskCapacity, capacity.taskCapacity);
  EXPECT_EQ(event.candidateReservation.size, capacity.candidates);
  EXPECT_EQ(event.daughterReservation.size, capacity.daughterIds);
  EXPECT_EQ(event.selectedReservation.size, capacity.selectedCandidates);
}

TEST(KFParticleGpuHostInterfaceTest, KeepsEmptyAndMalformedEventsIsolatedInBatchBuilder)
{
  Fixture empty;
  empty.first.Resize(0);
  empty.last.Resize(0);
  empty.chi.clear();
  Fixture valid;
  Fixture malformed;
  malformed.chi.pop_back();

  const auto validCapacity = FullPlanCapacity(valid, 85u);
  const unsigned int validTasks = validCapacity.taskCapacity;
  ::kfparticle::gpu::GpuBatchBuilder builder(
    BatchLimits(2u, 10u, 2u, validCapacity));
  EXPECT_EQ(builder.TryAppend(1101u, empty.Input(84u)), ::kfparticle::gpu::GpuBatchAppendResult::Added);
  EXPECT_EQ(builder.TryAppend(1102u, valid.Input(85u)), ::kfparticle::gpu::GpuBatchAppendResult::Added);
  EXPECT_THROW(builder.TryAppend(1103u, malformed.Input(86u)), std::invalid_argument);
  EXPECT_EQ(builder.EventCount(), 2u);

  auto batch = builder.Seal();
  EXPECT_EQ(batch.Event(0u).taskCapacity, 0u);
  EXPECT_EQ(batch.Event(1u).trackOffset, 0u);
  EXPECT_EQ(batch.Event(1u).vertexOffset, 1u);
  EXPECT_THROW(builder.TryAppend(1104u, valid.Input(87u)), std::logic_error);
}

TEST(KFParticleGpuHostInterfaceTest, SplitsDeterministicallyAtEveryBatchLimit)
{
  Fixture fixture;
  const auto sufficient = BatchLimits(
    1u, 10u, 1u, FullPlanCapacity(fixture, 88u));
  const std::array<unsigned int ::kfparticle::gpu::GpuBatchLimits::*, 7u> limits{{
    &::kfparticle::gpu::GpuBatchLimits::events,
    &::kfparticle::gpu::GpuBatchLimits::tracks,
    &::kfparticle::gpu::GpuBatchLimits::vertices,
    &::kfparticle::gpu::GpuBatchLimits::twoDaughterTasks,
    &::kfparticle::gpu::GpuBatchLimits::candidates,
    &::kfparticle::gpu::GpuBatchLimits::daughterIds,
    &::kfparticle::gpu::GpuBatchLimits::selectedCandidates,
  }};

  for (const auto limit : limits) {
    auto tooSmall = sufficient;
    tooSmall.*limit = 0u;
    ::kfparticle::gpu::GpuBatchBuilder builder(tooSmall);
    EXPECT_EQ(builder.TryAppend(1201u, fixture.Input(88u)), ::kfparticle::gpu::GpuBatchAppendResult::CapacityExceeded);
    EXPECT_TRUE(builder.Empty());
    EXPECT_EQ(builder.EventCount(), 0u);
  }
}

TEST(KFParticleGpuHostInterfaceTest, AdmitsOneOversizedEventOnlyForIsolatedExecution)
{
  Fixture fixture;
  const auto capacity = FullPlanCapacity(fixture, 89u);
  auto tooSmall = BatchLimits(2u, 10u, 2u, capacity);
  ASSERT_GT(tooSmall.twoDaughterTasks, 0u);
  --tooSmall.twoDaughterTasks;

  ::kfparticle::gpu::GpuBatchBuilder builder(tooSmall);
  EXPECT_EQ(builder.TryAppend(1202u, fixture.Input(89u)),
            ::kfparticle::gpu::GpuBatchAppendResult::CapacityExceeded);
  EXPECT_EQ(builder.TryAppend(
              1202u,
              fixture.Input(89u),
              ::kfparticle::gpu::GpuBatchAppendPolicy::AllowSingleEventBeyondResourceLimits),
            ::kfparticle::gpu::GpuBatchAppendResult::Added);
  EXPECT_EQ(builder.EventCount(), 1u);
  EXPECT_EQ(builder.TryAppend(
              1203u,
              fixture.Input(90u),
              ::kfparticle::gpu::GpuBatchAppendPolicy::AllowSingleEventBeyondResourceLimits),
            ::kfparticle::gpu::GpuBatchAppendResult::CapacityExceeded);

  const auto batch = builder.Seal();
  EXPECT_EQ(batch.EventCount(), 1u);
  EXPECT_EQ(batch.TaskCapacity(), capacity.taskCapacity);
}

TEST(KFParticleGpuHostInterfaceTest, EmptyInputSkipsGpuRuntimeAttachment)
{
  Fixture fixture;
  fixture.first.Resize(0);
  fixture.last.Resize(0);
  fixture.chi.clear();

  const auto result = ::kfparticle::gpu::GpuExecutor().Run(1234u,
                                                fixture.first,
                                                fixture.last,
                                                fixture.chi,
                                                fixture.vertices);
  EXPECT_EQ(result.sourceEventId, 1234u);
  EXPECT_EQ(result.status, ::kfparticle::gpu::GpuExecutionStatus::EmptyInput);
  EXPECT_TRUE(result.Succeeded());
  EXPECT_EQ(result.tracks, 0u);
  EXPECT_EQ(result.taskCapacity, 0u);
  EXPECT_TRUE(result.fullDiagnosticSnapshot);
}

TEST(KFParticleGpuHostInterfaceTest, PublishableOutputSkipsFullDiagnosticSnapshot)
{
  Fixture fixture;
  fixture.first.Resize(0);
  fixture.last.Resize(0);
  fixture.chi.clear();

  const auto result = ::kfparticle::gpu::GpuExecutor().Run(
    1235u,
    fixture.first,
    fixture.last,
    fixture.chi,
    fixture.vertices,
    -1.f,
    ::kfparticle::gpu::GpuOutputMode::PublishableOutput);

  EXPECT_EQ(result.status, ::kfparticle::gpu::GpuExecutionStatus::EmptyInput);
  EXPECT_FALSE(result.fullDiagnosticSnapshot);
  EXPECT_TRUE(result.rawCandidates.empty());
  EXPECT_TRUE(result.channels.empty());
}

TEST(KFParticleGpuHostInterfaceTest, SingleEventWithoutVerticesSkipsGpuAttachment)
{
  Fixture fixture;
  fixture.vertices.clear();
  const auto result = ::kfparticle::gpu::GpuExecutor().Run(
    1236u, fixture.first, fixture.last, fixture.chi, fixture.vertices);
  EXPECT_EQ(result.status, ::kfparticle::gpu::GpuExecutionStatus::EmptyInput);
  EXPECT_GT(result.tracks, 0u);
  EXPECT_EQ(result.vertices, 0u);
  EXPECT_TRUE(result.Succeeded());
}

TEST(KFParticleGpuHostInterfaceTest, SingleEventHasNoBatchAdmissionCeiling)
{
  Fixture fixture;
  fixture.first.Resize(0);
  fixture.last.Resize(0);
  fixture.chi.clear();
  fixture.vertices.resize(::kfparticle::gpu::DefaultGpuBatchLimits(1u).vertices + 1u);
  const auto result = ::kfparticle::gpu::GpuExecutor().Run(
    1237u, fixture.first, fixture.last, fixture.chi, fixture.vertices);
  EXPECT_EQ(result.status, ::kfparticle::gpu::GpuExecutionStatus::EmptyInput);
  EXPECT_EQ(result.vertices, fixture.vertices.size());
  EXPECT_TRUE(result.Succeeded());
}

TEST(KFParticleGpuHostInterfaceTest, ReturnsOneResultPerEmptyBatchEventWithoutGpuAttachment)
{
  Fixture first;
  Fixture second;
  first.first.Resize(0);
  first.last.Resize(0);
  first.chi.clear();
  second.first.Resize(0);
  second.last.Resize(0);
  second.chi.clear();

  ::kfparticle::gpu::GpuBatchBuilder builder(BatchLimits(2u, 0u, 2u, 0u));
  ASSERT_EQ(builder.TryAppend(1301u, first.Input(91u)), ::kfparticle::gpu::GpuBatchAppendResult::Added);
  ASSERT_EQ(builder.TryAppend(1302u, second.Input(92u)), ::kfparticle::gpu::GpuBatchAppendResult::Added);
  const auto results = ::kfparticle::gpu::GpuExecutor().RunBatch(builder.Seal());

  ASSERT_EQ(results.size(), 2u);
  EXPECT_EQ(results[0].sourceEventId, 1301u);
  EXPECT_EQ(results[1].sourceEventId, 1302u);
  EXPECT_EQ(results[0].status, ::kfparticle::gpu::GpuExecutionStatus::EmptyInput);
  EXPECT_EQ(results[1].status, ::kfparticle::gpu::GpuExecutionStatus::EmptyInput);
  EXPECT_TRUE(results[0].Succeeded());
  EXPECT_TRUE(results[1].Succeeded());
  EXPECT_TRUE(results[0].fullDiagnosticSnapshot);
  EXPECT_TRUE(results[1].fullDiagnosticSnapshot);
}

TEST(KFParticleGpuHostInterfaceTest, ReplaysOnePreparedBatchInsideTheRunner)
{
  Fixture first;
  Fixture second;
  first.first.Resize(0);
  first.last.Resize(0);
  first.chi.clear();
  second.first.Resize(0);
  second.last.Resize(0);
  second.chi.clear();

  ::kfparticle::gpu::GpuBatchBuilder builder(BatchLimits(2u, 0u, 2u, 0u));
  ASSERT_EQ(builder.TryAppend(1311u, first.Input(94u)), ::kfparticle::gpu::GpuBatchAppendResult::Added);
  ASSERT_EQ(builder.TryAppend(1312u, second.Input(95u)), ::kfparticle::gpu::GpuBatchAppendResult::Added);
  const auto batch = builder.Seal();
  ScopedEnvironment replay("KFPARTICLE_GPU_IN_PROCESS_REPLAY", "1");
  ScopedEnvironment monitoring("KFPARTICLE_GPU_PERFORMANCE_MONITORING", "1");
  ScopedEnvironment repetitions("KFPARTICLE_GPU_IN_PROCESS_REPEATS", "3");

  testing::internal::CaptureStderr();
  const auto results = ::kfparticle::gpu::GpuExecutor().RunBatch(batch);
  const std::string output = testing::internal::GetCapturedStderr();

  ASSERT_EQ(results.size(), 2u);
  EXPECT_EQ(results[0].sourceEventId, 1311u);
  EXPECT_EQ(results[1].sourceEventId, 1312u);
  EXPECT_TRUE(results[0].Succeeded());
  EXPECT_TRUE(results[1].Succeeded());
  std::size_t samples = 0u;
  std::size_t offset = 0u;
  while ((offset = output.find("KFP_IN_PROCESS_SAMPLE ", offset))
         != std::string::npos) {
    ++samples;
    ++offset;
  }
  EXPECT_EQ(samples, 3u);
  EXPECT_NE(output.find("sample=1/3"), std::string::npos);
  EXPECT_NE(output.find("sample=3/3"), std::string::npos);
}

TEST(KFParticleGpuHostInterfaceTest, IgnoresStrayRepeatCountOutsideExplicitReplay)
{
  Fixture fixture;
  fixture.first.Resize(0);
  fixture.last.Resize(0);
  fixture.chi.clear();

  ::kfparticle::gpu::GpuBatchBuilder builder(BatchLimits(1u, 0u, 1u, 0u));
  ASSERT_EQ(builder.TryAppend(1313u, fixture.Input(96u)),
            ::kfparticle::gpu::GpuBatchAppendResult::Added);
  ScopedEnvironment replay("KFPARTICLE_GPU_IN_PROCESS_REPLAY", "0");
  ScopedEnvironment monitoring("KFPARTICLE_GPU_PERFORMANCE_MONITORING", "0");
  ScopedEnvironment repetitions("KFPARTICLE_GPU_IN_PROCESS_REPEATS", "5");

  testing::internal::CaptureStderr();
  const auto results = ::kfparticle::gpu::GpuExecutor().RunBatch(builder.Seal());
  const std::string output = testing::internal::GetCapturedStderr();

  ASSERT_EQ(results.size(), 1u);
  EXPECT_TRUE(results.front().Succeeded());
  EXPECT_EQ(output.find("KFP_IN_PROCESS_SAMPLE "), std::string::npos);
}

TEST(KFParticleGpuHostInterfaceTest, ExplicitReplayRequiresPerformanceMonitoring)
{
  Fixture fixture;
  fixture.first.Resize(0);
  fixture.last.Resize(0);
  fixture.chi.clear();

  ::kfparticle::gpu::GpuBatchBuilder builder(BatchLimits(1u, 0u, 1u, 0u));
  ASSERT_EQ(builder.TryAppend(1314u, fixture.Input(97u)),
            ::kfparticle::gpu::GpuBatchAppendResult::Added);
  ScopedEnvironment replay("KFPARTICLE_GPU_IN_PROCESS_REPLAY", "1");
  ScopedEnvironment monitoring("KFPARTICLE_GPU_PERFORMANCE_MONITORING", "0");
  ScopedEnvironment repetitions("KFPARTICLE_GPU_IN_PROCESS_REPEATS", "2");

  EXPECT_THROW(::kfparticle::gpu::GpuExecutor().RunBatch(builder.Seal()), std::invalid_argument);
}

TEST(KFParticleGpuHostInterfaceTest, PublishableBatchOutputSkipsFullDiagnosticSnapshot)
{
  Fixture fixture;
  fixture.first.Resize(0);
  fixture.last.Resize(0);
  fixture.chi.clear();

  ::kfparticle::gpu::GpuBatchBuilder builder(BatchLimits(1u, 0u, 1u, 0u));
  ASSERT_EQ(builder.TryAppend(1303u, fixture.Input(93u)),
            ::kfparticle::gpu::GpuBatchAppendResult::Added);
  const auto results = ::kfparticle::gpu::GpuExecutor().RunBatch(
    builder.Seal(),
    ::kfparticle::gpu::GpuOutputMode::PublishableOutput);

  ASSERT_EQ(results.size(), 1u);
  EXPECT_FALSE(results.front().fullDiagnosticSnapshot);
  EXPECT_TRUE(results.front().rawCandidates.empty());
  EXPECT_TRUE(results.front().channels.empty());
}
