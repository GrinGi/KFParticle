/* Copyright (C) 2026 GSI Helmholtzzentrum fuer Schwerionenforschung, Darmstadt
   SPDX-License-Identifier: GPL-3.0-only */

#include "KFParticleGpuTestFixture.h"

using namespace kfp_gpu_test;

TEST(KFParticleGpuHostInterfaceTest, PreservesPhysicalTracksAndFieldAwareMetadata)
{
  Fixture fixture;
  auto input = fixture.Input();
  input.minSecondaryTrackChiToPrimaryVertex = 10.f;
  auto prepared = ::kfparticle::gpu::GpuInputPacker::Prepare(input);
  const auto capacities = ::kfparticle::gpu::GpuInputPacker::InputCapacities(prepared);

  ASSERT_EQ(prepared.TrackCount(), 2u * kTrackCount);
  ASSERT_EQ(prepared.VertexCount(), 1u);
  EXPECT_TRUE(prepared.HasValidatedPublicationInput());
  EXPECT_EQ(capacities.tracks, 2u * kTrackCount);
  EXPECT_EQ(capacities.vertices, 1u);
  EXPECT_EQ(capacities.events, 1u);
  EXPECT_TRUE(capacities.nonhomogeneousField);

  HostStorage storage;
  KFParticleGpuEventDesc event;
  ::kfparticle::gpu::GpuInputPacker::Write(prepared, storage.Tracks(prepared.TrackCount()), storage.Vertices(prepared.VertexCount()),
                             event);

  EXPECT_EQ(event.eventId, 73u);
  EXPECT_FLOAT_EQ(event.minSecondaryTrackChiToPrimaryVertex, 10.f);
  EXPECT_EQ(event.TrackSet(SecondaryPositiveFirst).tracks.offset, 0u);
  EXPECT_EQ(event.TrackSet(SecondaryPositiveFirst).tracks.size, 2u);
  EXPECT_EQ(event.TrackSet(SecondaryPositiveFirst).Species(Pion).offset, 0u);
  EXPECT_EQ(event.TrackSet(SecondaryPositiveFirst).Species(Pion).size, 1u);
  EXPECT_EQ(event.TrackSet(SecondaryNegativeFirst).tracks.offset, 2u);
  EXPECT_EQ(event.TrackSet(PrimaryPositiveFirst).tracks.offset, 3u);
  EXPECT_EQ(event.TrackSet(PrimaryPositiveFirst).Species(Proton).size, 1u);
  EXPECT_EQ(event.TrackSet(PrimaryNegativeFirst).tracks.offset, 4u);
  EXPECT_EQ(event.TrackSet(SecondaryPositiveLast).tracks.offset, 5u);
  EXPECT_EQ(event.primaryVertices.size, 1u);

  // The unsupported PID remains a physical secondary-positive track after the pion range.
  EXPECT_EQ(storage.Tracks(prepared.TrackCount()).SourceId(0u), 100);
  EXPECT_EQ(storage.Tracks(prepared.TrackCount()).SourceId(1u), 104);
  EXPECT_EQ(storage.Tracks(prepared.TrackCount()).SourceId(5u), 100);
  EXPECT_FLOAT_EQ(storage.Tracks(prepared.TrackCount()).Numerical().Parameter(0u, 0u), 0.f);
  EXPECT_FLOAT_EQ(storage.Tracks(prepared.TrackCount()).Numerical().Parameter(0u, 5u), 1000.f);
  EXPECT_FLOAT_EQ(storage.Tracks(prepared.TrackCount()).Numerical().Covariance(20u, 3u), 140.f);
  EXPECT_FLOAT_EQ(storage.Tracks(prepared.TrackCount()).FieldCoefficient(9u, 5u), 1209.f);
  EXPECT_FLOAT_EQ(storage.Tracks(prepared.TrackCount()).ChiToPrimaryVertex(4u), 4.f);
  EXPECT_EQ(storage.Tracks(prepared.TrackCount()).PrimaryVertexIndex(3u), 0);
  EXPECT_EQ(storage.Tracks(prepared.TrackCount()).NumberOfPixelHits(4u), 6);

  const auto vertices = storage.Vertices(prepared.VertexCount());
  EXPECT_FLOAT_EQ(vertices.Parameter(0u, 0u), 1.f);
  EXPECT_FLOAT_EQ(vertices.Parameter(2u, 0u), 3.f);
  EXPECT_FLOAT_EQ(vertices.Covariance(5u, 0u), 3.f);
  EXPECT_FLOAT_EQ(vertices.Chi2(0u), 4.f);
  EXPECT_EQ(vertices.NDF(0u), 9);
  EXPECT_EQ(vertices.NContributors(0u), 6);
}

TEST(KFParticleGpuHostInterfaceTest, CountPrefixFillMatchesReferenceBucketLayout)
{
  constexpr unsigned int trackCount = 16u;
  const std::array<int, trackCount> pdgs{{
    999, 2212, -11, 321, 13, 1000010020, 1000010030, 1000020030,
    1000020040, 1000020060, 1000030060, 1000030070, 1000040070,
    211, 1, 3}};
  KFPTrackVector first;
  KFPTrackVector last;
  first.Resize(trackCount);
  last.Resize(trackCount);
  std::vector<float> chi(trackCount);
  for (unsigned int index = 0u; index < trackCount; ++index) {
    const int charge = index >= 13u ? 1 : (index % 2u == 0u ? 1 : -1);
    const int primaryVertexIndex = index >= 13u
      ? -1 : (index % 3u == 0u ? 0 : -1);
    FillTrack(first, index, 10.f * static_cast<float>(index),
              500 + static_cast<int>(index),
              pdgs[index], charge, primaryVertexIndex);
    FillTrack(last, index,
              1000.f + 10.f * static_cast<float>(index),
              500 + static_cast<int>(index), pdgs[index], charge,
              primaryVertexIndex);
    // Layout follows first-point metadata, while packed last-point payload
    // must still retain the actual last-point metadata.
    last.SetPDG(index % 2u == 0u ? 888 : -888, index);
    last.SetQ(-charge, index);
    last.SetPVIndex(primaryVertexIndex >= 0 ? -1 : 0, index);
    chi[index] = 0.5f + static_cast<float>(index);
  }
  const std::vector<KFVertex> vertices(1u);
  ::kfparticle::gpu::GpuEventInput input;
  input.eventId = 82u;
  input.first = &first;
  input.last = &last;
  input.chiToPrimaryVertex = &chi;
  input.primaryVertices = &vertices;
  input.minSecondaryTrackChiToPrimaryVertex = 12.5f;

  const auto reference = ReferenceBucketPreparation(input);
  const auto prepared = ::kfparticle::gpu::GpuInputPacker::Prepare(input);
  ASSERT_EQ(prepared.TrackCount(), 2u * trackCount);
  ASSERT_EQ(prepared.TrackCount(), reference.sourceTracks.size());

  HostStorage storage;
  KFParticleGpuEventDesc actualDescriptor;
  ::kfparticle::gpu::GpuInputPacker::Write(
    prepared, storage.Tracks(prepared.TrackCount()), storage.Vertices(prepared.VertexCount()),
    actualDescriptor);
  EXPECT_EQ(actualDescriptor.eventId, reference.descriptor.eventId);
  EXPECT_FLOAT_EQ(actualDescriptor.minSecondaryTrackChiToPrimaryVertex,
                  reference.descriptor.minSecondaryTrackChiToPrimaryVertex);
  EXPECT_EQ(actualDescriptor.primaryVertices.offset,
            reference.descriptor.primaryVertices.offset);
  EXPECT_EQ(actualDescriptor.primaryVertices.size,
            reference.descriptor.primaryVertices.size);
  for (unsigned int set = 0u; set < NumberOfTrackSets; ++set) {
    EXPECT_EQ(actualDescriptor.trackSets[set].tracks.offset,
              reference.descriptor.trackSets[set].tracks.offset);
    EXPECT_EQ(actualDescriptor.trackSets[set].tracks.size,
              reference.descriptor.trackSets[set].tracks.size);
    for (unsigned int species = 0u;
         species < NumberOfTrackSpecies; ++species) {
      EXPECT_EQ(actualDescriptor.trackSets[set].species[species].offset,
                reference.descriptor.trackSets[set].species[species].offset);
      EXPECT_EQ(actualDescriptor.trackSets[set].species[species].size,
                reference.descriptor.trackSets[set].species[species].size);
    }
  }
  const auto packed = storage.Tracks(prepared.TrackCount());
  for (unsigned int packedIndex = 0u;
       packedIndex < prepared.TrackCount(); ++packedIndex) {
    const auto sourceReference = reference.sourceTracks[packedIndex];
    const KFPTrackVector& source = sourceReference.atLastPoint ? last : first;
    const unsigned int sourceIndex = sourceReference.index;
    for (unsigned int component = 0u; component < 6u; ++component) {
      EXPECT_FLOAT_EQ(packed.Numerical().Parameter(component, packedIndex),
                      source.Parameter(component)[sourceIndex]);
    }
    for (unsigned int component = 0u; component < 21u; ++component) {
      EXPECT_FLOAT_EQ(packed.Numerical().Covariance(component, packedIndex),
                      source.Covariance(component)[sourceIndex]);
    }
#ifdef NonhomogeneousField
    for (unsigned int component = 0u; component < 10u; ++component) {
      EXPECT_FLOAT_EQ(packed.FieldCoefficient(component, packedIndex),
                      source.FieldCoefficient(component)[sourceIndex]);
    }
#endif
    EXPECT_FLOAT_EQ(packed.ChiToPrimaryVertex(packedIndex), chi[sourceIndex]);
    EXPECT_EQ(packed.SourceId(packedIndex), source.Id()[sourceIndex]);
    EXPECT_EQ(packed.Pdg(packedIndex), source.PDG()[sourceIndex]);
    EXPECT_EQ(packed.Charge(packedIndex), source.Q()[sourceIndex]);
    EXPECT_EQ(packed.PrimaryVertexIndex(packedIndex),
              source.PVIndex()[sourceIndex]);
    EXPECT_EQ(packed.NumberOfPixelHits(packedIndex),
              source.NPixelHits()[sourceIndex]);
  }
}

TEST(KFParticleGpuHostInterfaceTest, PacksPrimaryVerticesIntoAbsoluteEventRanges)
{
  Fixture fixture;
  fixture.vertices.push_back(fixture.vertices.front());
  fixture.first.SetPVIndex(1, 2u);
  fixture.last.SetPVIndex(0, 2u);
  fixture.first.SetPVIndex(1, 3u);
  fixture.last.SetPVIndex(1, 3u);
  fixture.first.SetPVIndex(-7, 0u);
  fixture.last.SetPVIndex(-7, 0u);
  const auto prepared = ::kfparticle::gpu::GpuInputPacker::Prepare(fixture.Input());
  constexpr unsigned int trackOffset = 2u;
  constexpr unsigned int vertexOffset = 5u;
  HostStorage storage;
  const auto tracks = storage.Tracks(trackOffset + prepared.TrackCount());
  KFParticleGpuEventDesc event;
  ::kfparticle::gpu::GpuInputPacker::Write(prepared, trackOffset, vertexOffset, tracks,
                       storage.Vertices(vertexOffset + prepared.VertexCount()), event);
  EXPECT_EQ(event.primaryVertices.offset, vertexOffset);
  EXPECT_EQ(event.primaryVertices.size, 2u);
  for (unsigned int set = 0u; set < NumberOfTrackSets; ++set) {
    const auto range = event.trackSets[set].tracks;
    for (unsigned int index = range.offset; index < range.End(); ++index) {
      const int sourceId = tracks.SourceId(index);
      int expected = sourceId == 102 || sourceId == 103 ? 6 : -1;
      if (sourceId == 102 && set == PrimaryPositiveLast) { expected = 5; }
      if (sourceId == 100) { expected = -7; }
      EXPECT_EQ(tracks.PrimaryVertexIndex(index), expected);
    }
  }
  EXPECT_EQ(fixture.first.PVIndex()[2u], 1);
  EXPECT_EQ(fixture.last.PVIndex()[2u], 0);

  fixture.last.SetPVIndex(2, 2u);
  EXPECT_THROW(::kfparticle::gpu::GpuInputPacker::Write(prepared, trackOffset, vertexOffset, tracks,
    storage.Vertices(vertexOffset + prepared.VertexCount()), event), std::invalid_argument);
  fixture.last.SetPVIndex(0, 2u);
  fixture.first.SetPVIndex(2, 2u);
  EXPECT_THROW(::kfparticle::gpu::GpuInputPacker::Write(prepared, trackOffset, vertexOffset, tracks,
    storage.Vertices(vertexOffset + prepared.VertexCount()), event), std::invalid_argument);
}

TEST(KFParticleGpuHostInterfaceTest, PacksHeavyHypernuclearTrackSpecies)
{
  constexpr unsigned int trackCount = 4u;
  KFPTrackVector first;
  KFPTrackVector last;
  first.Resize(trackCount);
  last.Resize(trackCount);
  const std::array<int, trackCount> pdgs{
    1000020060, 1000030060, 1000030070, 1000040070};
  for (unsigned int index = 0u; index < trackCount; ++index) {
    FillTrack(first, index, 10.f * static_cast<float>(index),
              200 + static_cast<int>(index),
              pdgs[index], 1, -1);
    FillTrack(last, index, 100.f + 10.f * static_cast<float>(index),
              200 + static_cast<int>(index),
              pdgs[index], 1, -1);
  }
  const std::vector<float> chi(trackCount, 20.f);
  const std::vector<KFVertex> vertices;
  ::kfparticle::gpu::GpuEventInput input;
  input.eventId = 81u;
  input.first = &first;
  input.last = &last;
  input.chiToPrimaryVertex = &chi;
  input.primaryVertices = &vertices;

  const auto prepared = ::kfparticle::gpu::GpuInputPacker::Prepare(input);
  ASSERT_EQ(prepared.TrackCount(), 2u * trackCount);
  HostStorage storage;
  KFParticleGpuEventDesc event;
  ::kfparticle::gpu::GpuInputPacker::Write(prepared, storage.Tracks(prepared.TrackCount()),
                             storage.Vertices(0u), event);

  const KFParticleGpuTrackSetDesc& firstSet =
    event.TrackSet(SecondaryPositiveFirst);
  EXPECT_EQ(firstSet.tracks.size, trackCount);
  EXPECT_EQ(firstSet.Species(Helium6).size, 1u);
  EXPECT_EQ(firstSet.Species(Lithium6).size, 1u);
  EXPECT_EQ(firstSet.Species(Lithium7).size, 1u);
  EXPECT_EQ(firstSet.Species(Beryllium7).size, 1u);
  EXPECT_EQ(event.TrackSet(SecondaryPositiveLast).tracks.size, trackCount);
}

TEST(KFParticleGpuHostInterfaceTest, RejectsInconsistentOrInvalidInput)
{
  Fixture fixture;
  fixture.chi.pop_back();
  EXPECT_THROW(::kfparticle::gpu::GpuInputPacker::Prepare(fixture.Input()), std::invalid_argument);

  Fixture zeroCharge;
  zeroCharge.first.SetQ(0, 0);
  EXPECT_THROW(::kfparticle::gpu::GpuInputPacker::Prepare(zeroCharge.Input()), std::invalid_argument);

  Fixture mismatchedPair;
  mismatchedPair.last.SetId(999, 0u);
  EXPECT_THROW(::kfparticle::gpu::GpuInputPacker::Prepare(mismatchedPair.Input()),
               std::invalid_argument);

  Fixture duplicateSource;
  duplicateSource.first.SetId(duplicateSource.first.Id()[0u], 1u);
  duplicateSource.last.SetId(duplicateSource.last.Id()[0u], 1u);
  EXPECT_THROW(::kfparticle::gpu::GpuInputPacker::Prepare(duplicateSource.Input()),
               std::invalid_argument);

  Fixture nonFiniteCut;
  auto nonFiniteInput = nonFiniteCut.Input();
  nonFiniteInput.minSecondaryTrackChiToPrimaryVertex =
    std::numeric_limits<float>::quiet_NaN();
  EXPECT_THROW(::kfparticle::gpu::GpuInputPacker::Prepare(nonFiniteInput), std::invalid_argument);
}

TEST(KFParticleGpuHostInterfaceTest, ValidatesNumericalPayloadDuringDirectWrite)
{
  Fixture changedIdentity;
  auto identityPrepared = ::kfparticle::gpu::GpuInputPacker::Prepare(changedIdentity.Input());
  changedIdentity.last.SetId(999, 0u);
  HostStorage identityStorage;
  KFParticleGpuEventDesc identityEvent;
  EXPECT_NO_THROW(::kfparticle::gpu::GpuInputPacker::Write(
    identityPrepared, identityStorage.Tracks(identityPrepared.TrackCount()),
    identityStorage.Vertices(identityPrepared.VertexCount()), identityEvent));
  EXPECT_EQ(identityStorage.Tracks(identityPrepared.TrackCount()).SourceId(5u),
            100);

  Fixture nonFiniteTrack;
  nonFiniteTrack.first.SetParameter(
    std::numeric_limits<float>::quiet_NaN(), 2u, 0u);
  auto prepared = ::kfparticle::gpu::GpuInputPacker::Prepare(nonFiniteTrack.Input());
  HostStorage storage;
  KFParticleGpuEventDesc event;
  EXPECT_THROW(
    ::kfparticle::gpu::GpuInputPacker::Write(
      prepared, storage.Tracks(prepared.TrackCount()),
      storage.Vertices(prepared.VertexCount()), event),
    std::invalid_argument);

  Fixture nonFiniteChi;
  nonFiniteChi.chi[1u] = std::numeric_limits<float>::infinity();
  auto chiPrepared = ::kfparticle::gpu::GpuInputPacker::Prepare(nonFiniteChi.Input());
  EXPECT_THROW(
    ::kfparticle::gpu::GpuInputPacker::Write(
      chiPrepared, storage.Tracks(chiPrepared.TrackCount()),
      storage.Vertices(chiPrepared.VertexCount()), event),
    std::invalid_argument);

  Fixture nonFiniteVertex;
  KFPVertex vertex;
  vertex.SetXYZ(std::numeric_limits<float>::quiet_NaN(), 2.f, 3.f);
  vertex.SetCovarianceMatrix(1.f, 0.1f, 2.f, 0.2f, 0.3f, 3.f);
  vertex.SetChi2(4.f);
  vertex.SetNDF(9);
  vertex.SetNContributors(6);
  nonFiniteVertex.vertices.clear();
  nonFiniteVertex.vertices.emplace_back(vertex);
  auto vertexPrepared = ::kfparticle::gpu::GpuInputPacker::Prepare(nonFiniteVertex.Input());
  EXPECT_THROW(
    ::kfparticle::gpu::GpuInputPacker::Write(
      vertexPrepared, storage.Tracks(vertexPrepared.TrackCount()),
      storage.Vertices(vertexPrepared.VertexCount()), event),
    std::invalid_argument);
}

TEST(KFParticleGpuHostInterfaceTest, ClassifiesLastPointDataUsingFirstPointMetadata)
{
  Fixture fixture;
  fixture.last.SetQ(-1, 0);
  fixture.last.SetPDG(2212, 0);

  auto prepared = ::kfparticle::gpu::GpuInputPacker::Prepare(fixture.Input());
  HostStorage storage;
  KFParticleGpuEventDesc event;
  ::kfparticle::gpu::GpuInputPacker::Write(prepared, storage.Tracks(prepared.TrackCount()), storage.Vertices(prepared.VertexCount()),
                             event);

  // CPU SortTracks chooses the last-point set and species range from first-point metadata.
  EXPECT_EQ(event.TrackSet(SecondaryPositiveLast).Species(Pion).size, 1u);
  EXPECT_EQ(event.TrackSet(SecondaryNegativeLast).tracks.size, 1u);
  EXPECT_EQ(storage.Tracks(prepared.TrackCount()).Pdg(5u), 2212);
  EXPECT_EQ(storage.Tracks(prepared.TrackCount()).Charge(5u), -1);
}

TEST(KFParticleGpuHostInterfaceTest, ParallelWriteReportsFirstSourceEventFailure)
{
  Fixture first;
  Fixture second;
  first.first.SetParameter(std::numeric_limits<float>::quiet_NaN(), 0u, 0u);
  second.first.SetCovariance(std::numeric_limits<float>::quiet_NaN(), 0u, 0u);

  const auto totalCapacity = SumCapacity(
    {FullPlanCapacity(first, 181u), FullPlanCapacity(second, 182u)});
  ::kfparticle::gpu::GpuBatchBuilder builder(
    BatchLimits(2u, 20u, 2u, totalCapacity));
  ASSERT_EQ(builder.TryAppend(2001u, first.Input(181u)),
            ::kfparticle::gpu::GpuBatchAppendResult::Added);
  ASSERT_EQ(builder.TryAppend(2002u, second.Input(182u)),
            ::kfparticle::gpu::GpuBatchAppendResult::Added);
  auto batch = builder.Seal();
  HostStorage storage;
  std::array<KFParticleGpuEventDesc, 2u> events;

  try {
    batch.Write(storage.Tracks(batch.TrackCount()),
                storage.Vertices(batch.VertexCount()), events.data(), 4u);
    FAIL() << "parallel packing accepted non-finite track data";
  }
  catch (const std::invalid_argument& error) {
    EXPECT_STREQ(error.what(),
                 "KFParticle GPU packer received a non-finite track parameter");
  }
}

TEST(KFParticleGpuHostInterfaceTest, CountsCpuCompatiblePionPairHypotheses)
{
  Fixture fixture;
  // Retain pi+, pi-, and the two primary tracks. The same physical pi+ pi-
  // pair enters K0S and the CPU-compatible alternate-proton Lambda channel.
  fixture.first.Resize(4u);
  fixture.last.Resize(4u);
  fixture.chi.resize(4u);

  const auto expectedCapacity = FullPlanCapacity(fixture, 84u);
  const unsigned int expectedTasks = expectedCapacity.taskCapacity;
  EXPECT_GT(expectedTasks, 3u);
  ::kfparticle::gpu::GpuBatchBuilder builder(
    BatchLimits(1u, 8u, 1u, expectedCapacity));
  ASSERT_EQ(builder.TryAppend(1004u, fixture.Input(84u)),
            ::kfparticle::gpu::GpuBatchAppendResult::Added);
  const auto batch = builder.Seal();
  ASSERT_EQ(batch.EventCount(), 1u);
  EXPECT_EQ(batch.Event(0u).taskCapacity, expectedTasks);
  EXPECT_EQ(batch.TaskCapacity(), expectedTasks);
}
