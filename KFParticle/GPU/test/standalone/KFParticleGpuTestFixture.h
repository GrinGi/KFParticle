#ifndef KFPARTICLE_GPU_TEST_FIXTURE_H
#define KFPARTICLE_GPU_TEST_FIXTURE_H

/* Copyright (C) 2026 GSI Helmholtzzentrum fuer Schwerionenforschung, Darmstadt
   SPDX-License-Identifier: GPL-3.0-only */

#include "data/KFParticleGpuInputPacker.h"
#include "planning/KFParticleGpuBatch.h"
#include "execution/KFParticleGpuFinder.h"
#include "execution/KFParticleGpuMaterializationSchedule.h"

#include "KFPVertex.h"
#include "KFPTrackVector.h"
#include "KFParticle.h"
#include "algorithms/KFParticleGpuSelection.h"
#include "planning/KFParticleGpuCpuChannelCatalogue.h"
#include "planning/KFParticleGpuDecayPlan.h"
#include "algorithms/KFParticleGpuV0Track.h"
#include "KFVertex.h"

#include "gtest/gtest.h"

#include <array>
#include <algorithm>
#include <cstdlib>
#include <initializer_list>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace kfp_gpu_test
{
  using ::kfparticle::gpu::GpuEventInput;
  using ::kfparticle::gpu::GpuBatchAppendResult;
  using ::kfparticle::gpu::GpuBatchAppendPolicy;
  using ::kfparticle::gpu::GpuBatchBuilder;
  using ::kfparticle::gpu::GpuBatchLimits;
  using ::kfparticle::gpu::DefaultGpuBatchLimits;
  using ::kfparticle::gpu::GpuInputPacker;
  using ::kfparticle::gpu::GpuExecutor;
  using ::kfparticle::gpu::GpuExecutionStatus;
  using ::kfparticle::gpu::GpuMaterializationEventWeight;
  using ::kfparticle::gpu::GpuMaterializationScheduleMode;
  using ::kfparticle::gpu::GpuMaterializationScheduleName;
  using ::kfparticle::gpu::ParseGpuMaterializationSchedule;
  using ::kfparticle::gpu::PlanGpuMaterializationWorkers;
  using ::kfparticle::gpu::ResolveGpuMaterializationSchedule;

  constexpr unsigned int kTrackCount = 5u;
  constexpr unsigned int kStride = 32u;

  class ScopedEnvironment {
   public:
    ScopedEnvironment(const char* name, const char* value)
      : fName(name)
    {
      const char* previous = std::getenv(name);
      if (previous != nullptr) {
        fHadPrevious = true;
        fPrevious = previous;
      }
      setenv(name, value, 1);
    }

    ~ScopedEnvironment()
    {
      if (fHadPrevious) { setenv(fName.c_str(), fPrevious.c_str(), 1); }
      else { unsetenv(fName.c_str()); }
    }

   private:
    std::string fName;
    std::string fPrevious;
    bool fHadPrevious = false;
  };

  inline void FillTrack(KFPTrackVector& tracks,
                 unsigned int index,
                 float base,
                 int sourceId,
                 int pdg,
                 int charge,
                 int primaryVertexIndex)
  {
    for (unsigned int component = 0; component < 6u; ++component) {
      tracks.SetParameter(base + static_cast<float>(component), component, index);
    }
    for (unsigned int component = 0; component < 21u; ++component) {
      tracks.SetCovariance(base + 100.f + static_cast<float>(component), component, index);
    }
#ifdef NonhomogeneousField
    for (unsigned int component = 0; component < 10u; ++component) {
      tracks.SetFieldCoefficient(base + 200.f + static_cast<float>(component), component, index);
    }
#endif
    tracks.SetId(sourceId, index);
    tracks.SetPDG(pdg, index);
    tracks.SetQ(charge, index);
    tracks.SetPVIndex(primaryVertexIndex, index);
    tracks.SetNPixelHits(static_cast<int>(index) + 3, index);
  }

  struct Fixture {
    KFPTrackVector first;
    KFPTrackVector last;
    std::vector<float> chi{1.f, 2.f, 3.f, 4.f, 5.f};
    std::vector<KFVertex> vertices;

    Fixture()
    {
      first.Resize(kTrackCount);
      last.Resize(kTrackCount);

      // secondary positive pion, secondary negative pion, primary proton,
      // primary electron, and an unsupported secondary-positive hypothesis.
      const std::array<int, kTrackCount> pdg{211, -211, 2212, -11, 999};
      const std::array<int, kTrackCount> charge{1, -1, 1, -1, 1};
      const std::array<int, kTrackCount> pvIndex{-1, -1, 0, 0, -1};
      for (unsigned int index = 0; index < kTrackCount; ++index) {
        FillTrack(first, index, 10.f * static_cast<float>(index), 100 + static_cast<int>(index), pdg[index],
                  charge[index], pvIndex[index]);
        FillTrack(last, index, 1000.f + 10.f * static_cast<float>(index), 100 + static_cast<int>(index), pdg[index],
                  charge[index], pvIndex[index]);
      }

      KFPVertex source;
      source.SetXYZ(1.f, 2.f, 3.f);
      source.SetCovarianceMatrix(1.f, 0.1f, 2.f, 0.2f, 0.3f, 3.f);
      source.SetChi2(4.f);
      source.SetNDF(9);
      source.SetNContributors(6);
      vertices.emplace_back(source);
    }

    ::kfparticle::gpu::GpuEventInput Input(unsigned int eventId = 73u) const
    {
      ::kfparticle::gpu::GpuEventInput input;
      input.eventId             = eventId;
      input.first               = &first;
      input.last                = &last;
      input.chiToPrimaryVertex  = &chi;
      input.primaryVertices     = &vertices;
      return input;
    }
  };

  inline KFParticleGpuEventCapacityPlan FullPlanCapacity(
    const Fixture& fixture, unsigned int eventId = 73u)
  {
    const auto prepared = ::kfparticle::gpu::GpuInputPacker::Prepare(fixture.Input(eventId));
    KFParticleGpuDecayPlan plan;
    AddCompleteCpuFinderChannels(plan);
    return KFParticleGpuCpuFinderEventCapacityPlan(prepared.Descriptor(), plan);
  }

  inline unsigned int FullPlanTaskCapacity(const Fixture& fixture,
                                    unsigned int eventId = 73u)
  {
    return FullPlanCapacity(fixture, eventId).taskCapacity;
  }

  struct HostStorage {
    std::vector<float> parameters = std::vector<float>(6u * kStride);
    std::vector<float> covariances = std::vector<float>(21u * kStride);
    std::vector<float> field = std::vector<float>(10u * kStride);
    std::vector<float> chi = std::vector<float>(kStride);
    std::vector<int> integers =
      std::vector<int>(KFParticleGpuTrackInputLayout::NumberOfIntegerComponents * kStride);
    std::vector<float> vertexParameters = std::vector<float>(3u * kStride);
    std::vector<float> vertexCovariances = std::vector<float>(6u * kStride);
    std::vector<float> vertexChi2 = std::vector<float>(kStride);
    std::vector<int> vertexIntegers =
      std::vector<int>(KFParticleGpuVertexSoALayout::NumberOfIntegerComponents * kStride);

    KFParticleGpuInputTrackSoAView Tracks(unsigned int size)
    {
      return {parameters.data(), covariances.data(), field.data(), chi.data(), integers.data(), size, kStride};
    }

    KFParticleGpuVertexSoAView Vertices(unsigned int size)
    {
      return {vertexParameters.data(), vertexCovariances.data(), vertexChi2.data(), vertexIntegers.data(), size,
              kStride};
    }
  };

  struct ReferenceSourceTrack {
    unsigned int index = 0u;
    bool atLastPoint = false;
  };

  inline unsigned int ReferenceGpuSpecies(int pdg)
  {
    switch (std::abs(pdg)) {
      case 11: return Electron;
      case 13:
      case 19: return Muon;
      case 1:
      case 3:
      case 211: return Pion;
      case 321: return Kaon;
      case 2212: return Proton;
      case 1000010020: return Deuteron;
      case 1000010030: return Triton;
      case 1000020030: return Helium3;
      case 1000020040: return Helium4;
      case 1000020060: return Helium6;
      case 1000030060: return Lithium6;
      case 1000030070: return Lithium7;
      case 1000040070: return Beryllium7;
      default: return NumberOfTrackSpecies;
    }
  }

  inline KFParticleGpuTrackSet ReferenceTrackSet(bool atLastPoint,
                                          int charge,
                                          int primaryVertexIndex)
  {
    const unsigned int pointOffset = atLastPoint
      ? SecondaryPositiveLast : SecondaryPositiveFirst;
    const unsigned int signOffset = charge < 0 ? 1u : 0u;
    const unsigned int primaryOffset = primaryVertexIndex >= 0 ? 2u : 0u;
    return static_cast<KFParticleGpuTrackSet>(
      pointOffset + signOffset + primaryOffset);
  }

  struct ReferencePreparedLayout {
    KFParticleGpuEventDesc descriptor;
    std::vector<ReferenceSourceTrack> sourceTracks;
  };

  inline ReferencePreparedLayout ReferenceBucketPreparation(
    const ::kfparticle::gpu::GpuEventInput& input)
  {
    ReferencePreparedLayout reference;
    reference.descriptor.eventId = input.eventId;
    reference.descriptor.minSecondaryTrackChiToPrimaryVertex =
      input.minSecondaryTrackChiToPrimaryVertex;
    reference.descriptor.primaryVertices = {
      0u, static_cast<unsigned int>(input.primaryVertices->size())};

    std::array<std::array<std::vector<ReferenceSourceTrack>,
                          NumberOfTrackSpecies + 1u>,
               NumberOfTrackSets> buckets;
    for (unsigned int sourceIndex = 0u;
         sourceIndex < static_cast<unsigned int>(input.first->Size());
         ++sourceIndex) {
      const unsigned int species =
        ReferenceGpuSpecies(input.first->PDG()[sourceIndex]);
      for (const bool atLastPoint : {false, true}) {
        const auto set = ReferenceTrackSet(
          atLastPoint, input.first->Q()[sourceIndex],
          input.first->PVIndex()[sourceIndex]);
        buckets[static_cast<unsigned int>(set)][species].push_back(
          {sourceIndex, atLastPoint});
      }
    }

    unsigned int offset = 0u;
    for (unsigned int set = 0u; set < NumberOfTrackSets; ++set) {
      auto& setDescriptor = reference.descriptor.trackSets[set];
      const unsigned int setOffset = offset;
      for (unsigned int species = 0u;
           species < NumberOfTrackSpecies; ++species) {
        const auto& bucket = buckets[set][species];
        setDescriptor.species[species] = {
          offset, static_cast<unsigned int>(bucket.size())};
        reference.sourceTracks.insert(reference.sourceTracks.end(),
                                      bucket.begin(), bucket.end());
        offset += static_cast<unsigned int>(bucket.size());
      }
      const auto& unsupported = buckets[set][NumberOfTrackSpecies];
      reference.sourceTracks.insert(reference.sourceTracks.end(),
                                    unsupported.begin(), unsupported.end());
      offset += static_cast<unsigned int>(unsupported.size());
      setDescriptor.tracks = {setOffset, offset - setOffset};
    }
    return reference;
  }

  inline ::kfparticle::gpu::GpuBatchLimits BatchLimits(unsigned int events,
                                        unsigned int tracks,
                                        unsigned int vertices,
                                        const KFParticleGpuEventCapacityPlan& capacity)
  {
    ::kfparticle::gpu::GpuBatchLimits limits;
    limits.events             = events;
    limits.tracks             = tracks;
    limits.vertices           = vertices;
    limits.twoDaughterTasks   = capacity.taskCapacity;
    limits.candidates         = capacity.candidates;
    limits.daughterIds        = capacity.daughterIds;
    limits.selectedCandidates = capacity.selectedCandidates;
    return limits;
  }

  inline ::kfparticle::gpu::GpuBatchLimits BatchLimits(unsigned int events,
                                        unsigned int tracks,
                                        unsigned int vertices,
                                        unsigned int tasks)
  {
    return BatchLimits(events, tracks, vertices,
                       KFParticleGpuEventCapacityPlan{tasks, 0u, 0u, tasks,
                                                      tasks, 2u * tasks, tasks});
  }

  inline KFParticleGpuEventCapacityPlan SumCapacity(
    std::initializer_list<KFParticleGpuEventCapacityPlan> plans)
  {
    KFParticleGpuEventCapacityPlan result;
    for (const auto& plan : plans) {
      result.twoDaughterCandidates += plan.twoDaughterCandidates;
      result.cascadeCandidates += plan.cascadeCandidates;
      result.graphCandidates += plan.graphCandidates;
      result.taskCapacity += plan.taskCapacity;
      result.candidates += plan.candidates;
      result.daughterIds += plan.daughterIds;
      result.selectedCandidates += plan.selectedCandidates;
    }
    return result;
  }

  inline void ExpectEventCapacity(
    const ::kfparticle::gpu::GpuBatchEvent& event,
    const KFParticleGpuEventCapacityPlan& expected)
  {
    EXPECT_EQ(event.taskCapacity, expected.taskCapacity);
    EXPECT_EQ(event.candidateReservation.size, expected.candidates);
    EXPECT_EQ(event.daughterReservation.size, expected.daughterIds);
    EXPECT_EQ(event.selectedReservation.size, expected.selectedCandidates);
  }

  inline void ExpectCapacityPlan(const KFParticleGpuEventCapacityPlan& actual,
                          const KFParticleGpuEventCapacityPlan& expected)
  {
    EXPECT_EQ(actual.twoDaughterCandidates, expected.twoDaughterCandidates);
    EXPECT_EQ(actual.cascadeCandidates, expected.cascadeCandidates);
    EXPECT_EQ(actual.graphCandidates, expected.graphCandidates);
    EXPECT_EQ(actual.taskCapacity, expected.taskCapacity);
    EXPECT_EQ(actual.candidates, expected.candidates);
    EXPECT_EQ(actual.daughterIds, expected.daughterIds);
    EXPECT_EQ(actual.selectedCandidates, expected.selectedCandidates);
  }

  struct ValidationStorage {
    std::vector<KFParticleGpuTransactionDesc> transactions{1u};
    std::vector<KFParticleGpuDomainDesc> domains;
    std::vector<KFParticleGpuWorkRange> workRanges;
    std::vector<KFParticleGpuWorkStatus> workStatuses;
    std::vector<KFParticleGpuDomainStatus> domainStatuses;

    explicit ValidationStorage(unsigned int count,
                               bool permuted = false,
                               unsigned int domainOffset = 0u)
      : domains(count + domainOffset),
        workRanges(count + domainOffset),
        workStatuses(count + domainOffset),
        domainStatuses(count + domainOffset)
    {
      transactions[0].decayPlanRevision = 91u;
      transactions[0].domains = {domainOffset, count};
      transactions[0].workRanges = {domainOffset, count};
      for (unsigned int local = 0u; local < count; ++local) {
        const unsigned int index = domainOffset + local;
        const unsigned int rank = permuted ? (local * 37u) % count : local;
        const unsigned int base = 4u * rank;
        auto& domain = domains[index];
        domain.inputDescriptorIndex = local;
        domain.statusIndex = index;
        domain.sourceTracks = {base, 2u};
        domain.primaryVertices = {base, 1u};
        domain.workRanges = {index, 1u};
        domain.taskReservation = {base, 2u};
        domain.candidateReservation = {base, 2u};
        domain.daughterReservation = {base, 2u};
        domain.selectedReservation = {base, 2u};
        auto& work = workRanges[index];
        work.domainIndex = index;
        work.family = KFParticleGpuTwoDaughterWork;
        work.sourceTracks = domain.sourceTracks;
        work.secondarySourceTracks = domain.sourceTracks;
        work.primaryVertices = domain.primaryVertices;
        work.outputReservation = domain.taskReservation;
      }
    }

    KFParticleGpuTransactionView View()
    {
      return {transactions.data(), domains.data(), workRanges.data(),
              workStatuses.data(), domainStatuses.data(),
              static_cast<unsigned int>(domains.size()),
              static_cast<unsigned int>(workRanges.size()),
              static_cast<unsigned int>(domainStatuses.size())};
    }

    KFParticleGpuTransactionValidation Validate(
      bool reference,
      KFParticleGpuTransactionValidationStats* stats = nullptr)
    {
      auto view = View();
      constexpr unsigned int capacity = 1000000u;
      return reference
               ? KFParticleGpuValidateTransactionReference(
                   view, 91u, capacity, capacity,
                   transactions[0].domains.size, capacity, capacity,
                   capacity, capacity, stats)
               : KFParticleGpuValidateTransaction(
                   view, 91u, capacity, capacity,
                   transactions[0].domains.size, capacity, capacity,
                   capacity, capacity, stats);
    }
  };
}  // namespace


#endif
