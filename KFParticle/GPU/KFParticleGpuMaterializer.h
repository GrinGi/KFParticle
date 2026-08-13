/*
 * This file is part of KFParticle package
 * Copyright (C) 2007-2026
 *
 * KFParticle is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * any later version.
 */

#ifndef KFPARTICLEGPUMATERIALIZER_H
#define KFPARTICLEGPUMATERIALIZER_H

#include "KFParticleGpuCandidateTransfer.h"
#include "KFParticleGpuDecayPlan.h"

#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <vector>

enum KFParticleGpuMaterializationStatus
{
  KFGpuMaterializationSucceeded = 0u,
  KFGpuMaterializationOverflow,
  KFGpuMaterializationInvalidRange,
  KFGpuMaterializationInvalidCandidate,
  KFGpuMaterializationInvalidDirectDaughter,
  KFGpuMaterializationCyclicAncestry,
  KFGpuMaterializationLineageMismatch,
  KFGpuMaterializationInvalidNumericalState
};

struct KFParticleGpuMaterializationRequest
{
  KFParticleGpuConstInputTrackSoAView tracks;
  KFParticleGpuConstCandidatePoolView candidates;
  const KFParticleGpuCandidateRange* ranges = nullptr;
  unsigned int numberOfRanges = 0u;
  unsigned int eventIndex = 0u;
};

struct KFParticleGpuMaterializedParticle
{
  KFParticleGpuFitState fit;
  std::vector<int> daughterParticleIds;
  std::vector<int> leafSourceIds;
  float fieldCoefficients[10] = {};
  unsigned int gpuCandidateIndex = std::numeric_limits<unsigned int>::max();
  unsigned int channelId = 0u;
  unsigned int topology = KFGpuGraphTopologyInvalid;
  unsigned int outputClass = KFGpuGraphOutputInvalid;
  unsigned int operationStatus = KFGpuCandidateOperationPending;
  int pdg = 0;
  int primaryVertexIndex = -1;
  int sourceId = -1;
  bool inputTrack = false;
};

struct KFParticleGpuMaterializedEvent
{
  std::vector<KFParticleGpuMaterializedParticle> particles;
  KFParticleGpuMaterializationStatus status = KFGpuMaterializationSucceeded;

  bool Succeeded() const { return status == KFGpuMaterializationSucceeded; }
};

/**
 * Validates one event and reconstructs its direct particle graph on the host.
 *
 * The compact result is independent of ROOT and is therefore testable in the
 * standalone XPU build. CommitToParticles performs the final atomic conversion
 * to the scalar KFParticle contract.
 */
class KFParticleGpuMaterializer
{
 public:
  static KFParticleGpuMaterializedEvent Build(const KFParticleGpuMaterializationRequest& request)
  {
    KFParticleGpuMaterializedEvent result;
    if (request.candidates.OverflowFlags() != 0u) {
      result.status = KFGpuMaterializationOverflow;
      return result;
    }

    const unsigned int candidateSize = request.candidates.Size();
    if (request.numberOfRanges != 0u && request.ranges == nullptr) {
      result.status = KFGpuMaterializationInvalidRange;
      return result;
    }
    std::vector<unsigned int> requested;
    for (unsigned int rangeIndex = 0u; rangeIndex < request.numberOfRanges; ++rangeIndex) {
      const KFParticleGpuCandidateRange& range = request.ranges[rangeIndex];
      if (range.overflowFlags != 0u) {
        result.status = KFGpuMaterializationOverflow;
        return result;
      }
      if (range.offset > candidateSize || range.size > candidateSize - range.offset) {
        result.status = KFGpuMaterializationInvalidRange;
        return result;
      }
      const unsigned int daughterSize = request.candidates.Daughters().Size();
      if (range.daughterOffset > daughterSize
          || range.daughterSize > daughterSize - range.daughterOffset) {
        result.status = KFGpuMaterializationInvalidRange;
        return result;
      }
      for (unsigned int candidate = range.offset; candidate < range.offset + range.size; ++candidate) {
        requested.push_back(candidate);
      }
    }
    std::vector<unsigned char> visit(candidateSize, 0u);
    std::vector<unsigned int> orderedCandidates;
    std::vector<unsigned int> referencedTracks;
    std::function<bool(unsigned int)> visitCandidate = [&](unsigned int candidate) {
      if (candidate >= candidateSize) {
        result.status = KFGpuMaterializationInvalidCandidate;
        return false;
      }
      if (visit[candidate] == 1u) {
        result.status = KFGpuMaterializationCyclicAncestry;
        return false;
      }
      if (visit[candidate] == 2u) {
        return true;
      }

      const auto& metadata = request.candidates.Metadata();
      if (metadata.EventIndex(candidate) != request.eventIndex
          || metadata.OperationStatus(candidate) != KFGpuCandidateOperationAccepted
          || (metadata.Flags(candidate) & KFGpuCandidateValid) == 0u
          || metadata.OutputClass(candidate) == KFGpuGraphOutputInvalid
          || metadata.OutputClass(candidate) >= KFGpuGraphOutputClassCount
          || metadata.DirectDaughterCount(candidate) == 0u
          || metadata.DirectDaughterCount(candidate) > 2u) {
        result.status = KFGpuMaterializationInvalidCandidate;
        return false;
      }

      visit[candidate] = 1u;
      for (unsigned int daughter = 0u; daughter < metadata.DirectDaughterCount(candidate); ++daughter) {
        const unsigned int kind = daughter == 0u
                                    ? metadata.DirectFirstKind(candidate)
                                    : metadata.DirectSecondKind(candidate);
        const unsigned int index = daughter == 0u
                                     ? metadata.DirectFirstIndex(candidate)
                                     : metadata.DirectSecondIndex(candidate);
        if (kind == KFGpuDirectDaughterInputTrack) {
          if (index >= request.tracks.Size()) {
            result.status = KFGpuMaterializationInvalidDirectDaughter;
            return false;
          }
          referencedTracks.push_back(index);
        }
        else if (kind == KFGpuDirectDaughterCandidate) {
          if (index == candidate) {
            result.status = KFGpuMaterializationCyclicAncestry;
            return false;
          }
          if (!visitCandidate(index)) {
            if (result.status == KFGpuMaterializationSucceeded) {
              result.status = KFGpuMaterializationInvalidDirectDaughter;
            }
            return false;
          }
        }
        else {
          result.status = KFGpuMaterializationInvalidDirectDaughter;
          return false;
        }
      }
      visit[candidate] = 2u;
      orderedCandidates.push_back(candidate);
      return true;
    };

    for (unsigned int candidate : requested) {
      if (!visitCandidate(candidate)) {
        return result;
      }
    }

    std::sort(referencedTracks.begin(), referencedTracks.end());
    referencedTracks.erase(
      std::unique(referencedTracks.begin(), referencedTracks.end()), referencedTracks.end());
    std::vector<int> trackParticleIds(request.tracks.Size(), -1);
    std::vector<int> candidateParticleIds(candidateSize, -1);
    result.particles.reserve(referencedTracks.size() + orderedCandidates.size());

    for (unsigned int trackIndex : referencedTracks) {
      KFParticleGpuMaterializedParticle particle;
      particle.inputTrack = true;
      particle.sourceId = request.tracks.SourceId(trackIndex);
      particle.pdg = request.tracks.Pdg(trackIndex);
      particle.primaryVertexIndex = request.tracks.PrimaryVertexIndex(trackIndex);
      particle.leafSourceIds.push_back(particle.sourceId);
      KFParticleGpuTrackState track;
      LoadTrackState(request.tracks.Numerical(), trackIndex, track);
      particle.fit.Initialize(track, request.tracks.Charge(trackIndex), TrackMass(particle.pdg));
      if (particle.sourceId < 0 || !Finite(particle.fit)) {
        result.status = KFGpuMaterializationInvalidNumericalState;
        result.particles.clear();
        return result;
      }
      if (request.tracks.FieldCoefficientsData()) {
        for (unsigned int component = 0u; component < 10u; ++component) {
          particle.fieldCoefficients[component] =
            request.tracks.FieldCoefficient(component, trackIndex);
        }
      }
      trackParticleIds[trackIndex] = static_cast<int>(result.particles.size());
      result.particles.push_back(particle);
    }

    for (unsigned int candidate : orderedCandidates) {
      KFParticleGpuMaterializedParticle particle;
      particle.gpuCandidateIndex = candidate;
      particle.channelId = request.candidates.Metadata().ChannelId(candidate);
      particle.topology = request.candidates.Metadata().Topology(candidate);
      particle.outputClass = request.candidates.Metadata().OutputClass(candidate);
      particle.operationStatus = request.candidates.Metadata().OperationStatus(candidate);
      particle.pdg = request.candidates.Metadata().Pdg(candidate);
      particle.primaryVertexIndex =
        request.candidates.Metadata().PrimaryVertexIndex(candidate);
      LoadCandidateFit(request.candidates, candidate, particle.fit);
      if (!Finite(particle.fit)) {
        result.status = KFGpuMaterializationInvalidNumericalState;
        result.particles.clear();
        return result;
      }

      const unsigned int directCount =
        request.candidates.Metadata().DirectDaughterCount(candidate);
      for (unsigned int daughter = 0u; daughter < directCount; ++daughter) {
        const unsigned int kind = daughter == 0u
                                    ? request.candidates.Metadata().DirectFirstKind(candidate)
                                    : request.candidates.Metadata().DirectSecondKind(candidate);
        const unsigned int index = daughter == 0u
                                     ? request.candidates.Metadata().DirectFirstIndex(candidate)
                                     : request.candidates.Metadata().DirectSecondIndex(candidate);
        const int particleId = kind == KFGpuDirectDaughterInputTrack
                                 ? trackParticleIds[index]
                                 : candidateParticleIds[index];
        if (particleId < 0) {
          result.status = KFGpuMaterializationInvalidDirectDaughter;
          result.particles.clear();
          return result;
        }
        particle.daughterParticleIds.push_back(particleId);
        const auto& directParticle =
          result.particles[static_cast<std::size_t>(particleId)];
        const auto& leaves = directParticle.leafSourceIds;
        particle.leafSourceIds.insert(
          particle.leafSourceIds.end(), leaves.begin(), leaves.end());
        if (daughter == 0u) {
          for (unsigned int component = 0u; component < 10u; ++component) {
            particle.fieldCoefficients[component] =
              directParticle.fieldCoefficients[component];
          }
        }
      }
      Canonicalize(particle.leafSourceIds);

      std::vector<int> storedLineage;
      const unsigned int offset = request.candidates.Metadata().DaughterOffset(candidate);
      const unsigned int count = request.candidates.Metadata().DaughterCount(candidate);
      if (!request.candidates.Daughters().CanStore(offset, count)
          || offset > request.candidates.Daughters().Size()
          || count > request.candidates.Daughters().Size() - offset) {
        result.status = KFGpuMaterializationInvalidRange;
        result.particles.clear();
        return result;
      }
      for (unsigned int daughter = 0u; daughter < count; ++daughter) {
        storedLineage.push_back(request.candidates.Daughters().SourceId(offset + daughter));
      }
      Canonicalize(storedLineage);
      if (storedLineage != particle.leafSourceIds) {
        result.status = KFGpuMaterializationLineageMismatch;
        result.particles.clear();
        return result;
      }

      candidateParticleIds[candidate] = static_cast<int>(result.particles.size());
      result.particles.push_back(particle);
    }
    return result;
  }

  template<typename Particle>
  static bool CommitToParticles(const KFParticleGpuMaterializationRequest& request,
                                std::vector<Particle>& destination,
                                std::vector<KFParticleGpuMaterializedParticle>* metadata = nullptr,
                                KFParticleGpuMaterializationStatus* status = nullptr)
  {
    KFParticleGpuMaterializedEvent event = Build(request);
    if (status) {
      *status = event.status;
    }
    if (!event.Succeeded()) {
      return false;
    }

    std::vector<Particle> temporary;
    temporary.reserve(event.particles.size());
    for (const auto& source : event.particles) {
      Particle particle;
      for (int component = 0; component < KFParticleGpuFitState::NumberOfParameters; ++component) {
        particle.Parameter(component) = source.fit.Parameter(component);
      }
      for (int component = 0; component < KFParticleGpuFitState::NumberOfCovarianceElements; ++component) {
        particle.Covariance(component) = source.fit.Covariance(component);
      }
      particle.Chi2() = source.fit.Chi2();
      particle.NDF() = source.fit.NDF();
      particle.Q() = static_cast<char>(source.fit.Q());
      particle.SetSFromDecay(source.fit.SFromDecay());
      particle.SetSumDaughterMass(source.fit.SumDaughterMass());
      particle.SetMassHypo(source.fit.MassHypo());
      particle.SetConstructMethod(source.fit.ConstructMethod());
      particle.SetPDG(source.pdg);
      particle.SetId(static_cast<int>(temporary.size()));
#ifdef NonhomogeneousField
      for (int component = 0; component < 10; ++component) {
        particle.SetFieldCoeff(source.fieldCoefficients[component], component);
      }
#endif
      if (source.inputTrack) {
        particle.AddDaughterId(source.sourceId);
      }
      else {
        for (int daughterId : source.daughterParticleIds) {
          particle.AddDaughterId(daughterId);
        }
      }
      temporary.push_back(particle);
    }

    destination.swap(temporary);
    if (metadata) {
      metadata->swap(event.particles);
    }
    return true;
  }

 private:
  static float TrackMass(int pdg)
  {
    switch (pdg < 0 ? -pdg : pdg) {
      case 11: return 0.00051099895f;
      case 13: return 0.1056583755f;
      case 321: return 0.493677f;
      case 2212: return 0.9382720813f;
      case 1000010020: return 1.87561294257f;
      case 1000010030: return 2.80892113298f;
      case 1000020030: return 2.80839160743f;
      case 1000020040: return 3.7273794066f;
      case 1000020060: return 5.6055375f;
      case 1000030060: return 5.6015181f;
      case 1000030070: return 6.5338336f;
      case 1000040070: return 6.5341844f;
      default: return 0.13957039f;
    }
  }

  static bool Finite(const KFParticleGpuFitState& fit)
  {
    for (int component = 0; component < KFParticleGpuFitState::NumberOfParameters; ++component) {
      if (!std::isfinite(fit.Parameter(component))) {
        return false;
      }
    }
    for (int component = 0; component < KFParticleGpuFitState::NumberOfCovarianceElements; ++component) {
      if (!std::isfinite(fit.Covariance(component))) {
        return false;
      }
    }
    return std::isfinite(fit.Chi2()) && std::isfinite(fit.SFromDecay())
           && std::isfinite(fit.SumDaughterMass()) && std::isfinite(fit.MassHypo())
           && fit.Q() >= static_cast<int>(std::numeric_limits<char>::min())
           && fit.Q() <= static_cast<int>(std::numeric_limits<char>::max());
  }

  static void Canonicalize(std::vector<int>& lineage)
  {
    std::sort(lineage.begin(), lineage.end());
    lineage.erase(std::unique(lineage.begin(), lineage.end()), lineage.end());
  }
};

#endif
