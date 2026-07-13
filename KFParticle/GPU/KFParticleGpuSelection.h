/*
 * This file is part of KFParticle package
 * Copyright (C) 2007-2026
 *
 * KFParticle is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 */

#ifndef KFPARTICLEGPUSELECTION_H
#define KFPARTICLEGPUSELECTION_H

#include "KFParticleGpuCandidateTransfer.h"
#include "KFParticleGpuInputDataTransfer.h"
#include "KFParticleGpuMath.h"
#include "KFParticleGpuVertexState.h"

/** Coarse CPU-Finder-compatible destination of a constructed default V0. */
enum KFParticleGpuV0SelectionClass
{
  KFGpuV0SelectionNotEvaluated = 0,
  KFGpuV0SelectionRejected,
  KFGpuV0SelectionSecondary,
  KFGpuV0SelectionPrimary
};

/**
 * Independent reasons why a constructed candidate cannot enter a selected
 * V0 output. More than one bit may be set by the future selection kernel.
 */
enum KFParticleGpuV0SelectionRejection
{
  KFGpuV0SelectionRejectNone = 0u,
  KFGpuV0SelectionRejectBuild = 1u << 0,
  KFGpuV0SelectionRejectNonFinite = 1u << 1,
  KFGpuV0SelectionRejectGeometricChi2 = 1u << 2,
  KFGpuV0SelectionRejectNoPrimaryVertex = 1u << 3,
  KFGpuV0SelectionRejectDistance = 1u << 4,
  KFGpuV0SelectionRejectDecayLength = 1u << 5,
  KFGpuV0SelectionRejectMass = 1u << 6,
  KFGpuV0SelectionRejectTopology = 1u << 7,
  KFGpuV0SelectionRejectOutputOverflow = 1u << 8
};

/**
 * Flat observables retained for diagnostics before selected-output compaction.
 * Defaults are neutral; the future topology stage writes the measured values.
 */
struct KFParticleGpuV0SelectionObservables
{
  float mass = 0.f;
  float massError = 0.f;
  float geometricChi2PerNdf = 0.f;
  float nearestPrimaryVertexDistance = 0.f;
  float nearestPrimaryVertexDistanceError = 0.f;
  float nearestPrimaryVertexLdL = 0.f;
  float bestPrimaryVertexTopoChi2PerNdf = 0.f;
  float bestPrimaryVertexDecayLength = 0.f;
  float bestPrimaryVertexDecayLengthError = 0.f;
  float bestPrimaryVertexLdL = 0.f;
};

/**
 * One non-owning selection record for a raw candidate-pool entry. The raw pool
 * remains the source of fit and daughter data until a later compact stage.
 */
struct KFParticleGpuV0SelectionResult
{
  unsigned int candidateIndex = 0u;
  unsigned int channelId = 0u;
  unsigned int eventIndex = 0u;
  int bestPrimaryVertexIndex = -1;
  unsigned int selectionClass = KFGpuV0SelectionNotEvaluated;
  unsigned int rejectionReasons = KFGpuV0SelectionRejectNone;
  KFParticleGpuV0SelectionObservables observables;
};

template<typename ResultValue>
class KFParticleGpuV0SelectionResultViewBase
{
 public:
  KFPARTICLE_GPU_HOST_DEVICE KFParticleGpuV0SelectionResultViewBase() = default;

  KFPARTICLE_GPU_HOST_DEVICE KFParticleGpuV0SelectionResultViewBase(ResultValue* results,
                                                                     unsigned int capacity)
    : fResults(results), fCapacity(capacity)
  {
  }

  KFPARTICLE_GPU_HOST_DEVICE ResultValue& Result(unsigned int candidateIndex) const
  {
    return fResults[candidateIndex];
  }
  KFPARTICLE_GPU_HOST_DEVICE unsigned int Capacity() const { return fCapacity; }
  KFPARTICLE_GPU_HOST_DEVICE bool CanStore(unsigned int candidateIndex) const
  {
    return fResults && candidateIndex < fCapacity;
  }
  KFPARTICLE_GPU_HOST_DEVICE ResultValue* Data() const { return fResults; }

 private:
  ResultValue* fResults;
  unsigned int fCapacity;
};

typedef KFParticleGpuV0SelectionResultViewBase<KFParticleGpuV0SelectionResult>
  KFParticleGpuV0SelectionResultView;
typedef KFParticleGpuV0SelectionResultViewBase<const KFParticleGpuV0SelectionResult>
  KFParticleGpuConstV0SelectionResultView;

KFPARTICLE_GPU_HOST_DEVICE inline KFParticleGpuConstV0SelectionResultView
MakeConstView(const KFParticleGpuV0SelectionResultView& view)
{
  return KFParticleGpuConstV0SelectionResultView(view.Data(), view.Capacity());
}

/**
 * Minimum candidate-to-primary-vertex observables needed by the first V0
 * selection stage. Topological chi2 is a spatial-residual approximation here;
 * it is not a replacement for the CPU constrained production-vertex fit.
 */
struct KFParticleGpuPrimaryVertexTopologyObservables
{
  float distance = 0.f;
  float distanceError = 1.e8f;
  float ldL = 0.f;
  float chi2PerNdf = 1.e8f;
  unsigned int valid = 0u;
};

/** Per-channel selection thresholds; negative floating limits disable a cut. */
struct KFParticleGpuV0SelectionConfig
{
  float expectedMass = 0.f;
  float expectedMassSigma = -1.f;
  float massSigmaCut = -1.f;
  float maxGeometricChi2PerNdf = -1.f;
  float maxPrimaryVertexDistance = -1.f;
  float minSecondaryLdL = -1.f;
  float maxPrimaryTopologyChi2PerNdf = -1.f;
  float maxSecondaryTopologyChi2PerNdf = -1.f;
  unsigned int requirePrimaryVertex = 0u;
};

enum KFParticleGpuSelectedCandidateOverflow
{
  KFGpuSelectedCandidateCapacityExceeded = 1u << 0
};

/** Non-owning compact output of raw candidate-pool indices. */
template<typename UnsignedValue>
class KFParticleGpuSelectedCandidateIndexViewBase
{
 public:
  KFPARTICLE_GPU_HOST_DEVICE KFParticleGpuSelectedCandidateIndexViewBase() = default;

  KFPARTICLE_GPU_HOST_DEVICE KFParticleGpuSelectedCandidateIndexViewBase(
    UnsignedValue* indices,
    UnsignedValue* channelIds,
    UnsignedValue* size,
    UnsignedValue* overflowFlags,
    unsigned int capacity)
    : fIndices(indices), fChannelIds(channelIds), fSize(size), fOverflowFlags(overflowFlags), fCapacity(capacity)
  {
  }

  KFPARTICLE_GPU_HOST_DEVICE KFParticleGpuSelectedCandidateIndexViewBase(
    UnsignedValue* indices,
    UnsignedValue* size,
    UnsignedValue* overflowFlags,
    unsigned int capacity)
    : KFParticleGpuSelectedCandidateIndexViewBase(indices, nullptr, size, overflowFlags, capacity)
  {
  }

  KFPARTICLE_GPU_HOST_DEVICE UnsignedValue& Index(unsigned int index) const { return fIndices[index]; }
  KFPARTICLE_GPU_HOST_DEVICE UnsignedValue& ChannelId(unsigned int index) const
  {
    return fChannelIds[index];
  }
  KFPARTICLE_GPU_HOST_DEVICE unsigned int Size() const
  {
    return fSize ? static_cast<unsigned int>(*fSize) : 0u;
  }
  KFPARTICLE_GPU_HOST_DEVICE unsigned int Capacity() const { return fCapacity; }
  KFPARTICLE_GPU_HOST_DEVICE unsigned int OverflowFlags() const
  {
    return fOverflowFlags ? static_cast<unsigned int>(*fOverflowFlags) : 0u;
  }
  KFPARTICLE_GPU_HOST_DEVICE bool CanStore(unsigned int index) const { return index < fCapacity; }
  KFPARTICLE_GPU_HOST_DEVICE UnsignedValue* IndicesData() const { return fIndices; }
  KFPARTICLE_GPU_HOST_DEVICE UnsignedValue* ChannelIdsData() const { return fChannelIds; }
  KFPARTICLE_GPU_HOST_DEVICE UnsignedValue* SizeData() const { return fSize; }
  KFPARTICLE_GPU_HOST_DEVICE UnsignedValue* OverflowFlagsData() const { return fOverflowFlags; }

 private:
  UnsignedValue* fIndices;
  UnsignedValue* fChannelIds;
  UnsignedValue* fSize;
  UnsignedValue* fOverflowFlags;
  unsigned int fCapacity;
};

typedef KFParticleGpuSelectedCandidateIndexViewBase<unsigned int>
  KFParticleGpuSelectedCandidateIndexView;
typedef KFParticleGpuSelectedCandidateIndexViewBase<const unsigned int>
  KFParticleGpuConstSelectedCandidateIndexView;

KFPARTICLE_GPU_HOST_DEVICE inline KFParticleGpuConstSelectedCandidateIndexView
MakeConstView(const KFParticleGpuSelectedCandidateIndexView& view)
{
  return KFParticleGpuConstSelectedCandidateIndexView(
    view.IndicesData(), view.ChannelIdsData(), view.SizeData(), view.OverflowFlagsData(), view.Capacity());
}

struct KFParticleGpuSelectedCandidateRange
{
  unsigned int offset = 0u;
  unsigned int size = 0u;
  unsigned int overflowFlags = 0u;

  KFPARTICLE_GPU_HOST_DEVICE bool Empty() const { return size == 0u; }
  KFPARTICLE_GPU_HOST_DEVICE unsigned int End() const { return offset + size; }
  KFPARTICLE_GPU_HOST_DEVICE bool Truncated() const
  {
    return (overflowFlags & KFGpuSelectedCandidateCapacityExceeded) != 0u;
  }
};

struct KFParticleGpuSelectedChannelRange
{
  unsigned int channelId = 0u;
  unsigned int eventIndex = 0u;
  KFParticleGpuSelectedCandidateRange candidates;
};

/**
 * Non-owning downstream view of compact selected V0s.
 *
 * Selection stores raw-pool indices only; this view resolves fit metadata and
 * ordered daughter lineage in the original SoA pools without a copy.
 */
class KFParticleGpuSelectedV0View
{
 public:
  KFPARTICLE_GPU_HOST_DEVICE KFParticleGpuSelectedV0View() = default;

  KFPARTICLE_GPU_HOST_DEVICE KFParticleGpuSelectedV0View(
    const KFParticleGpuConstCandidatePoolView& candidates,
    const KFParticleGpuConstSelectedCandidateIndexView& selected,
    const KFParticleGpuConstV0SelectionResultView& results = KFParticleGpuConstV0SelectionResultView())
    : fCandidates(candidates), fSelected(selected), fResults(results)
  {
  }

  KFPARTICLE_GPU_HOST_DEVICE unsigned int Size() const { return fSelected.Size(); }
  KFPARTICLE_GPU_HOST_DEVICE unsigned int OverflowFlags() const { return fSelected.OverflowFlags(); }
  KFPARTICLE_GPU_HOST_DEVICE bool HasSelectionResults() const
  {
    return fResults.Data() != nullptr;
  }
  KFPARTICLE_GPU_HOST_DEVICE unsigned int CandidateIndex(unsigned int selectedIndex) const
  {
    return fSelected.Index(selectedIndex);
  }
  KFPARTICLE_GPU_HOST_DEVICE int Pdg(unsigned int selectedIndex) const
  {
    return fCandidates.Metadata().Pdg(CandidateIndex(selectedIndex));
  }
  KFPARTICLE_GPU_HOST_DEVICE unsigned int ChannelId(unsigned int selectedIndex) const
  {
    return fSelected.ChannelIdsData() ? fSelected.ChannelId(selectedIndex)
                                      : fCandidates.Metadata().ChannelId(CandidateIndex(selectedIndex));
  }
  KFPARTICLE_GPU_HOST_DEVICE unsigned int EventIndex(unsigned int selectedIndex) const
  {
    return fCandidates.Metadata().EventIndex(CandidateIndex(selectedIndex));
  }
  KFPARTICLE_GPU_HOST_DEVICE unsigned int DaughterCount(unsigned int selectedIndex) const
  {
    return fCandidates.Metadata().DaughterCount(CandidateIndex(selectedIndex));
  }
  KFPARTICLE_GPU_HOST_DEVICE int DaughterSourceId(unsigned int selectedIndex,
                                                  unsigned int daughterIndex) const
  {
    const unsigned int candidateIndex = CandidateIndex(selectedIndex);
    return fCandidates.Daughters().SourceId(
      fCandidates.Metadata().DaughterOffset(candidateIndex) + daughterIndex);
  }
  KFPARTICLE_GPU_HOST_DEVICE void LoadFitState(unsigned int selectedIndex,
                                               KFParticleGpuFitState& destination) const
  {
    ::LoadCandidateFit(fCandidates, CandidateIndex(selectedIndex), destination);
  }
  KFPARTICLE_GPU_HOST_DEVICE const KFParticleGpuV0SelectionResult& Selection(
    unsigned int selectedIndex) const
  {
    return fResults.Result(CandidateIndex(selectedIndex));
  }

 private:
  KFParticleGpuConstCandidatePoolView fCandidates;
  KFParticleGpuConstSelectedCandidateIndexView fSelected;
  KFParticleGpuConstV0SelectionResultView fResults;
};

KFPARTICLE_GPU_HOST_DEVICE inline KFParticleGpuSelectedV0View MakeSelectedV0View(
  const KFParticleGpuConstCandidatePoolView& candidates,
  const KFParticleGpuConstSelectedCandidateIndexView& selected,
  const KFParticleGpuConstV0SelectionResultView& results = KFParticleGpuConstV0SelectionResultView())
{
  return KFParticleGpuSelectedV0View(candidates, selected, results);
}

namespace KFParticleGpuSelection
{
  KFPARTICLE_GPU_HOST_DEVICE inline bool IsFiniteValue(float value)
  {
    return value == value && value > -1.e30f && value < 1.e30f;
  }

  KFPARTICLE_GPU_HOST_DEVICE inline bool IsFiniteFitState(
    const KFParticleGpuFitState& candidate)
  {
    for (int i = 0; i < KFParticleGpuFitState::NumberOfParameters; ++i) {
      if (!IsFiniteValue(candidate.Parameter(i))) {
        return false;
      }
    }
    return IsFiniteValue(candidate.Chi2());
  }

  KFPARTICLE_GPU_HOST_DEVICE inline bool BuildPrimaryVertexTopologyObservables(
    const KFParticleGpuFitState& candidate,
    const KFParticleGpuVertexState& primaryVertex,
    KFParticleGpuPrimaryVertexTopologyObservables& observables)
  {
    const float dx = primaryVertex.X() - candidate.X();
    const float dy = primaryVertex.Y() - candidate.Y();
    const float dz = primaryVertex.Z() - candidate.Z();
    const float distance2 = dx * dx + dy * dy + dz * dz;
    observables.distance = distance2 > 1.e-16f ? KFParticleGpuSqrt(distance2) : 1.e-8f;

    const float c00 = candidate.Covariance(0, 0) + primaryVertex.Covariance(0);
    const float c10 = candidate.Covariance(1, 0) + primaryVertex.Covariance(1);
    const float c11 = candidate.Covariance(1, 1) + primaryVertex.Covariance(2);
    const float c20 = candidate.Covariance(2, 0) + primaryVertex.Covariance(3);
    const float c21 = candidate.Covariance(2, 1) + primaryVertex.Covariance(4);
    const float c22 = candidate.Covariance(2, 2) + primaryVertex.Covariance(5);
    const float variance = c00 * dx * dx + c11 * dy * dy + c22 * dz * dz
                           + 2.f * (c10 * dx * dy + c20 * dx * dz + c21 * dy * dz);
    if (variance <= 0.f) {
      observables.distanceError = 1.e8f;
      observables.ldL = 0.f;
      observables.chi2PerNdf = 1.e8f;
      observables.valid = 0u;
      return false;
    }

    observables.distanceError = KFParticleGpuSqrt(variance) / observables.distance;
    observables.ldL = observables.distance / observables.distanceError;

    const float determinant = c00 * (c11 * c22 - c21 * c21)
                              - c10 * (c10 * c22 - c20 * c21)
                              + c20 * (c10 * c21 - c20 * c11);
    if (determinant <= 1.e-12f) {
      observables.chi2PerNdf = 1.e8f;
      observables.valid = 0u;
      return false;
    }

    const float inverse00 = (c11 * c22 - c21 * c21) / determinant;
    const float inverse10 = (c20 * c21 - c10 * c22) / determinant;
    const float inverse20 = (c10 * c21 - c20 * c11) / determinant;
    const float inverse11 = (c00 * c22 - c20 * c20) / determinant;
    const float inverse21 = (c10 * c20 - c00 * c21) / determinant;
    const float inverse22 = (c00 * c11 - c10 * c10) / determinant;
    const float chi2 = dx * dx * inverse00 + dy * dy * inverse11 + dz * dz * inverse22
                       + 2.f * (dx * dy * inverse10 + dx * dz * inverse20
                                 + dy * dz * inverse21);
    if (chi2 < 0.f) {
      observables.chi2PerNdf = 1.e8f;
      observables.valid = 0u;
      return false;
    }

    observables.chi2PerNdf = chi2 / 3.f;
    observables.valid = 1u;
    return true;
  }

  KFPARTICLE_GPU_HOST_DEVICE inline bool FindBestPrimaryVertexTopology(
    const KFParticleGpuFitState& candidate,
    const KFParticleGpuConstVertexSoAView& primaryVertices,
    const KFParticleGpuRange& vertexRange,
    int& bestPrimaryVertexIndex,
    KFParticleGpuPrimaryVertexTopologyObservables& bestObservables)
  {
    bestPrimaryVertexIndex = -1;
    bestObservables = KFParticleGpuPrimaryVertexTopologyObservables();
    if (vertexRange.size == 0u || vertexRange.offset > primaryVertices.Size()
        || vertexRange.size > primaryVertices.Size() - vertexRange.offset) {
      return false;
    }

    for (unsigned int localIndex = 0; localIndex < vertexRange.size; ++localIndex) {
      const unsigned int vertexIndex = vertexRange.offset + localIndex;
      KFParticleGpuVertexState vertex;
      LoadVertexState(primaryVertices, vertexIndex, vertex);
      KFParticleGpuPrimaryVertexTopologyObservables observables;
      if (!BuildPrimaryVertexTopologyObservables(candidate, vertex, observables)) {
        continue;
      }
      if (bestPrimaryVertexIndex < 0
          || observables.chi2PerNdf < bestObservables.chi2PerNdf) {
        bestPrimaryVertexIndex = static_cast<int>(vertexIndex);
        bestObservables = observables;
      }
    }
    return bestPrimaryVertexIndex >= 0;
  }

  KFPARTICLE_GPU_HOST_DEVICE inline void EvaluateV0Selection(
    const KFParticleGpuFitState& candidate,
    unsigned int candidateFlags,
    unsigned int candidateIndex,
    unsigned int channelId,
    unsigned int eventIndex,
    const KFParticleGpuConstVertexSoAView& primaryVertices,
    const KFParticleGpuRange& vertexRange,
    const KFParticleGpuV0SelectionConfig& config,
    KFParticleGpuV0SelectionResult& result)
  {
    result = KFParticleGpuV0SelectionResult();
    result.candidateIndex = candidateIndex;
    result.channelId = channelId;
    result.eventIndex = eventIndex;
    result.selectionClass = KFGpuV0SelectionRejected;

    if ((candidateFlags & KFGpuCandidateValid) == 0u
        || (candidateFlags & KFGpuCandidateBuildFailed) != 0u) {
      result.rejectionReasons |= KFGpuV0SelectionRejectBuild;
    }
    if (!IsFiniteFitState(candidate)) {
      result.rejectionReasons |= KFGpuV0SelectionRejectNonFinite;
      return;
    }

    result.observables.geometricChi2PerNdf = candidate.NDF() > 0
                                                ? candidate.Chi2()
                                                    / static_cast<float>(candidate.NDF())
                                                : 1.e8f;
    if (config.maxGeometricChi2PerNdf >= 0.f
        && result.observables.geometricChi2PerNdf > config.maxGeometricChi2PerNdf) {
      result.rejectionReasons |= KFGpuV0SelectionRejectGeometricChi2;
    }

    float mass = 0.f;
    float massError = 0.f;
    const bool massIsValid = KFParticleGpuMath::GetMass(candidate, mass, massError);
    result.observables.mass = mass;
    result.observables.massError = massError;
    if (!massIsValid
        || (config.expectedMassSigma >= 0.f && config.massSigmaCut >= 0.f
            && KFParticleGpuMath::Abs(mass - config.expectedMass)
                 > config.expectedMassSigma * config.massSigmaCut)) {
      result.rejectionReasons |= KFGpuV0SelectionRejectMass;
    }

    const bool needsPrimaryVertex = config.requirePrimaryVertex != 0u
                                    || config.maxPrimaryVertexDistance >= 0.f
                                    || config.minSecondaryLdL >= 0.f
                                    || config.maxPrimaryTopologyChi2PerNdf >= 0.f
                                    || config.maxSecondaryTopologyChi2PerNdf >= 0.f;
    KFParticleGpuPrimaryVertexTopologyObservables nearestObservables;
    KFParticleGpuPrimaryVertexTopologyObservables bestObservables;
    int nearestPrimaryVertexIndex = -1;
    int bestPrimaryVertexIndex = -1;

    if (needsPrimaryVertex && vertexRange.size > 0u && vertexRange.offset <= primaryVertices.Size()
        && vertexRange.size <= primaryVertices.Size() - vertexRange.offset) {
      for (unsigned int localIndex = 0; localIndex < vertexRange.size; ++localIndex) {
        const unsigned int vertexIndex = vertexRange.offset + localIndex;
        KFParticleGpuVertexState vertex;
        LoadVertexState(primaryVertices, vertexIndex, vertex);
        KFParticleGpuPrimaryVertexTopologyObservables observables;
        if (!BuildPrimaryVertexTopologyObservables(candidate, vertex, observables)) {
          continue;
        }
        if (nearestPrimaryVertexIndex < 0 || observables.distance < nearestObservables.distance) {
          nearestPrimaryVertexIndex = static_cast<int>(vertexIndex);
          nearestObservables = observables;
        }
        if (bestPrimaryVertexIndex < 0
            || observables.chi2PerNdf < bestObservables.chi2PerNdf) {
          bestPrimaryVertexIndex = static_cast<int>(vertexIndex);
          bestObservables = observables;
        }
      }
    }

    if (needsPrimaryVertex && (nearestPrimaryVertexIndex < 0 || bestPrimaryVertexIndex < 0)) {
      result.rejectionReasons |= KFGpuV0SelectionRejectNoPrimaryVertex;
    }
    else if (needsPrimaryVertex) {
      result.bestPrimaryVertexIndex = bestPrimaryVertexIndex;
      result.observables.nearestPrimaryVertexDistance = nearestObservables.distance;
      result.observables.nearestPrimaryVertexDistanceError = nearestObservables.distanceError;
      result.observables.nearestPrimaryVertexLdL = nearestObservables.ldL;
      result.observables.bestPrimaryVertexTopoChi2PerNdf = bestObservables.chi2PerNdf;
      result.observables.bestPrimaryVertexDecayLength = nearestObservables.distance;
      result.observables.bestPrimaryVertexDecayLengthError = nearestObservables.distanceError;
      result.observables.bestPrimaryVertexLdL = nearestObservables.ldL;

      if (config.maxPrimaryVertexDistance >= 0.f
          && nearestObservables.distance >= config.maxPrimaryVertexDistance) {
        result.rejectionReasons |= KFGpuV0SelectionRejectDistance;
      }

      const bool primary = config.maxPrimaryTopologyChi2PerNdf >= 0.f
                           && bestObservables.chi2PerNdf < config.maxPrimaryTopologyChi2PerNdf;
      if (!primary && config.minSecondaryLdL >= 0.f
          && nearestObservables.ldL <= config.minSecondaryLdL) {
        result.rejectionReasons |= KFGpuV0SelectionRejectDecayLength;
      }
      if (!primary && config.maxSecondaryTopologyChi2PerNdf >= 0.f
          && bestObservables.chi2PerNdf >= config.maxSecondaryTopologyChi2PerNdf) {
        result.rejectionReasons |= KFGpuV0SelectionRejectTopology;
      }
      if (result.rejectionReasons == KFGpuV0SelectionRejectNone) {
        result.selectionClass = primary ? KFGpuV0SelectionPrimary : KFGpuV0SelectionSecondary;
      }
      return;
    }

    if (result.rejectionReasons == KFGpuV0SelectionRejectNone) {
      result.selectionClass = KFGpuV0SelectionSecondary;
    }
  }

  KFPARTICLE_GPU_HOST_DEVICE inline bool HasRejection(
    const KFParticleGpuV0SelectionResult& result,
    unsigned int rejection)
  {
    return (result.rejectionReasons & rejection) != 0u;
  }

  KFPARTICLE_GPU_HOST_DEVICE inline bool IsSelected(
    const KFParticleGpuV0SelectionResult& result)
  {
    return result.rejectionReasons == KFGpuV0SelectionRejectNone
           && (result.selectionClass == KFGpuV0SelectionSecondary
               || result.selectionClass == KFGpuV0SelectionPrimary);
  }

  KFPARTICLE_GPU_HOST_DEVICE inline bool IsInsideMassWindow(float mass,
                                                            float expectedMass,
                                                            float halfWidth)
  {
    return halfWidth >= 0.f && KFParticleGpuMath::Abs(mass - expectedMass) <= halfWidth;
  }

  KFPARTICLE_GPU_HOST_DEVICE inline float MassWindowHalfWidth(float expectedSigma,
                                                              float sigmaCut)
  {
    return expectedSigma >= 0.f && sigmaCut >= 0.f ? expectedSigma * sigmaCut : -1.f;
  }

  KFPARTICLE_GPU_HOST_DEVICE inline float Chi2PerNdf(const KFParticleGpuFitState& particle)
  {
    return particle.NDF() > 0 ? particle.Chi2() / static_cast<float>(particle.NDF()) : 1.e8f;
  }

  KFPARTICLE_GPU_HOST_DEVICE inline bool PassChi2PerNdf(const KFParticleGpuFitState& particle,
                                                        float maxChi2PerNdf)
  {
    return maxChi2PerNdf < 0.f || Chi2PerNdf(particle) <= maxChi2PerNdf;
  }

  KFPARTICLE_GPU_HOST_DEVICE inline bool PassMassWindow(const KFParticleGpuFitState& particle,
                                                        float expectedMass,
                                                        float halfWidth)
  {
    return IsInsideMassWindow(KFParticleGpuMath::Mass(particle), expectedMass, halfWidth);
  }

  KFPARTICLE_GPU_HOST_DEVICE inline bool PassSigmaMassWindow(const KFParticleGpuFitState& particle,
                                                             float expectedMass,
                                                             float expectedSigma,
                                                             float sigmaCut)
  {
    return PassMassWindow(
      particle, expectedMass, MassWindowHalfWidth(expectedSigma, sigmaCut));
  }

  KFPARTICLE_GPU_HOST_DEVICE inline bool HasDistinctDaughterSourceIds(
    const KFParticleGpuConstDaughterStorageView& daughters,
    unsigned int offset,
    unsigned int count)
  {
    if (!daughters.CanStore(offset, count)) {
      return false;
    }

    // The flattened lineage is small for the first decay kernels, so an
    // in-place quadratic check avoids temporary device storage.
    for (unsigned int i = 0; i < count; ++i) {
      for (unsigned int j = i + 1; j < count; ++j) {
        if (daughters.SourceId(offset + i) == daughters.SourceId(offset + j)) {
          return false;
        }
      }
    }
    return true;
  }

  KFPARTICLE_GPU_HOST_DEVICE inline bool HasDistinctDaughterSourceIds(
    const KFParticleGpuConstCandidatePoolView& candidates,
    unsigned int candidateIndex)
  {
    if (candidateIndex >= candidates.Size() || candidateIndex >= candidates.Capacity()) {
      return false;
    }
    return HasDistinctDaughterSourceIds(
      candidates.Daughters(),
      candidates.Metadata().DaughterOffset(candidateIndex),
      candidates.Metadata().DaughterCount(candidateIndex));
  }
}

#endif
