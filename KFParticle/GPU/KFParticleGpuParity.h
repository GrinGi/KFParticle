/*
 * This file is part of KFParticle package
 * Copyright (C) 2007-2026
 *
 * KFParticle is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 */

#ifndef KFPARTICLEGPUPARITY_H
#define KFPARTICLEGPUPARITY_H

#include "KFParticleGpuCandidateTransfer.h"
#include "KFParticleGpuMath.h"
#include "KFParticleGpuSelection.h"

#include <cstdint>
#include <type_traits>

static const unsigned int KFGpuParityMaximumLineageSize = 16u;

enum KFParticleGpuParityAvailable : unsigned int
{
  KFGpuParityKeyAvailable = 1u << 0u,
  KFGpuParityParametersAvailable = 1u << 1u,
  KFGpuParityCovarianceAvailable = 1u << 2u,
  KFGpuParityFitQualityAvailable = 1u << 3u,
  KFGpuParityFitMetadataAvailable = 1u << 4u,
  KFGpuParityMassAvailable = 1u << 5u,
  KFGpuParitySelectionAvailable = 1u << 6u,
  KFGpuParityOperationAvailable = 1u << 7u
};

/** Stable order-independent identity shared by CPU and GPU candidates. */
struct KFParticleGpuParityCandidateKey
{
  std::uint64_t eventId = 0u;
  unsigned int channelId = 0u;
  int pdg = 0;
  unsigned int lineageSize = 0u;
  int lineage[KFGpuParityMaximumLineageSize] = {};
};

/** Flat comparison record; no pointer in this type may cross the device ABI. */
struct KFParticleGpuParitySnapshot
{
  KFParticleGpuParityCandidateKey key;
  float parameters[KFParticleGpuFitState::NumberOfParameters] = {};
  float covariance[KFParticleGpuFitState::NumberOfCovarianceElements] = {};
  float chi2 = 0.f;
  float sFromDecay = 0.f;
  float sumDaughterMass = 0.f;
  float massHypothesis = -1.f;
  float mass = 0.f;
  float massError = 0.f;
  int ndf = 0;
  int charge = 0;
  int atProductionVertex = 0;
  int constructMethod = 0;
  int primaryVertexIndex = -1;
  int bestPrimaryVertexIndex = -1;
  unsigned int candidateFlags = 0u;
  unsigned int topology = KFGpuGraphTopologyInvalid;
  unsigned int outputClass = KFGpuGraphOutputInvalid;
  unsigned int operationStatus = KFGpuCandidateOperationPending;
  unsigned int selectionClass = KFGpuV0SelectionNotEvaluated;
  unsigned int rejectionReasons = KFGpuV0SelectionRejectNone;
  unsigned int topologyStatus = KFGpuV0LineTopologyNone;
  KFParticleGpuV0SelectionObservables selectionObservables;
  unsigned int selected = 0u;
  unsigned int massValid = 0u;
  unsigned int available = 0u;
};

struct KFParticleGpuParityTolerance
{
  float parameterAbsolute = 1.e-4f;
  float parameterRelative = 0.f;
  float covarianceAbsolute = 1.e-3f;
  float covarianceRelative = 0.f;
  float fitScalarAbsolute = 1.e-4f;
  float fitScalarRelative = 0.f;
  float massAbsolute = 1.e-4f;
  float massRelative = 0.f;
  float massErrorAbsolute = 1.e-3f;
  float massErrorRelative = 0.f;
  float selectionAbsolute = 1.e-3f;
  float selectionRelative = 0.f;
};

enum KFParticleGpuParityIssue : unsigned int
{
  KFGpuParityNoIssue = 0u,
  KFGpuParityKeyMismatch = 1u << 0u,
  KFGpuParityParameterMismatch = 1u << 1u,
  KFGpuParityCovarianceMismatch = 1u << 2u,
  KFGpuParityFitScalarMismatch = 1u << 3u,
  KFGpuParityFitIntegerMismatch = 1u << 4u,
  KFGpuParityMassMismatch = 1u << 5u,
  KFGpuParitySelectionMismatch = 1u << 6u,
  KFGpuParityOperationMismatch = 1u << 7u,
  KFGpuParityNonFinite = 1u << 8u
};

struct KFParticleGpuParityDifference
{
  unsigned int issues = KFGpuParityNoIssue;
  unsigned int firstParameter = 0xffffffffu;
  unsigned int firstCovariance = 0xffffffffu;
  float maxParameterResidual = 0.f;
  float maxCovarianceResidual = 0.f;
  float maxFitScalarResidual = 0.f;
  float maxMassResidual = 0.f;
  float maxMassErrorResidual = 0.f;
  float maxSelectionResidual = 0.f;

  KFPARTICLE_GPU_HOST_DEVICE bool Equivalent() const { return issues == KFGpuParityNoIssue; }
};

enum KFParticleGpuPromotionBlocker : unsigned int
{
  KFGpuPromotionReady = 0u,
  KFGpuPromotionUnsupportedTopology = 1u << 0u,
  KFGpuPromotionOverflow = 1u << 1u,
  KFGpuPromotionInvalidInput = 1u << 2u,
  KFGpuPromotionInvalidField = 1u << 3u,
  KFGpuPromotionIncompleteLineage = 1u << 4u,
  KFGpuPromotionPhysicsMismatch = 1u << 5u,
  KFGpuPromotionRuntimeUnavailable = 1u << 6u,
  KFGpuPromotionExecutionFailure = 1u << 7u
};

struct KFParticleGpuPromotionVerdict
{
  unsigned int blockers = KFGpuPromotionReady;
  unsigned int comparedCandidates = 0u;
  unsigned int mismatchedCandidates = 0u;

  KFPARTICLE_GPU_HOST_DEVICE bool CanPromote() const
  {
    return blockers == KFGpuPromotionReady && mismatchedCandidates == 0u;
  }
};

namespace KFParticleGpuParity
{
  KFPARTICLE_GPU_HOST_DEVICE inline float Abs(float value)
  {
    return value < 0.f ? -value : value;
  }

  KFPARTICLE_GPU_HOST_DEVICE inline bool IsFinite(float value)
  {
    return value == value && value > -1.e30f && value < 1.e30f;
  }

  KFPARTICLE_GPU_HOST_DEVICE inline float Max(float left, float right)
  {
    return left > right ? left : right;
  }

  KFPARTICLE_GPU_HOST_DEVICE inline void Canonicalize(KFParticleGpuParityCandidateKey& key)
  {
    for (unsigned int i = 1u; i < key.lineageSize; ++i) {
      const int value = key.lineage[i];
      unsigned int position = i;
      while (position != 0u && key.lineage[position - 1u] > value) {
        key.lineage[position] = key.lineage[position - 1u];
        --position;
      }
      key.lineage[position] = value;
    }
  }

  KFPARTICLE_GPU_HOST_DEVICE inline bool Equal(const KFParticleGpuParityCandidateKey& left,
                                                const KFParticleGpuParityCandidateKey& right)
  {
    if (left.eventId != right.eventId || left.channelId != right.channelId
        || left.pdg != right.pdg || left.lineageSize != right.lineageSize) {
      return false;
    }
    for (unsigned int i = 0u; i < left.lineageSize; ++i) {
      if (left.lineage[i] != right.lineage[i]) { return false; }
    }
    return true;
  }

  KFPARTICLE_GPU_HOST_DEVICE inline bool Less(const KFParticleGpuParityCandidateKey& left,
                                               const KFParticleGpuParityCandidateKey& right)
  {
    if (left.eventId != right.eventId) { return left.eventId < right.eventId; }
    if (left.channelId != right.channelId) { return left.channelId < right.channelId; }
    if (left.pdg != right.pdg) { return left.pdg < right.pdg; }
    if (left.lineageSize != right.lineageSize) { return left.lineageSize < right.lineageSize; }
    for (unsigned int i = 0u; i < left.lineageSize; ++i) {
      if (left.lineage[i] != right.lineage[i]) { return left.lineage[i] < right.lineage[i]; }
    }
    return false;
  }

  KFPARTICLE_GPU_HOST_DEVICE inline bool BuildSnapshot(
    const KFParticleGpuConstCandidatePoolView& candidates,
    unsigned int candidateIndex,
    std::uint64_t sourceEventId,
    const KFParticleGpuV0SelectionResult* selection,
    bool selected,
    KFParticleGpuParitySnapshot& snapshot)
  {
    snapshot = KFParticleGpuParitySnapshot();
    if (candidateIndex >= candidates.Size() || candidateIndex >= candidates.Capacity()) {
      return false;
    }

    const auto& metadata = candidates.Metadata();
    const unsigned int lineageSize = metadata.DaughterCount(candidateIndex);
    const unsigned int lineageOffset = metadata.DaughterOffset(candidateIndex);
    if (lineageSize == 0u || lineageSize > KFGpuParityMaximumLineageSize
        || !candidates.Daughters().CanStore(lineageOffset, lineageSize)) {
      return false;
    }

    snapshot.key.eventId = sourceEventId;
    snapshot.key.channelId = metadata.ChannelId(candidateIndex);
    snapshot.key.pdg = metadata.Pdg(candidateIndex);
    snapshot.key.lineageSize = lineageSize;
    for (unsigned int i = 0u; i < lineageSize; ++i) {
      snapshot.key.lineage[i] = candidates.Daughters().SourceId(lineageOffset + i);
    }
    Canonicalize(snapshot.key);
    snapshot.available |= KFGpuParityKeyAvailable;

    KFParticleGpuFitState fit;
    LoadCandidateFit(candidates, candidateIndex, fit);
    for (int i = 0; i < KFParticleGpuFitState::NumberOfParameters; ++i) {
      snapshot.parameters[i] = fit.Parameter(i);
    }
    for (int i = 0; i < KFParticleGpuFitState::NumberOfCovarianceElements; ++i) {
      snapshot.covariance[i] = fit.Covariance(i);
    }
    snapshot.chi2 = fit.Chi2();
    snapshot.sFromDecay = fit.SFromDecay();
    snapshot.sumDaughterMass = fit.SumDaughterMass();
    snapshot.massHypothesis = fit.MassHypo();
    snapshot.ndf = fit.NDF();
    snapshot.charge = fit.Q();
    snapshot.atProductionVertex = fit.AtProductionVertex();
    snapshot.constructMethod = fit.ConstructMethod();
    snapshot.available |= KFGpuParityParametersAvailable | KFGpuParityCovarianceAvailable
                          | KFGpuParityFitQualityAvailable | KFGpuParityFitMetadataAvailable;

    snapshot.massValid = KFParticleGpuMath::GetMass(fit, snapshot.mass, snapshot.massError) ? 1u : 0u;
    snapshot.available |= KFGpuParityMassAvailable;
    snapshot.primaryVertexIndex = metadata.PrimaryVertexIndex(candidateIndex);
    snapshot.candidateFlags = metadata.Flags(candidateIndex);
    snapshot.topology = metadata.Topology(candidateIndex);
    snapshot.outputClass = metadata.OutputClass(candidateIndex);
    snapshot.operationStatus = metadata.OperationStatus(candidateIndex);
    snapshot.available |= KFGpuParityOperationAvailable;
    snapshot.selected = selected ? 1u : 0u;

    if (selection) {
      snapshot.bestPrimaryVertexIndex = selection->bestPrimaryVertexIndex;
      snapshot.selectionClass = selection->selectionClass;
      snapshot.rejectionReasons = selection->rejectionReasons;
      snapshot.topologyStatus = selection->topologyStatus;
      snapshot.selectionObservables = selection->observables;
      snapshot.available |= KFGpuParitySelectionAvailable;
    }
    return true;
  }

  KFPARTICLE_GPU_HOST_DEVICE inline void CompareFloat(float expected,
                                                       float actual,
                                                       float absoluteTolerance,
                                                       float relativeTolerance,
                                                       unsigned int issue,
                                                       unsigned int& issues,
                                                       float& maximum)
  {
    if (!IsFinite(expected) || !IsFinite(actual)) {
      issues |= KFGpuParityNonFinite | issue;
      return;
    }
    const float residual = Abs(expected - actual);
    maximum = Max(maximum, residual);
    const float scale = Max(Abs(expected), Abs(actual));
    const float tolerance = Max(absoluteTolerance, relativeTolerance * scale);
    if (residual > tolerance) { issues |= issue; }
  }

  KFPARTICLE_GPU_HOST_DEVICE inline bool MassValidityEquivalent(
    const KFParticleGpuParitySnapshot& expected,
    const KFParticleGpuParitySnapshot& actual,
    const KFParticleGpuParityTolerance& tolerance)
  {
    if (expected.massValid == actual.massValid) { return true; }
    const KFParticleGpuParitySnapshot& valid =
      expected.massValid != 0u ? expected : actual;
    return IsFinite(expected.mass) && IsFinite(actual.mass)
           && IsFinite(valid.massError)
           && Abs(expected.mass - actual.mass) <= tolerance.massAbsolute
           && valid.massError <= tolerance.massErrorAbsolute;
  }

  KFPARTICLE_GPU_HOST_DEVICE inline KFParticleGpuParityDifference Compare(
    const KFParticleGpuParitySnapshot& expected,
    const KFParticleGpuParitySnapshot& actual,
    const KFParticleGpuParityTolerance& tolerance)
  {
    KFParticleGpuParityDifference difference;
    if (!Equal(expected.key, actual.key)) { difference.issues |= KFGpuParityKeyMismatch; }

    const unsigned int common = expected.available & actual.available;
    if ((common & KFGpuParityParametersAvailable) != 0u) {
      for (unsigned int i = 0u; i < KFParticleGpuFitState::NumberOfParameters; ++i) {
        const unsigned int before = difference.issues;
        CompareFloat(expected.parameters[i],
                     actual.parameters[i],
                     tolerance.parameterAbsolute,
                     tolerance.parameterRelative,
                     KFGpuParityParameterMismatch,
                     difference.issues,
                     difference.maxParameterResidual);
        if (difference.firstParameter == 0xffffffffu && difference.issues != before) {
          difference.firstParameter = i;
        }
      }
    }
    if ((common & KFGpuParityCovarianceAvailable) != 0u) {
      for (unsigned int i = 0u; i < KFParticleGpuFitState::NumberOfCovarianceElements; ++i) {
        const unsigned int before = difference.issues;
        CompareFloat(expected.covariance[i],
                     actual.covariance[i],
                     tolerance.covarianceAbsolute,
                     tolerance.covarianceRelative,
                     KFGpuParityCovarianceMismatch,
                     difference.issues,
                     difference.maxCovarianceResidual);
        if (difference.firstCovariance == 0xffffffffu && difference.issues != before) {
          difference.firstCovariance = i;
        }
      }
    }
    if ((common & KFGpuParityFitQualityAvailable) != 0u) {
      CompareFloat(expected.chi2,
                   actual.chi2,
                   tolerance.fitScalarAbsolute,
                   tolerance.fitScalarRelative,
                   KFGpuParityFitScalarMismatch,
                   difference.issues,
                   difference.maxFitScalarResidual);
      if (expected.ndf != actual.ndf || expected.charge != actual.charge) {
        difference.issues |= KFGpuParityFitIntegerMismatch;
      }
    }
    if ((common & KFGpuParityFitMetadataAvailable) != 0u) {
      CompareFloat(expected.sFromDecay,
                   actual.sFromDecay,
                   tolerance.fitScalarAbsolute,
                   tolerance.fitScalarRelative,
                   KFGpuParityFitScalarMismatch,
                   difference.issues,
                   difference.maxFitScalarResidual);
      CompareFloat(expected.sumDaughterMass,
                   actual.sumDaughterMass,
                   tolerance.fitScalarAbsolute,
                   tolerance.fitScalarRelative,
                   KFGpuParityFitScalarMismatch,
                   difference.issues,
                   difference.maxFitScalarResidual);
      CompareFloat(expected.massHypothesis,
                   actual.massHypothesis,
                   tolerance.fitScalarAbsolute,
                   tolerance.fitScalarRelative,
                   KFGpuParityFitScalarMismatch,
                   difference.issues,
                   difference.maxFitScalarResidual);
      if (expected.atProductionVertex != actual.atProductionVertex
          || expected.constructMethod != actual.constructMethod) {
        difference.issues |= KFGpuParityFitIntegerMismatch;
      }
    }
    if ((common & KFGpuParityMassAvailable) != 0u) {
      if (!MassValidityEquivalent(expected, actual, tolerance)) {
        difference.issues |= KFGpuParityMassMismatch;
      }
      else if (expected.massValid != 0u && actual.massValid != 0u) {
        CompareFloat(expected.mass,
                     actual.mass,
                     tolerance.massAbsolute,
                     tolerance.massRelative,
                     KFGpuParityMassMismatch,
                     difference.issues,
                     difference.maxMassResidual);
        CompareFloat(expected.massError,
                     actual.massError,
                     tolerance.massErrorAbsolute,
                     tolerance.massErrorRelative,
                     KFGpuParityMassMismatch,
                     difference.issues,
                     difference.maxMassErrorResidual);
      }
    }
    if ((common & KFGpuParitySelectionAvailable) != 0u) {
      if (expected.selected != actual.selected
          || expected.bestPrimaryVertexIndex != actual.bestPrimaryVertexIndex
          || expected.selectionClass != actual.selectionClass
          || expected.rejectionReasons != actual.rejectionReasons
          || expected.topologyStatus != actual.topologyStatus) {
        difference.issues |= KFGpuParitySelectionMismatch;
      }
      const float* expectedValues = &expected.selectionObservables.mass;
      const float* actualValues = &actual.selectionObservables.mass;
      for (unsigned int i = 0u; i < 10u; ++i) {
        CompareFloat(expectedValues[i],
                     actualValues[i],
                     tolerance.selectionAbsolute,
                     tolerance.selectionRelative,
                     KFGpuParitySelectionMismatch,
                     difference.issues,
                     difference.maxSelectionResidual);
      }
    }
    if ((common & KFGpuParityOperationAvailable) != 0u
        && (expected.primaryVertexIndex != actual.primaryVertexIndex
            || expected.candidateFlags != actual.candidateFlags
            || expected.topology != actual.topology
            || expected.outputClass != actual.outputClass
            || expected.operationStatus != actual.operationStatus)) {
      difference.issues |= KFGpuParityOperationMismatch;
    }
    return difference;
  }

  KFPARTICLE_GPU_HOST_DEVICE inline KFParticleGpuPromotionVerdict PromotionVerdict(
    unsigned int blockers,
    unsigned int comparedCandidates,
    unsigned int mismatchedCandidates)
  {
    KFParticleGpuPromotionVerdict verdict;
    verdict.blockers = blockers;
    verdict.comparedCandidates = comparedCandidates;
    verdict.mismatchedCandidates = mismatchedCandidates;
    if (mismatchedCandidates != 0u) { verdict.blockers |= KFGpuPromotionPhysicsMismatch; }
    return verdict;
  }
}

static_assert(std::is_trivially_copyable<KFParticleGpuParityCandidateKey>::value,
              "KFParticle GPU parity keys must remain flat values");
static_assert(sizeof(KFParticleGpuV0SelectionObservables) == 10u * sizeof(float),
              "KFParticle GPU parity comparison expects ten packed selection observables");
static_assert(std::is_trivially_copyable<KFParticleGpuParitySnapshot>::value,
              "KFParticle GPU parity snapshots must remain flat values");
static_assert(std::is_trivially_copyable<KFParticleGpuParityTolerance>::value,
              "KFParticle GPU parity tolerances must remain flat values");
static_assert(std::is_trivially_copyable<KFParticleGpuPromotionVerdict>::value,
              "KFParticle GPU promotion verdicts must remain flat values");

#endif
