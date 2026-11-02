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

/**
 * @file
 * @brief Flat CPU/GPU comparison records, numerical tolerances, and qualification results.
 *
 * Candidates are identified by event, decay channel, PDG code, and sorted source-track
 * ancestry. Comparison checks this identity and the data groups available on both sides;
 * it does not reconstruct particles, match candidate collections, or transfer GPU memory.
 * The caller is responsible for requiring complete evidence before accepting GPU output.
 */

#include "data/KFParticleGpuCandidateTransfer.h"
#include "algorithms/KFParticleGpuMath.h"
#include "algorithms/KFParticleGpuSelection.h"

#include <cstdint>
#include <type_traits>

/// Maximum number of source-track ancestry entries stored in a comparison key.
static const unsigned int KFGpuParityMaximumLineageSize = 16u;

/**
 * @brief Bit mask describing populated snapshot groups, not their numerical validity.
 */
enum KFParticleGpuParityAvailable : unsigned int
{
  KFGpuParityKeyAvailable = 1u << 0u, ///< Identity populated; Compare() checks keys regardless of this bit.
  KFGpuParityParametersAvailable = 1u << 1u, ///< Eight fitted state parameters.
  KFGpuParityCovarianceAvailable = 1u << 2u, ///< Packed symmetric state covariance.
  KFGpuParityFitQualityAvailable = 1u << 3u, ///< Chi-square, NDF, and charge.
  KFGpuParityFitMetadataAvailable = 1u << 4u, ///< Path/mass metadata, vertex flag, and construction method.
  KFGpuParityMassAvailable = 1u << 5u, ///< Mass, uncertainty, and validity flag.
  KFGpuParitySelectionAvailable = 1u << 6u, ///< Selection decision, topology checks, and observables.
  KFGpuParityOperationAvailable = 1u << 7u ///< Candidate flags, graph classification, vertex, and operation status.
};

/**
 * @brief Candidate identity independent of ancestry order after Canonicalize().
 *
 * Lineage stores physical source-track IDs, not intermediate candidate indices.
 * Repeated IDs are retained: sorting removes order dependence, not multiplicity.
 * Callers must keep lineageSize within the fixed array capacity.
 */
struct KFParticleGpuParityCandidateKey
{
  std::uint64_t eventId = 0u; ///< Source-event identity, distinct from a batch-local slot.
  unsigned int channelId = 0u; ///< Decay-channel identity.
  int pdg = 0; ///< Mother-particle PDG hypothesis.
  unsigned int lineageSize = 0u; ///< Number of used ancestry entries.
  int lineage[KFGpuParityMaximumLineageSize] = {}; ///< Source IDs, sorted before comparison.
};

/**
 * @brief Pointer-free snapshot of one candidate for CPU/GPU equivalence checks.
 *
 * Stores fit values and reconstruction decisions without owning a particle or its ancestry.
 * The availability mask identifies meaningful groups; default values alone are not evidence
 * that a quantity was evaluated. All storage is inline and trivially copyable.
 */
struct KFParticleGpuParitySnapshot
{
  KFParticleGpuParityCandidateKey key; ///< Canonical candidate identity.
  float parameters[KFParticleGpuFitState::NumberOfParameters] = {}; ///< State (x,y,z,px,py,pz,E,S).
  float covariance[KFParticleGpuFitState::NumberOfCovarianceElements] = {}; ///< Packed lower 8x8 triangle.
  float chi2 = 0.f; ///< Fit chi-square.
  float sFromDecay = 0.f; ///< Accumulated transport parameter from the decay vertex, not a path length.
  float sumDaughterMass = 0.f; ///< Stored sum of daughter masses.
  float massHypothesis = -1.f; ///< Fit mass hypothesis; negative means unset.
  float mass = 0.f; ///< Reconstructed invariant mass from GetMass().
  float massError = 0.f; ///< Propagated mass uncertainty, or GetMass() fallback value.
  int ndf = 0; ///< Fit degrees of freedom.
  int charge = 0; ///< Particle charge in elementary-charge units.
  int atProductionVertex = 0; ///< Stored flag that the state is at the production vertex.
  int constructMethod = 0; ///< Stored fit-construction method identifier.
  int primaryVertexIndex = -1; ///< Candidate metadata vertex index; -1 means unassigned.
  int bestPrimaryVertexIndex = -1; ///< Vertex chosen by selection; -1 means none.
  unsigned int candidateFlags = 0u; ///< Candidate metadata flags.
  unsigned int topology = KFGpuGraphTopologyInvalid; ///< Decay-graph topology classification.
  unsigned int outputClass = KFGpuGraphOutputInvalid; ///< Decay-graph output classification.
  unsigned int operationStatus = KFGpuCandidateOperationPending; ///< Candidate operation outcome.
  unsigned int selectionClass = KFGpuV0SelectionNotEvaluated; ///< Selection classification.
  unsigned int rejectionReasons = KFGpuV0SelectionRejectNone; ///< Selection rejection bit mask.
  unsigned int topologyStatus = KFGpuV0LineTopologyNone; ///< Line-to-vertex topology status mask.
  KFParticleGpuV0SelectionObservables selectionObservables; ///< Ten scalar selection observables.
  unsigned int selected = 0u; ///< Caller-supplied acceptance flag, compared only with selection available.
  unsigned int massValid = 0u; ///< Nonzero when GetMass() reported a valid mass and uncertainty.
  unsigned int available = 0u; ///< KFParticleGpuParityAvailable mask.
};

/**
 * @brief Absolute and relative tolerances for each floating-point comparison group.
 *
 * For finite values a and b, accept |a-b| <= max(absolute, relative * max(|a|,|b|)).
 * Absolute tolerances use the units of the corresponding quantity; relative ones are
 * dimensionless. Defaults use absolute tolerances only. Supply finite, nonnegative values;
 * the comparison does not validate the tolerance configuration.
 */
struct KFParticleGpuParityTolerance
{
  float parameterAbsolute = 1.e-4f; ///< Absolute tolerance shared by the eight parameters.
  float parameterRelative = 0.f; ///< Relative parameter tolerance.
  float covarianceAbsolute = 1.e-3f; ///< Absolute tolerance shared by covariance entries.
  float covarianceRelative = 0.f; ///< Relative covariance tolerance.
  float fitScalarAbsolute = 1.e-4f; ///< Absolute tolerance for chi2 and path/mass metadata.
  float fitScalarRelative = 0.f; ///< Relative fit-scalar tolerance.
  float massAbsolute = 1.e-4f; ///< Absolute mass tolerance, also used at the validity boundary.
  float massRelative = 0.f; ///< Relative mass tolerance when both masses are valid.
  float massErrorAbsolute = 1.e-3f; ///< Absolute mass-error tolerance and validity-boundary threshold.
  float massErrorRelative = 0.f; ///< Relative mass-error tolerance when both masses are valid.
  float selectionAbsolute = 1.e-3f; ///< Absolute tolerance shared by ten selection observables.
  float selectionRelative = 0.f; ///< Relative selection-observable tolerance.
};

/**
 * @brief Bit mask of differences found in the fields actually compared.
 */
enum KFParticleGpuParityIssue : unsigned int
{
  KFGpuParityNoIssue = 0u, ///< No detected difference; does not imply complete coverage.
  KFGpuParityKeyMismatch = 1u << 0u, ///< Candidate identities differ.
  KFGpuParityParameterMismatch = 1u << 1u, ///< A state parameter failed comparison.
  KFGpuParityCovarianceMismatch = 1u << 2u, ///< A covariance entry failed comparison.
  KFGpuParityFitScalarMismatch = 1u << 3u, ///< Chi2 or floating-point fit metadata differ.
  KFGpuParityFitIntegerMismatch = 1u << 4u, ///< NDF, charge, vertex flag, or construction method differ.
  KFGpuParityMassMismatch = 1u << 5u, ///< Mass validity, mass, or mass uncertainty differ.
  KFGpuParitySelectionMismatch = 1u << 6u, ///< Selection decisions or observables differ.
  KFGpuParityOperationMismatch = 1u << 7u, ///< Candidate operation metadata differ.
  KFGpuParityNonFinite = 1u << 8u ///< CompareFloat() encountered NaN or a value outside (-1e30, 1e30).
};

/**
 * @brief Comparison outcome with absolute residual maxima and first failing array entries.
 *
 * Maxima include accepted finite comparisons, not just failures. Values rejected by
 * IsFinite() do not update maxima; skipped groups retain zero residuals.
 */
struct KFParticleGpuParityDifference
{
  unsigned int issues = KFGpuParityNoIssue; ///< KFParticleGpuParityIssue mask.
  unsigned int firstParameter = 0xffffffffu; ///< First failing parameter, or unsigned sentinel if none.
  unsigned int firstCovariance = 0xffffffffu; ///< First failing packed covariance entry, or sentinel.
  float maxParameterResidual = 0.f; ///< Largest compared absolute parameter difference.
  float maxCovarianceResidual = 0.f; ///< Largest compared absolute covariance difference.
  float maxFitScalarResidual = 0.f; ///< Largest compared chi2/path/mass-metadata difference.
  float maxMassResidual = 0.f; ///< Absolute mass difference when both masses are valid.
  float maxMassErrorResidual = 0.f; ///< Absolute mass-error difference when both masses are valid.
  float maxSelectionResidual = 0.f; ///< Largest compared selection-observable difference.

  /**
   * @brief Return whether no issue was recorded, without checking evidence completeness.
   */
  KFPARTICLE_GPU_HOST_DEVICE bool Equivalent() const { return issues == KFGpuParityNoIssue; }
};

/**
 * @brief Reasons supplied by qualification code to block acceptance of GPU results.
 */
enum KFParticleGpuPromotionBlocker : unsigned int
{
  KFGpuPromotionReady = 0u, ///< No blocker reported.
  KFGpuPromotionUnsupportedTopology = 1u << 0u, ///< Required decay topology is unsupported.
  KFGpuPromotionOverflow = 1u << 1u, ///< Reconstruction exceeded a storage capacity.
  KFGpuPromotionInvalidInput = 1u << 2u, ///< Input validation failed.
  KFGpuPromotionInvalidField = 1u << 3u, ///< Magnetic-field input is invalid.
  KFGpuPromotionIncompleteLineage = 1u << 4u, ///< Candidate source-track ancestry is incomplete.
  KFGpuPromotionPhysicsMismatch = 1u << 5u, ///< Candidate comparison found a discrepancy.
  KFGpuPromotionRuntimeUnavailable = 1u << 6u, ///< Required GPU runtime is unavailable.
  KFGpuPromotionExecutionFailure = 1u << 7u ///< GPU execution failed.
};

/**
 * @brief Aggregate qualification flags and comparison counts, without enforcing a sampling policy.
 */
struct KFParticleGpuPromotionVerdict
{
  unsigned int blockers = KFGpuPromotionReady; ///< KFParticleGpuPromotionBlocker mask.
  unsigned int comparedCandidates = 0u; ///< Caller-reported comparison count; not validated here.
  unsigned int mismatchedCandidates = 0u; ///< Caller-reported count of candidate discrepancies.

  /**
   * @brief Return true when blockers and mismatches are absent, even if no candidates were compared.
   */
  KFPARTICLE_GPU_HOST_DEVICE bool CanPromote() const
  {
    return blockers == KFGpuPromotionReady && mismatchedCandidates == 0u;
  }
};

/**
 * @brief Host/device helpers for canonical candidate identities and tolerant comparisons.
 */
namespace KFParticleGpuParity
{
  /**
   * @brief Return the comparison-based absolute value; NaN remains NaN and negative zero is preserved.
   */
  KFPARTICLE_GPU_HOST_DEVICE inline float Abs(float value)
  {
    return value < 0.f ? -value : value;
  }

  /**
   * @brief Accept non-NaN values strictly between -1e30 and 1e30.
   * This diagnostic bound is stricter than IEEE floating-point finiteness.
   */
  KFPARTICLE_GPU_HOST_DEVICE inline bool IsFinite(float value)
  {
    return value == value && value > -1.e30f && value < 1.e30f;
  }

  /**
   * @brief Return left if left > right, otherwise right; no special NaN handling is performed.
   */
  KFPARTICLE_GPU_HOST_DEVICE inline float Max(float left, float right)
  {
    return left > right ? left : right;
  }

  /**
   * @brief Sort used lineage entries in place by source ID, retaining duplicates.
   * @pre key.lineageSize does not exceed KFGpuParityMaximumLineageSize.
   * Uses insertion sort on the small fixed array; other identity fields are unchanged.
   */
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

  /**
   * @brief Compare event/channel/PDG identity and used lineage entries exactly.
   * @pre Both lineage sizes fit their arrays; canonicalize first for order-independent equality.
   * Does not sort, validate source IDs, or inspect unused array entries.
   */
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

  /**
   * @brief Read one candidate into a canonical snapshot and calculate its mass and uncertainty.
   *
   * @param candidates Candidate-pool view accessible in the calling execution context.
   * @param candidateIndex Pool slot, checked against both logical size and capacity.
   * @param sourceEventId Event identity supplied by the caller, not inferred from the pool.
   * @param selection Optional selection result for this candidate; null leaves that group unavailable.
   * @param selected Acceptance flag stored independently of whether selection is supplied.
   * @param snapshot Output, reset to defaults before any validation.
   * @return False for an out-of-range candidate or empty/oversized/out-of-bounds lineage.
   *
   * The view must reference valid backing storage. Success means the record was assembled,
   * not that the fit is finite, its ancestry is unique, or its mass is valid. Without a
   * selection result, Compare() does not compare the stored acceptance flag.
   */
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

  /**
   * @brief Accumulate a tolerance failure and the largest absolute residual seen so far.
   *
   * Accepts |expected-actual| <= max(absoluteTolerance, relativeTolerance * max(|expected|,|actual|)).
   * Values failing IsFinite() add both the supplied issue and KFGpuParityNonFinite,
   * leaving maximum unchanged. Otherwise maximum is updated even for an accepted pair.
   * Existing issue bits are preserved; tolerances must be finite and nonnegative.
   */
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

  /**
   * @brief Test mass-validity flags with an absolute-tolerance exception for a small uncertainty.
   *
   * Equal flags pass without inspecting the values. If flags differ, both masses and the
   * valid side's error must pass IsFinite(), the mass difference must not exceed massAbsolute,
   * and that error must not exceed massErrorAbsolute. The invalid side's error and relative
   * tolerances are ignored. This is a comparison policy, not an independent mass validation.
   */
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

  /**
   * @brief Compare two canonical snapshots and return issue flags and absolute residual maxima.
   *
   * Keys are always checked; other groups are checked only in the intersection of the
   * availability masks. Missing groups are not mismatches. Integer metadata and decisions
   * must agree exactly; floating-point fields use the corresponding tolerance pair.
   * Mass values/errors are compared only when both validity flags are nonzero; equal invalid
   * flags, or a tolerated validity difference, skip those numerical comparisons.
   * All ten selection observables are compared whenever selection is available on both sides.
   *
   * @pre Keys have bounded lineage sizes and are sorted; tolerances are finite and nonnegative.
   * @note Equivalent() does not establish that every required field or candidate was compared.
   */
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

  /**
   * @brief Package caller-supplied flags/counts and add the physics blocker for nonzero mismatches.
   * Counts are not checked for consistency or a minimum sample size; existing blockers remain set.
   */
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
