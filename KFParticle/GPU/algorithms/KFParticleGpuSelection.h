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

/**
 * @file
 * @brief V0 selection records, non-owning output views, and host/device selection helpers.
 *
 * Evaluates already fitted candidates against mass, fit-quality, and primary-vertex cuts.
 * Spatial topology uses a position-residual approximation; line topology additionally
 * constrains a neutral copy to a production vertex to estimate decay length. The original
 * fit is unchanged. Compact outputs reference the raw candidate pool rather than copy fits.
 * No helper here allocates device buffers, launches kernels, or reserves output slots.
 */

#include "data/KFParticleGpuCandidateTransfer.h"
#include "data/KFParticleGpuInputDataTransfer.h"
#include "algorithms/KFParticleGpuMath.h"
#include "data/KFParticleGpuFitState.h"

/** @brief Coarse CPU-Finder-compatible selection destination of a constructed V0. */
enum KFParticleGpuV0SelectionClass
{
  KFGpuV0SelectionNotEvaluated = 0, ///< Selection has not assigned a decision.
  KFGpuV0SelectionRejected, ///< Candidate failed one or more requirements.
  KFGpuV0SelectionSecondary, ///< Accepted without satisfying the configured primary-topology criterion.
  KFGpuV0SelectionPrimary ///< Accepted and below the configured primary-topology chi2 threshold.
};

/**
 * @brief Independent rejection reasons; multiple bits may be set for one candidate.
 * EvaluateV0Selection() supplies physics/build reasons; output writers may add overflow.
 */
enum KFParticleGpuV0SelectionRejection
{
  KFGpuV0SelectionRejectNone = 0u, ///< No rejection recorded; alone does not imply selection was evaluated.
  KFGpuV0SelectionRejectBuild = 1u << 0, ///< Missing valid-build flag or explicit build failure.
  KFGpuV0SelectionRejectNonFinite = 1u << 1, ///< A checked fit parameter or chi2 failed the numerical bound.
  KFGpuV0SelectionRejectGeometricChi2 = 1u << 2, ///< Fit chi2/NDF exceeds its enabled maximum.
  KFGpuV0SelectionRejectNoPrimaryVertex = 1u << 3, ///< No usable PV was found when topology requires one.
  KFGpuV0SelectionRejectDistance = 1u << 4, ///< Nearest usable PV distance is at or above its maximum.
  KFGpuV0SelectionRejectDecayLength = 1u << 5, ///< Secondary decay significance is at or below its minimum.
  KFGpuV0SelectionRejectMass = 1u << 6, ///< Invalid mass/error or mass outside the enabled window.
  KFGpuV0SelectionRejectTopology = 1u << 7, ///< Unsupported requested topology or failed secondary chi2 cut.
  KFGpuV0SelectionRejectOutputOverflow = 1u << 8, ///< Selected-output storage could not retain the candidate.
  KFGpuV0SelectionRejectPointing = 1u << 9 ///< No usable PV passes the line-mode compatibility/pointing test.
};

/**
 * @brief Ten scalar observables retained for diagnostics before selected-output compaction.
 * Defaults do not indicate successful evaluation; consult the selection class and status.
 * Historical names are retained: the three bestPrimaryVertex decay-length fields actually
 * describe the nearest usable PV, not necessarily the PV with the smallest topology chi2.
 */
struct KFParticleGpuV0SelectionObservables
{
  float mass = 0.f; ///< GetMass() invariant mass, including its invalid-result convention.
  float massError = 0.f; ///< GetMass() uncertainty or fallback value.
  float geometricChi2PerNdf = 0.f; ///< Candidate fit chi2/NDF; 1e8 for nonpositive NDF.
  float nearestPrimaryVertexDistance = 0.f; ///< Three-dimensional separation from the nearest usable PV.
  float nearestPrimaryVertexDistanceError = 0.f; ///< Uncertainty projected along that displacement.
  float nearestPrimaryVertexLdL = 0.f; ///< Displacement divided by its uncertainty, with mode-specific guards.
  float bestPrimaryVertexTopoChi2PerNdf = 0.f; ///< Smallest usable spatial-residual chi2/3 across PVs.
  float bestPrimaryVertexDecayLength = 0.f; ///< Nearest-PV constrained length (line) or separation (spatial).
  float bestPrimaryVertexDecayLengthError = 0.f; ///< Error of the preceding nearest-PV length.
  float bestPrimaryVertexLdL = 0.f; ///< Nearest-PV length significance used for the secondary cut.
};

/**
 * @brief Selection decision and observables associated with one raw candidate-pool entry.
 * Does not contain or own the fit or daughters; indices refer to external pools.
 */
struct KFParticleGpuV0SelectionResult
{
  unsigned int candidateIndex = 0u; ///< Raw candidate-pool slot, not a compact-output position.
  unsigned int channelId = 0u; ///< Channel identity supplied by the caller.
  unsigned int eventIndex = 0u; ///< Event index supplied by the caller.
  int bestPrimaryVertexIndex = -1; ///< Absolute vertex-pool index minimizing topology chi2; -1 if absent.
  unsigned int selectionClass = KFGpuV0SelectionNotEvaluated; ///< KFParticleGpuV0SelectionClass value.
  unsigned int rejectionReasons = KFGpuV0SelectionRejectNone; ///< KFParticleGpuV0SelectionRejection mask.
  unsigned int topologyStatus = 0u; ///< Line-topology status; remains zero for the spatial route.
  KFParticleGpuV0SelectionObservables observables; ///< Quantities computed before acceptance or early return.
};

/**
 * @brief Non-owning, candidate-indexed array view of selection records.
 * @tparam ResultValue Mutable or const KFParticleGpuV0SelectionResult.
 * Accessors do not synchronize or validate storage lifetime. Constness of the view object
 * does not make records read-only; that is determined by ResultValue.
 */
template<typename ResultValue>
class KFParticleGpuV0SelectionResultViewBase
{
 public:
  /// @brief Default constructor; plain default-initialization leaves members unset. Bind storage before use.
  KFPARTICLE_GPU_HOST_DEVICE KFParticleGpuV0SelectionResultViewBase() = default;

  /// @brief Bind an external result array and its capacity without allocating or validating it.
  KFPARTICLE_GPU_HOST_DEVICE KFParticleGpuV0SelectionResultViewBase(ResultValue* results,
                                                                 unsigned int capacity)
    : fResults(results), fCapacity(capacity)
  {
  }

  /// @brief Return the raw-candidate-indexed record; requires nonnull storage and an in-range index.
  KFPARTICLE_GPU_HOST_DEVICE ResultValue& Result(unsigned int candidateIndex) const
  {
    return fResults[candidateIndex];
  }

  /// @brief Return the declared number of addressable result slots.
  KFPARTICLE_GPU_HOST_DEVICE unsigned int Capacity() const { return fCapacity; }

  /// @brief Check for nonnull storage and an index below capacity; no backing-allocation check is made.
  KFPARTICLE_GPU_HOST_DEVICE bool CanStore(unsigned int candidateIndex) const
  {
    return fResults && candidateIndex < fCapacity;
  }

  /// @brief Borrow the underlying result pointer without extending its lifetime.
  KFPARTICLE_GPU_HOST_DEVICE ResultValue* Data() const { return fResults; }

 private:
  ResultValue* fResults; ///< Borrowed candidate-indexed result array.
  unsigned int fCapacity; ///< Declared array capacity.
};

/// Mutable view of externally owned selection records.
typedef KFParticleGpuV0SelectionResultViewBase<KFParticleGpuV0SelectionResult>
  KFParticleGpuV0SelectionResultView;

/// Read-only view of externally owned selection records.
typedef KFParticleGpuV0SelectionResultViewBase<const KFParticleGpuV0SelectionResult>
  KFParticleGpuConstV0SelectionResultView;

/// @brief Create a read-only alias of the same result storage without copying records.
KFPARTICLE_GPU_HOST_DEVICE inline KFParticleGpuConstV0SelectionResultView
MakeConstView(const KFParticleGpuV0SelectionResultView& view)
{
  return KFParticleGpuConstV0SelectionResultView(view.Data(), view.Capacity());
}

/**
 * @brief Candidate-to-PV separation, projected error, and spatial-residual chi2/3.
 * Uses the sum of the position covariance matrices, assuming independent inputs.
 * No production-vertex fit is performed; valid reflects algebraic checks, not a full finite-state validation.
 */
struct KFParticleGpuPrimaryVertexTopologyObservables
{
  float distance = 0.f; ///< PV-to-decay-vertex separation, floored at 1e-8 by the builder.
  float distanceError = 1.e8f; ///< Projected position uncertainty; 1e8 is the fallback.
  float ldL = 0.f; ///< Separation divided by its projected uncertainty.
  float chi2PerNdf = 1.e8f; ///< Spatial Mahalanobis residual divided by three; 1e8 is the fallback.
  unsigned int valid = 0u; ///< Nonzero if the builder's variance/determinant/chi2 checks passed.
};

/**
 * @brief Status bits for the bounded neutral-V0 line-topology calculation.
 * Valid may coexist with DegenerateCovariance when only the projected distance error failed.
 * The historical "line" distance is the ordinary 3D separation, not distance to the momentum line.
 */
enum KFParticleGpuV0LineTopologyStatus
{
  KFGpuV0LineTopologyNone = 0u, ///< Not evaluated or no usable PV found.
  KFGpuV0LineTopologyValid = 1u << 0, ///< Calculation completed; not a guarantee that every observable is finite.
  KFGpuV0LineTopologyNonFinite = 1u << 1, ///< A checked input or squared momentum/distance failed its bound.
  KFGpuV0LineTopologyDegenerateMomentum = 1u << 2, ///< Squared momentum is at most 1e-16.
  KFGpuV0LineTopologyDegenerateCovariance = 1u << 3 ///< Projected variance or constrained length calculation failed.
};

/**
 * @brief Geometry and constrained decay-length observables for one neutral candidate/PV pair.
 * Distinguishes raw displacement quantities from decay length after constraining a copy to the PV.
 * The reported chi2 remains a spatial residual, not the chi2 of that constrained fit.
 */
struct KFParticleGpuV0LineTopologyResult
{
  int primaryVertexIndex = -1; ///< Absolute vertex-pool index assigned by search helpers, not the pair builder.
  float lineDistance = 0.f; ///< Raw 3D vertex separation, floored at 1e-8.
  float lineDistanceError = 1.e8f; ///< Projected position error; may remain the 1e8 sentinel.
  float lineLdL = 0.f; ///< Distance/error when error < 1e7; zero error can yield infinity.
  float pathToPrimaryVertex = 0.f; ///< (decay position - PV).p / |p|^2, not a geometric length.
  float decayLength = 0.f; ///< Signed length from the production-vertex-constrained copy.
  float decayLengthError = 1.e8f; ///< Uncertainty returned by GetDecayLength().
  float decayLdL = 0.f; ///< Constrained length/error for positive error, otherwise zero.
  float pointingCosine = 0.f; ///< Cosine between momentum and PV-to-decay displacement; zero for tiny distance.
  float lineChi2PerNdf = 1.e8f; ///< Raw spatial residual chi2/3, or 1e8 when inversion/residual checks fail.
  unsigned int status = KFGpuV0LineTopologyNone; ///< KFParticleGpuV0LineTopologyStatus mask.
};

/** @brief Select the spatial approximation or neutral production-vertex/decay-length route. */
enum KFParticleGpuV0TopologyMode
{
  KFGpuV0TopologySpatial = 0, ///< Use raw separation and its uncertainty; no production-vertex constraint.
  KFGpuV0TopologyLine = 1 ///< Also calculate constrained neutral decay length and require PV pointing compatibility.
};

/**
 * @brief Per-channel cuts interpreted by EvaluateV0Selection().
 * Negative cut thresholds disable the corresponding cut. The mass window is enabled only
 * when both sigma settings are nonnegative, but valid GetMass() output is always required.
 * Any enabled PV-related cut, or requirePrimaryVertex, requests PV topology evaluation.
 * Supply finite settings; this structure and the evaluator do not validate configuration finiteness.
 */
struct KFParticleGpuV0SelectionConfig
{
  float expectedMass = 0.f; ///< Center of the optional mass window.
  float expectedMassSigma = -1.f; ///< Fixed expected mass width, not the fitted candidate's mass error.
  float massSigmaCut = -1.f; ///< Width multiplier; equality at the resulting mass-window boundary is accepted.
  float maxGeometricChi2PerNdf = -1.f; ///< Inclusive maximum candidate chi2/NDF.
  float maxPrimaryVertexDistance = -1.f; ///< Exclusive maximum distance to the nearest usable PV.
  float minSecondaryLdL = -1.f; ///< Exclusive minimum nearest-PV length significance for nonprimary candidates.
  float maxPrimaryTopologyChi2PerNdf = -1.f; ///< Exclusive chi2/3 threshold defining the primary class.
  float maxSecondaryTopologyChi2PerNdf = -1.f; ///< Exclusive chi2/3 maximum for nonprimary candidates.
  unsigned int requirePrimaryVertex = 0u; ///< Nonzero requests a usable PV even with all PV cuts disabled.
  int topologyMode = KFGpuV0TopologySpatial; ///< KFParticleGpuV0TopologyMode, consulted only when a PV is needed.
};

/** @brief Compact selected-output overflow flags. */
enum KFParticleGpuSelectedCandidateOverflow
{
  KFGpuSelectedCandidateCapacityExceeded = 1u << 0 ///< Output capacity was insufficient to retain all entries.
};

/**
 * @brief Non-owning selected-index storage with optional per-entry channel IDs.
 * @tparam UnsignedValue unsigned int or const unsigned int, controlling storage mutability.
 * Size and overflow pointers refer to external counters. This view performs no atomic reservation,
 * allocation, synchronization, or clamping of size to capacity. Callers must provide valid storage.
 * In event-reserved batches, Size() is the high-water end of populated event ranges, not
 * their total entry count. Iterate actual event/channel ranges; gaps have no defined contents.
 */
template<typename UnsignedValue>
class KFParticleGpuSelectedCandidateIndexViewBase
{
 public:
  /// @brief Default constructor; plain default-initialization leaves members unset. Bind storage before use.
  KFPARTICLE_GPU_HOST_DEVICE KFParticleGpuSelectedCandidateIndexViewBase() = default;

  /**
   * @brief Bind index/channel arrays and external counters without taking ownership.
   * channelIds may be null when raw candidate metadata supplies the channel identity.
   * Null size/overflow pointers read as zero; this does not make missing index storage usable.
   */
  KFPARTICLE_GPU_HOST_DEVICE KFParticleGpuSelectedCandidateIndexViewBase(
    UnsignedValue* indices,
    UnsignedValue* channelIds,
    UnsignedValue* size,
    UnsignedValue* overflowFlags,
    unsigned int capacity)
    : fIndices(indices), fChannelIds(channelIds), fSize(size), fOverflowFlags(overflowFlags), fCapacity(capacity)
  {
  }

  /// @brief Bind compact indices and counters without a separate channel-ID array.
  KFPARTICLE_GPU_HOST_DEVICE KFParticleGpuSelectedCandidateIndexViewBase(
    UnsignedValue* indices,
    UnsignedValue* size,
    UnsignedValue* overflowFlags,
    unsigned int capacity)
    : KFParticleGpuSelectedCandidateIndexViewBase(indices, nullptr, size, overflowFlags, capacity)
  {
  }

  /// @brief Return the raw candidate index at a compact slot; storage and bounds are unchecked.
  KFPARTICLE_GPU_HOST_DEVICE UnsignedValue& Index(unsigned int index) const { return fIndices[index]; }

  /// @brief Return a compact slot's channel ID; requires a nonnull channel array and a valid index.
  KFPARTICLE_GPU_HOST_DEVICE UnsignedValue& ChannelId(unsigned int index) const
  {
    return fChannelIds[index];
  }

  /// @brief Read the external extent (possibly with inter-event gaps), or zero; not clamped to capacity.
  KFPARTICLE_GPU_HOST_DEVICE unsigned int Size() const
  {
    return fSize ? static_cast<unsigned int>(*fSize) : 0u;
  }

  /// @brief Return the declared number of compact output slots.
  KFPARTICLE_GPU_HOST_DEVICE unsigned int Capacity() const { return fCapacity; }

  /// @brief Read the external overflow mask, or zero when its pointer is absent.
  KFPARTICLE_GPU_HOST_DEVICE unsigned int OverflowFlags() const
  {
    return fOverflowFlags ? static_cast<unsigned int>(*fOverflowFlags) : 0u;
  }

  /// @brief Check only index < capacity; does not check that any backing pointer is nonnull.
  KFPARTICLE_GPU_HOST_DEVICE bool CanStore(unsigned int index) const { return index < fCapacity; }

  /// @brief Borrow the compact raw-candidate-index array.
  KFPARTICLE_GPU_HOST_DEVICE UnsignedValue* IndicesData() const { return fIndices; }

  /// @brief Borrow the optional per-entry channel-ID array.
  KFPARTICLE_GPU_HOST_DEVICE UnsignedValue* ChannelIdsData() const { return fChannelIds; }

  /// @brief Borrow the optional external size-counter pointer.
  KFPARTICLE_GPU_HOST_DEVICE UnsignedValue* SizeData() const { return fSize; }

  /// @brief Borrow the optional external overflow-mask pointer.
  KFPARTICLE_GPU_HOST_DEVICE UnsignedValue* OverflowFlagsData() const { return fOverflowFlags; }

 private:
  UnsignedValue* fIndices; ///< Borrowed array mapping compact slots to raw candidate slots.
  UnsignedValue* fChannelIds; ///< Optional borrowed channel IDs for compact entries.
  UnsignedValue* fSize; ///< Optional borrowed size counter.
  UnsignedValue* fOverflowFlags; ///< Optional borrowed overflow mask.
  unsigned int fCapacity; ///< Declared array capacity.
};

/// Mutable compact selected-index view.
typedef KFParticleGpuSelectedCandidateIndexViewBase<unsigned int>
  KFParticleGpuSelectedCandidateIndexView;

/// Read-only compact selected-index view.
typedef KFParticleGpuSelectedCandidateIndexViewBase<const unsigned int>
  KFParticleGpuConstSelectedCandidateIndexView;

/// @brief Create a read-only alias of the same compact arrays and counters, without copying data.
KFPARTICLE_GPU_HOST_DEVICE inline KFParticleGpuConstSelectedCandidateIndexView
MakeConstView(const KFParticleGpuSelectedCandidateIndexView& view)
{
  return KFParticleGpuConstSelectedCandidateIndexView(
    view.IndicesData(), view.ChannelIdsData(), view.SizeData(), view.OverflowFlagsData(), view.Capacity());
}

/** @brief Half-open interval of compact selected entries, with a range-specific overflow mask. */
struct KFParticleGpuSelectedCandidateRange
{
  unsigned int offset = 0u; ///< First compact slot, not a raw candidate index.
  unsigned int size = 0u; ///< Number of entries in the interval.
  unsigned int overflowFlags = 0u; ///< KFParticleGpuSelectedCandidateOverflow mask.

  /// @brief Return whether the interval contains no entries, independently of overflow status.
  KFPARTICLE_GPU_HOST_DEVICE bool Empty() const { return size == 0u; }

  /// @brief Return the exclusive end; caller must ensure offset + size does not overflow.
  KFPARTICLE_GPU_HOST_DEVICE unsigned int End() const { return offset + size; }

  /// @brief Test the capacity-exceeded bit, not whether size happens to equal capacity.
  KFPARTICLE_GPU_HOST_DEVICE bool Truncated() const
  {
    return (overflowFlags & KFGpuSelectedCandidateCapacityExceeded) != 0u;
  }
};

/** @brief Associate one event/channel identity with its interval in compact selected output. */
struct KFParticleGpuSelectedChannelRange
{
  unsigned int channelId = 0u; ///< Channel identity for the interval.
  unsigned int eventIndex = 0u; ///< Event index for the interval.
  KFParticleGpuSelectedCandidateRange candidates; ///< Compact selected-entry interval and overflow state.
};

/**
 * @brief Non-owning downstream view resolving compact selected entries into raw V0 data.
 *
 * Selection stores raw-pool indices only; this view resolves fit metadata and
 * stored daughter lineage in the original SoA pools without a copy.
 * Accessors are unchecked: all selected slots, raw indices, daughter indices, and optional
 * result accesses must address live storage in the caller's execution context.
 */
class KFParticleGpuSelectedV0View
{
 public:
  /// @brief Default constructor; bind valid component views before accessing data.
  KFPARTICLE_GPU_HOST_DEVICE KFParticleGpuSelectedV0View() = default;

  /// @brief Alias the raw candidate pool and compact indices, with optional candidate-indexed results.
  KFPARTICLE_GPU_HOST_DEVICE KFParticleGpuSelectedV0View(
    const KFParticleGpuConstCandidatePoolView& candidates,
    const KFParticleGpuConstSelectedCandidateIndexView& selected,
    const KFParticleGpuConstV0SelectionResultView& results = KFParticleGpuConstV0SelectionResultView())
    : fCandidates(candidates), fSelected(selected), fResults(results)
  {
  }

  /// @brief Read the compact size counter without clamping to its capacity.
  KFPARTICLE_GPU_HOST_DEVICE unsigned int Size() const { return fSelected.Size(); }

  /// @brief Read the compact output's overflow flags.
  KFPARTICLE_GPU_HOST_DEVICE unsigned int OverflowFlags() const { return fSelected.OverflowFlags(); }

  /// @brief Check whether result storage is present; does not verify its capacity.
  KFPARTICLE_GPU_HOST_DEVICE bool HasSelectionResults() const
  {
    return fResults.Data() != nullptr;
  }

  /// @brief Map a compact selected slot to its raw candidate-pool index.
  KFPARTICLE_GPU_HOST_DEVICE unsigned int CandidateIndex(unsigned int selectedIndex) const
  {
    return fSelected.Index(selectedIndex);
  }

  /// @brief Read the selected candidate's mother PDG code from raw metadata.
  KFPARTICLE_GPU_HOST_DEVICE int Pdg(unsigned int selectedIndex) const
  {
    return fCandidates.Metadata().Pdg(CandidateIndex(selectedIndex));
  }

  /// @brief Prefer the compact entry's channel ID when present; otherwise use raw metadata.
  KFPARTICLE_GPU_HOST_DEVICE unsigned int ChannelId(unsigned int selectedIndex) const
  {
    return fSelected.ChannelIdsData() ? fSelected.ChannelId(selectedIndex)
                                      : fCandidates.Metadata().ChannelId(CandidateIndex(selectedIndex));
  }

  /// @brief Read the selected candidate's event index from raw metadata.
  KFPARTICLE_GPU_HOST_DEVICE unsigned int EventIndex(unsigned int selectedIndex) const
  {
    return fCandidates.Metadata().EventIndex(CandidateIndex(selectedIndex));
  }

  /// @brief Return the number of stored source-track ancestry entries, not necessarily direct daughters.
  KFPARTICLE_GPU_HOST_DEVICE unsigned int DaughterCount(unsigned int selectedIndex) const
  {
    return fCandidates.Metadata().DaughterCount(CandidateIndex(selectedIndex));
  }

  /// @brief Read a source-track ID by its ordinal in the candidate's stored ancestry interval.
  KFPARTICLE_GPU_HOST_DEVICE int DaughterSourceId(unsigned int selectedIndex,
                                                unsigned int daughterIndex) const
  {
    const unsigned int candidateIndex = CandidateIndex(selectedIndex);
    return fCandidates.Daughters().SourceId(
      fCandidates.Metadata().DaughterOffset(candidateIndex) + daughterIndex);
  }

  /// @brief Copy the selected candidate's stored fit into destination, without refitting or transport.
  KFPARTICLE_GPU_HOST_DEVICE void LoadFitState(unsigned int selectedIndex,
                                             KFParticleGpuFitState& destination) const
  {
    ::LoadCandidateFit(fCandidates, CandidateIndex(selectedIndex), destination);
  }

  /// @brief Borrow the raw-candidate-indexed selection record; requires present, sufficiently large result storage.
  KFPARTICLE_GPU_HOST_DEVICE const KFParticleGpuV0SelectionResult& Selection(
    unsigned int selectedIndex) const
  {
    return fResults.Result(CandidateIndex(selectedIndex));
  }

 private:
  KFParticleGpuConstCandidatePoolView fCandidates; ///< Borrowed raw fits, metadata, and source ancestry.
  KFParticleGpuConstSelectedCandidateIndexView fSelected; ///< Borrowed compact selection indices.
  KFParticleGpuConstV0SelectionResultView fResults; ///< Optional borrowed candidate-indexed decisions.
};

/// @brief Assemble a downstream view from existing storage; no validation, allocation, or data copy is performed.
KFPARTICLE_GPU_HOST_DEVICE inline KFParticleGpuSelectedV0View MakeSelectedV0View(
  const KFParticleGpuConstCandidatePoolView& candidates,
  const KFParticleGpuConstSelectedCandidateIndexView& selected,
  const KFParticleGpuConstV0SelectionResultView& results = KFParticleGpuConstV0SelectionResultView())
{
  return KFParticleGpuSelectedV0View(candidates, selected, results);
}

/**
 * @brief Host/device geometry, selection, mass-window, and ancestry checks for fitted candidates.
 * PV search helpers require live vertex storage and absolute indices representable as int.
 * They check ranges against logical Size(), not backing capacity or pointer validity.
 */
namespace KFParticleGpuSelection
{
  /**
   * @brief Accept non-NaN values strictly between -1e30 and 1e30, a stricter bound than IEEE finiteness.
   */
  KFPARTICLE_GPU_HOST_DEVICE inline bool IsFiniteValue(float value)
  {
    return value == value && value > -1.e30f && value < 1.e30f;
  }

  /**
   * @brief Check all eight parameters and chi2 with IsFiniteValue().
   * Does not inspect covariance, charge, NDF, other fit metadata, or physical consistency.
   */
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

  /// @brief Restore default observables, sentinel errors/chi2, no status bits, and PV index -1.
  KFPARTICLE_GPU_HOST_DEVICE inline void ResetLineTopology(
    KFParticleGpuV0LineTopologyResult& result)
  {
    result = KFParticleGpuV0LineTopologyResult();
  }

  /**
   * @brief Calculate raw PV geometry and production-vertex-constrained length for a neutral candidate.
   *
   * For displacement d = PV - decay position and summed position covariance C, computes
   * distance |d| (floored at 1e-8), error sqrt(d^T C d)/distance, and spatial chi2 = d^T C^-1 d.
   * Reports chi2/3, not the constrained fit chi2. The pointing cosine uses -d, so outward
   * momentum has positive cosine. A local fit copy supplies constrained decay length/error;
   * neither candidate nor PV is modified. The result is reset first and its PV index stays -1.
   *
   * @return False for failed input/squared-value bounds, momentum squared <= 1e-16, or a
   * failed neutral constraint/decay-length calculation. Failure may leave partial observables.
   * @note An invalid projected variance keeps error 1e8 and sets DegenerateCovariance alongside
   * Valid if the remaining calculation succeeds. Zero projected variance can produce infinite
   * lineLdL without rejection. Failed spatial inversion leaves chi2/3 at 1e8 without invalidating
   * the result. Success is therefore not a guarantee of finite values or a positive-definite covariance.
   */
  KFPARTICLE_GPU_HOST_DEVICE inline bool BuildV0LineTopology(
    const KFParticleGpuFitState& candidate,
    const KFParticleGpuVertexState& primaryVertex,
    KFParticleGpuV0LineTopologyResult& result)
  {
    ResetLineTopology(result);
    if (!IsFiniteFitState(candidate) || !IsFiniteValue(primaryVertex.X())
        || !IsFiniteValue(primaryVertex.Y()) || !IsFiniteValue(primaryVertex.Z())) {
      result.status = KFGpuV0LineTopologyNonFinite;
      return false;
    }

    const float px = candidate.Px();
    const float py = candidate.Py();
    const float pz = candidate.Pz();
    const float momentum2 = px * px + py * py + pz * pz;

    // Despite its historical name, CPU GetDistanceToVertexLine() measures
    // the ordinary decay-vertex-to-PV displacement, not a perpendicular to
    // the momentum line. Preserve the CPU sign convention for the pointing
    // test: dx points from the decay vertex to the primary vertex.
    const float dx = primaryVertex.X() - candidate.X();
    const float dy = primaryVertex.Y() - candidate.Y();
    const float dz = primaryVertex.Z() - candidate.Z();
    const float distance2 = dx * dx + dy * dy + dz * dz;
    if (!IsFiniteValue(momentum2) || !IsFiniteValue(distance2)) {
      result.status = KFGpuV0LineTopologyNonFinite;
      return false;
    }
    if (momentum2 <= 1.e-16f) {
      result.status = KFGpuV0LineTopologyDegenerateMomentum;
      return false;
    }

    const float c00 = candidate.Covariance(0, 0) + primaryVertex.Covariance(0);
    const float c10 = candidate.Covariance(1, 0) + primaryVertex.Covariance(1);
    const float c11 = candidate.Covariance(1, 1) + primaryVertex.Covariance(2);
    const float c20 = candidate.Covariance(2, 0) + primaryVertex.Covariance(3);
    const float c21 = candidate.Covariance(2, 1) + primaryVertex.Covariance(4);
    const float c22 = candidate.Covariance(2, 2) + primaryVertex.Covariance(5);
    result.lineDistance = distance2 > 1.e-16f ? KFParticleGpuSqrt(distance2) : 1.e-8f;
    result.pathToPrimaryVertex = momentum2 > 1.e-16f
                                   ? -(dx * px + dy * py + dz * pz) / momentum2 : 0.f;
    const float momentum = momentum2 > 1.e-16f ? KFParticleGpuSqrt(momentum2) : 0.f;
    result.pointingCosine = result.lineDistance > 1.e-8f && momentum > 1.e-8f
                              ? -(dx * px + dy * py + dz * pz)
                                  / (result.lineDistance * momentum)
                              : 0.f;

    const float distanceVariance = c00 * dx * dx + c11 * dy * dy + c22 * dz * dz
                                   + 2.f * (c10 * dx * dy + c20 * dx * dz
                                           + c21 * dy * dz);
    const bool degenerateLineCovariance =
      !IsFiniteValue(distanceVariance) || distanceVariance < 0.f;
    if (!degenerateLineCovariance) {
      result.lineDistanceError = KFParticleGpuSqrt(distanceVariance) / result.lineDistance;
      result.lineLdL = result.lineDistanceError < 1.e7f
                          ? result.lineDistance / result.lineDistanceError : 0.f;
    }
    // CPU GetDistanceToVertexLine() keeps the geometric distance usable when
    // its projected covariance is invalid and reports a sentinel error.  Do
    // the same here: the subsequent constrained decay-length and pointing
    // checks still carry valid selection information for this PV.

    KFParticleGpuFitState constrained = candidate;
    if (!KFParticleGpuMath::SetNeutralProductionVertex(constrained, primaryVertex)
        || !KFParticleGpuMath::GetDecayLength(
          constrained, result.decayLength, result.decayLengthError)) {
      result.status = KFGpuV0LineTopologyDegenerateCovariance;
      return false;
    }
    result.decayLdL = result.decayLengthError > 0.f
                        ? result.decayLength / result.decayLengthError : 0.f;

    const float determinant = c00 * (c11 * c22 - c21 * c21)
                              - c10 * (c10 * c22 - c20 * c21)
                              + c20 * (c10 * c21 - c20 * c11);
    if (IsFiniteValue(determinant) && determinant > 1.e-12f) {
      const float inverse00 = (c11 * c22 - c21 * c21) / determinant;
      const float inverse10 = (c20 * c21 - c10 * c22) / determinant;
      const float inverse20 = (c10 * c21 - c20 * c11) / determinant;
      const float inverse11 = (c00 * c22 - c20 * c20) / determinant;
      const float inverse21 = (c10 * c20 - c00 * c21) / determinant;
      const float inverse22 = (c00 * c11 - c10 * c10) / determinant;
      const float chi2 = dx * dx * inverse00 + dy * dy * inverse11 + dz * dz * inverse22
                         + 2.f * (dx * dy * inverse10 + dx * dz * inverse20
                                  + dy * dz * inverse21);
      if (IsFiniteValue(chi2) && chi2 >= 0.f) {
        result.lineChi2PerNdf = chi2 / 3.f;
      }
    }
    result.status = KFGpuV0LineTopologyValid;
    if (degenerateLineCovariance) {
      result.status |= KFGpuV0LineTopologyDegenerateCovariance;
    }
    return true;
  }

  /**
   * @brief Find the usable PV with the smallest line-topology spatial chi2/3 in the given range.
   * Resets the output, skips failed pair calculations, retains the first PV on ties, and stores
   * the absolute vertex index. Valid results with sentinel chi2 or covariance warnings remain eligible.
   * @return False for an empty/out-of-range interval or no successful pair calculation.
   * Failure leaves defaults; statuses from failed individual PV attempts are not accumulated.
   */
  KFPARTICLE_GPU_HOST_DEVICE inline bool FindBestV0LineTopology(
    const KFParticleGpuFitState& candidate,
    const KFParticleGpuConstVertexSoAView& primaryVertices,
    const KFParticleGpuRange& vertexRange,
    KFParticleGpuV0LineTopologyResult& bestResult)
  {
    ResetLineTopology(bestResult);
    if (vertexRange.size == 0u || vertexRange.offset > primaryVertices.Size()
        || vertexRange.size > primaryVertices.Size() - vertexRange.offset) {
      return false;
    }

    for (unsigned int localIndex = 0u; localIndex < vertexRange.size; ++localIndex) {
      const unsigned int vertexIndex = vertexRange.offset + localIndex;
      KFParticleGpuVertexState vertex;
      LoadVertexState(primaryVertices, vertexIndex, vertex);
      KFParticleGpuV0LineTopologyResult current;
      if (!BuildV0LineTopology(candidate, vertex, current)) {
        continue;
      }
      current.primaryVertexIndex = static_cast<int>(vertexIndex);
      if ((bestResult.status & KFGpuV0LineTopologyValid) == 0u
          || current.lineChi2PerNdf < bestResult.lineChi2PerNdf) {
        bestResult = current;
      }
    }
    return (bestResult.status & KFGpuV0LineTopologyValid) != 0u;
  }

  /**
   * @brief Find the smallest 3D separation among successful line-topology candidate/PV calculations.
   * Resets the output and records the absolute vertex index, retaining the first PV on ties.
   * The geometrically nearest PV is skipped if its constrained calculation fails.
   * @return False for an empty/out-of-range interval or no usable PV; output then remains at defaults.
   */
  KFPARTICLE_GPU_HOST_DEVICE inline bool FindNearestV0LineTopology(
    const KFParticleGpuFitState& candidate,
    const KFParticleGpuConstVertexSoAView& primaryVertices,
    const KFParticleGpuRange& vertexRange,
    KFParticleGpuV0LineTopologyResult& nearestResult)
  {
    ResetLineTopology(nearestResult);
    if (vertexRange.size == 0u || vertexRange.offset > primaryVertices.Size()
        || vertexRange.size > primaryVertices.Size() - vertexRange.offset) {
      return false;
    }

    for (unsigned int localIndex = 0u; localIndex < vertexRange.size; ++localIndex) {
      const unsigned int vertexIndex = vertexRange.offset + localIndex;
      KFParticleGpuVertexState vertex;
      LoadVertexState(primaryVertices, vertexIndex, vertex);
      KFParticleGpuV0LineTopologyResult current;
      if (!BuildV0LineTopology(candidate, vertex, current)) {
        continue;
      }
      current.primaryVertexIndex = static_cast<int>(vertexIndex);
      if ((nearestResult.status & KFGpuV0LineTopologyValid) == 0u
          || current.lineDistance < nearestResult.lineDistance) {
        nearestResult = current;
      }
    }
    return (nearestResult.status & KFGpuV0LineTopologyValid) != 0u;
  }

  /**
   * @brief Accept if any usable PV is within three distance errors or has outward-pointing momentum.
   * Uses strict distance < 3*error OR pointingCosine > 0, after a successful BuildV0LineTopology().
   * A retained sentinel distance error participates in the test. Returns false for an empty/invalid
   * vertex interval or when no PV passes; the matching PV need not be the nearest or best-chi2 one.
   */
  KFPARTICLE_GPU_HOST_DEVICE inline bool PointsFromAnyPrimaryVertex(
    const KFParticleGpuFitState& candidate,
    const KFParticleGpuConstVertexSoAView& primaryVertices,
    const KFParticleGpuRange& vertexRange)
  {
    if (vertexRange.size == 0u || vertexRange.offset > primaryVertices.Size()
        || vertexRange.size > primaryVertices.Size() - vertexRange.offset) {
      return false;
    }
    for (unsigned int localIndex = 0u; localIndex < vertexRange.size; ++localIndex) {
      KFParticleGpuVertexState vertex;
      LoadVertexState(primaryVertices, vertexRange.offset + localIndex, vertex);
      KFParticleGpuV0LineTopologyResult topology;
      if (!BuildV0LineTopology(candidate, vertex, topology)) { continue; }
      // CPU GetDistanceToVertexLine accepts a candidate which is compatible
      // with the PV within 3 sigma or whose momentum points away from it.
      if (topology.lineDistance < 3.f * topology.lineDistanceError
          || topology.pointingCosine > 0.f) {
        return true;
      }
    }
    return false;
  }

  /**
   * @brief Compute a spatial PV compatibility approximation without constraining or transporting the fit.
   * Uses d = PV - decay position, C = C_candidate(position) + C_PV, distance = max(|d|,1e-8),
   * projected error sqrt(d^T C d)/distance, and chi2/3 = (d^T C^-1 d)/3.
   * @return False for projected variance <= 0, determinant <= 1e-12, or chi2 < 0;
   * valid is cleared and chi2 set to 1e8, while previously computed fields may remain.
   * @note No explicit finite-value checks are made: NaNs can bypass the ordered rejection
   * comparisons and yield true. Callers requiring finite results must validate them separately.
   */
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

  /**
   * @brief Find the lowest spatial chi2/3 among PVs whose spatial builder returns true.
   * Resets outputs, records the absolute vertex index, and keeps the first PV on ties.
   * @return False for an empty/out-of-range interval or no builder success; index remains -1.
   * Inherits the spatial builder's lack of explicit finite-value validation.
   */
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

  /**
   * @brief Evaluate one fitted candidate and write its classification, rejection bits, and observables.
   *
   * Resets the result, copies caller-supplied candidate/channel/event IDs, and starts as rejected.
   * Checks build flags, bounded parameters/chi2, geometric fit quality, and GetMass() validity.
   * The optional mass window uses expectedMassSigma * massSigmaCut, not the fitted mass error.
   * PV topology runs only when requested by a PV cut or requirePrimaryVertex; otherwise a
   * candidate passing the non-PV checks is secondary, regardless of topologyMode.
   *
   * With topology enabled, nearest distance and best spatial chi2 may come from different PVs.
   * The best-chi2 PV determines primary classification; the nearest usable PV supplies decay
   * length/significance even in fields named bestPrimaryVertexDecayLength. Line mode adds the
   * any-PV pointing test and uses constrained neutral decay length; spatial mode uses separation.
   *
   * Mass and geometric-chi2 boundaries are inclusive. Distance and secondary-topology maxima
   * are exclusive, as is the primary-class chi2 threshold; secondary significance must strictly
   * exceed its minimum. Primary candidates skip the secondary significance/topology cuts.
   * Multiple reasons may be recorded, but early returns mean not all checks necessarily ran.
   * No output compaction or overflow detection is performed here.
   *
   * @pre Live vertex storage when used, finite configuration, and representable absolute PV indices.
   * @note Uses the limited IsFiniteFitState() check and inherits the topology helpers' numerical
   * limitations. It does not validate all covariance entries or all computed observables.
   */
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

    if (needsPrimaryVertex && config.topologyMode == KFGpuV0TopologyLine) {
      KFParticleGpuV0LineTopologyResult nearestLine;
      KFParticleGpuV0LineTopologyResult bestLine;
      const bool nearestFound = FindNearestV0LineTopology(
        candidate, primaryVertices, vertexRange, nearestLine);
      const bool bestFound = FindBestV0LineTopology(
        candidate, primaryVertices, vertexRange, bestLine);
      result.topologyStatus = bestFound ? bestLine.status : nearestLine.status;
      if (!nearestFound || !bestFound) {
        result.rejectionReasons |= KFGpuV0SelectionRejectNoPrimaryVertex;
        return;
      }
      if (!PointsFromAnyPrimaryVertex(candidate, primaryVertices, vertexRange)) {
        result.rejectionReasons |= KFGpuV0SelectionRejectPointing;
      }

      result.bestPrimaryVertexIndex = bestLine.primaryVertexIndex;
      result.observables.nearestPrimaryVertexDistance = nearestLine.lineDistance;
      result.observables.nearestPrimaryVertexDistanceError = nearestLine.lineDistanceError;
      result.observables.nearestPrimaryVertexLdL = nearestLine.lineLdL;
      result.observables.bestPrimaryVertexTopoChi2PerNdf = bestLine.lineChi2PerNdf;
      result.observables.bestPrimaryVertexDecayLength = nearestLine.decayLength;
      result.observables.bestPrimaryVertexDecayLengthError = nearestLine.decayLengthError;
      result.observables.bestPrimaryVertexLdL = nearestLine.decayLdL;

      if (config.maxPrimaryVertexDistance >= 0.f
          && nearestLine.lineDistance >= config.maxPrimaryVertexDistance) {
        result.rejectionReasons |= KFGpuV0SelectionRejectDistance;
      }
      const bool primary = config.maxPrimaryTopologyChi2PerNdf >= 0.f
                           && bestLine.lineChi2PerNdf < config.maxPrimaryTopologyChi2PerNdf;
      if (!primary && config.minSecondaryLdL >= 0.f
          && nearestLine.decayLdL <= config.minSecondaryLdL) {
        result.rejectionReasons |= KFGpuV0SelectionRejectDecayLength;
      }
      if (!primary && config.maxSecondaryTopologyChi2PerNdf >= 0.f
          && bestLine.lineChi2PerNdf >= config.maxSecondaryTopologyChi2PerNdf) {
        result.rejectionReasons |= KFGpuV0SelectionRejectTopology;
      }
      if (result.rejectionReasons == KFGpuV0SelectionRejectNone) {
        result.selectionClass = primary ? KFGpuV0SelectionPrimary : KFGpuV0SelectionSecondary;
      }
      return;
    }
    if (needsPrimaryVertex && config.topologyMode != KFGpuV0TopologySpatial) {
      result.rejectionReasons |= KFGpuV0SelectionRejectTopology;
      return;
    }

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

  /// @brief Test whether any bit in the supplied rejection mask is present; zero always returns false.
  KFPARTICLE_GPU_HOST_DEVICE inline bool HasRejection(
    const KFParticleGpuV0SelectionResult& result,
    unsigned int rejection)
  {
    return (result.rejectionReasons & rejection) != 0u;
  }

  /// @brief Require both an accepted primary/secondary class and an empty rejection mask.
  KFPARTICLE_GPU_HOST_DEVICE inline bool IsSelected(
    const KFParticleGpuV0SelectionResult& result)
  {
    return result.rejectionReasons == KFGpuV0SelectionRejectNone
           && (result.selectionClass == KFGpuV0SelectionSecondary
               || result.selectionClass == KFGpuV0SelectionPrimary);
  }

  /**
   * @brief Test |mass - expectedMass| <= halfWidth with a nonnegative half-width.
   * A negative width rejects rather than disables this helper's test. No explicit finite checks are made.
   */
  KFPARTICLE_GPU_HOST_DEVICE inline bool IsInsideMassWindow(float mass,
                                                          float expectedMass,
                                                          float halfWidth)
  {
    return halfWidth >= 0.f && KFParticleGpuMath::Abs(mass - expectedMass) <= halfWidth;
  }

  /// @brief Multiply nonnegative sigma and cut settings; return -1 otherwise, without finite/overflow checks.
  KFPARTICLE_GPU_HOST_DEVICE inline float MassWindowHalfWidth(float expectedSigma,
                                                            float sigmaCut)
  {
    return expectedSigma >= 0.f && sigmaCut >= 0.f ? expectedSigma * sigmaCut : -1.f;
  }

  /// @brief Return chi2/NDF for positive NDF, otherwise the 1e8 sentinel; no finite-state check is made.
  KFPARTICLE_GPU_HOST_DEVICE inline float Chi2PerNdf(const KFParticleGpuFitState& particle)
  {
    return particle.NDF() > 0 ? particle.Chi2() / static_cast<float>(particle.NDF()) : 1.e8f;
  }

  /**
   * @brief Require a KFParticleGpuMath::IsFinite() ratio and an inclusive maximum unless the cut is negative.
   * Nonpositive NDF maps to the finite 1e8 sentinel and is not independently rejected.
   */
  KFPARTICLE_GPU_HOST_DEVICE inline bool PassChi2PerNdf(const KFParticleGpuFitState& particle,
                                                      float maxChi2PerNdf)
  {
    const float chi2PerNdf = Chi2PerNdf(particle);
    return KFParticleGpuMath::IsFinite(chi2PerNdf)
           && (maxChi2PerNdf < 0.f || chi2PerNdf <= maxChi2PerNdf);
  }

  /**
   * @brief Apply an inclusive window to Math::Mass(), which returns a signed root for negative mass squared.
   * Does not call GetMass(), check its uncertainty, or validate the fit. Negative half-width rejects.
   */
  KFPARTICLE_GPU_HOST_DEVICE inline bool PassMassWindow(const KFParticleGpuFitState& particle,
                                                      float expectedMass,
                                                      float halfWidth)
  {
    return IsInsideMassWindow(KFParticleGpuMath::Mass(particle), expectedMass, halfWidth);
  }

  /**
   * @brief Apply PassMassWindow() with half-width expectedSigma * sigmaCut.
   * Negative sigma/cut settings reject here, unlike the optional window in EvaluateV0Selection().
   */
  KFPARTICLE_GPU_HOST_DEVICE inline bool PassSigmaMassWindow(const KFParticleGpuFitState& particle,
                                                           float expectedMass,
                                                           float expectedSigma,
                                                           float sigmaCut)
  {
    return PassMassWindow(
      particle, expectedMass, MassWindowHalfWidth(expectedSigma, sigmaCut));
  }

  /**
   * @brief Check that a bounded source-ancestry interval contains no repeated source IDs.
   * Uses O(count^2) comparisons without allocation, sorting, or modifying storage. Does not
   * require positive IDs; a valid empty or one-entry interval is distinct.
   * @return False when CanStore() fails or any pair of stored IDs is equal.
   * @pre Backing source-ID storage is live and readable for the requested nonempty interval.
   */
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

  /**
   * @brief Check a candidate's flattened source ancestry for duplicates after validating its pool index.
   * Rejects indices outside either Size() or Capacity(), then delegates the metadata range check.
   * Does not validate candidate build flags or relationships among direct daughter candidates.
   */
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
