/*
 * This file is part of KFParticle package
 * Copyright (C) 2007-2026
 *
 * KFParticle is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 */

/**
 * @file
 * @brief Candidate metadata, daughter lineage, and non-owning host/device pool views.
 *
 * These descriptors expose existing storage; they do not allocate memory,
 * transfer data between CPU and GPU, or synchronize access. The caller supplies
 * pointers valid in the execution memory space and keeps the storage alive.
 */

#ifndef KFPARTICLEGPUCANDIDATETRANSFER_H
#define KFPARTICLEGPUCANDIDATETRANSFER_H

#include "planning/KFParticleGpuDecayGraph.h"
#include "data/KFParticleGpuSoAView.h"

/**
 * @brief Component indices for the signed and unsigned candidate metadata arrays.
 *
 * Each component occupies one stride of the structure-of-arrays (SoA) storage.
 * Flattened source-track lineage and direct daughter references are separate:
 * the former identifies final tracks, the latter preserves the decay structure.
 */
struct KFParticleGpuCandidateMetadataLayout
{
  /// @brief Components stored in the signed-integer metadata array.
  enum IntegerComponent
  {
    Pdg = 0,                   ///< Candidate particle hypothesis.
    PrimaryVertexIndex,        ///< Associated primary vertex index, or the applicable unassigned sentinel.
    NumberOfIntegerComponents  ///< Number of signed component arrays, not candidate slots.
  };

  /// @brief Components stored in the unsigned-integer metadata array.
  enum UnsignedComponent
  {
    EventIndex = 0,             ///< Index of the input event descriptor.
    DaughterOffset,             ///< Beginning of flattened source-track lineage in daughter storage.
    DaughterCount,              ///< Number of source-track identifiers in that lineage.
    Flags,                      ///< KFParticleGpuCandidateFlags bit mask.
    ChannelId,                  ///< Stable reconstruction channel ID, not a transient routing bit.
    Topology,                   ///< Decay graph topology of the candidate.
    OutputClass,                ///< Decay graph output category used by later reconstruction stages.
    OperationStatus,            ///< KFParticleGpuCandidateOperationStatus value.
    DirectDaughterCount,        ///< Number of valid direct references, from zero to two.
    DirectFirstKind,            ///< KFParticleGpuDirectDaughterKind for the first reference.
    DirectFirstIndex,           ///< Input-track or candidate-pool index according to DirectFirstKind.
    DirectSecondKind,           ///< KFParticleGpuDirectDaughterKind for the second reference.
    DirectSecondIndex,          ///< Input-track or candidate-pool index according to DirectSecondKind.
    NumberOfUnsignedComponents  ///< Number of unsigned component arrays, not candidate slots.
  };
};

/** @brief Storage addressed by a candidate's direct daughter reference. */
enum KFParticleGpuDirectDaughterKind : unsigned int
{
  KFGpuDirectDaughterNone = 0u,        ///< No valid daughter reference.
  KFGpuDirectDaughterInputTrack = 1u,  ///< Index in input-track storage, not a source-track ID.
  KFGpuDirectDaughterCandidate = 2u    ///< Index in the candidate pool.
};

/**
 * @brief Non-owning SoA view of candidate identity, selection status, and lineage metadata.
 * @tparam IntegerValue int for writable data, or const int for read-only data.
 * @tparam UnsignedValue unsigned int or const unsigned int, matching the signed array's mutability.
 *
 * Accessors address component * stride + candidate and do not check bounds.
 * The stride is allocated capacity, not the active candidate count. A const
 * view object can still modify data if its element types are non-const;
 * MakeConstView() creates a read-only view without copying the arrays.
 *
 * @pre Bind the descriptor before use. Referenced arrays must remain valid in
 * the current memory space; callers validate indices and coordinate access.
 */
template<typename IntegerValue, typename UnsignedValue>
class KFParticleGpuCandidateMetadataSoAViewBase
{
 public:
  /// @brief Defaulted descriptor constructor; assign a bound view before accessing storage.
  KFPARTICLE_GPU_HOST_DEVICE KFParticleGpuCandidateMetadataSoAViewBase() = default;

  /**
   * @brief Binds the two metadata arrays without allocating or copying data.
   * @param integers Signed component arrays in KFParticleGpuCandidateMetadataLayout order.
   * @param unsignedValues Unsigned component arrays in the same layout's order.
   * @param stride Allocated candidate slots per component in both arrays.
   */
  KFPARTICLE_GPU_HOST_DEVICE KFParticleGpuCandidateMetadataSoAViewBase(
    IntegerValue* integers,
    UnsignedValue* unsignedValues,
    unsigned int stride)
    : fIntegers(integers), fUnsigned(unsignedValues), fStride(stride)
  {
  }

  /// @brief Returns a signed component reference; component and candidate indices are unchecked.
  KFPARTICLE_GPU_HOST_DEVICE IntegerValue& Integer(unsigned int component, unsigned int candidate) const
  {
    return fIntegers[component * fStride + candidate];
  }

  /// @brief Returns an unsigned component reference; component and candidate indices are unchecked.
  KFPARTICLE_GPU_HOST_DEVICE UnsignedValue& Unsigned(unsigned int component, unsigned int candidate) const
  {
    return fUnsigned[component * fStride + candidate];
  }

  /// @brief Returns the candidate PDG hypothesis reference.
  KFPARTICLE_GPU_HOST_DEVICE IntegerValue& Pdg(unsigned int candidate) const
  {
    return Integer(KFParticleGpuCandidateMetadataLayout::Pdg, candidate);
  }

  /// @brief Returns the associated primary-vertex index or unassigned sentinel reference.
  KFPARTICLE_GPU_HOST_DEVICE IntegerValue& PrimaryVertexIndex(unsigned int candidate) const
  {
    return Integer(KFParticleGpuCandidateMetadataLayout::PrimaryVertexIndex, candidate);
  }

  /// @brief Returns the input event descriptor index reference.
  KFPARTICLE_GPU_HOST_DEVICE UnsignedValue& EventIndex(unsigned int candidate) const
  {
    return Unsigned(KFParticleGpuCandidateMetadataLayout::EventIndex, candidate);
  }

  /// @brief Returns the offset of flattened source-track lineage in daughter storage.
  KFPARTICLE_GPU_HOST_DEVICE UnsignedValue& DaughterOffset(unsigned int candidate) const
  {
    return Unsigned(KFParticleGpuCandidateMetadataLayout::DaughterOffset, candidate);
  }

  /// @brief Returns the number of final source-track identifiers, not the number of direct daughters.
  KFPARTICLE_GPU_HOST_DEVICE UnsignedValue& DaughterCount(unsigned int candidate) const
  {
    return Unsigned(KFParticleGpuCandidateMetadataLayout::DaughterCount, candidate);
  }

  /// @brief Returns the KFParticleGpuCandidateFlags bit-mask reference.
  KFPARTICLE_GPU_HOST_DEVICE UnsignedValue& Flags(unsigned int candidate) const
  {
    return Unsigned(KFParticleGpuCandidateMetadataLayout::Flags, candidate);
  }

  /// @brief Returns the stable reconstruction channel ID reference.
  KFPARTICLE_GPU_HOST_DEVICE UnsignedValue& ChannelId(unsigned int candidate) const
  {
    return Unsigned(KFParticleGpuCandidateMetadataLayout::ChannelId, candidate);
  }

  /// @brief Returns the decay graph output category reference.
  KFPARTICLE_GPU_HOST_DEVICE UnsignedValue& OutputClass(unsigned int candidate) const
  {
    return Unsigned(KFParticleGpuCandidateMetadataLayout::OutputClass, candidate);
  }

  /// @brief Returns the decay graph topology reference.
  KFPARTICLE_GPU_HOST_DEVICE UnsignedValue& Topology(unsigned int candidate) const
  {
    return Unsigned(KFParticleGpuCandidateMetadataLayout::Topology, candidate);
  }

  /// @brief Returns the construction/operation outcome, separate from later selection flags.
  KFPARTICLE_GPU_HOST_DEVICE UnsignedValue& OperationStatus(unsigned int candidate) const
  {
    return Unsigned(KFParticleGpuCandidateMetadataLayout::OperationStatus, candidate);
  }

  /// @brief Returns the number of valid direct daughter references, at most two.
  KFPARTICLE_GPU_HOST_DEVICE UnsignedValue& DirectDaughterCount(unsigned int candidate) const
  {
    return Unsigned(KFParticleGpuCandidateMetadataLayout::DirectDaughterCount, candidate);
  }

  /// @brief Returns the storage kind addressed by the first direct daughter.
  KFPARTICLE_GPU_HOST_DEVICE UnsignedValue& DirectFirstKind(unsigned int candidate) const
  {
    return Unsigned(KFParticleGpuCandidateMetadataLayout::DirectFirstKind, candidate);
  }

  /// @brief Returns the first direct daughter's index, interpreted according to DirectFirstKind().
  KFPARTICLE_GPU_HOST_DEVICE UnsignedValue& DirectFirstIndex(unsigned int candidate) const
  {
    return Unsigned(KFParticleGpuCandidateMetadataLayout::DirectFirstIndex, candidate);
  }

  /// @brief Returns the storage kind addressed by the second direct daughter.
  KFPARTICLE_GPU_HOST_DEVICE UnsignedValue& DirectSecondKind(unsigned int candidate) const
  {
    return Unsigned(KFParticleGpuCandidateMetadataLayout::DirectSecondKind, candidate);
  }

  /// @brief Returns the second direct daughter's index, interpreted according to DirectSecondKind().
  KFPARTICLE_GPU_HOST_DEVICE UnsignedValue& DirectSecondIndex(unsigned int candidate) const
  {
    return Unsigned(KFParticleGpuCandidateMetadataLayout::DirectSecondIndex, candidate);
  }

  /// @brief Returns the borrowed signed-component array pointer.
  KFPARTICLE_GPU_HOST_DEVICE IntegerValue* IntegersData() const { return fIntegers; }

  /// @brief Returns the borrowed unsigned-component array pointer.
  KFPARTICLE_GPU_HOST_DEVICE UnsignedValue* UnsignedData() const { return fUnsigned; }

  /// @brief Returns allocated slots per component, not the active candidate count.
  KFPARTICLE_GPU_HOST_DEVICE unsigned int Stride() const { return fStride; }

 private:
  IntegerValue* fIntegers;   ///< Borrowed signed-component storage.
  UnsignedValue* fUnsigned;  ///< Borrowed unsigned-component storage.
  unsigned int fStride;      ///< Distance in elements between adjacent component arrays.
};

/// @brief Writable candidate metadata view; owns no memory.
typedef KFParticleGpuCandidateMetadataSoAViewBase<int, unsigned int>
  KFParticleGpuCandidateMetadataSoAView;

/// @brief Read-only candidate metadata view over the same layout.
typedef KFParticleGpuCandidateMetadataSoAViewBase<const int, const unsigned int>
  KFParticleGpuConstCandidateMetadataSoAView;

/// @brief Creates a read-only metadata descriptor sharing the original arrays and their lifetime.
KFPARTICLE_GPU_HOST_DEVICE inline KFParticleGpuConstCandidateMetadataSoAView
MakeConstView(const KFParticleGpuCandidateMetadataSoAView& view)
{
  return KFParticleGpuConstCandidateMetadataSoAView(
    view.IntegersData(), view.UnsignedData(), view.Stride());
}

/**
 * @brief Non-owning view of flattened final source-track identifiers and their shared size counter.
 * @tparam IntegerValue int or const int, controlling access to source IDs.
 * @tparam UnsignedValue unsigned int or const unsigned int, controlling access to the counter.
 *
 * Each candidate refers to a contiguous range given by DaughterOffset and
 * DaughterCount in its metadata. This CSR-style representation stores leaf
 * source IDs, not direct candidate indices. It neither reserves slots nor
 * updates the counter automatically. Only CanStore() checks capacity; it does
 * not validate pointers or confirm that a range has been filled.
 *
 * @pre Bind the descriptor before use and keep its arrays/counter alive.
 * Access and counter updates must be synchronized by the caller.
 */
template<typename IntegerValue, typename UnsignedValue>
class KFParticleGpuDaughterStorageViewBase
{
 public:
  /// @brief Defaulted descriptor constructor; assign a bound view before accessing storage.
  KFPARTICLE_GPU_HOST_DEVICE KFParticleGpuDaughterStorageViewBase() = default;

  /**
   * @brief Binds source IDs and their counter without allocating or copying data.
   * @param sourceIds Flattened source-track identifier array.
   * @param size Shared element counter; nullptr makes Size() return zero.
   * @param capacity Allocated source-ID slots, not the number of candidates.
   */
  KFPARTICLE_GPU_HOST_DEVICE KFParticleGpuDaughterStorageViewBase(
    IntegerValue* sourceIds,
    UnsignedValue* size,
    unsigned int capacity)
    : fSourceIds(sourceIds), fSize(size), fCapacity(capacity)
  {
  }

  /// @brief Returns a source-ID reference at an unchecked flattened index.
  KFPARTICLE_GPU_HOST_DEVICE IntegerValue& SourceId(unsigned int index) const { return fSourceIds[index]; }

  /// @brief Reads the shared counter, or zero for a null counter pointer; does not clamp to capacity.
  KFPARTICLE_GPU_HOST_DEVICE unsigned int Size() const
  {
    return fSize ? static_cast<unsigned int>(*fSize) : 0;
  }

  /// @brief Returns the allocated number of source-ID slots.
  KFPARTICLE_GPU_HOST_DEVICE unsigned int Capacity() const { return fCapacity; }

  /// @brief Tests whether [offset, offset + count) fits capacity without unsigned addition overflow; reserves nothing.
  KFPARTICLE_GPU_HOST_DEVICE bool CanStore(unsigned int offset, unsigned int count) const
  {
    return offset <= fCapacity && count <= fCapacity - offset;
  }

  /// @brief Returns the borrowed flattened source-ID array pointer.
  KFPARTICLE_GPU_HOST_DEVICE IntegerValue* SourceIdsData() const { return fSourceIds; }

  /// @brief Returns the borrowed shared element-counter pointer.
  KFPARTICLE_GPU_HOST_DEVICE UnsignedValue* SizeData() const { return fSize; }

 private:
  IntegerValue* fSourceIds;  ///< Borrowed flattened source-ID storage.
  UnsignedValue* fSize;      ///< Borrowed shared element counter, optionally null.
  unsigned int fCapacity;    ///< Allocated source-ID slots.
};

/// @brief Writable flattened source-track lineage view.
typedef KFParticleGpuDaughterStorageViewBase<int, unsigned int> KFParticleGpuDaughterStorageView;

/// @brief Read-only flattened source-track lineage view.
typedef KFParticleGpuDaughterStorageViewBase<const int, const unsigned int>
  KFParticleGpuConstDaughterStorageView;

/// @brief Creates a read-only lineage descriptor sharing the source-ID array and counter.
KFPARTICLE_GPU_HOST_DEVICE inline KFParticleGpuConstDaughterStorageView
MakeConstView(const KFParticleGpuDaughterStorageView& view)
{
  return KFParticleGpuConstDaughterStorageView(
    view.SourceIdsData(), view.SizeData(), view.Capacity());
}

/** @brief Bit flags reporting insufficient candidate or daughter storage. */
enum KFParticleGpuPoolOverflow
{
  CandidateCapacityExceeded = 1u << 0,  ///< A candidate/task reservation exceeded its available slots.
  DaughterCapacityExceeded = 1u << 1    ///< A flattened daughter reservation exceeded its available slots.
};

/** @brief Candidate validity, construction-mode, and selection-result bits. */
enum KFParticleGpuCandidateFlags
{
  KFGpuCandidateValid = 1u << 0,             ///< Candidate has a valid constructed state.
  KFGpuCandidateBuildFailed = 1u << 1,       ///< Candidate construction failed.
  KFGpuCandidateLineDca = 1u << 2,           ///< Line-DCA construction mode flag.
  KFGpuCandidateEnergyFit = 1u << 3,         ///< Energy-fit construction mode flag.
  KFGpuCandidateSelectionRejected = 1u << 4  ///< Candidate failed a subsequent selection step.
};

/** @brief Persistent construction/operation outcome, independent of topology and later selection. */
enum KFParticleGpuCandidateOperationStatus : unsigned int
{
  KFGpuCandidateOperationPending = 0u,   ///< Outcome not yet finalized.
  KFGpuCandidateOperationAccepted = 1u,  ///< Construction/operation accepted the candidate.
  KFGpuCandidateOperationRejected = 2u   ///< Construction/operation rejected the candidate.
};

/**
 * @brief Non-owning candidate pool combining fit states, metadata, and flattened daughter lineage.
 * @tparam FloatValue float or const float for numerical fit storage.
 * @tparam IntegerValue int or const int for signed fit/metadata and source IDs.
 * @tparam UnsignedValue unsigned int or const unsigned int for metadata and counters.
 *
 * Kernels reserve candidate and daughter slots separately, then use this view
 * to access them. The view does not allocate, reserve, copy, reset, or synchronize
 * storage. Size() reads a shared counter and is not clamped to Capacity().
 *
 * Constness of the descriptor does not make writable element types read-only;
 * use MakeConstView() for read-only access. Replacing or releasing backing
 * storage invalidates the corresponding pointers in all copies of the view.
 *
 * @pre Bind mutually consistent arrays and counters before use. Callers ensure
 * accessible memory, valid indices, and exclusive slots or atomic counter updates.
 */
template<typename FloatValue, typename IntegerValue, typename UnsignedValue>
class KFParticleGpuCandidatePoolViewBase
{
 public:
  /// @brief Numerical fit-state SoA descriptor with the pool's element mutability.
  typedef KFParticleGpuFitSoAViewBase<FloatValue, IntegerValue> FitView;

  /// @brief Candidate metadata descriptor with the pool's element mutability.
  typedef KFParticleGpuCandidateMetadataSoAViewBase<IntegerValue, UnsignedValue> MetadataView;

  /// @brief Flattened source-track lineage descriptor with the pool's element mutability.
  typedef KFParticleGpuDaughterStorageViewBase<IntegerValue, UnsignedValue> DaughterView;

  /// @brief Defaulted descriptor constructor; assign a bound view before accessing storage.
  KFPARTICLE_GPU_HOST_DEVICE KFParticleGpuCandidatePoolViewBase() = default;

  /**
   * @brief Binds pool descriptors and counters without copying the underlying data.
   * @param fit Numerical storage with at least @p capacity candidate slots.
   * @param metadata Metadata storage with at least @p capacity candidate slots.
   * @param daughters Flattened lineage storage with its own capacity and counter.
   * @param size Shared candidate counter; nullptr makes Size() return zero.
   * @param overflowFlags Shared overflow mask; nullptr makes OverflowFlags() return zero.
   * @param capacity Candidate slot limit; no consistency checks are performed here.
   */
  KFPARTICLE_GPU_HOST_DEVICE KFParticleGpuCandidatePoolViewBase(
    const FitView& fit,
    const MetadataView& metadata,
    const DaughterView& daughters,
    UnsignedValue* size,
    UnsignedValue* overflowFlags,
    unsigned int capacity)
    : fFit(fit)
    , fMetadata(metadata)
    , fDaughters(daughters)
    , fSize(size)
    , fOverflowFlags(overflowFlags)
    , fCapacity(capacity)
  {
  }

  /// @brief Returns the stored fit descriptor by reference.
  KFPARTICLE_GPU_HOST_DEVICE FitView& Fit() { return fFit; }

  /// @brief Returns a const fit descriptor reference; element mutability still follows FloatValue/IntegerValue.
  KFPARTICLE_GPU_HOST_DEVICE const FitView& Fit() const { return fFit; }

  /// @brief Returns the stored metadata descriptor by reference.
  KFPARTICLE_GPU_HOST_DEVICE MetadataView& Metadata() { return fMetadata; }

  /// @brief Returns a const metadata descriptor reference; element mutability still follows its template types.
  KFPARTICLE_GPU_HOST_DEVICE const MetadataView& Metadata() const { return fMetadata; }

  /// @brief Returns the stored daughter-lineage descriptor by reference.
  KFPARTICLE_GPU_HOST_DEVICE DaughterView& Daughters() { return fDaughters; }

  /// @brief Returns a const lineage descriptor reference; element mutability still follows its template types.
  KFPARTICLE_GPU_HOST_DEVICE const DaughterView& Daughters() const { return fDaughters; }

  /// @brief Reads the shared candidate counter, or zero for a null pointer; does not clamp to capacity.
  KFPARTICLE_GPU_HOST_DEVICE unsigned int Size() const
  {
    return fSize ? static_cast<unsigned int>(*fSize) : 0;
  }

  /// @brief Returns the allocated candidate slot limit supplied by the caller.
  KFPARTICLE_GPU_HOST_DEVICE unsigned int Capacity() const { return fCapacity; }

  /// @brief Tests only whether @p index is below capacity; does not reserve a slot or validate its data.
  KFPARTICLE_GPU_HOST_DEVICE bool CanStoreCandidate(unsigned int index) const { return index < fCapacity; }

  /// @brief Reads the shared overflow mask, or zero for a null pointer; performs no synchronization.
  KFPARTICLE_GPU_HOST_DEVICE unsigned int OverflowFlags() const
  {
    return fOverflowFlags ? static_cast<unsigned int>(*fOverflowFlags) : 0;
  }

  /// @brief Returns the borrowed candidate-counter pointer for externally coordinated updates.
  KFPARTICLE_GPU_HOST_DEVICE UnsignedValue* SizeData() const { return fSize; }

  /// @brief Returns the borrowed overflow-mask pointer; atomic updates belong to the kernel layer.
  KFPARTICLE_GPU_HOST_DEVICE UnsignedValue* OverflowFlagsData() const { return fOverflowFlags; }

 private:
  FitView fFit;                   ///< Non-owning numerical fit descriptor.
  MetadataView fMetadata;         ///< Non-owning identity/status/lineage metadata descriptor.
  DaughterView fDaughters;        ///< Non-owning flattened source-ID descriptor.
  UnsignedValue* fSize;           ///< Borrowed candidate counter, optionally null.
  UnsignedValue* fOverflowFlags;  ///< Borrowed overflow mask, optionally null.
  unsigned int fCapacity;         ///< Allocated candidate slots.
};

/// @brief Writable candidate pool descriptor; does not own the underlying arrays.
typedef KFParticleGpuCandidatePoolViewBase<float, int, unsigned int> KFParticleGpuCandidatePoolView;

/// @brief Read-only candidate pool descriptor with const element types throughout.
typedef KFParticleGpuCandidatePoolViewBase<const float, const int, const unsigned int>
  KFParticleGpuConstCandidatePoolView;

/// @brief Creates a read-only pool descriptor sharing all arrays and counters with the original pool.
KFPARTICLE_GPU_HOST_DEVICE inline KFParticleGpuConstCandidatePoolView
MakeConstView(const KFParticleGpuCandidatePoolView& view)
{
  return KFParticleGpuConstCandidatePoolView(
    MakeConstView(view.Fit()),
    MakeConstView(view.Metadata()),
    MakeConstView(view.Daughters()),
    view.SizeData(),
    view.OverflowFlagsData(),
    view.Capacity());
}

/**
 * @brief Clears both direct daughter references and their count at an existing candidate slot.
 * @note Does not change the flattened source-ID lineage. The candidate index is unchecked.
 */
KFPARTICLE_GPU_HOST_DEVICE inline void ClearCandidateDirectDaughters(
  const KFParticleGpuCandidateMetadataSoAView& metadata,
  unsigned int candidate)
{
  metadata.DirectDaughterCount(candidate) = 0u;
  metadata.DirectFirstKind(candidate) = KFGpuDirectDaughterNone;
  metadata.DirectFirstIndex(candidate) = 0u;
  metadata.DirectSecondKind(candidate) = KFGpuDirectDaughterNone;
  metadata.DirectSecondIndex(candidate) = 0u;
}

/**
 * @brief Writes one direct daughter reference without changing DirectDaughterCount().
 * @param metadata Writable candidate metadata view.
 * @param candidate Existing candidate slot; not bounds-checked.
 * @param daughter Direct slot, either zero or one; every nonzero value addresses the second slot.
 * @param kind KFParticleGpuDirectDaughterKind identifying the referenced storage.
 * @param index Input-track or candidate-pool index, interpreted according to @p kind.
 * @note Does not validate the reference or modify flattened source-track lineage.
 */
KFPARTICLE_GPU_HOST_DEVICE inline void SetCandidateDirectDaughter(
  const KFParticleGpuCandidateMetadataSoAView& metadata,
  unsigned int candidate,
  unsigned int daughter,
  unsigned int kind,
  unsigned int index)
{
  if (daughter == 0u) {
    metadata.DirectFirstKind(candidate) = kind;
    metadata.DirectFirstIndex(candidate) = index;
  }
  else {
    metadata.DirectSecondKind(candidate) = kind;
    metadata.DirectSecondIndex(candidate) = index;
  }
}

/**
 * @brief Copies one candidate's numerical fit fields into a local fit state, without metadata or lineage.
 * @pre The source slot is populated and accessible; no bounds check, allocation, or H2D/D2H transfer is performed.
 */
KFPARTICLE_GPU_HOST_DEVICE inline void LoadCandidateFit(
  const KFParticleGpuConstCandidatePoolView& source,
  unsigned int candidate,
  KFParticleGpuFitState& destination)
{
  LoadFitState(source.Fit(), candidate, destination);
}

/**
 * @brief Stores numerical fit fields at a candidate slot already reserved by the caller.
 * @param source Local fit state to store.
 * @param destination Bound pool with fit storage consistent with its capacity.
 * @param candidate Reserved candidate index.
 * @return False without writing if the index exceeds pool capacity; true after storing the fit.
 * @note Does not write metadata/lineage or update size/overflow counters. The
 * kernel layer owns reservation, synchronization, and atomic overflow updates.
 */
KFPARTICLE_GPU_HOST_DEVICE inline bool StoreCandidateFit(
  const KFParticleGpuFitState& source,
  const KFParticleGpuCandidatePoolView& destination,
  unsigned int candidate)
{
  if (!destination.CanStoreCandidate(candidate)) {
    return false;
  }
  StoreFitState(source, destination.Fit(), candidate);
  return true;
}

#endif
