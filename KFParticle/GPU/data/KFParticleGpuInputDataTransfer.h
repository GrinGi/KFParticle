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
 * @brief Packed event descriptors, borrowed track/vertex views, and local state-copy helpers.
 *
 * Ranges refer to shared packed arrays; views hold pointers without owning memory.
 * Load/Store helpers copy numerical values in the calling execution context, not
 * between host and device address spaces. Allocation, transfers, and synchronization
 * are caller responsibilities. Unless explicitly stated, access is unchecked.
 */

#ifndef KFPARTICLEGPUINPUTDATATRANSFER_H
#define KFPARTICLEGPUINPUTDATATRANSFER_H

#include "data/KFParticleGpuFitState.h"
#include "data/KFParticleGpuSoAView.h"

#include <climits>

/**
 * @brief Half-open interval [offset, offset + size) in a packed array.
 * @note Contains no pointer or capacity. The caller must ensure that the interval
 * fits its backing storage and its unsigned end is representable.
 */
struct KFParticleGpuRange
{
  unsigned int offset;  ///< Absolute start index in the associated packed array.
  unsigned int size;    ///< Logical element count, not allocation capacity.

  /// @brief Creates an empty range at offset zero.
  KFPARTICLE_GPU_HOST_DEVICE KFParticleGpuRange() : offset(0), size(0) {}

  /// @brief Stores an offset and count without checking bounds or unsigned overflow.
  KFPARTICLE_GPU_HOST_DEVICE KFParticleGpuRange(unsigned int rangeOffset, unsigned int rangeSize)
    : offset(rangeOffset), size(rangeSize)
  {
  }

  /// @brief Returns the exclusive end by unchecked unsigned addition.
  KFPARTICLE_GPU_HOST_DEVICE unsigned int End() const { return offset + size; }

  /// @brief Tests half-open range membership; assumes End() does not overflow.
  KFPARTICLE_GPU_HOST_DEVICE bool Contains(unsigned int index) const
  {
    return index >= offset && index < End();
  }
};

/**
 * @brief Convert an event-local PV association to the shared packed PV index.
 * @param localIndex Local PV slot, or any negative unassigned value to preserve.
 * @param vertices Event range in the shared PV buffer; its storage is caller-owned.
 * @param[out] packedIndex Converted index; unchanged on failure.
 * @return False for a nonnegative index outside the event or an int overflow.
 * @note This is an index conversion, not a memory transfer or range-allocation check.
 */
KFPARTICLE_GPU_HOST_DEVICE inline bool KFParticleGpuPackPrimaryVertexIndex(
  int localIndex, const KFParticleGpuRange& vertices, int& packedIndex)
{
  if (localIndex < 0) {
    packedIndex = localIndex;
    return true;
  }
  const unsigned int local = static_cast<unsigned int>(localIndex);
  if (local >= vertices.size
      || vertices.offset > static_cast<unsigned int>(INT_MAX) - local) {
    return false;
  }
  packedIndex = static_cast<int>(vertices.offset + local);
  return true;
}

/**
 * @brief Check that a packed PV index belongs to a bounded event range.
 * Negative/unassigned indices, empty or invalid ranges, and foreign-event PVs fail.
 * Uses subtraction-based checks so malformed unsigned ranges cannot wrap around.
 * Supply vertexCount when checking against actual vertex storage; the default
 * checks event membership and unsigned representability without a storage view.
 */
KFPARTICLE_GPU_HOST_DEVICE inline bool KFParticleGpuIsEventPrimaryVertex(
  int packedIndex, const KFParticleGpuRange& vertices, unsigned int vertexCount = UINT_MAX)
{
  if (packedIndex < 0 || vertices.offset > vertexCount
      || vertices.size > vertexCount - vertices.offset) {
    return false;
  }
  const unsigned int index = static_cast<unsigned int>(packedIndex);
  return index >= vertices.offset && index - vertices.offset < vertices.size;
}

/**
 * @brief Packed CPU finder track-vector roles by PV classification, charge sign, and track-state endpoint.
 * First/Last identify track-state representations, not event order or separate physical track identities.
 */
enum KFParticleGpuTrackSet
{
  SecondaryPositiveFirst = 0,  ///< Secondary positive tracks at the first endpoint.
  SecondaryNegativeFirst,      ///< Secondary negative tracks at the first endpoint.
  PrimaryPositiveFirst,        ///< Primary positive tracks at the first endpoint.
  PrimaryNegativeFirst,        ///< Primary negative tracks at the first endpoint.
  SecondaryPositiveLast,       ///< Secondary positive tracks at the last endpoint.
  SecondaryNegativeLast,       ///< Secondary negative tracks at the last endpoint.
  PrimaryPositiveLast,         ///< Primary positive tracks at the last endpoint.
  PrimaryNegativeLast,         ///< Primary negative tracks at the last endpoint.
  NumberOfTrackSets            ///< Role count; not a valid TrackSet() index.
};

/**
 * @brief Species-range roles within one packed track set, not signed PDG codes.
 * The count sentinel also denotes the whole track set in graph source IDs, but
 * is not a valid index into KFParticleGpuTrackSetDesc::species.
 */
enum KFParticleGpuTrackSpecies
{
  Electron = 0,         ///< Electron/positron species role.
  Muon,                 ///< Muon species role.
  Pion,                 ///< Charged-pion species role.
  Kaon,                 ///< Charged-kaon species role.
  Proton,               ///< Proton/antiproton species role.
  Deuteron,             ///< Deuteron species role.
  Triton,               ///< Triton species role.
  Helium3,              ///< Helium-3 species role.
  Helium4,              ///< Helium-4 species role.
  Helium6,              ///< Helium-6 species role.
  Lithium6,             ///< Lithium-6 species role.
  Lithium7,             ///< Lithium-7 species role.
  Beryllium7,           ///< Beryllium-7 species role.
  NumberOfTrackSpecies  ///< Species count and graph-source whole-set sentinel.
};

/**
 * @brief Encodes a track-set/species role as a nonzero graph source ID.
 * @param set Valid track-set role, strictly below NumberOfTrackSets.
 * @param species Species role or NumberOfTrackSpecies for the whole set.
 * @return One-based role ID; this is neither a physical track SourceId nor an event index.
 * @note Performs no validation; the caller must provide values in the stated ranges.
 */
KFPARTICLE_GPU_HOST_DEVICE inline unsigned int KFParticleGpuGraphTrackSourceId(
  KFParticleGpuTrackSet set,
  KFParticleGpuTrackSpecies species)
{
  return 1u + static_cast<unsigned int>(set)
    * (static_cast<unsigned int>(NumberOfTrackSpecies) + 1u)
    + static_cast<unsigned int>(species);
}

/**
 * @brief Decodes a valid graph track-source role, accepting the whole-set species sentinel.
 * @param sourceId One-based role encoding produced by KFParticleGpuGraphTrackSourceId().
 * @param[out] set Decoded track-set role on success.
 * @param[out] species Decoded species or NumberOfTrackSpecies on success.
 * @return False for zero or an out-of-range role ID, leaving both outputs unchanged.
 */
KFPARTICLE_GPU_HOST_DEVICE inline bool KFParticleGpuDecodeGraphTrackSourceId(
  unsigned int sourceId,
  KFParticleGpuTrackSet& set,
  KFParticleGpuTrackSpecies& species)
{
  if (sourceId == 0u) return false;
  const unsigned int value = sourceId - 1u;
  const unsigned int stride = static_cast<unsigned int>(NumberOfTrackSpecies) + 1u;
  const unsigned int setIndex = value / stride;
  const unsigned int speciesIndex = value % stride;
  if (setIndex >= static_cast<unsigned int>(NumberOfTrackSets)
      || speciesIndex > static_cast<unsigned int>(NumberOfTrackSpecies)) {
    return false;
  }
  set = static_cast<KFParticleGpuTrackSet>(setIndex);
  species = static_cast<KFParticleGpuTrackSpecies>(speciesIndex);
  return true;
}

/**
 * @brief Absolute ranges of one CPU KFPTrackVector after packing into shared SoA.
 *
 * GPU ranges contain physical tracks only and do not include CPU SIMD padding.
 * Species offsets use the same shared-array coordinates as tracks, not offsets
 * relative to this track set. This descriptor does not validate range consistency.
 */
struct KFParticleGpuTrackSetDesc
{
  KFParticleGpuRange tracks;  ///< Complete packed interval for this track-set role.
  KFParticleGpuRange species[NumberOfTrackSpecies];  ///< Absolute per-species subranges.

  /// @brief Borrows a read-only species range; value must be strictly below NumberOfTrackSpecies.
  KFPARTICLE_GPU_HOST_DEVICE const KFParticleGpuRange& Species(KFParticleGpuTrackSpecies value) const
  {
    return species[static_cast<int>(value)];
  }

  /// @brief Borrows a writable species range; does not accept the whole-set sentinel or check bounds.
  KFPARTICLE_GPU_HOST_DEVICE KFParticleGpuRange& Species(KFParticleGpuTrackSpecies value)
  {
    return species[static_cast<int>(value)];
  }
};

/**
 * @brief Event identity, track-set ranges, PV range, and the secondary-track input cut.
 *
 * Describes one event within shared packed input without owning or copying its
 * tracks or vertices. All offsets are absolute in their corresponding buffers.
 * eventId is supplied by the producer and is distinct from this descriptor's
 * packed-array index used by tasks and candidate metadata.
 */
struct KFParticleGpuEventDesc
{
  unsigned int eventId;  ///< Producer-supplied event identity, not necessarily the packed event index.
  KFParticleGpuTrackSetDesc trackSets[NumberOfTrackSets];  ///< Input ranges for all finder track-set roles.
  KFParticleGpuRange primaryVertices;  ///< Event-local interval in the shared packed PV buffer.
  float minSecondaryTrackChiToPrimaryVertex;  ///< Secondary-track chi lower cut; -1 leaves it unspecified.

  /// @brief Initializes event ID zero, empty ranges, and an unspecified secondary-track cut.
  KFPARTICLE_GPU_HOST_DEVICE KFParticleGpuEventDesc()
    : eventId(0), trackSets(), primaryVertices(), minSecondaryTrackChiToPrimaryVertex(-1.f)
  {
  }

  /// @brief Borrows a read-only track-set descriptor; requires a valid role below NumberOfTrackSets.
  KFPARTICLE_GPU_HOST_DEVICE const KFParticleGpuTrackSetDesc& TrackSet(KFParticleGpuTrackSet value) const
  {
    return trackSets[static_cast<int>(value)];
  }

  /// @brief Borrows a writable track-set descriptor without checking the role index.
  KFPARTICLE_GPU_HOST_DEVICE KFParticleGpuTrackSetDesc& TrackSet(KFParticleGpuTrackSet value)
  {
    return trackSets[static_cast<int>(value)];
  }
};

/// @brief Component indices for packed track integer metadata, separate from numerical track state.
struct KFParticleGpuTrackInputLayout
{
  /// @brief Component-major integer-array layout; each component occupies one stride.
  enum IntegerComponent
  {
    SourceId = 0,              ///< Producer's physical track ID for lineage matching, not a packed index.
    Pdg,                       ///< Input PID hypothesis.
    Charge,                    ///< Electric charge.
    PrimaryVertexIndex,        ///< Absolute packed PV index in this event's range, or a negative unassigned value.
    NumberOfPixelHits,          ///< Pixel-hit count used by optional track cuts.
    NumberOfIntegerComponents  ///< Number of integer component blocks.
  };
};

/**
 * @brief Non-owning component-major view of numerical finder tracks and their input metadata.
 *
 * The field pointer may be null for homogeneous-field builds. All other arrays
 * use the numerical view stride for component-major storage; chi-to-PV is a
 * single scalar array indexed directly by track. Size is the active track count;
 * stride is the spacing between components and must be at least Size().
 *
 * Accessors do not check indices, pointer validity, or finite values. Memory must
 * remain alive and accessible in the calling host/device context. A const view
 * object does not make its pointees const; the template arguments control that.
 * @tparam FloatValue float for writable data, const float for read-only data.
 * @tparam IntegerValue int for writable metadata, const int for read-only metadata.
 */
template<typename FloatValue, typename IntegerValue>
class KFParticleGpuInputTrackSoAViewBase
{
 public:
  /**
   * @brief Keeps the view trivially default-constructible without binding storage.
   * @note Plain default-initialization leaves members indeterminate; {} value-initializes
   * them to zero. Neither form supplies arrays that can be dereferenced.
   */
  KFPARTICLE_GPU_HOST_DEVICE KFParticleGpuInputTrackSoAViewBase() = default;

  /**
   * @brief Binds borrowed input arrays and layout dimensions without allocating, copying, or validating data.
   * @param parameters Six component blocks of track coordinates/momenta.
   * @param covariances Twenty-one packed covariance component blocks.
   * @param fieldCoefficients Ten field-coefficient blocks, or null when no local field regions are supplied.
   * @param chiToPrimaryVertex One scalar per track slot.
   * @param integers Integer component blocks defined by KFParticleGpuTrackInputLayout.
   * @param size Logical number of active tracks.
   * @param stride Element spacing between component blocks, at least size.
   * @note Component arrays must cover their component count times stride. Scalar
   * access requires readable/writable entries for the requested track indices.
   */
  KFPARTICLE_GPU_HOST_DEVICE KFParticleGpuInputTrackSoAViewBase(
    FloatValue* parameters,
    FloatValue* covariances,
    FloatValue* fieldCoefficients,
    FloatValue* chiToPrimaryVertex,
    IntegerValue* integers,
    unsigned int size,
    unsigned int stride)
    : fNumerical(parameters, covariances, size, stride)
    , fFieldCoefficients(fieldCoefficients)
    , fChiToPrimaryVertex(chiToPrimaryVertex)
    , fIntegers(integers)
  {
  }

  /**
   * @brief Borrows the writable numerical-view object.
   * @note Layout changes must remain consistent with the field and metadata arrays.
   */
  KFPARTICLE_GPU_HOST_DEVICE KFParticleGpuTrackSoAViewBase<FloatValue>& Numerical()
  {
    return fNumerical;
  }

  /// @brief Borrows the read-only numerical-view object; pointee mutability still follows FloatValue.
  KFPARTICLE_GPU_HOST_DEVICE const KFParticleGpuTrackSoAViewBase<FloatValue>& Numerical() const
  {
    return fNumerical;
  }

  /// @brief Accesses a field coefficient at component * Stride() + track; requires a nonnull field and component < 10.
  KFPARTICLE_GPU_HOST_DEVICE FloatValue& FieldCoefficient(unsigned int component,
                                                          unsigned int track) const
  {
    return fFieldCoefficients[component * Stride() + track];
  }

  /// @brief Accesses a track's chi-to-PV scalar; requires track < Size() and a valid scalar array.
  KFPARTICLE_GPU_HOST_DEVICE FloatValue& ChiToPrimaryVertex(unsigned int track) const
  {
    return fChiToPrimaryVertex[track];
  }

  /// @brief Accesses an integer component using the shared stride; component and element indices must be valid.
  KFPARTICLE_GPU_HOST_DEVICE IntegerValue& Integer(unsigned int component, unsigned int track) const
  {
    return fIntegers[component * Stride() + track];
  }

  /// @brief Accesses the input track's lineage ID, not its packed-array index; requires track < Size().
  KFPARTICLE_GPU_HOST_DEVICE IntegerValue& SourceId(unsigned int track) const
  {
    return Integer(KFParticleGpuTrackInputLayout::SourceId, track);
  }

  /// @brief Accesses the input PID hypothesis for a valid track index.
  KFPARTICLE_GPU_HOST_DEVICE IntegerValue& Pdg(unsigned int track) const
  {
    return Integer(KFParticleGpuTrackInputLayout::Pdg, track);
  }

  /// @brief Accesses the electric charge for a valid track index.
  KFPARTICLE_GPU_HOST_DEVICE IntegerValue& Charge(unsigned int track) const
  {
    return Integer(KFParticleGpuTrackInputLayout::Charge, track);
  }

  /// @brief Accesses the packed PV association, which may be negative for an unassigned track.
  KFPARTICLE_GPU_HOST_DEVICE IntegerValue& PrimaryVertexIndex(unsigned int track) const
  {
    return Integer(KFParticleGpuTrackInputLayout::PrimaryVertexIndex, track);
  }

  /// @brief Accesses the pixel-hit count for a valid track index.
  KFPARTICLE_GPU_HOST_DEVICE IntegerValue& NumberOfPixelHits(unsigned int track) const
  {
    return Integer(KFParticleGpuTrackInputLayout::NumberOfPixelHits, track);
  }

  /// @brief Borrows the field-coefficient array; may return null when no field regions are supplied.
  KFPARTICLE_GPU_HOST_DEVICE FloatValue* FieldCoefficientsData() const { return fFieldCoefficients; }

  /// @brief Borrows the scalar chi-to-PV array without copying or validating it.
  KFPARTICLE_GPU_HOST_DEVICE FloatValue* ChiToPrimaryVertexData() const { return fChiToPrimaryVertex; }

  /// @brief Borrows the component-major integer array with template-controlled pointee constness.
  KFPARTICLE_GPU_HOST_DEVICE IntegerValue* IntegersData() const { return fIntegers; }

  /// @brief Returns the logical number of active elements, not the component stride.
  KFPARTICLE_GPU_HOST_DEVICE unsigned int Size() const { return fNumerical.Size(); }

  /// @brief Returns the element spacing between component blocks.
  KFPARTICLE_GPU_HOST_DEVICE unsigned int Stride() const { return fNumerical.Stride(); }

 private:
  /// Borrowed numerical track view carrying shared size and stride.
  KFParticleGpuTrackSoAViewBase<FloatValue> fNumerical;
  FloatValue* fFieldCoefficients;  ///< Borrowed optional component-major field coefficients.
  FloatValue* fChiToPrimaryVertex;  ///< Borrowed per-track chi-to-PV scalars.
  IntegerValue* fIntegers;  ///< Borrowed component-major integer data.
};

/// @brief Borrowed packed track view with writable numerical and integer data.
typedef KFParticleGpuInputTrackSoAViewBase<float, int> KFParticleGpuInputTrackSoAView;

/// @brief Borrowed packed track view whose numerical and integer data are read-only.
typedef KFParticleGpuInputTrackSoAViewBase<const float, const int> KFParticleGpuConstInputTrackSoAView;

/**
 * @brief Creates a read-only track view over exactly the same arrays, size, and stride.
 * @note Copies view metadata only. The result shares the original storage and lifetime;
 * it neither freezes data against other writers nor performs a host/device transfer.
 */
KFPARTICLE_GPU_HOST_DEVICE inline KFParticleGpuConstInputTrackSoAView
MakeConstView(const KFParticleGpuInputTrackSoAView& view)
{
  return KFParticleGpuConstInputTrackSoAView(view.Numerical().ParametersData(),
                                             view.Numerical().CovariancesData(),
                                             view.FieldCoefficientsData(),
                                             view.ChiToPrimaryVertexData(),
                                             view.IntegersData(),
                                             view.Size(),
                                             view.Stride());
}

/// @brief Integer component layout for packed primary vertices.
struct KFParticleGpuVertexSoALayout
{
  /// @brief Fit-counter components stored in separate stride-sized integer blocks.
  enum IntegerComponent
  {
    Ndf = 0,                   ///< Vertex fit degrees of freedom.
    NumberOfContributors,      ///< Number of tracks contributing to the vertex.
    NumberOfIntegerComponents  ///< Number of integer component blocks.
  };
};

/**
 * @brief Borrowed component-major view of vertex position, covariance, chi2, and fit counters.
 *
 * Position has three components and packed symmetric covariance has six. Chi2
 * is indexed directly by vertex; parameter, covariance, and integer blocks use
 * component * stride + vertex. Size is the logical count and must not exceed stride.
 * Indexed access is unchecked. This value neither owns memory nor transfers data;
 * the caller must preserve pointer validity and coordinate concurrent access.
 * @tparam FloatValue float or const float, controlling numerical data mutability.
 * @tparam IntegerValue int or const int, controlling counter mutability independently of view constness.
 */
template<typename FloatValue, typename IntegerValue>
class KFParticleGpuVertexSoAViewBase
{
 public:
  /**
   * @brief Keeps construction trivial; does not bind or validate backing arrays.
   * @note Plain default-initialization leaves members indeterminate; {} zeroes them.
   */
  KFPARTICLE_GPU_HOST_DEVICE KFParticleGpuVertexSoAViewBase() = default;

  /**
   * @brief Binds vertex arrays and layout dimensions without taking ownership or copying values.
   * @param parameters Three position component blocks.
   * @param covariances Six lower-triangular covariance component blocks.
   * @param chi2 One fit chi2 value per vertex slot.
   * @param integers Component blocks defined by KFParticleGpuVertexSoALayout.
   * @param size Active vertex count.
   * @param stride Element spacing between component blocks, at least size.
   * @note Callers supply arrays large enough for these dimensions in the current address space.
   */
  KFPARTICLE_GPU_HOST_DEVICE KFParticleGpuVertexSoAViewBase(FloatValue* parameters,
                                                            FloatValue* covariances,
                                                            FloatValue* chi2,
                                                            IntegerValue* integers,
                                                            unsigned int size,
                                                            unsigned int stride)
    : fParameters(parameters)
    , fCovariances(covariances)
    , fChi2(chi2)
    , fIntegers(integers)
    , fSize(size)
    , fStride(stride)
  {
  }

  /// @brief Accesses a vertex coordinate; requires component < 3 and vertex < Size().
  KFPARTICLE_GPU_HOST_DEVICE FloatValue& Parameter(unsigned int component, unsigned int vertex) const
  {
    return fParameters[component * fStride + vertex];
  }

  /// @brief Accesses one packed covariance component; requires component < 6 and vertex < Size().
  KFPARTICLE_GPU_HOST_DEVICE FloatValue& Covariance(unsigned int component, unsigned int vertex) const
  {
    return fCovariances[component * fStride + vertex];
  }

  /// @brief Accesses the fit chi2 scalar for a valid vertex index.
  KFPARTICLE_GPU_HOST_DEVICE FloatValue& Chi2(unsigned int vertex) const { return fChi2[vertex]; }

  /// @brief Accesses an integer component using the shared stride; component and element indices must be valid.
  KFPARTICLE_GPU_HOST_DEVICE IntegerValue& Integer(unsigned int component, unsigned int vertex) const
  {
    return fIntegers[component * fStride + vertex];
  }

  /// @brief Accesses the fit degrees of freedom for a valid vertex index.
  KFPARTICLE_GPU_HOST_DEVICE IntegerValue& NDF(unsigned int vertex) const
  {
    return Integer(KFParticleGpuVertexSoALayout::Ndf, vertex);
  }

  /// @brief Accesses the contributor count for a valid vertex index.
  KFPARTICLE_GPU_HOST_DEVICE IntegerValue& NContributors(unsigned int vertex) const
  {
    return Integer(KFParticleGpuVertexSoALayout::NumberOfContributors, vertex);
  }

  /// @brief Borrows the three position component blocks with template-controlled pointee constness.
  KFPARTICLE_GPU_HOST_DEVICE FloatValue* ParametersData() const { return fParameters; }

  /// @brief Borrows the six covariance component blocks with template-controlled pointee constness.
  KFPARTICLE_GPU_HOST_DEVICE FloatValue* CovariancesData() const { return fCovariances; }

  /// @brief Borrows the per-vertex chi2 array.
  KFPARTICLE_GPU_HOST_DEVICE FloatValue* Chi2Data() const { return fChi2; }

  /// @brief Borrows the component-major integer array with template-controlled pointee constness.
  KFPARTICLE_GPU_HOST_DEVICE IntegerValue* IntegersData() const { return fIntegers; }

  /// @brief Returns the logical number of active elements, not the component stride.
  KFPARTICLE_GPU_HOST_DEVICE unsigned int Size() const { return fSize; }

  /// @brief Returns the element spacing between component blocks.
  KFPARTICLE_GPU_HOST_DEVICE unsigned int Stride() const { return fStride; }

 private:
  FloatValue* fParameters;  ///< Borrowed position component array.
  FloatValue* fCovariances;  ///< Borrowed packed covariance component array.
  FloatValue* fChi2;  ///< Borrowed per-vertex fit chi2 values.
  IntegerValue* fIntegers;  ///< Borrowed component-major integer data.
  unsigned int fSize;  ///< Logical number of active vertices.
  unsigned int fStride;  ///< Spacing between component blocks in elements.
};

/// @brief Borrowed packed vertex view with writable numerical values and fit counters.
typedef KFParticleGpuVertexSoAViewBase<float, int> KFParticleGpuVertexSoAView;

/// @brief Borrowed packed vertex view with read-only numerical values and fit counters.
typedef KFParticleGpuVertexSoAViewBase<const float, const int> KFParticleGpuConstVertexSoAView;

/**
 * @brief Creates a read-only vertex view without copying its backing arrays.
 * @note Preserves pointers, size, and stride; lifetime and synchronization remain the caller's responsibility.
 */
KFPARTICLE_GPU_HOST_DEVICE inline KFParticleGpuConstVertexSoAView
MakeConstView(const KFParticleGpuVertexSoAView& view)
{
  return KFParticleGpuConstVertexSoAView(view.ParametersData(),
                                         view.CovariancesData(),
                                         view.Chi2Data(),
                                         view.IntegersData(),
                                         view.Size(),
                                         view.Stride());
}

/**
 * @brief Copies one track's six numerical parameters and 21 covariance entries into an owned local state.
 * @param source Readable packed input in the calling execution context.
 * @param track Valid index below source.Size().
 * @param[out] destination Receives numerical values only; no PID, field, charge, or PV metadata is loaded.
 * @note Delegates to the numerical-view overload without bounds or finite-value checks.
 */
KFPARTICLE_GPU_HOST_DEVICE inline void LoadTrackState(const KFParticleGpuConstInputTrackSoAView& source,
                                                      unsigned int track,
                                                      KFParticleGpuTrackState& destination)
{
  LoadTrackState(source.Numerical(), track, destination);
}

/**
 * @brief Copies numerical track parameters and covariance into an existing packed slot.
 * @param source Local numerical state to copy.
 * @param[in,out] destination Writable packed input; field and integer metadata remain unchanged.
 * @param track Valid destination index below destination.Size().
 * @note Does not reserve storage, change Size(), validate values, or enqueue a transfer.
 */
KFPARTICLE_GPU_HOST_DEVICE inline void StoreTrackState(const KFParticleGpuTrackState& source,
                                                       KFParticleGpuInputTrackSoAView& destination,
                                                       unsigned int track)
{
  StoreTrackState(source, destination.Numerical(), track);
}

/**
 * @brief Tests only whether the field-coefficient pointer is nonnull.
 * @note Does not check track count, array extent, coefficient validity, or whether the field is nonzero.
 */
KFPARTICLE_GPU_HOST_DEVICE inline bool HasFieldRegions(
  const KFParticleGpuConstInputTrackSoAView& source)
{
  return source.FieldCoefficientsData() != nullptr;
}

/**
 * @brief Copies all ten local field coefficients for one track into a field-region value.
 * @param source Packed input with a nonnull, readable field array.
 * @param track Valid index below source.Size().
 * @param[out] destination Receives the three quadratic field polynomials and their reference z.
 * @note Access is unchecked; use LoadFieldRegionOrZero() when the field array may be absent.
 */
KFPARTICLE_GPU_HOST_DEVICE inline void LoadFieldRegion(const KFParticleGpuConstInputTrackSoAView& source,
                                                       unsigned int track,
                                                       KFParticleGpuFieldRegion& destination)
{
  for (unsigned int component = 0; component < KFParticleGpuFieldRegion::NumberOfCoefficients; ++component) {
    destination.Coefficient(component) = source.FieldCoefficient(component, track);
  }
}

/**
 * @brief Resets a field region, then loads coefficients if the field pointer and track index are available.
 * @param source Packed input, optionally without field coefficients.
 * @param track Requested packed track index.
 * @param[out] destination Loaded field on success; all coefficients, including reference z, zero on failure.
 * @return False for a null field pointer or track >= source.Size(); true after copying coefficients.
 * @note Does not validate coefficient values or the backing allocation extent.
 */
KFPARTICLE_GPU_HOST_DEVICE inline bool LoadFieldRegionOrZero(
  const KFParticleGpuConstInputTrackSoAView& source,
  unsigned int track,
  KFParticleGpuFieldRegion& destination)
{
  destination.Reset();
  if (!HasFieldRegions(source) || track >= source.Size()) {
    return false;
  }
  LoadFieldRegion(source, track, destination);
  return true;
}

/**
 * @brief Loads a track's local field approximation and evaluates it at z.
 * @param source Packed input with optional field coefficients.
 * @param track Packed track index; an unavailable track/field selects a zero-coefficient region.
 * @param z Evaluation coordinate in the field polynomial's coordinate system.
 * @return Evaluated field components; there is no error-status return or finite-value validation.
 * @note Evaluates the fallback polynomial as well, so invalid z can still produce nonfinite values.
 */
KFPARTICLE_GPU_HOST_DEVICE inline KFParticleGpuFieldValue EvaluateTrackField(
  const KFParticleGpuConstInputTrackSoAView& source,
  unsigned int track,
  float z)
{
  KFParticleGpuFieldRegion field;
  LoadFieldRegionOrZero(source, track, field);
  return field.Get(z);
}

/**
 * @brief Evaluates the selected track's field approximation at state.Z().
 * @note Other state parameters are ignored; fallback and validation behavior match EvaluateTrackField().
 */
KFPARTICLE_GPU_HOST_DEVICE inline KFParticleGpuFieldValue EvaluateTrackFieldAtState(
  const KFParticleGpuConstInputTrackSoAView& source,
  unsigned int track,
  const KFParticleGpuFitState& state)
{
  return EvaluateTrackField(source, track, state.Z());
}

/**
 * @brief Copies ten field-region coefficients into one existing packed track slot.
 * @param source Local field approximation to copy without validation.
 * @param[in,out] destination Writable input view with a nonnull field array of adequate extent.
 * @param track Valid destination track index.
 * @note Does not allocate field storage or modify other track data; accesses are unchecked.
 */
KFPARTICLE_GPU_HOST_DEVICE inline void StoreFieldRegion(const KFParticleGpuFieldRegion& source,
                                                        KFParticleGpuInputTrackSoAView& destination,
                                                        unsigned int track)
{
  for (unsigned int component = 0; component < KFParticleGpuFieldRegion::NumberOfCoefficients; ++component) {
    destination.FieldCoefficient(component, track) = source.Coefficient(component);
  }
}

/**
 * @brief Copies one packed PV into an owned numerical vertex state.
 * @param source Readable vertex arrays in the calling execution context.
 * @param vertex Valid index below source.Size().
 * @param[out] destination Receives position, packed covariance, chi2, NDF, and contributor count.
 * @note Does not validate bounds, covariance, or fit quality and does not attach event identity.
 */
KFPARTICLE_GPU_HOST_DEVICE inline void LoadVertexState(const KFParticleGpuConstVertexSoAView& source,
                                                       unsigned int vertex,
                                                       KFParticleGpuVertexState& destination)
{
  for (unsigned int component = 0; component < KFParticleGpuVertexState::NumberOfParameters; ++component) {
    destination.Parameter(component) = source.Parameter(component, vertex);
  }
  for (unsigned int component = 0;
       component < KFParticleGpuVertexState::NumberOfCovarianceElements;
       ++component) {
    destination.Covariance(component) = source.Covariance(component, vertex);
  }
  destination.Chi2() = source.Chi2(vertex);
  destination.NDF() = source.NDF(vertex);
  destination.NContributors() = source.NContributors(vertex);
}

/**
 * @brief Copies position, covariance, and all numerical vertex fit counters into an existing packed slot.
 * @param source Local numerical vertex state.
 * @param[in,out] destination Writable vertex arrays with valid backing storage.
 * @param vertex Valid destination index below destination.Size().
 * @note Performs no bounds/value checks, allocation, logical-size update, or host/device transfer.
 */
KFPARTICLE_GPU_HOST_DEVICE inline void StoreVertexState(const KFParticleGpuVertexState& source,
                                                        KFParticleGpuVertexSoAView& destination,
                                                        unsigned int vertex)
{
  for (unsigned int component = 0; component < KFParticleGpuVertexState::NumberOfParameters; ++component) {
    destination.Parameter(component, vertex) = source.Parameter(component);
  }
  for (unsigned int component = 0;
       component < KFParticleGpuVertexState::NumberOfCovarianceElements;
       ++component) {
    destination.Covariance(component, vertex) = source.Covariance(component);
  }
  destination.Chi2(vertex) = source.Chi2();
  destination.NDF(vertex) = source.NDF();
  destination.NContributors(vertex) = source.NContributors();
}

#endif
