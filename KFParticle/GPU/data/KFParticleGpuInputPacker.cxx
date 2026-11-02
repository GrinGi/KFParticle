/* Copyright (C) 2026 GSI Helmholtzzentrum fuer Schwerionenforschung, Darmstadt
   SPDX-License-Identifier: GPL-3.0-only */

#include "data/KFParticleGpuInputPacker.h"

#include "KFVertex.h"
#include "KFPTrackVector.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace
{
  using kfparticle::gpu::GpuEventInput;

  constexpr unsigned int kUnsupportedSpecies = NumberOfTrackSpecies;
  constexpr unsigned int kSpeciesBuckets = NumberOfTrackSpecies + 1u;
  constexpr unsigned int kTrackBuckets = NumberOfTrackSets * kSpeciesBuckets;

  struct SourceTrack {
    unsigned int index = 0u;
    bool atLastPoint = false;
  };

  unsigned int ToGpuSpecies(int pdg)
  {
    const int absPdg = std::abs(pdg);
    switch (absPdg) {
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
      default: return kUnsupportedSpecies;
    }
  }

  KFParticleGpuTrackSet ToTrackSet(bool atLastPoint,
                                   int charge,
                                   int primaryVertexIndex)
  {
    const unsigned int pointOffset = atLastPoint
      ? SecondaryPositiveLast : SecondaryPositiveFirst;
    const unsigned int signOffset = charge < 0 ? 1u : 0u;
    const unsigned int primaryOffset = primaryVertexIndex >= 0 ? 2u : 0u;
    return static_cast<KFParticleGpuTrackSet>(
      pointOffset + primaryOffset + signOffset);
  }

  constexpr unsigned int TrackBucket(KFParticleGpuTrackSet set,
                                     unsigned int species)
  {
    return static_cast<unsigned int>(set) * kSpeciesBuckets + species;
  }

  void RequireFinite(float value, const char* what)
  {
    if (!std::isfinite(value)) { throw std::invalid_argument(what); }
  }

  void ValidateTrackMetadata(const KFPTrackVector& tracks, unsigned int index)
  {
    if (tracks.Id()[index] < 0) { throw std::invalid_argument("KFParticle GPU packer received a negative source ID"); }
    if (tracks.Q()[index] == 0) { throw std::invalid_argument("KFParticle GPU packer received a zero track charge"); }
  }
}  // namespace

namespace kfparticle::gpu
{
  struct GpuPreparedEvent::Impl {
    GpuEventInput input;
    KFParticleGpuEventDesc descriptor;
    std::vector<SourceTrack> packingOrder;
    std::vector<int> sourceIds;
    bool validatedPublicationInput = false;
  };

  GpuPreparedEvent::GpuPreparedEvent(std::unique_ptr<Impl> impl) : fImpl(std::move(impl)) {}

  GpuPreparedEvent::GpuPreparedEvent(GpuPreparedEvent&&) noexcept = default;

  GpuPreparedEvent& GpuPreparedEvent::operator=(GpuPreparedEvent&&) noexcept = default;

  GpuPreparedEvent::~GpuPreparedEvent() = default;

  unsigned int GpuPreparedEvent::TrackCount() const
  {
    return static_cast<unsigned int>(fImpl->packingOrder.size());
  }

  unsigned int GpuPreparedEvent::VertexCount() const { return fImpl->descriptor.primaryVertices.size; }

  const KFParticleGpuEventDesc& GpuPreparedEvent::Descriptor() const { return fImpl->descriptor; }

  bool GpuPreparedEvent::HasValidatedPublicationInput() const
  {
    return fImpl->validatedPublicationInput;
  }

  GpuPreparedEvent GpuInputPacker::Prepare(const GpuEventInput& input)
  {
    if (input.first == nullptr || input.last == nullptr || input.chiToPrimaryVertex == nullptr
        || input.primaryVertices == nullptr) {
      throw std::invalid_argument("KFParticle GPU packer requires first/last tracks, chi-to-PV, and vertices");
    }

    const int sourceTrackCount = input.first->Size();
    if (sourceTrackCount < 0 || input.last->Size() != sourceTrackCount
        || input.chiToPrimaryVertex->size() != static_cast<std::size_t>(sourceTrackCount)) {
      throw std::invalid_argument("KFParticle GPU packer received inconsistent track-vector sizes");
    }
    if (static_cast<unsigned long long>(sourceTrackCount) * 2ull
        > std::numeric_limits<unsigned int>::max()) {
      throw std::overflow_error("KFParticle GPU packer track count exceeds the GPU ABI");
    }
    RequireFinite(input.minSecondaryTrackChiToPrimaryVertex,
                  "KFParticle GPU packer received a non-finite minimum secondary-track chi-to-PV");

    auto impl                 = std::make_unique<GpuPreparedEvent::Impl>();
    impl->input               = input;
    impl->descriptor.eventId  = input.eventId;
    impl->descriptor.minSecondaryTrackChiToPrimaryVertex =
      input.minSecondaryTrackChiToPrimaryVertex;
    if (input.primaryVertices->size() > std::numeric_limits<unsigned int>::max()) {
      throw std::overflow_error("KFParticle GPU packer vertex count exceeds the GPU ABI");
    }
    impl->descriptor.primaryVertices = KFParticleGpuRange(0u, static_cast<unsigned int>(input.primaryVertices->size()));

    impl->sourceIds.resize(static_cast<std::size_t>(sourceTrackCount));
    std::array<unsigned int, kTrackBuckets> bucketCounts{};
    for (unsigned int sourceIndex = 0; sourceIndex < static_cast<unsigned int>(sourceTrackCount); ++sourceIndex) {
      ValidateTrackMetadata(*input.first, sourceIndex);
      ValidateTrackMetadata(*input.last, sourceIndex);
      const int sourceId = input.first->Id()[sourceIndex];
      if (input.last->Id()[sourceIndex] != sourceId) {
        throw std::invalid_argument(
          "KFParticle GPU packer received mismatched first/last source IDs");
      }
      impl->sourceIds[sourceIndex] = sourceId;

      const int charge = input.first->Q()[sourceIndex];
      const int primaryVertexIndex = input.first->PVIndex()[sourceIndex];
      const unsigned int species = ToGpuSpecies(input.first->PDG()[sourceIndex]);
      ++bucketCounts[TrackBucket(ToTrackSet(false, charge, primaryVertexIndex), species)];
      ++bucketCounts[TrackBucket(ToTrackSet(true, charge, primaryVertexIndex), species)];
    }

    auto sortedSourceIds = impl->sourceIds;
    std::sort(sortedSourceIds.begin(), sortedSourceIds.end());
    if (std::adjacent_find(sortedSourceIds.begin(), sortedSourceIds.end())
        != sortedSourceIds.end()) {
      throw std::invalid_argument(
        "KFParticle GPU packer received a duplicate physical source ID");
    }

    std::array<unsigned int, kTrackBuckets> bucketOffsets{};
    unsigned int offset = 0;
    for (unsigned int set = 0; set < NumberOfTrackSets; ++set) {
      auto& setDesc = impl->descriptor.trackSets[set];
      const unsigned int setOffset = offset;
      for (unsigned int species = 0; species < NumberOfTrackSpecies; ++species) {
        const unsigned int bucket = TrackBucket(
          static_cast<KFParticleGpuTrackSet>(set), species);
        bucketOffsets[bucket] = offset;
        setDesc.species[species] = KFParticleGpuRange(
          offset, bucketCounts[bucket]);
        offset += bucketCounts[bucket];
      }
      const unsigned int unsupportedBucket = TrackBucket(
        static_cast<KFParticleGpuTrackSet>(set), kUnsupportedSpecies);
      bucketOffsets[unsupportedBucket] = offset;
      offset += bucketCounts[unsupportedBucket];
      setDesc.tracks = KFParticleGpuRange(setOffset, offset - setOffset);
    }
    const unsigned int expectedTrackCount =
      2u * static_cast<unsigned int>(sourceTrackCount);
    if (offset != expectedTrackCount) {
      throw std::logic_error("KFParticle GPU packer track count is inconsistent");
    }

    impl->packingOrder.resize(offset);
    auto bucketCursors = bucketOffsets;
    for (unsigned int sourceIndex = 0;
         sourceIndex < static_cast<unsigned int>(sourceTrackCount);
         ++sourceIndex) {
      const int charge = input.first->Q()[sourceIndex];
      const int primaryVertexIndex = input.first->PVIndex()[sourceIndex];
      const unsigned int species = ToGpuSpecies(input.first->PDG()[sourceIndex]);
      const unsigned int firstBucket = TrackBucket(
        ToTrackSet(false, charge, primaryVertexIndex), species);
      const unsigned int lastBucket = TrackBucket(
        ToTrackSet(true, charge, primaryVertexIndex), species);
      impl->packingOrder[bucketCursors[firstBucket]++] = {sourceIndex, false};
      impl->packingOrder[bucketCursors[lastBucket]++] = {sourceIndex, true};
    }
    impl->validatedPublicationInput = true;

    return GpuPreparedEvent(std::move(impl));
  }

  KFParticleGpuBufferCapacities GpuInputPacker::InputCapacities(const GpuPreparedEvent& prepared)
  {
    KFParticleGpuBufferCapacities capacities;
    capacities.tracks              = prepared.TrackCount();
    capacities.vertices            = prepared.VertexCount();
    capacities.events              = 1;
    capacities.nonhomogeneousField = true;
    return capacities;
  }

  void GpuInputPacker::Write(const GpuPreparedEvent& prepared,
                                  const KFParticleGpuInputTrackSoAView& tracks,
                                  const KFParticleGpuVertexSoAView& vertices,
                                  KFParticleGpuEventDesc& event)
  {
    Write(prepared, 0u, 0u, tracks, vertices, event);
  }

  void GpuInputPacker::Write(const GpuPreparedEvent& prepared,
                                  unsigned int trackOffset,
                                  unsigned int vertexOffset,
                                  const KFParticleGpuInputTrackSoAView& tracks,
                                  const KFParticleGpuVertexSoAView& vertices,
                                  KFParticleGpuEventDesc& event)
  {
    if (trackOffset > tracks.Size() || prepared.TrackCount() > tracks.Size() - trackOffset
        || vertexOffset > vertices.Size() || prepared.VertexCount() > vertices.Size() - vertexOffset) {
      throw std::invalid_argument("KFParticle GPU packer received views without the requested event range");
    }
    if (prepared.TrackCount() > 0u && tracks.FieldCoefficientsData() == nullptr) {
      throw std::logic_error("KFParticle GPU packer requires allocated nonhomogeneous-field storage");
    }
    if (prepared.TrackCount() > 0u
        && (tracks.Numerical().ParametersData() == nullptr || tracks.Numerical().CovariancesData() == nullptr
            || tracks.ChiToPrimaryVertexData() == nullptr || tracks.IntegersData() == nullptr)) {
      throw std::logic_error("KFParticle GPU packer received an incomplete track host view");
    }
    if (prepared.VertexCount() > 0u
        && (vertices.ParametersData() == nullptr || vertices.CovariancesData() == nullptr
            || vertices.Chi2Data() == nullptr || vertices.IntegersData() == nullptr)) {
      throw std::logic_error("KFParticle GPU packer received an incomplete primary-vertex host view");
    }

    const auto& input = prepared.fImpl->input;
    for (unsigned int packedIndex = 0u;
         packedIndex < prepared.TrackCount(); ++packedIndex) {
      const SourceTrack sourceReference =
        prepared.fImpl->packingOrder[packedIndex];
      const KFPTrackVector& source = sourceReference.atLastPoint
        ? *input.last : *input.first;
      const unsigned int sourceIndex = sourceReference.index;
      const unsigned int outputIndex = trackOffset + packedIndex;
      int packedPrimaryVertexIndex = -1;
      if (!KFParticleGpuPackPrimaryVertexIndex(
            source.PVIndex()[sourceIndex],
            KFParticleGpuRange(vertexOffset, prepared.VertexCount()),
            packedPrimaryVertexIndex)) {
        throw std::invalid_argument(
          "KFParticle GPU packer received an out-of-range primary-vertex association");
      }

      for (unsigned int component = 0; component < 6u; ++component) {
        const float value = source.Parameter(component)[sourceIndex];
        RequireFinite(value, "KFParticle GPU packer received a non-finite track parameter");
        tracks.Numerical().Parameter(component, outputIndex) = value;
      }
      for (unsigned int component = 0; component < 21u; ++component) {
        const float value = source.Covariance(component)[sourceIndex];
        RequireFinite(value, "KFParticle GPU packer received a non-finite track covariance");
        tracks.Numerical().Covariance(component, outputIndex) = value;
      }
#ifdef NonhomogeneousField
      for (unsigned int component = 0; component < 10u; ++component) {
        const float value = source.FieldCoefficient(component)[sourceIndex];
        RequireFinite(value, "KFParticle GPU packer received a non-finite field coefficient");
        tracks.FieldCoefficient(component, outputIndex) = value;
      }
#else
      throw std::logic_error("KFParticle GPU diagnostic packing requires NonhomogeneousField");
#endif
      const float chi = (*input.chiToPrimaryVertex)[sourceIndex];
      RequireFinite(chi, "KFParticle GPU packer received a non-finite chi-to-PV");
      tracks.ChiToPrimaryVertex(outputIndex) = chi;
      tracks.SourceId(outputIndex) = prepared.fImpl->sourceIds[sourceIndex];
      tracks.Pdg(outputIndex) = source.PDG()[sourceIndex];
      tracks.Charge(outputIndex) = source.Q()[sourceIndex];
      tracks.PrimaryVertexIndex(outputIndex) = packedPrimaryVertexIndex;
      tracks.NumberOfPixelHits(outputIndex) = source.NPixelHits()[sourceIndex];
    }

    for (unsigned int vertexIndex = 0; vertexIndex < prepared.VertexCount(); ++vertexIndex) {
      const auto& source = (*input.primaryVertices)[vertexIndex];
      const unsigned int outputIndex = vertexOffset + vertexIndex;
      for (unsigned int component = 0; component < 3u; ++component) {
        const float value = source.GetParameter(component);
        RequireFinite(value, "KFParticle GPU packer received a non-finite primary vertex");
        vertices.Parameter(component, outputIndex) = value;
      }
      for (unsigned int component = 0; component < 6u; ++component) {
        const float value = source.GetCovariance(component);
        RequireFinite(value, "KFParticle GPU packer received a non-finite primary-vertex covariance");
        vertices.Covariance(component, outputIndex) = value;
      }
      const float chi2 = source.GetChi2();
      RequireFinite(chi2, "KFParticle GPU packer received a non-finite primary-vertex chi2");
      vertices.Chi2(outputIndex) = chi2;
      vertices.NDF(outputIndex) = source.GetNDF();
      vertices.NContributors(outputIndex) = source.GetNContributors();
    }

    event = prepared.Descriptor();
    for (unsigned int set = 0u; set < NumberOfTrackSets; ++set) {
      auto& trackSet = event.trackSets[set];
      trackSet.tracks.offset += trackOffset;
      for (unsigned int species = 0u; species < NumberOfTrackSpecies; ++species) {
        trackSet.species[species].offset += trackOffset;
      }
    }
    event.primaryVertices.offset += vertexOffset;
  }
}  // namespace kfparticle::gpu
