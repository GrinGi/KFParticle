/*
 * This file is part of KFParticle package
 * Copyright (C) 2007-2026
 *
 * KFParticle is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 */

#ifndef KFPARTICLEGPUCHANNELROUTING_H
#define KFPARTICLEGPUCHANNELROUTING_H

#include "KFParticleGpuInputData.h"
#include "KFParticleGpuPlatform.h"
#include "KFParticleGpuV0Track.h"

#include <type_traits>

/**
 * Bounded channel-bit set used only for device-side hypothesis routing.
 *
 * Stable channel IDs remain in descriptors and candidates. Nine words cover
 * the complete audited CPU catalogue while keeping routing bounded and free of
 * host-side branches per decay hypothesis.
 */
struct KFParticleGpuChannelMask
{
  static constexpr unsigned int WordBits = 32u;
  static constexpr unsigned int WordCount = 9u;
  static constexpr unsigned int BitCapacity = WordBits * WordCount;
  static constexpr unsigned int InvalidBit = BitCapacity;

  unsigned int words[WordCount] = {};

  KFPARTICLE_GPU_HOST_DEVICE void Clear()
  {
    for (unsigned int word = 0u; word < WordCount; ++word) {
      words[word] = 0u;
    }
  }

  KFPARTICLE_GPU_HOST_DEVICE bool Empty() const
  {
    unsigned int value = 0u;
    for (unsigned int word = 0u; word < WordCount; ++word) {
      value |= words[word];
    }
    return value == 0u;
  }

  KFPARTICLE_GPU_HOST_DEVICE bool Set(unsigned int bit)
  {
    if (bit >= BitCapacity) {
      return false;
    }
    words[bit / WordBits] |= 1u << (bit % WordBits);
    return true;
  }

  KFPARTICLE_GPU_HOST_DEVICE bool Reset(unsigned int bit)
  {
    if (bit >= BitCapacity) {
      return false;
    }
    words[bit / WordBits] &= ~(1u << (bit % WordBits));
    return true;
  }

  KFPARTICLE_GPU_HOST_DEVICE bool Test(unsigned int bit) const
  {
    return bit < BitCapacity
           && (words[bit / WordBits] & (1u << (bit % WordBits))) != 0u;
  }

  KFPARTICLE_GPU_HOST_DEVICE void UnionWith(const KFParticleGpuChannelMask& other)
  {
    for (unsigned int word = 0u; word < WordCount; ++word) {
      words[word] |= other.words[word];
    }
  }

  KFPARTICLE_GPU_HOST_DEVICE KFParticleGpuChannelMask
  Intersected(const KFParticleGpuChannelMask& other) const
  {
    KFParticleGpuChannelMask result;
    for (unsigned int word = 0u; word < WordCount; ++word) {
      result.words[word] = words[word] & other.words[word];
    }
    return result;
  }

  KFPARTICLE_GPU_HOST_DEVICE unsigned int NextSetBit(unsigned int first) const
  {
    for (unsigned int bit = first; bit < BitCapacity; ++bit) {
      if (Test(bit)) {
        return bit;
      }
    }
    return InvalidBit;
  }

  KFPARTICLE_GPU_HOST_DEVICE unsigned int Count() const
  {
    unsigned int count = 0u;
    for (unsigned int bit = NextSetBit(0u);
         bit != InvalidBit;
         bit = NextSetBit(bit + 1u)) {
      ++count;
    }
    return count;
  }

  KFPARTICLE_GPU_HOST_DEVICE bool operator==(const KFParticleGpuChannelMask& other) const
  {
    for (unsigned int word = 0u; word < WordCount; ++word) {
      if (words[word] != other.words[word]) {
        return false;
      }
    }
    return true;
  }
};

enum KFParticleGpuSelectedV0Role : unsigned int
{
  KFGpuSelectedV0RoleInvalid = 0u,
  KFGpuSelectedV0RoleLambda = 1u,
  KFGpuSelectedV0RoleAntiLambda = 2u
};

enum KFParticleGpuBachelorRole : unsigned int
{
  KFGpuBachelorRoleInvalid = 0u,
  KFGpuBachelorRolePiMinus = 1u,
  KFGpuBachelorRolePiPlus = 2u,
  KFGpuBachelorRoleKMinus = 3u,
  KFGpuBachelorRoleKPlus = 4u
};

enum KFParticleGpuV0TrackRoutingMode : unsigned int
{
  KFGpuV0TrackRoutingAtomic = 0u,
  KFGpuV0TrackRoutingBlockScan = 1u
};

KFPARTICLE_GPU_HOST_DEVICE inline unsigned int KFParticleGpuRoutingRoleBit(unsigned int role)
{
  return role < 32u ? 1u << role : 0u;
}

KFPARTICLE_GPU_HOST_DEVICE inline KFParticleGpuSelectedV0Role
KFParticleGpuClassifySelectedV0Role(int pdg)
{
  if (pdg == 3122) {
    return KFGpuSelectedV0RoleLambda;
  }
  if (pdg == -3122) {
    return KFGpuSelectedV0RoleAntiLambda;
  }
  return KFGpuSelectedV0RoleInvalid;
}

KFPARTICLE_GPU_HOST_DEVICE inline KFParticleGpuBachelorRole
KFParticleGpuClassifyBachelorRole(int pdg)
{
  if (pdg == -211) {
    return KFGpuBachelorRolePiMinus;
  }
  if (pdg == 211) {
    return KFGpuBachelorRolePiPlus;
  }
  if (pdg == -321) {
    return KFGpuBachelorRoleKMinus;
  }
  if (pdg == 321) {
    return KFGpuBachelorRoleKPlus;
  }
  return KFGpuBachelorRoleInvalid;
}

/** Complete device descriptor referenced by a transient routing bit. */
struct KFParticleGpuV0TrackRoutingDescriptor
{
  unsigned int channelBit = KFParticleGpuChannelMask::InvalidBit;
  unsigned int channelId = 0u;
  unsigned int parentChannelId = 0u;
  unsigned int family = KFGpuCpuFamilyTrackComposite;
  unsigned int generation = 2u;
  unsigned int outputClass = KFGpuGraphOutputSecondary;
  unsigned int selectedV0Role = KFGpuSelectedV0RoleInvalid;
  unsigned int bachelorRole = KFGpuBachelorRoleInvalid;
  KFParticleGpuTrackSet bachelorTrackSet = SecondaryNegativeFirst;
  KFParticleGpuTrackSpecies bachelorSpecies = Pion;
  int v0Pdg = 0;
  int bachelorPdg = 0;
  int motherPdg = 0;
  int primaryVertexIndex = -1;
  unsigned int flags = KFGpuV0TrackUseLineDca | KFGpuV0TrackUseEnergyFit;
  int transportMode = KFGpuTransportFullField;
  float bachelorMass = 0.f;
  float motherMass = 0.f;
  float motherMassSigma = -1.f;
  float secondaryMassSigmaCut = -1.f;
  float maxSecondaryTopoChi2PerNdf = -1.f;
  float minBachelorChiToPrimaryVertex = -1.f;
  float maxV0TrackDistance = -1.f;
  float minBachelorPt = -1.f;
  int minBachelorPixelHits = -1;
  float parentMassConstraint = -1.f;
  float parentMassConstraintSigma = 0.f;
};

/** One compatibility-table cell can activate several physical hypotheses. */
struct KFParticleGpuV0TrackCompatibilityEntry
{
  unsigned int parentChannelId = 0u;
  int selectedPdg = 0;
  int bachelorPdg = 0;
  unsigned int selectedV0Role = KFGpuSelectedV0RoleInvalid;
  unsigned int bachelorRole = KFGpuBachelorRoleInvalid;
  KFParticleGpuChannelMask channels;
};

/**
 * Plan-level group with common event-local source geometry.
 *
 * Actual ranges are resolved from KFParticleGpuEventDesc immediately before a
 * Stage 15.2 launch; no event data is stored in this persistent table.
 */
struct KFParticleGpuV0TrackExecutionGroup
{
  unsigned int groupIndex = 0u;
  unsigned int generation = 2u;
  KFParticleGpuTrackSet bachelorTrackSet = SecondaryNegativeFirst;
  unsigned int selectedV0Roles = 0u;
  unsigned int bachelorRoles = 0u;
  unsigned int bachelorSpecies = 0u;
  KFParticleGpuChannelMask channels;

  KFPARTICLE_GPU_HOST_DEVICE bool AcceptsSelectedV0Role(unsigned int role) const
  {
    return (selectedV0Roles & KFParticleGpuRoutingRoleBit(role)) != 0u;
  }

  KFPARTICLE_GPU_HOST_DEVICE bool AcceptsBachelorRole(unsigned int role) const
  {
    return (bachelorRoles & KFParticleGpuRoutingRoleBit(role)) != 0u;
  }

  KFPARTICLE_GPU_HOST_DEVICE bool AcceptsBachelorSpecies(
    KFParticleGpuTrackSpecies species) const
  {
    const unsigned int value = static_cast<unsigned int>(species);
    return value < 32u && (bachelorSpecies & (1u << value)) != 0u;
  }
};

/** Minimal routed work item; the descriptor owns all physical constants. */
struct KFParticleGpuV0TrackRoutedTask
{
  unsigned int descriptorIndex = 0u;
  unsigned int selectedV0Index = 0u;
  unsigned int bachelorTrackIndex = 0u;
  unsigned int eventIndex = 0u;
};

/** Scalar diagnostics for one bounded fused-routing transaction. */
struct KFParticleGpuV0TrackRoutingStatus
{
  unsigned int visitedPairs = 0u;
  unsigned int activeChannelBits = 0u;
  unsigned int acceptedTasks = 0u;
  unsigned int storedTasks = 0u;
  unsigned int blockReservations = 0u;
  unsigned int overflowFlags = 0u;

  KFPARTICLE_GPU_HOST_DEVICE bool Truncated() const
  {
    return acceptedTasks > storedTasks;
  }
};

/** Flat non-owning routing state suitable for XPU constant memory. */
class KFParticleGpuV0TrackRoutingView
{
 public:
  KFPARTICLE_GPU_HOST_DEVICE KFParticleGpuV0TrackRoutingView() = default;

  KFPARTICLE_GPU_HOST_DEVICE KFParticleGpuV0TrackRoutingView(
    const KFParticleGpuV0TrackRoutingDescriptor* descriptors,
    unsigned int descriptorCount,
    const KFParticleGpuV0TrackCompatibilityEntry* compatibility,
    unsigned int compatibilityCount,
    const KFParticleGpuV0TrackExecutionGroup* groups,
    unsigned int groupCount,
    const KFParticleGpuChannelMask* enabledChannels,
    unsigned int* channelVisitedCounters,
    unsigned int* channelAcceptedCounters,
    unsigned int* channelStoredCounters,
    unsigned int* channelConstructedCounters)
    : fDescriptors(descriptors)
    , fDescriptorCount(descriptorCount)
    , fCompatibility(compatibility)
    , fCompatibilityCount(compatibilityCount)
    , fGroups(groups)
    , fGroupCount(groupCount)
    , fEnabledChannels(enabledChannels)
    , fChannelVisitedCounters(channelVisitedCounters)
    , fChannelAcceptedCounters(channelAcceptedCounters)
    , fChannelStoredCounters(channelStoredCounters)
    , fChannelConstructedCounters(channelConstructedCounters)
  {
  }

  KFPARTICLE_GPU_HOST_DEVICE const KFParticleGpuV0TrackRoutingDescriptor*
  Descriptors() const
  {
    return fDescriptors;
  }
  KFPARTICLE_GPU_HOST_DEVICE unsigned int DescriptorCount() const { return fDescriptorCount; }
  KFPARTICLE_GPU_HOST_DEVICE const KFParticleGpuV0TrackCompatibilityEntry*
  CompatibilityEntries() const
  {
    return fCompatibility;
  }
  KFPARTICLE_GPU_HOST_DEVICE unsigned int CompatibilityCount() const
  {
    return fCompatibilityCount;
  }
  KFPARTICLE_GPU_HOST_DEVICE const KFParticleGpuV0TrackExecutionGroup* Groups() const
  {
    return fGroups;
  }
  KFPARTICLE_GPU_HOST_DEVICE unsigned int GroupCount() const { return fGroupCount; }
  KFPARTICLE_GPU_HOST_DEVICE const KFParticleGpuChannelMask* EnabledChannelsData() const
  {
    return fEnabledChannels;
  }
  KFPARTICLE_GPU_HOST_DEVICE unsigned int* ChannelVisitedCounters() const
  {
    return fChannelVisitedCounters;
  }
  KFPARTICLE_GPU_HOST_DEVICE unsigned int* ChannelAcceptedCounters() const
  {
    return fChannelAcceptedCounters;
  }
  // Retain the Stage 15.1 accessor name as the accepted-task counter.
  KFPARTICLE_GPU_HOST_DEVICE unsigned int* ChannelCounters() const
  {
    return ChannelAcceptedCounters();
  }
  KFPARTICLE_GPU_HOST_DEVICE unsigned int* ChannelStoredCounters() const
  {
    return fChannelStoredCounters;
  }
  KFPARTICLE_GPU_HOST_DEVICE unsigned int* ChannelConstructedCounters() const
  {
    return fChannelConstructedCounters;
  }

  KFPARTICLE_GPU_HOST_DEVICE KFParticleGpuChannelMask CompatibleChannels(
    unsigned int selectedV0Role,
    unsigned int bachelorRole) const
  {
    KFParticleGpuChannelMask result;
    for (unsigned int index = 0u; index < fCompatibilityCount; ++index) {
      const KFParticleGpuV0TrackCompatibilityEntry& entry = fCompatibility[index];
      if (entry.selectedV0Role == selectedV0Role && entry.bachelorRole == bachelorRole) {
        result.UnionWith(entry.channels);
      }
    }
    return fEnabledChannels ? result.Intersected(fEnabledChannels[0]) : result;
  }

  KFPARTICLE_GPU_HOST_DEVICE KFParticleGpuChannelMask CompatibleChannelsByPdg(
    unsigned int parentChannelId,
    int selectedPdg,
    int bachelorPdg) const
  {
    const KFParticleGpuChannelMask result =
      KnownChannelsByPdg(parentChannelId, selectedPdg, bachelorPdg);
    return fEnabledChannels ? result.Intersected(fEnabledChannels[0]) : result;
  }

  KFPARTICLE_GPU_HOST_DEVICE KFParticleGpuChannelMask KnownChannelsByPdg(
    unsigned int parentChannelId,
    int selectedPdg,
    int bachelorPdg) const
  {
    KFParticleGpuChannelMask result;
    for (unsigned int index = 0u; index < fCompatibilityCount; ++index) {
      const KFParticleGpuV0TrackCompatibilityEntry& entry = fCompatibility[index];
      if ((entry.parentChannelId == 0u || entry.parentChannelId == parentChannelId)
          && entry.selectedPdg == selectedPdg
          && entry.bachelorPdg == bachelorPdg) {
        result.UnionWith(entry.channels);
      }
    }
    return result;
  }

 private:
  const KFParticleGpuV0TrackRoutingDescriptor* fDescriptors;
  unsigned int fDescriptorCount;
  const KFParticleGpuV0TrackCompatibilityEntry* fCompatibility;
  unsigned int fCompatibilityCount;
  const KFParticleGpuV0TrackExecutionGroup* fGroups;
  unsigned int fGroupCount;
  const KFParticleGpuChannelMask* fEnabledChannels;
  unsigned int* fChannelVisitedCounters;
  unsigned int* fChannelAcceptedCounters;
  unsigned int* fChannelStoredCounters;
  unsigned int* fChannelConstructedCounters;
};

/** Published non-owning work buffers for one composite-track generation. */
class KFParticleGpuV0TrackGenerationStorageView
{
 public:
  KFPARTICLE_GPU_HOST_DEVICE KFParticleGpuV0TrackGenerationStorageView() = default;

  KFPARTICLE_GPU_HOST_DEVICE KFParticleGpuV0TrackGenerationStorageView(
    KFParticleGpuV0TrackRoutedTask* tasks,
    unsigned int taskCapacity,
    unsigned int* visitedPairs,
    unsigned int* activeChannelBits,
    unsigned int* acceptedTasks,
    unsigned int* storedTasks,
    unsigned int* blockReservations,
    unsigned int* overflowFlags)
    : fTasks(tasks)
    , fTaskCapacity(taskCapacity)
    , fVisitedPairs(visitedPairs)
    , fActiveChannelBits(activeChannelBits)
    , fAcceptedTasks(acceptedTasks)
    , fStoredTasks(storedTasks)
    , fBlockReservations(blockReservations)
    , fOverflowFlags(overflowFlags)
  {
  }

  KFPARTICLE_GPU_HOST_DEVICE KFParticleGpuV0TrackRoutedTask* Tasks() const
  {
    return fTasks;
  }
  KFPARTICLE_GPU_HOST_DEVICE unsigned int TaskCapacity() const
  {
    return fTaskCapacity;
  }
  KFPARTICLE_GPU_HOST_DEVICE unsigned int* VisitedPairs() const
  {
    return fVisitedPairs;
  }
  KFPARTICLE_GPU_HOST_DEVICE unsigned int* ActiveChannelBits() const
  {
    return fActiveChannelBits;
  }
  KFPARTICLE_GPU_HOST_DEVICE unsigned int* AcceptedTasks() const
  {
    return fAcceptedTasks;
  }
  KFPARTICLE_GPU_HOST_DEVICE unsigned int* StoredTasks() const
  {
    return fStoredTasks;
  }
  KFPARTICLE_GPU_HOST_DEVICE unsigned int* BlockReservations() const
  {
    return fBlockReservations;
  }
  KFPARTICLE_GPU_HOST_DEVICE unsigned int* OverflowFlags() const
  {
    return fOverflowFlags;
  }

 private:
  KFParticleGpuV0TrackRoutedTask* fTasks;
  unsigned int fTaskCapacity;
  unsigned int* fVisitedPairs;
  unsigned int* fActiveChannelBits;
  unsigned int* fAcceptedTasks;
  unsigned int* fStoredTasks;
  unsigned int* fBlockReservations;
  unsigned int* fOverflowFlags;
};

static_assert(sizeof(unsigned int) == 4u, "KFParticle GPU channel masks require 32-bit words");
static_assert(std::is_trivially_copyable<KFParticleGpuChannelMask>::value,
              "KFParticle GPU channel masks must remain flat device values");
static_assert(std::is_trivially_copyable<KFParticleGpuV0TrackRoutingDescriptor>::value,
              "KFParticle GPU routing descriptors must remain flat device values");
static_assert(std::is_trivially_copyable<KFParticleGpuV0TrackCompatibilityEntry>::value,
              "KFParticle GPU compatibility entries must remain flat device values");
static_assert(std::is_trivially_copyable<KFParticleGpuV0TrackExecutionGroup>::value,
              "KFParticle GPU execution groups must remain flat device values");
static_assert(std::is_trivially_copyable<KFParticleGpuV0TrackRoutedTask>::value,
              "KFParticle GPU routed tasks must remain flat device values");
static_assert(std::is_trivially_copyable<KFParticleGpuV0TrackRoutingStatus>::value,
              "KFParticle GPU routing status must remain a flat device value");
static_assert(std::is_trivially_copyable<KFParticleGpuV0TrackRoutingView>::value,
              "KFParticle GPU routing views must not own device memory");
static_assert(
  std::is_trivially_copyable<KFParticleGpuV0TrackGenerationStorageView>::value,
  "KFParticle GPU composite-track generation storage must be a non-owning view");
static_assert(
  std::is_trivially_default_constructible<
    KFParticleGpuV0TrackGenerationStorageView>::value,
  "KFParticle GPU composite-track generation storage must remain a flat ABI");

#endif
