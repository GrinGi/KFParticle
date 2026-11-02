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
 * @brief Two-track channel lookup, compact routed tasks, and generation workspace views.
 *
 * Immutable plan tables are separate from mutable routing/selection counters.
 * Views borrow storage and perform no allocation or synchronization. Accessors
 * require valid pointers and in-range indices unless an explicit check is provided.
 */
#ifndef KFPARTICLEGPUTWODAUGHTERROUTING_H
#define KFPARTICLEGPUTWODAUGHTERROUTING_H

#include "planning/KFParticleGpuChannelRouting.h"

#include <type_traits>

/** @brief Compact charge/species roles for pion, proton, and kaon compatibility lookup. */
enum KFParticleGpuTrackRole : unsigned int
{
  KFGpuTrackRoleInvalid = 0u,
  KFGpuTrackRolePiPlus = 1u,
  KFGpuTrackRoleProton = 2u,
  KFGpuTrackRolePiMinus = 3u,
  KFGpuTrackRoleAntiProton = 4u,
  KFGpuTrackRoleKPlus = 5u,
  KFGpuTrackRoleKMinus = 6u
};

/** @brief Task reservation strategy: individual atomics or block-wide scan/compaction. */
enum KFParticleGpuTwoDaughterRoutingMode : unsigned int
{
  KFGpuTwoDaughterRoutingAtomic = 0u,
  KFGpuTwoDaughterRoutingBlockScan = 1u
};

/** @brief Maps supported signed PDG codes to compact roles; other species return Invalid. */
KFPARTICLE_GPU_HOST_DEVICE inline KFParticleGpuTrackRole
KFParticleGpuClassifyTrackRole(int pdg)
{
  if (pdg == 211) {
    return KFGpuTrackRolePiPlus;
  }
  if (pdg == 2212) {
    return KFGpuTrackRoleProton;
  }
  if (pdg == -211) {
    return KFGpuTrackRolePiMinus;
  }
  if (pdg == -2212) {
    return KFGpuTrackRoleAntiProton;
  }
  if (pdg == 321) {
    return KFGpuTrackRoleKPlus;
  }
  if (pdg == -321) {
    return KFGpuTrackRoleKMinus;
  }
  return KFGpuTrackRoleInvalid;
}

/** @brief Removes charge from a supported role; returns NumberOfTrackSpecies for unknown roles. */
KFPARTICLE_GPU_HOST_DEVICE inline KFParticleGpuTrackSpecies
KFParticleGpuTrackRoleSpecies(unsigned int role)
{
  if (role == KFGpuTrackRolePiPlus || role == KFGpuTrackRolePiMinus) {
    return Pion;
  }
  if (role == KFGpuTrackRoleProton || role == KFGpuTrackRoleAntiProton) {
    return Proton;
  }
  if (role == KFGpuTrackRoleKPlus || role == KFGpuTrackRoleKMinus) {
    return Kaon;
  }
  return NumberOfTrackSpecies;
}

/**
 * @brief Immutable two-track channel hypothesis and cuts addressed by a routing bit.
 *
 * Track sets/species/roles select inputs. Source PDGs constrain packed identities;
 * daughter masses/PDGs define the fitted hypothesis. selection holds later V0 cuts.
 * Negative optional floating thresholds disable cuts; zero PDG/charge leaves that
 * identity unrestricted where the corresponding track-cut helper is used.
 */
struct KFParticleGpuTwoDaughterRoutingDescriptor
{
  unsigned int channelBit = KFParticleGpuChannelMask::InvalidBit;
  unsigned int channelId = 0u;
  unsigned int firstRole = KFGpuTrackRoleInvalid;
  unsigned int secondRole = KFGpuTrackRoleInvalid;
  KFParticleGpuTrackSet firstTrackSet = SecondaryPositiveFirst;
  KFParticleGpuTrackSet secondTrackSet = SecondaryNegativeFirst;
  KFParticleGpuTrackSpecies firstSpecies = Pion;
  KFParticleGpuTrackSpecies secondSpecies = Pion;
  unsigned int flags = KFGpuTwoDaughterUseLineDca | KFGpuTwoDaughterUseEnergyFit;
  unsigned int outputClass = KFGpuGraphOutputPrimaryAndSecondary;
  int motherPdg = 0;
  int firstDaughterPdg = 0;
  int secondDaughterPdg = 0;
  int firstSourcePdg = 0;
  int firstAlternateSourcePdg = 0;
  int secondSourcePdg = 0;
  int secondAlternateSourcePdg = 0;
  int primaryVertexIndex = -1;
  float firstMass = 0.f;
  float secondMass = 0.f;
  float motherMass = 0.f;
  float motherMassSigma = 0.f;
  float secondaryMassSigmaCut = -1.f;
  float maxSecondaryTopoChi2PerNdf = -1.f;
  float minSecondaryLdL = -1.f;
  KFParticleGpuV0SelectionConfig selection;
  int transportMode = KFGpuTransportStraightLine;
  int firstCharge = 0;
  int secondCharge = 0;
  int minFirstPixelHits = 0;
  int minSecondPixelHits = 0;
  float maxFirstChiToPrimaryVertex = -1.f;
  float maxSecondChiToPrimaryVertex = -1.f;
  float minFirstChiToPrimaryVertex = -1.f;
  float minSecondChiToPrimaryVertex = -1.f;
  float minFirstPt = -1.f;
  float minSecondPt = -1.f;
  float maxDaughterDistance = -1.f;
};

/** @brief Maps an ordered source-PDG/role pair to the channels accepting that combination. */
struct KFParticleGpuTwoDaughterCompatibilityEntry
{
  int firstPdg = 0;
  int secondPdg = 0;
  unsigned int firstRole = KFGpuTrackRoleInvalid;
  unsigned int secondRole = KFGpuTrackRoleInvalid;
  KFParticleGpuChannelMask channels;
};

/**
 * @brief Source track-set geometry and role/species masks shared by channel hypotheses.
 * @note firstRoles/secondRoles and firstSpecies/secondSpecies are bit sets, not scalar IDs.
 */
struct KFParticleGpuTwoDaughterExecutionGroup
{
  unsigned int groupIndex = 0u;
  KFParticleGpuTrackSet firstTrackSet = SecondaryPositiveFirst;
  KFParticleGpuTrackSet secondTrackSet = SecondaryNegativeFirst;
  unsigned int firstRoles = 0u;
  unsigned int secondRoles = 0u;
  unsigned int firstSpecies = 0u;
  unsigned int secondSpecies = 0u;
  KFParticleGpuChannelMask channels;

  /** @brief Tests membership in the first-daughter role mask. */
  KFPARTICLE_GPU_HOST_DEVICE bool AcceptsFirstRole(unsigned int role) const
  {
    return (firstRoles & KFParticleGpuRoutingRoleBit(role)) != 0u;
  }

  /** @brief Tests membership in the second-daughter role mask. */
  KFPARTICLE_GPU_HOST_DEVICE bool AcceptsSecondRole(unsigned int role) const
  {
    return (secondRoles & KFParticleGpuRoutingRoleBit(role)) != 0u;
  }

  /** @brief Tests the first-species mask, rejecting values that cannot index a 32-bit mask. */
  KFPARTICLE_GPU_HOST_DEVICE bool AcceptsFirstSpecies(
    KFParticleGpuTrackSpecies species) const
  {
    const unsigned int value = static_cast<unsigned int>(species);
    return value < 32u && (firstSpecies & (1u << value)) != 0u;
  }

};

/**
 * @brief Compact task linking two absolute track indices to an immutable descriptor.
 *
 * Event and domain indices retain batch ownership. Numerical fit/cut constants
 * remain in the descriptor table and are not copied into every routed task.
 */
struct KFParticleGpuTwoDaughterRoutedTask
{
  unsigned int descriptorIndex = 0u;
  unsigned int firstTrackIndex = 0u;
  unsigned int secondTrackIndex = 0u;
  unsigned int eventIndex = 0u;
  unsigned int domainIndex = 0u;
};

/** @brief Routing counters distinguishing accepted hypotheses from capacity-bounded stored tasks. */
struct KFParticleGpuTwoDaughterRoutingStatus
{
  unsigned int visitedPairs = 0u;
  unsigned int activeChannelBits = 0u;
  unsigned int acceptedTasks = 0u;
  unsigned int storedTasks = 0u;
  unsigned int blockReservations = 0u;
  unsigned int overflowFlags = 0u;

  /** @brief Reports accepted tasks that could not be retained; does not inspect other overflow bits. */
  KFPARTICLE_GPU_HOST_DEVICE bool Truncated() const
  {
    return acceptedTasks > storedTasks;
  }
};

/**
 * @brief Non-owning candidate-to-descriptor index array for constant-time selection lookup.
 * @tparam UnsignedValue unsigned int for writable storage, const unsigned int for read-only storage.
 * @note Default initialization leaves members unbound; bind before access. A const
 * view object does not make writable underlying elements read-only.
 */
template<typename UnsignedValue>
class KFParticleGpuCandidateDescriptorIndexViewBase
{
 public:
  /** @brief Preserves trivial default construction without binding an index array. */
  KFPARTICLE_GPU_HOST_DEVICE KFParticleGpuCandidateDescriptorIndexViewBase() = default;

  /** @brief Borrows an index array and its capacity, without initializing elements. */
  KFPARTICLE_GPU_HOST_DEVICE KFParticleGpuCandidateDescriptorIndexViewBase(
    UnsignedValue* indices,
    unsigned int capacity)
    : fIndices(indices), fCapacity(capacity)
  {
  }

  /** @brief Returns the descriptor tag at a valid candidate index, without checking bounds. */
  KFPARTICLE_GPU_HOST_DEVICE UnsignedValue& Index(unsigned int candidateIndex) const
  {
    return fIndices[candidateIndex];
  }
  /** @brief Returns the borrowed tag-array pointer. */
  KFPARTICLE_GPU_HOST_DEVICE UnsignedValue* Data() const { return fIndices; }

  /** @brief Returns the number of addressable candidate tags. */
  KFPARTICLE_GPU_HOST_DEVICE unsigned int Capacity() const { return fCapacity; }

  /** @brief Checks only index capacity, not pointer validity or element mutability. */
  KFPARTICLE_GPU_HOST_DEVICE bool CanStore(unsigned int candidateIndex) const
  {
    return candidateIndex < fCapacity;
  }

 private:
  UnsignedValue* fIndices;
  unsigned int fCapacity;
};

/** @brief Writable candidate-to-descriptor tags. */
typedef KFParticleGpuCandidateDescriptorIndexViewBase<unsigned int>
  KFParticleGpuCandidateDescriptorIndexView;
/** @brief Read-only candidate-to-descriptor tags over the same storage layout. */
typedef KFParticleGpuCandidateDescriptorIndexViewBase<const unsigned int>
  KFParticleGpuConstCandidateDescriptorIndexView;

/** @brief Creates a read-only alias without copying tags or extending storage lifetime. */
KFPARTICLE_GPU_HOST_DEVICE inline KFParticleGpuConstCandidateDescriptorIndexView
MakeConstView(const KFParticleGpuCandidateDescriptorIndexView& view)
{
  return KFParticleGpuConstCandidateDescriptorIndexView(view.Data(), view.Capacity());
}

/**
 * @brief Non-owning routing tables, optional enabled-channel mask, and diagnostic counters.
 *
 * Compatibility lookups union all matching entries; role and PDG arguments are
 * ordered. A null enabled mask means no channel restriction. Lookup does not
 * update counters. Default-initialized views must be bound before access.
 */
class KFParticleGpuTwoDaughterRoutingView
{
 public:
  /** @brief Preserves trivial default construction without binding tables. */
  KFPARTICLE_GPU_HOST_DEVICE KFParticleGpuTwoDaughterRoutingView() = default;

  /** @brief Binds plan arrays and mutable counter arrays without copying or resetting them. */
  KFPARTICLE_GPU_HOST_DEVICE KFParticleGpuTwoDaughterRoutingView(
    const KFParticleGpuTwoDaughterRoutingDescriptor* descriptors,
    unsigned int descriptorCount,
    const KFParticleGpuTwoDaughterCompatibilityEntry* compatibility,
    unsigned int compatibilityCount,
    const KFParticleGpuTwoDaughterExecutionGroup* groups,
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

  /** @brief Returns the immutable descriptor array. */
  KFPARTICLE_GPU_HOST_DEVICE const KFParticleGpuTwoDaughterRoutingDescriptor*
  Descriptors() const
  {
    return fDescriptors;
  }
  /** @brief Returns the number of descriptors. */
  KFPARTICLE_GPU_HOST_DEVICE unsigned int DescriptorCount() const { return fDescriptorCount; }

  /** @brief Returns ordered role/PDG compatibility entries. */
  KFPARTICLE_GPU_HOST_DEVICE const KFParticleGpuTwoDaughterCompatibilityEntry*
  CompatibilityEntries() const
  {
    return fCompatibility;
  }
  /** @brief Returns the number of compatibility entries. */
  KFPARTICLE_GPU_HOST_DEVICE unsigned int CompatibilityCount() const
  {
    return fCompatibilityCount;
  }
  /** @brief Returns immutable execution groups sharing source geometry. */
  KFPARTICLE_GPU_HOST_DEVICE const KFParticleGpuTwoDaughterExecutionGroup* Groups() const
  {
    return fGroups;
  }
  /** @brief Returns the number of execution groups. */
  KFPARTICLE_GPU_HOST_DEVICE unsigned int GroupCount() const { return fGroupCount; }

  /** @brief Returns the optional single enabled-channel mask pointer. */
  KFPARTICLE_GPU_HOST_DEVICE const KFParticleGpuChannelMask* EnabledChannelsData() const
  {
    return fEnabledChannels;
  }
  /** @brief Returns mutable per-channel visited-combination counters. */
  KFPARTICLE_GPU_HOST_DEVICE unsigned int* ChannelVisitedCounters() const
  {
    return fChannelVisitedCounters;
  }
  /** @brief Returns mutable per-channel accepted-task counters before capacity limiting. */
  KFPARTICLE_GPU_HOST_DEVICE unsigned int* ChannelAcceptedCounters() const
  {
    return fChannelAcceptedCounters;
  }
  /** @brief Returns mutable per-channel retained-task counters. */
  KFPARTICLE_GPU_HOST_DEVICE unsigned int* ChannelStoredCounters() const
  {
    return fChannelStoredCounters;
  }
  /** @brief Returns mutable per-channel constructed-candidate counters. */
  KFPARTICLE_GPU_HOST_DEVICE unsigned int* ChannelConstructedCounters() const
  {
    return fChannelConstructedCounters;
  }

  /** @brief Unions channels matching the ordered role pair and applies the optional enabled mask. */
  KFPARTICLE_GPU_HOST_DEVICE KFParticleGpuChannelMask CompatibleChannels(
    unsigned int firstRole,
    unsigned int secondRole) const
  {
    KFParticleGpuChannelMask result;
    for (unsigned int index = 0u; index < fCompatibilityCount; ++index) {
      const KFParticleGpuTwoDaughterCompatibilityEntry& entry = fCompatibility[index];
      if (entry.firstRole == firstRole && entry.secondRole == secondRole) {
        result.UnionWith(entry.channels);
      }
    }
    return fEnabledChannels ? result.Intersected(fEnabledChannels[0]) : result;
  }

  /** @brief Returns known channels for an ordered PDG pair, restricted to enabled channels. */
  KFPARTICLE_GPU_HOST_DEVICE KFParticleGpuChannelMask CompatibleChannelsByPdg(
    int firstPdg,
    int secondPdg) const
  {
    const KFParticleGpuChannelMask result = KnownChannelsByPdg(firstPdg, secondPdg);
    return fEnabledChannels ? result.Intersected(fEnabledChannels[0]) : result;
  }

  /** @brief Unions channels matching an ordered PDG pair without applying the enabled mask. */
  KFPARTICLE_GPU_HOST_DEVICE KFParticleGpuChannelMask KnownChannelsByPdg(
    int firstPdg,
    int secondPdg) const
  {
    KFParticleGpuChannelMask result;
    for (unsigned int index = 0u; index < fCompatibilityCount; ++index) {
      const KFParticleGpuTwoDaughterCompatibilityEntry& entry = fCompatibility[index];
      if (entry.firstPdg == firstPdg && entry.secondPdg == secondPdg) {
        result.UnionWith(entry.channels);
      }
    }
    return result;
  }

 private:
  const KFParticleGpuTwoDaughterRoutingDescriptor* fDescriptors;
  unsigned int fDescriptorCount;
  const KFParticleGpuTwoDaughterCompatibilityEntry* fCompatibility;
  unsigned int fCompatibilityCount;
  const KFParticleGpuTwoDaughterExecutionGroup* fGroups;
  unsigned int fGroupCount;
  const KFParticleGpuChannelMask* fEnabledChannels;
  unsigned int* fChannelVisitedCounters;
  unsigned int* fChannelAcceptedCounters;
  unsigned int* fChannelStoredCounters;
  unsigned int* fChannelConstructedCounters;
};

/**
 * @brief Non-owning per-descriptor counts, segment offsets, and scatter cursors for selection.
 *
 * Accepted counts precede capacity limiting; stored counts bound reserved output
 * segments. All arrays have Capacity() entries. Index access is unchecked and
 * mutable even through a const view. Bind a default-initialized view before use.
 */
class KFParticleGpuTwoDaughterSelectionWorkspaceView
{
 public:
  /** @brief Preserves trivial default construction without binding workspace arrays. */
  KFPARTICLE_GPU_HOST_DEVICE KFParticleGpuTwoDaughterSelectionWorkspaceView() = default;

  /** @brief Borrows four counter/offset arrays of the supplied common capacity. */
  KFPARTICLE_GPU_HOST_DEVICE KFParticleGpuTwoDaughterSelectionWorkspaceView(
    unsigned int* accepted,
    unsigned int* stored,
    unsigned int* offsets,
    unsigned int* cursors,
    unsigned int capacity)
    : fAccepted(accepted)
    , fStored(stored)
    , fOffsets(offsets)
    , fCursors(cursors)
    , fCapacity(capacity)
  {
  }

  /** @brief Returns the number of descriptor slots in each workspace array. */
  KFPARTICLE_GPU_HOST_DEVICE unsigned int Capacity() const { return fCapacity; }

  /** @brief Returns accepted-count storage. */
  KFPARTICLE_GPU_HOST_DEVICE unsigned int* AcceptedData() const { return fAccepted; }

  /** @brief Returns capacity-bounded stored-count storage. */
  KFPARTICLE_GPU_HOST_DEVICE unsigned int* StoredData() const { return fStored; }

  /** @brief Returns selected-output segment offsets. */
  KFPARTICLE_GPU_HOST_DEVICE unsigned int* OffsetsData() const { return fOffsets; }

  /** @brief Returns mutable scatter cursors. */
  KFPARTICLE_GPU_HOST_DEVICE unsigned int* CursorsData() const { return fCursors; }

  /** @brief Accesses an accepted count at a valid descriptor index. */
  KFPARTICLE_GPU_HOST_DEVICE unsigned int& Accepted(unsigned int index) const
  {
    return fAccepted[index];
  }
  /** @brief Accesses a stored count at a valid descriptor index. */
  KFPARTICLE_GPU_HOST_DEVICE unsigned int& Stored(unsigned int index) const
  {
    return fStored[index];
  }
  /** @brief Accesses a segment offset at a valid descriptor index. */
  KFPARTICLE_GPU_HOST_DEVICE unsigned int& Offset(unsigned int index) const
  {
    return fOffsets[index];
  }
  /** @brief Accesses a scatter cursor at a valid descriptor index. */
  KFPARTICLE_GPU_HOST_DEVICE unsigned int& Cursor(unsigned int index) const
  {
    return fCursors[index];
  }

 private:
  unsigned int* fAccepted;
  unsigned int* fStored;
  unsigned int* fOffsets;
  unsigned int* fCursors;
  unsigned int fCapacity;
};

/**
 * @brief Non-owning retained tasks, scalar counters, and selection workspace for one generation.
 *
 * Counter pointers refer to caller-managed scalar storage. Access does not reset
 * counters, reserve tasks, or supply atomic synchronization. Bind before use;
 * default construction preserves the trivial device ABI without initializing members.
 */
class KFParticleGpuTwoDaughterGenerationStorageView
{
 public:
  /** @brief Preserves trivial default construction without binding generation storage. */
  KFPARTICLE_GPU_HOST_DEVICE
  KFParticleGpuTwoDaughterGenerationStorageView() = default;

  /** @brief Binds task storage, scalar counter pointers, and a borrowed selection workspace. */
  KFPARTICLE_GPU_HOST_DEVICE KFParticleGpuTwoDaughterGenerationStorageView(
    KFParticleGpuTwoDaughterRoutedTask* tasks,
    unsigned int taskCapacity,
    unsigned int* visitedPairs,
    unsigned int* activeChannelBits,
    unsigned int* acceptedTasks,
    unsigned int* storedTasks,
    unsigned int* blockReservations,
    unsigned int* overflowFlags,
    const KFParticleGpuTwoDaughterSelectionWorkspaceView& selection)
    : fTasks(tasks)
    , fTaskCapacity(taskCapacity)
    , fVisitedPairs(visitedPairs)
    , fActiveChannelBits(activeChannelBits)
    , fAcceptedTasks(acceptedTasks)
    , fStoredTasks(storedTasks)
    , fBlockReservations(blockReservations)
    , fOverflowFlags(overflowFlags)
    , fSelection(selection)
  {
  }

  /** @brief Returns the mutable retained-task array. */
  KFPARTICLE_GPU_HOST_DEVICE KFParticleGpuTwoDaughterRoutedTask* Tasks() const
  {
    return fTasks;
  }
  /** @brief Returns retained-task capacity, not the current task count. */
  KFPARTICLE_GPU_HOST_DEVICE unsigned int TaskCapacity() const
  {
    return fTaskCapacity;
  }
  /** @brief Returns the visited-pair counter pointer. */
  KFPARTICLE_GPU_HOST_DEVICE unsigned int* VisitedPairs() const
  {
    return fVisitedPairs;
  }
  /** @brief Returns the active-channel-bit counter pointer. */
  KFPARTICLE_GPU_HOST_DEVICE unsigned int* ActiveChannelBits() const
  {
    return fActiveChannelBits;
  }
  /** @brief Returns the accepted-task counter before capacity limiting. */
  KFPARTICLE_GPU_HOST_DEVICE unsigned int* AcceptedTasks() const
  {
    return fAcceptedTasks;
  }
  /** @brief Returns the retained-task counter. */
  KFPARTICLE_GPU_HOST_DEVICE unsigned int* StoredTasks() const
  {
    return fStoredTasks;
  }
  /** @brief Returns the block-level task reservation counter. */
  KFPARTICLE_GPU_HOST_DEVICE unsigned int* BlockReservations() const
  {
    return fBlockReservations;
  }
  /** @brief Returns accumulated capacity-overflow flags. */
  KFPARTICLE_GPU_HOST_DEVICE unsigned int* OverflowFlags() const
  {
    return fOverflowFlags;
  }
  /** @brief Returns the borrowed selection workspace view; underlying arrays remain mutable. */
  KFPARTICLE_GPU_HOST_DEVICE const KFParticleGpuTwoDaughterSelectionWorkspaceView&
  Selection() const
  {
    return fSelection;
  }

 private:
  KFParticleGpuTwoDaughterRoutedTask* fTasks;
  unsigned int fTaskCapacity;
  unsigned int* fVisitedPairs;
  unsigned int* fActiveChannelBits;
  unsigned int* fAcceptedTasks;
  unsigned int* fStoredTasks;
  unsigned int* fBlockReservations;
  unsigned int* fOverflowFlags;
  KFParticleGpuTwoDaughterSelectionWorkspaceView fSelection;
};

static_assert(std::is_trivially_copyable<KFParticleGpuTwoDaughterRoutingDescriptor>::value,
              "KFParticle GPU two-daughter descriptors must remain flat device values");
static_assert(std::is_trivially_copyable<KFParticleGpuTwoDaughterCompatibilityEntry>::value,
              "KFParticle GPU two-daughter compatibility entries must remain flat values");
static_assert(std::is_trivially_copyable<KFParticleGpuTwoDaughterExecutionGroup>::value,
              "KFParticle GPU two-daughter execution groups must remain flat values");
static_assert(std::is_trivially_copyable<KFParticleGpuTwoDaughterRoutedTask>::value,
              "KFParticle GPU two-daughter routed tasks must remain flat values");
static_assert(std::is_trivially_copyable<KFParticleGpuTwoDaughterRoutingStatus>::value,
              "KFParticle GPU two-daughter routing status must remain a flat value");
static_assert(std::is_trivially_copyable<KFParticleGpuCandidateDescriptorIndexView>::value,
              "KFParticle GPU candidate routing tags must be non-owning views");
static_assert(std::is_trivially_copyable<KFParticleGpuTwoDaughterRoutingView>::value,
              "KFParticle GPU two-daughter routing views must not own memory");
static_assert(
  std::is_trivially_copyable<KFParticleGpuTwoDaughterSelectionWorkspaceView>::value,
  "KFParticle GPU two-daughter selection workspace must be a non-owning view");
static_assert(
  std::is_trivially_copyable<KFParticleGpuTwoDaughterGenerationStorageView>::value,
  "KFParticle GPU two-daughter generation storage must be a non-owning view");
static_assert(
  std::is_trivially_default_constructible<
    KFParticleGpuTwoDaughterGenerationStorageView>::value,
  "KFParticle GPU two-daughter generation storage must remain a flat ABI");

#endif
