/*
 * This file is part of KFParticle package
 * Copyright (C) 2007-2026
 *
 * KFParticle is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 */

#ifndef KFPARTICLEGPUTWODAUGHTERROUTING_H
#define KFPARTICLEGPUTWODAUGHTERROUTING_H

#include "KFParticleGpuChannelRouting.h"

#include <type_traits>

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

enum KFParticleGpuTwoDaughterRoutingMode : unsigned int
{
  KFGpuTwoDaughterRoutingAtomic = 0u,
  KFGpuTwoDaughterRoutingBlockScan = 1u
};

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

KFPARTICLE_GPU_HOST_DEVICE inline int KFParticleGpuTrackRoleCharge(unsigned int role)
{
  if (role == KFGpuTrackRolePiPlus || role == KFGpuTrackRoleProton
      || role == KFGpuTrackRoleKPlus) {
    return 1;
  }
  if (role == KFGpuTrackRolePiMinus || role == KFGpuTrackRoleAntiProton
      || role == KFGpuTrackRoleKMinus) {
    return -1;
  }
  return 0;
}

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

/** Complete immutable two-track channel description referenced by a routing bit. */
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

struct KFParticleGpuTwoDaughterCompatibilityEntry
{
  int firstPdg = 0;
  int secondPdg = 0;
  unsigned int firstRole = KFGpuTrackRoleInvalid;
  unsigned int secondRole = KFGpuTrackRoleInvalid;
  KFParticleGpuChannelMask channels;
};

/** Plan-level source geometry shared by compatible two-track hypotheses. */
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

  KFPARTICLE_GPU_HOST_DEVICE bool AcceptsFirstRole(unsigned int role) const
  {
    return (firstRoles & KFParticleGpuRoutingRoleBit(role)) != 0u;
  }

  KFPARTICLE_GPU_HOST_DEVICE bool AcceptsSecondRole(unsigned int role) const
  {
    return (secondRoles & KFParticleGpuRoutingRoleBit(role)) != 0u;
  }

  KFPARTICLE_GPU_HOST_DEVICE bool AcceptsFirstSpecies(
    KFParticleGpuTrackSpecies species) const
  {
    const unsigned int value = static_cast<unsigned int>(species);
    return value < 32u && (firstSpecies & (1u << value)) != 0u;
  }

  KFPARTICLE_GPU_HOST_DEVICE bool AcceptsSecondSpecies(
    KFParticleGpuTrackSpecies species) const
  {
    const unsigned int value = static_cast<unsigned int>(species);
    return value < 32u && (secondSpecies & (1u << value)) != 0u;
  }
};

/** Minimal generation task; all numerical constants remain in the descriptor table. */
struct KFParticleGpuTwoDaughterRoutedTask
{
  unsigned int descriptorIndex = 0u;
  unsigned int firstTrackIndex = 0u;
  unsigned int secondTrackIndex = 0u;
  unsigned int eventIndex = 0u;
};

struct KFParticleGpuTwoDaughterRoutingStatus
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

/** Candidate-indexed transient routing tag used for O(1) selection lookup. */
template<typename UnsignedValue>
class KFParticleGpuCandidateDescriptorIndexViewBase
{
 public:
  KFPARTICLE_GPU_HOST_DEVICE KFParticleGpuCandidateDescriptorIndexViewBase() = default;

  KFPARTICLE_GPU_HOST_DEVICE KFParticleGpuCandidateDescriptorIndexViewBase(
    UnsignedValue* indices,
    unsigned int capacity)
    : fIndices(indices), fCapacity(capacity)
  {
  }

  KFPARTICLE_GPU_HOST_DEVICE UnsignedValue& Index(unsigned int candidateIndex) const
  {
    return fIndices[candidateIndex];
  }
  KFPARTICLE_GPU_HOST_DEVICE UnsignedValue* Data() const { return fIndices; }
  KFPARTICLE_GPU_HOST_DEVICE unsigned int Capacity() const { return fCapacity; }
  KFPARTICLE_GPU_HOST_DEVICE bool CanStore(unsigned int candidateIndex) const
  {
    return candidateIndex < fCapacity;
  }

 private:
  UnsignedValue* fIndices;
  unsigned int fCapacity;
};

typedef KFParticleGpuCandidateDescriptorIndexViewBase<unsigned int>
  KFParticleGpuCandidateDescriptorIndexView;
typedef KFParticleGpuCandidateDescriptorIndexViewBase<const unsigned int>
  KFParticleGpuConstCandidateDescriptorIndexView;

KFPARTICLE_GPU_HOST_DEVICE inline KFParticleGpuConstCandidateDescriptorIndexView
MakeConstView(const KFParticleGpuCandidateDescriptorIndexView& view)
{
  return KFParticleGpuConstCandidateDescriptorIndexView(view.Data(), view.Capacity());
}

/** Flat non-owning two-daughter routing state published to XPU kernels. */
class KFParticleGpuTwoDaughterRoutingView
{
 public:
  KFPARTICLE_GPU_HOST_DEVICE KFParticleGpuTwoDaughterRoutingView() = default;

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

  KFPARTICLE_GPU_HOST_DEVICE const KFParticleGpuTwoDaughterRoutingDescriptor*
  Descriptors() const
  {
    return fDescriptors;
  }
  KFPARTICLE_GPU_HOST_DEVICE unsigned int DescriptorCount() const { return fDescriptorCount; }
  KFPARTICLE_GPU_HOST_DEVICE const KFParticleGpuTwoDaughterCompatibilityEntry*
  CompatibilityEntries() const
  {
    return fCompatibility;
  }
  KFPARTICLE_GPU_HOST_DEVICE unsigned int CompatibilityCount() const
  {
    return fCompatibilityCount;
  }
  KFPARTICLE_GPU_HOST_DEVICE const KFParticleGpuTwoDaughterExecutionGroup* Groups() const
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
  KFPARTICLE_GPU_HOST_DEVICE unsigned int* ChannelStoredCounters() const
  {
    return fChannelStoredCounters;
  }
  KFPARTICLE_GPU_HOST_DEVICE unsigned int* ChannelConstructedCounters() const
  {
    return fChannelConstructedCounters;
  }

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

  KFPARTICLE_GPU_HOST_DEVICE KFParticleGpuChannelMask CompatibleChannelsByPdg(
    int firstPdg,
    int secondPdg) const
  {
    const KFParticleGpuChannelMask result = KnownChannelsByPdg(firstPdg, secondPdg);
    return fEnabledChannels ? result.Intersected(fEnabledChannels[0]) : result;
  }

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

/** Per-descriptor workspace for generation-wide selected-output compaction. */
class KFParticleGpuTwoDaughterSelectionWorkspaceView
{
 public:
  KFPARTICLE_GPU_HOST_DEVICE KFParticleGpuTwoDaughterSelectionWorkspaceView() = default;

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

  KFPARTICLE_GPU_HOST_DEVICE unsigned int Capacity() const { return fCapacity; }
  KFPARTICLE_GPU_HOST_DEVICE unsigned int* AcceptedData() const { return fAccepted; }
  KFPARTICLE_GPU_HOST_DEVICE unsigned int* StoredData() const { return fStored; }
  KFPARTICLE_GPU_HOST_DEVICE unsigned int* OffsetsData() const { return fOffsets; }
  KFPARTICLE_GPU_HOST_DEVICE unsigned int* CursorsData() const { return fCursors; }
  KFPARTICLE_GPU_HOST_DEVICE unsigned int& Accepted(unsigned int index) const
  {
    return fAccepted[index];
  }
  KFPARTICLE_GPU_HOST_DEVICE unsigned int& Stored(unsigned int index) const
  {
    return fStored[index];
  }
  KFPARTICLE_GPU_HOST_DEVICE unsigned int& Offset(unsigned int index) const
  {
    return fOffsets[index];
  }
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

/** Published non-owning work buffers for one two-daughter generation. */
class KFParticleGpuTwoDaughterGenerationStorageView
{
 public:
  KFPARTICLE_GPU_HOST_DEVICE
  KFParticleGpuTwoDaughterGenerationStorageView() = default;

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

  KFPARTICLE_GPU_HOST_DEVICE KFParticleGpuTwoDaughterRoutedTask* Tasks() const
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
