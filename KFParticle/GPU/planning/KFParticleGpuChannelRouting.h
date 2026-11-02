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
 * @brief Channel masks and routing descriptors for selected composites paired with bachelor tracks.
 *
 * The historical V0 names also cover later composite-plus-track generations.
 * Plan tables describe hypotheses; routed tasks carry indices rather than
 * particle states. Views borrow host/device storage and perform no allocation,
 * data transfer, kernel launch, or synchronization.
 */

#ifndef KFPARTICLEGPUCHANNELROUTING_H
#define KFPARTICLEGPUCHANNELROUTING_H

#include "data/KFParticleGpuInputDataTransfer.h"
#include "KFParticleGpuPlatform.h"
#include "algorithms/KFParticleGpuV0Track.h"

#include <type_traits>

/**
 * @brief Fixed 288-bit set of transient channel positions for host/device hypothesis routing.
 *
 * Bits refer to positions in a compiled routing plan, not stable channel IDs;
 * IDs remain in descriptors and candidates. Nine 32-bit words make the mask a
 * flat, trivially copyable value with no dynamic memory. All words start at zero.
 *
 * @note Mutating operations are not atomic. Each writer must own its mask or
 * be externally synchronized. Masks from different plans need not share bit meanings.
 */
struct KFParticleGpuChannelMask
{
  static constexpr unsigned int WordBits = 32u;                      ///< Bits per unsigned storage word.
  static constexpr unsigned int WordCount = 9u;                      ///< Fixed storage words per mask.
  static constexpr unsigned int BitCapacity = WordBits * WordCount;  ///< Exclusive upper bound for valid bit indices.
  static constexpr unsigned int InvalidBit = BitCapacity;            ///< Sentinel returned when no set bit is found.

  unsigned int words[WordCount] = {};  ///< Inline zero-initialized bit storage.

  /// @brief Clears every channel bit in place.
  KFPARTICLE_GPU_HOST_DEVICE void Clear()
  {
    for (unsigned int word = 0u; word < WordCount; ++word) {
      words[word] = 0u;
    }
  }

  /// @brief Returns true when no channel bit is set.
  KFPARTICLE_GPU_HOST_DEVICE bool Empty() const
  {
    unsigned int value = 0u;
    for (unsigned int word = 0u; word < WordCount; ++word) {
      value |= words[word];
    }
    return value == 0u;
  }

  /// @brief Sets @p bit; returns false without changing the mask when the index is out of range.
  KFPARTICLE_GPU_HOST_DEVICE bool Set(unsigned int bit)
  {
    if (bit >= BitCapacity) {
      return false;
    }
    words[bit / WordBits] |= 1u << (bit % WordBits);
    return true;
  }

  /// @brief Clears @p bit; returns false without changing the mask when the index is out of range.
  KFPARTICLE_GPU_HOST_DEVICE bool Reset(unsigned int bit)
  {
    if (bit >= BitCapacity) {
      return false;
    }
    words[bit / WordBits] &= ~(1u << (bit % WordBits));
    return true;
  }

  /// @brief Tests @p bit; an out-of-range index returns false.
  KFPARTICLE_GPU_HOST_DEVICE bool Test(unsigned int bit) const
  {
    return bit < BitCapacity
           && (words[bit / WordBits] & (1u << (bit % WordBits))) != 0u;
  }

  /// @brief Adds every bit from @p other to this mask; both masks must use the same plan's bit mapping.
  KFPARTICLE_GPU_HOST_DEVICE void UnionWith(const KFParticleGpuChannelMask& other)
  {
    for (unsigned int word = 0u; word < WordCount; ++word) {
      words[word] |= other.words[word];
    }
  }

  /// @brief Returns common bits without modifying either mask; both masks must use the same bit mapping.
  KFPARTICLE_GPU_HOST_DEVICE KFParticleGpuChannelMask
  Intersected(const KFParticleGpuChannelMask& other) const
  {
    KFParticleGpuChannelMask result;
    for (unsigned int word = 0u; word < WordCount; ++word) {
      result.words[word] = words[word] & other.words[word];
    }
    return result;
  }

  /// @brief Finds the first set bit at or after @p first, or InvalidBit if none exists within capacity.
  KFPARTICLE_GPU_HOST_DEVICE unsigned int NextSetBit(unsigned int first) const
  {
    for (unsigned int bit = first; bit < BitCapacity; ++bit) {
      if (Test(bit)) {
        return bit;
      }
    }
    return InvalidBit;
  }

  /// @brief Returns the number of set channel bits.
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

  /// @brief Compares raw mask words; does not establish that the masks belong to the same routing plan.
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

/** @brief Coarse roles for Lambda routing; other composites use PDG-based matching. */
enum KFParticleGpuSelectedV0Role : unsigned int
{
  KFGpuSelectedV0RoleInvalid = 0u,    ///< PDG is not represented by this coarse role set.
  KFGpuSelectedV0RoleLambda = 1u,     ///< Lambda hypothesis, PDG 3122.
  KFGpuSelectedV0RoleAntiLambda = 2u  ///< Anti-Lambda hypothesis, PDG -3122.
};

/** @brief Coarse charged pion/kaon roles for a track paired with a selected composite. */
enum KFParticleGpuBachelorRole : unsigned int
{
  KFGpuBachelorRoleInvalid = 0u,  ///< PDG is not represented by this coarse role set.
  KFGpuBachelorRolePiMinus = 1u,  ///< Negative pion hypothesis, PDG -211.
  KFGpuBachelorRolePiPlus = 2u,   ///< Positive pion hypothesis, PDG 211.
  KFGpuBachelorRoleKMinus = 3u,   ///< Negative kaon hypothesis, PDG -321.
  KFGpuBachelorRoleKPlus = 4u     ///< Positive kaon hypothesis, PDG 321.
};

/** @brief Task-slot reservation strategies for routed composite-plus-track generation. */
enum KFParticleGpuV0TrackRoutingMode : unsigned int
{
  KFGpuV0TrackRoutingAtomic = 0u,    ///< Reserve a slot atomically for each accepted task.
  KFGpuV0TrackRoutingBlockScan = 1u  ///< Combine block-local counts before reserving task slots.
};

/**
 * @brief Converts a role number below 32 to a single-bit mask; returns zero otherwise.
 * @note This only checks the shift range, not role validity; role zero produces bit zero.
 */
KFPARTICLE_GPU_HOST_DEVICE inline unsigned int KFParticleGpuRoutingRoleBit(unsigned int role)
{
  return role < 32u ? 1u << role : 0u;
}

/// @brief Classifies PDG 3122/-3122 as Lambda/anti-Lambda; all others return the invalid role.
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

/// @brief Classifies signed pion/kaon PDGs; all others return the invalid bachelor role.
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

/**
 * @brief Flat compiled hypothesis for building a selected composite plus a bachelor track.
 *
 * Contains stable channel identity, input requirements, fit settings, and cuts.
 * A routed task references this descriptor instead of repeating its constants.
 * The routing plan assigns channelBit and validates the descriptor; default
 * field values alone do not constitute a ready-to-run physical hypothesis.
 */
struct KFParticleGpuV0TrackRoutingDescriptor
{
  unsigned int channelBit = KFParticleGpuChannelMask::InvalidBit;  ///< Position in this plan's channel mask.
  unsigned int channelId = 0u;  ///< Stable ID of the produced channel.
  unsigned int parentChannelId = 0u;  ///< Required parent channel ID; zero allows any matching parent PDG.
  unsigned int family = KFGpuCpuFamilyTrackComposite;  ///< CPU catalogue family of the hypothesis.
  unsigned int generation = 2u;  ///< Dependency generation in the decay graph.
  unsigned int outputClass = KFGpuGraphOutputSecondary;  ///< Output category for subsequent reconstruction stages.
  unsigned int selectedV0Role = KFGpuSelectedV0RoleInvalid;  ///< Optional coarse Lambda/anti-Lambda role.
  unsigned int bachelorRole = KFGpuBachelorRoleInvalid;  ///< Optional coarse charged pion/kaon role.
  KFParticleGpuTrackSet bachelorTrackSet = SecondaryNegativeFirst;  ///< Event-local source track set.
  KFParticleGpuTrackSpecies bachelorSpecies = Pion;  ///< Track species used by plan grouping.
  int v0Pdg = 0;  ///< Required selected composite hypothesis.
  int bachelorPdg = 0;  ///< Required bachelor-track hypothesis.
  int motherPdg = 0;  ///< Produced mother hypothesis.
  int primaryVertexIndex = -1;  ///< Primary-vertex requirement or a supported selection sentinel.
  unsigned int flags = KFGpuV0TrackUseLineDca | KFGpuV0TrackUseEnergyFit;  ///< Construction-mode flags.
  int transportMode = KFGpuTransportFullField;  ///< Transport model for the fit.
  float bachelorMass = 0.f;  ///< Bachelor mass hypothesis.
  float motherMass = 0.f;  ///< Expected mother mass for selection.
  float motherMassSigma = -1.f;  ///< Mother mass resolution used by the selection window.
  float secondaryMassSigmaCut = -1.f;  ///< Secondary mass-window width in resolution units.
  float maxSecondaryTopoChi2PerNdf = -1.f;  ///< Secondary topology chi2/NDF limit.
  float minBachelorChiToPrimaryVertex = -1.f;  ///< Minimum bachelor-to-primary-vertex chi value.
  float maxV0TrackDistance = -1.f;  ///< Upper distance cut between composite and bachelor.
  float minBachelorPt = -1.f;  ///< Minimum bachelor transverse momentum.
  int minBachelorPixelHits = -1;  ///< Minimum required bachelor pixel hits.
  float parentMassConstraint = -1.f;  ///< Optional mass constraint for the selected composite.
  float parentMassConstraintSigma = 0.f;  ///< Uncertainty supplied with the parent mass constraint.
};

/**
 * @brief Compatibility-table entry mapping input identities to one or more channel bits.
 *
 * Role-based lookup compares only the two roles. PDG-based lookup compares both
 * PDGs and the parent channel, with a zero parentChannelId acting as a wildcard.
 * The enabled-channel mask is applied by the routing view, not stored here.
 */
struct KFParticleGpuV0TrackCompatibilityEntry
{
  unsigned int parentChannelId = 0u;  ///< Required stable parent ID; zero is an entry-side wildcard.
  int selectedPdg = 0;  ///< Selected composite PDG required by PDG-based lookup.
  int bachelorPdg = 0;  ///< Bachelor PDG required by PDG-based lookup.
  unsigned int selectedV0Role = KFGpuSelectedV0RoleInvalid;  ///< Composite role for role-based lookup.
  unsigned int bachelorRole = KFGpuBachelorRoleInvalid;  ///< Bachelor role for role-based lookup.
  KFParticleGpuChannelMask channels;  ///< Compatible hypotheses in the owning plan's bit mapping.
};

/**
 * @brief Plan-level group of hypotheses sharing a bachelor source set and dependency generation.
 *
 * Actual track ranges are resolved from KFParticleGpuEventDesc at execution;
 * no event data or track copies are stored in this persistent table. Role and
 * species masks summarize the group; exact PDG/channel checks remain separate.
 */
struct KFParticleGpuV0TrackExecutionGroup
{
  unsigned int groupIndex = 0u;  ///< Index in the compiled execution-group table.
  unsigned int generation = 2u;  ///< Dependency generation shared by the group.
  KFParticleGpuTrackSet bachelorTrackSet = SecondaryNegativeFirst;  ///< Event-local bachelor source set.
  unsigned int selectedV0Roles = 0u;  ///< Bit mask of coarse composite role numbers.
  unsigned int bachelorRoles = 0u;  ///< Bit mask of coarse bachelor role numbers.
  unsigned int bachelorSpecies = 0u;  ///< Bit mask indexed by KFParticleGpuTrackSpecies values.
  KFParticleGpuChannelMask channels;  ///< Plan channel bits belonging to this group.

  /// @brief Tests the composite-role bit; values outside the 32-bit mask return false.
  KFPARTICLE_GPU_HOST_DEVICE bool AcceptsSelectedV0Role(unsigned int role) const
  {
    return (selectedV0Roles & KFParticleGpuRoutingRoleBit(role)) != 0u;
  }

  /// @brief Tests the bachelor-role bit; values outside the 32-bit mask return false.
  KFPARTICLE_GPU_HOST_DEVICE bool AcceptsBachelorRole(unsigned int role) const
  {
    return (bachelorRoles & KFParticleGpuRoutingRoleBit(role)) != 0u;
  }

  /// @brief Tests the species bit; values outside the 32-bit mask return false.
  KFPARTICLE_GPU_HOST_DEVICE bool AcceptsBachelorSpecies(
    KFParticleGpuTrackSpecies species) const
  {
    const unsigned int value = static_cast<unsigned int>(species);
    return value < 32u && (bachelorSpecies & (1u << value)) != 0u;
  }
};

/**
 * @brief Compact construction task referencing a plan hypothesis and event-local inputs.
 *
 * Contains indices only, not fit states or copied tracks. In particular,
 * selectedV0Index addresses the selected-index list, not the raw candidate pool.
 * Transaction kernels additionally use domainIndex and workIndex for isolation.
 */
struct KFParticleGpuV0TrackRoutedTask
{
  unsigned int descriptorIndex = 0u;     ///< Hypothesis index in the compiled routing descriptor table.
  unsigned int selectedV0Index = 0u;     ///< Index in selected-candidate storage; resolve it to a pool index.
  unsigned int bachelorTrackIndex = 0u;  ///< Index in the batch input-track arrays, not a source-track ID.
  unsigned int eventIndex = 0u;          ///< Input event descriptor index.
  unsigned int domainIndex = 0u;         ///< Transaction domain index when used by transaction kernels.
  unsigned int workIndex = 0u;           ///< Transaction work-range index when used by transaction kernels.
};

/**
 * @brief Scalar snapshot of a bounded composite-plus-track routing pass.
 *
 * Accepted tasks count successful hypotheses before storage clipping; stored
 * tasks count those retained in the task buffer. Inspect overflowFlags as well
 * as Truncated(), since a flag need not imply an accepted/stored count difference.
 */
struct KFParticleGpuV0TrackRoutingStatus
{
  unsigned int visitedPairs = 0u;       ///< Pairs counted after basic routing eligibility checks.
  unsigned int activeChannelBits = 0u;  ///< Sum of enabled compatible hypotheses over counted pairs, before cuts.
  unsigned int acceptedTasks = 0u;      ///< Hypotheses passing generation cuts, including those lost to capacity.
  unsigned int storedTasks = 0u;        ///< Accepted tasks retained within storage capacity.
  unsigned int blockReservations = 0u;  ///< Reservation operations: per task or per block, depending on routing mode.
  unsigned int overflowFlags = 0u;      ///< Overflow/error bits reported by the producing kernel.

  /// @brief Returns whether accepted tasks outnumber stored tasks; does not inspect overflowFlags.
  KFPARTICLE_GPU_HOST_DEVICE bool Truncated() const
  {
    return acceptedTasks > storedTasks;
  }
};

/**
 * @brief Non-owning host/device view of a compiled routing plan and its mutable channel counters.
 *
 * Tables are read-only through this view. Lookup methods union matching channel
 * masks; CompatibleChannels() and CompatibleChannelsByPdg() also apply the
 * optional enabled mask, while KnownChannelsByPdg() deliberately ignores it.
 * A null enabled mask leaves all compatible plan channels available.
 *
 * The flat descriptor can be published in XPU constant memory without owning
 * its arrays. Const methods may return mutable counter pointers; updating those
 * counters and synchronizing access are responsibilities of the calling kernels.
 * @pre Bind valid arrays/counts in the execution memory space before querying
 * the view, and keep all referenced storage alive. Construction performs no validation.
 */
class KFParticleGpuV0TrackRoutingView
{
 public:
  /// @brief Defaulted descriptor constructor; assign a bound view before querying it.
  KFPARTICLE_GPU_HOST_DEVICE KFParticleGpuV0TrackRoutingView() = default;

  /**
   * @brief Binds routing tables, an optional enabled mask, and per-descriptor counter arrays.
   * @param descriptors Compiled hypotheses, indexed by transient channel bit.
   * @param descriptorCount Number of valid hypothesis descriptors.
   * @param compatibility Input-identity to channel-mask table.
   * @param compatibilityCount Number of valid compatibility entries.
   * @param groups Execution-group table.
   * @param groupCount Number of valid execution groups.
   * @param enabledChannels Single enabled-channel mask, or nullptr to omit filtering.
   * @param channelVisitedCounters Per-descriptor compatible-pair counters.
   * @param channelAcceptedCounters Per-descriptor accepted-task counters.
   * @param channelStoredCounters Per-descriptor retained-task counters.
   * @param channelConstructedCounters Per-descriptor constructed-candidate counters.
   * @note Copies pointers/counts only; no counters are initialized or downloaded.
   */
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

  /// @brief Returns the borrowed hypothesis table, indexed below DescriptorCount().
  KFPARTICLE_GPU_HOST_DEVICE const KFParticleGpuV0TrackRoutingDescriptor*
  Descriptors() const
  {
    return fDescriptors;
  }

  /// @brief Returns the number of valid routing descriptors, not allocated capacity.
  KFPARTICLE_GPU_HOST_DEVICE unsigned int DescriptorCount() const { return fDescriptorCount; }

  /// @brief Returns the borrowed compatibility table, indexed below CompatibilityCount().
  KFPARTICLE_GPU_HOST_DEVICE const KFParticleGpuV0TrackCompatibilityEntry*
  CompatibilityEntries() const
  {
    return fCompatibility;
  }

  /// @brief Returns the number of valid compatibility entries.
  KFPARTICLE_GPU_HOST_DEVICE unsigned int CompatibilityCount() const
  {
    return fCompatibilityCount;
  }

  /// @brief Returns the borrowed execution-group table, indexed below GroupCount().
  KFPARTICLE_GPU_HOST_DEVICE const KFParticleGpuV0TrackExecutionGroup* Groups() const
  {
    return fGroups;
  }

  /// @brief Returns the number of valid execution groups.
  KFPARTICLE_GPU_HOST_DEVICE unsigned int GroupCount() const { return fGroupCount; }

  /// @brief Returns the borrowed enabled mask pointer; nullptr means no enabled-mask filtering.
  KFPARTICLE_GPU_HOST_DEVICE const KFParticleGpuChannelMask* EnabledChannelsData() const
  {
    return fEnabledChannels;
  }

  /// @brief Returns mutable per-descriptor compatible-pair counters without reading or synchronizing them.
  KFPARTICLE_GPU_HOST_DEVICE unsigned int* ChannelVisitedCounters() const
  {
    return fChannelVisitedCounters;
  }

  /// @brief Returns mutable per-descriptor accepted-task counters, before capacity clipping.
  KFPARTICLE_GPU_HOST_DEVICE unsigned int* ChannelAcceptedCounters() const
  {
    return fChannelAcceptedCounters;
  }

  /// @brief Returns mutable per-descriptor retained-task counters.
  KFPARTICLE_GPU_HOST_DEVICE unsigned int* ChannelStoredCounters() const
  {
    return fChannelStoredCounters;
  }

  /// @brief Returns mutable per-descriptor constructed-candidate counters.
  KFPARTICLE_GPU_HOST_DEVICE unsigned int* ChannelConstructedCounters() const
  {
    return fChannelConstructedCounters;
  }

  /**
   * @brief Unions entries matching both coarse roles, then intersects with the enabled mask if present.
   * @note Does not match parent channel IDs or PDGs, validate role enums, or apply physical cuts.
   */
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

  /// @brief Returns KnownChannelsByPdg() intersected with the enabled mask, when present; applies no physical cuts.
  KFPARTICLE_GPU_HOST_DEVICE KFParticleGpuChannelMask CompatibleChannelsByPdg(
    unsigned int parentChannelId,
    int selectedPdg,
    int bachelorPdg) const
  {
    const KFParticleGpuChannelMask result =
      KnownChannelsByPdg(parentChannelId, selectedPdg, bachelorPdg);
    return fEnabledChannels ? result.Intersected(fEnabledChannels[0]) : result;
  }

  /**
   * @brief Unions channels matching both PDGs and the parent requirement, ignoring the enabled mask.
   * @param parentChannelId Stable channel ID of the selected composite.
   * @param selectedPdg Selected composite PDG hypothesis.
   * @param bachelorPdg Bachelor-track PDG hypothesis.
   * @note A table entry with parentChannelId zero matches any input parent ID;
   * an input ID of zero does not itself act as a wildcard. No physical cuts are applied.
   */
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
  const KFParticleGpuV0TrackRoutingDescriptor* fDescriptors;  ///< Borrowed hypothesis table.
  unsigned int fDescriptorCount;  ///< Active hypothesis count.
  const KFParticleGpuV0TrackCompatibilityEntry* fCompatibility;  ///< Borrowed compatibility table.
  unsigned int fCompatibilityCount;  ///< Active compatibility-entry count.
  const KFParticleGpuV0TrackExecutionGroup* fGroups;  ///< Borrowed execution-group table.
  unsigned int fGroupCount;  ///< Active execution-group count.
  const KFParticleGpuChannelMask* fEnabledChannels;  ///< Optional borrowed enabled mask.
  unsigned int* fChannelVisitedCounters;  ///< Borrowed compatible-pair counters indexed by descriptor.
  unsigned int* fChannelAcceptedCounters;  ///< Borrowed accepted-task counters indexed by descriptor.
  unsigned int* fChannelStoredCounters;  ///< Borrowed retained-task counters indexed by descriptor.
  unsigned int* fChannelConstructedCounters;  ///< Borrowed constructed-candidate counters indexed by descriptor.
};

/**
 * @brief Non-owning task storage and scalar counters for one composite-plus-track generation.
 *
 * Publishes pointers and capacity to the routing/construction kernels, without
 * reserving slots, resetting counters, transferring results, or synchronizing.
 * Const accessors intentionally expose writable storage for kernel updates.
 * AcceptedTasks() may count beyond TaskCapacity(); StoredTasks() tracks retained
 * entries and OverflowFlags() reports storage failures.
 *
 * @pre Bind the descriptor before use and keep the referenced allocations alive
 * in the execution memory space. The producer owns atomic reservations and
 * counter initialization; consumers must wait for production to complete.
 */
class KFParticleGpuV0TrackGenerationStorageView
{
 public:
  /// @brief Defaulted descriptor constructor; assign a bound view before querying it.
  KFPARTICLE_GPU_HOST_DEVICE KFParticleGpuV0TrackGenerationStorageView() = default;

  /**
   * @brief Binds routed-task storage and scalar counter pointers without initializing their contents.
   * @param tasks Writable routed-task array.
   * @param taskCapacity Allocated task slots, not accepted task count.
   * @param visitedPairs Counter of pairs passing basic routing eligibility checks.
   * @param activeChannelBits Counter of enabled compatible hypotheses before generation cuts.
   * @param acceptedTasks Counter of tasks passing generation cuts, including capacity overflow.
   * @param storedTasks Counter of accepted tasks retained in the array.
   * @param blockReservations Counter of reservation operations, dependent on routing mode.
   * @param overflowFlags Bit mask updated by the producing kernels on storage failure.
   */
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

  /// @brief Returns the borrowed writable task array; valid entries are determined by the producer's counters.
  KFPARTICLE_GPU_HOST_DEVICE KFParticleGpuV0TrackRoutedTask* Tasks() const
  {
    return fTasks;
  }

  /// @brief Returns the allocated task slot limit, not the number of valid tasks.
  KFPARTICLE_GPU_HOST_DEVICE unsigned int TaskCapacity() const
  {
    return fTaskCapacity;
  }

  /// @brief Returns the mutable scalar visited-pair counter pointer.
  KFPARTICLE_GPU_HOST_DEVICE unsigned int* VisitedPairs() const
  {
    return fVisitedPairs;
  }

  /// @brief Returns the mutable scalar compatible-hypothesis counter pointer.
  KFPARTICLE_GPU_HOST_DEVICE unsigned int* ActiveChannelBits() const
  {
    return fActiveChannelBits;
  }

  /// @brief Returns the mutable scalar accepted-task counter pointer; its value may exceed task capacity.
  KFPARTICLE_GPU_HOST_DEVICE unsigned int* AcceptedTasks() const
  {
    return fAcceptedTasks;
  }

  /// @brief Returns the mutable scalar retained-task counter pointer.
  KFPARTICLE_GPU_HOST_DEVICE unsigned int* StoredTasks() const
  {
    return fStoredTasks;
  }

  /// @brief Returns the mutable reservation-counter pointer; it is not necessarily a GPU block count.
  KFPARTICLE_GPU_HOST_DEVICE unsigned int* BlockReservations() const
  {
    return fBlockReservations;
  }

  /// @brief Returns the mutable scalar overflow-mask pointer without reading or synchronizing it.
  KFPARTICLE_GPU_HOST_DEVICE unsigned int* OverflowFlags() const
  {
    return fOverflowFlags;
  }

 private:
  KFParticleGpuV0TrackRoutedTask* fTasks;  ///< Borrowed routed-task storage.
  unsigned int fTaskCapacity;              ///< Allocated task slots.
  unsigned int* fVisitedPairs;             ///< Borrowed scalar visited-pair counter.
  unsigned int* fActiveChannelBits;        ///< Borrowed scalar compatible-hypothesis counter.
  unsigned int* fAcceptedTasks;            ///< Borrowed scalar accepted-task counter.
  unsigned int* fStoredTasks;              ///< Borrowed scalar retained-task counter.
  unsigned int* fBlockReservations;        ///< Borrowed scalar reservation counter.
  unsigned int* fOverflowFlags;            ///< Borrowed scalar overflow mask.
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
