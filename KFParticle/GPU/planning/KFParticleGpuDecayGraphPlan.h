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
 * @brief Host-owned decay-graph and channel-routing plans, plus event-domain task tiling.
 *
 * Manifests collect declarations; compilation validates and arranges them into
 * flat tables for later device upload. Routing plans compile channel hypotheses
 * separately from graph dependencies. None of these helpers launches GPU work
 * or transfers data; device storage and synchronization belong to their callers.
 */

#ifndef KFPARTICLEGPUDECAYGRAPHPLAN_H
#define KFPARTICLEGPUDECAYGRAPHPLAN_H

#include "planning/KFParticleGpuChannelRouting.h"
#include "planning/KFParticleGpuDecayGraph.h"
#include "data/KFParticleGpuTransaction.h"
#include "planning/KFParticleGpuTwoDaughterRouting.h"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <vector>

class KFParticleGpuDecayPlan;

/**
 * @brief Owns an uncompiled inventory of channel nodes and CPU finder family coverage.
 *
 * Add methods copy entries in insertion order without checking IDs, dependencies,
 * or duplicates. KFParticleGpuDecayGraphPlan::Compile() performs graph validation.
 * The manifest contains channel descriptions, not event tracks or candidates.
 * Mutations must be serialized; appending or clearing can invalidate element pointers.
 */
class KFParticleGpuDecayGraphManifest
{
 public:
  /// @brief Removes all declarations while retaining the vectors' allocated capacity.
  void Clear();

  /// @brief Appends a node by value without validation or duplicate-ID checks.
  void AddNode(const KFParticleGpuGraphNode& node);

  /// @brief Appends a family coverage record by value without validation or deduplication.
  void AddFamilyCoverage(const KFParticleGpuGraphFamilyCoverage& coverage);

  /// @brief Returns the owned node declarations in insertion order, without copying them.
  const std::vector<KFParticleGpuGraphNode>& Nodes() const { return fNodes; }

  /// @brief Returns the owned, uncompiled family coverage records.
  const std::vector<KFParticleGpuGraphFamilyCoverage>& FamilyCoverage() const
  {
    return fFamilyCoverage;
  }

 private:
  std::vector<KFParticleGpuGraphNode> fNodes;  ///< Owned channel declarations, not yet sorted or validated.
  std::vector<KFParticleGpuGraphFamilyCoverage> fFamilyCoverage;  ///< Owned coverage declarations.
};

/**
 * @brief Maximum element counts accepted by graph compilation, not preallocated storage sizes.
 * Defaults permit the full unsigned index range; they do not impose a practical
 * memory budget. Node limits include unsupported declarations retained in the graph.
 */
struct KFParticleGpuDecayGraphCompileLimits
{
  unsigned int nodes = std::numeric_limits<unsigned int>::max();     ///< Maximum manifest nodes.
  unsigned int groups = std::numeric_limits<unsigned int>::max();    ///< Maximum compiled execution groups.
  unsigned int families = std::numeric_limits<unsigned int>::max();  ///< Maximum family coverage records.
};

/**
 * @brief Validates a manifest and owns the resulting generation-ordered host graph tables.
 *
 * Supported nodes are sorted by generation and execution signature, then grouped
 * into contiguous ranges. Unsupported nodes remain outside execution groups.
 * Coverage records are sorted by family; a content hash identifies the result.
 * This class validates structural contracts, not numerical CPU/GPU equivalence.
 *
 * Input declarations are copied, so the manifest need not outlive the plan.
 * Compile() and Clear() invalidate element pointers and require serialized access.
 * Uploading these tables and managing device-side views are separate responsibilities.
 */
class KFParticleGpuDecayGraphPlan
{
 public:
  /**
   * @brief Replaces the current graph with validated, sorted declarations and execution groups.
   * @param manifest Nodes and exactly one coverage record for each CPU finder family.
   * @param limits Upper bounds on node, group, and family record counts.
   * @throws std::length_error If a compile limit is exceeded.
   * @throws std::invalid_argument If coverage, IDs, node shape, payload indices, or dependencies are invalid.
   * @note Clears the previous result first. Failure leaves revision zero and may
   * leave copied/partially processed tables; consume tables only after successful compilation.
   * Does not validate external payload storage or the numerical settings it contains.
   */
  void Compile(
    const KFParticleGpuDecayGraphManifest& manifest,
    const KFParticleGpuDecayGraphCompileLimits& limits =
      KFParticleGpuDecayGraphCompileLimits());

  /// @brief Empties all tables and resets the revision to zero, retaining vector capacity.
  void Clear();

  /// @brief Returns the nonzero compiled content hash, or zero before compilation, after Clear(), or on failure.
  unsigned long long SourceRevision() const { return fSourceRevision; }

  /// @brief Returns owned nodes in compiled order, including unsupported declarations after supported ones.
  const std::vector<KFParticleGpuGraphNode>& Nodes() const { return fNodes; }

  /// @brief Returns execution groups whose ranges index Nodes(); only supported nodes are grouped.
  const std::vector<KFParticleGpuGraphExecutionGroup>& Groups() const { return fGroups; }

  /// @brief Returns owned coverage records, sorted by family after successful compilation.
  const std::vector<KFParticleGpuGraphFamilyCoverage>& FamilyCoverage() const
  {
    return fFamilyCoverage;
  }

 private:
  unsigned long long fSourceRevision = 0u;  ///< Graph-content hash, not the source decay plan's revision counter.
  std::vector<KFParticleGpuGraphNode> fNodes;  ///< Owned copy of graph nodes.
  std::vector<KFParticleGpuGraphExecutionGroup> fGroups;  ///< Owned ranges over supported nodes.
  std::vector<KFParticleGpuGraphFamilyCoverage> fFamilyCoverage;  ///< Owned family coverage records.
};

/**
 * @brief Builds a graph manifest from the supplied decay plan, with catalogue-based family coverage.
 *
 * Copies two-daughter, cascade, and generic graph-operation declarations; it
 * does not add missing catalogue channels. Payload indices remain tied to their
 * source tables. Family counts describe implementation coverage, not real-data qualification.
 * @throws std::invalid_argument If a generic graph-operation descriptor or its topology is unsupported.
 * @note Cascade parents use the explicit channel ID or the first matching two-daughter
 * mother PDG, with generation set to the cascade generation minus one. Full dependency
 * and manifest validation is deferred to KFParticleGpuDecayGraphPlan::Compile().
 */
KFParticleGpuDecayGraphManifest MakeDefaultCpuFinderDecayGraphManifest(
  const KFParticleGpuDecayPlan& plan);

/**
 * @brief Owns host routing tables for selected-composite plus bachelor-track channels.
 *
 * Compilation assigns mask bits in source-plan order, combines compatibility
 * entries by parent ID and input PDGs, and groups channels by bachelor track set
 * and dependency generation. The historical V0 name includes later composites.
 * Lookup methods select compatible hypotheses, not accepted physical candidates.
 *
 * Tables own copies and do not retain the source plan. SourceRevision() records
 * its revision for caller-managed caching; Compile() always rebuilds. Uploads
 * and event input resolution happen elsewhere. Serialize mutations and refresh
 * element pointers after Compile() or Clear().
 */
class KFParticleGpuV0TrackRoutingPlan
{
 public:
  /**
   * @brief Rebuilds routing tables from the plan's composite-track channels, enabling all compiled bits.
   * @throws std::length_error If the channel count exceeds KFParticleGpuChannelMask::BitCapacity.
   * @throws std::invalid_argument If IDs, charged source geometry, species, or signs are inconsistent.
   * @note Failure leaves revision zero and may leave partial tables; use only a successfully compiled result.
   * This helper does not validate candidate-parent graph dependencies or every numerical fit setting.
   */
  void Compile(const KFParticleGpuDecayPlan& plan);

  /// @brief Empties tables and the enabled mask, resets revision to zero, and retains vector capacity.
  void Clear();

  /// @brief Returns the source decay-plan revision captured on success, or zero for a cleared/failed compilation.
  unsigned long long SourceRevision() const { return fSourceRevision; }

  /// @brief Returns hypotheses in source-plan order; descriptor indices equal their transient channel bits.
  const std::vector<KFParticleGpuV0TrackRoutingDescriptor>& Descriptors() const
  {
    return fDescriptors;
  }

  /// @brief Returns input-identity entries and their channel masks; multiple hypotheses may share one entry.
  const std::vector<KFParticleGpuV0TrackCompatibilityEntry>& CompatibilityEntries() const
  {
    return fCompatibility;
  }

  /// @brief Returns groups sharing a bachelor track set and dependency generation, without event ranges.
  const std::vector<KFParticleGpuV0TrackExecutionGroup>& Groups() const { return fGroups; }

  /// @brief Returns the enabled mask; successful compilation enables every declared channel.
  const KFParticleGpuChannelMask& EnabledChannels() const { return fEnabledChannels; }

  /// @brief Unions entries matching both coarse roles and applies the enabled mask; ignores PDGs and parent IDs.
  KFParticleGpuChannelMask CompatibleChannels(
    unsigned int selectedV0Role,
    unsigned int bachelorRole) const;

  /**
   * @brief Unions enabled hypotheses matching both signed PDGs and the parent requirement.
   * @note A table entry with parentChannelId zero is a wildcard; an input ID of zero is not.
   * Performs a linear table scan without physical cuts or event-input validation.
   */
  KFParticleGpuChannelMask CompatibleChannelsByPdg(
    unsigned int parentChannelId,
    int selectedPdg,
    int bachelorPdg) const;

 private:
  unsigned long long fSourceRevision = 0u;  ///< Captured source-plan revision, not a graph-content hash.
  std::vector<KFParticleGpuV0TrackRoutingDescriptor> fDescriptors;  ///< Owned hypothesis settings.
  std::vector<KFParticleGpuV0TrackCompatibilityEntry> fCompatibility;  ///< Owned identity-to-channel mapping.
  std::vector<KFParticleGpuV0TrackExecutionGroup> fGroups;  ///< Owned source-set/generation groups.
  KFParticleGpuChannelMask fEnabledChannels;  ///< Enabled transient bits for this plan only.
};

/**
 * @brief Owns host routing tables for raw two-track channel hypotheses.
 *
 * Compilation assigns bits in source-plan order, includes configured alternate
 * source PDGs in compatibility entries, and groups channels by the ordered pair
 * of track sets. Role/PDG lookup is a coarse routing step; fit and selection cuts
 * are applied later during reconstruction.
 *
 * The source plan is copied into tables, not borrowed. Its captured revision
 * supports caller-managed caching; Compile() always rebuilds and does not upload.
 * Mutations must be serialized and can invalidate pointers to table elements.
 */
class KFParticleGpuTwoDaughterRoutingPlan
{
 public:
  /**
   * @brief Rebuilds and enables the plan's two-daughter routing hypotheses and compatibility tables.
   * @throws std::length_error If the channel count exceeds KFParticleGpuChannelMask::BitCapacity.
   * @throws std::invalid_argument If IDs, source species/signs, transport, or checked numerical settings are invalid.
   * @note Clears the previous result first; failure leaves revision zero and may leave partial tables.
   */
  void Compile(const KFParticleGpuDecayPlan& plan);

  /// @brief Empties tables and the enabled mask, resets revision to zero, and retains vector capacity.
  void Clear();

  /// @brief Returns the captured decay-plan revision after success, or zero before compilation/after failure.
  unsigned long long SourceRevision() const { return fSourceRevision; }

  /// @brief Returns hypothesis descriptors in source-plan order, indexed by transient channel bit.
  const std::vector<KFParticleGpuTwoDaughterRoutingDescriptor>& Descriptors() const
  {
    return fDescriptors;
  }

  /// @brief Returns the ordered source-PDG pair table, including configured alternate hypotheses.
  const std::vector<KFParticleGpuTwoDaughterCompatibilityEntry>& CompatibilityEntries() const
  {
    return fCompatibility;
  }

  /// @brief Returns groups sharing the ordered pair of input track sets, without concrete event ranges.
  const std::vector<KFParticleGpuTwoDaughterExecutionGroup>& Groups() const
  {
    return fGroups;
  }

  /// @brief Returns the enabled channel mask; successful compilation enables all compiled hypotheses.
  const KFParticleGpuChannelMask& EnabledChannels() const { return fEnabledChannels; }

  /// @brief Unions enabled entries matching the ordered coarse-role pair; applies no physical cuts.
  KFParticleGpuChannelMask CompatibleChannels(
    unsigned int firstRole,
    unsigned int secondRole) const;

  /// @brief Linearly matches the exact ordered source-PDG pair, including alternate entries, and filters enabled bits.
  KFParticleGpuChannelMask CompatibleChannelsByPdg(
    int firstPdg,
    int secondPdg) const;

 private:
  unsigned long long fSourceRevision = 0u;  ///< Captured source-plan revision, not a graph-content hash.
  std::vector<KFParticleGpuTwoDaughterRoutingDescriptor> fDescriptors;  ///< Owned channel settings.
  std::vector<KFParticleGpuTwoDaughterCompatibilityEntry> fCompatibility;  ///< Owned source-PDG/role mapping.
  std::vector<KFParticleGpuTwoDaughterExecutionGroup> fGroups;  ///< Owned ordered-track-set groups.
  KFParticleGpuChannelMask fEnabledChannels;  ///< Enabled transient channel bits.
};

/**
 * @brief Builds host-side construction tiles over each domain's retained task prefix.
 * @param transaction Host-readable transaction, domain, and status arrays; all three pointers must be nonnull.
 * @param taskCapacity Total task-array capacity against which each domain reservation is checked.
 * @return Owned tiles in domain order, with task offsets spaced by KFParticleGpuPairTileSize.
 * Each tile identifies a domain and a task-array offset; the final tile may be partial.
 * @throws std::logic_error If required pointers, the domain range, status indices, or task reservations are invalid.
 * @throws std::overflow_error If the tile count exceeds the unsigned GPU ABI.
 * @pre The caller has completed any device work and readback needed to make counters current on the host.
 * @note Uses min(storedTasks, reservation.size), preserving a bounded prefix on overflow.
 * Does not copy tasks, upload tiles, validate task contents, or synchronize a queue.
 */
inline std::vector<KFParticleGpuTaskTile> KFParticleGpuBuildTaskTiles(
  KFParticleGpuTransactionView transaction, unsigned int taskCapacity)
{
  if (!transaction.Transaction() || !transaction.Domains() || !transaction.DomainStatuses()) {
    throw std::logic_error("missing construction task-map input");
  }
  const auto domains = transaction.Transaction()->domains;
  if (!KFParticleGpuRangeFits(domains, transaction.DomainCapacity())) {
    throw std::logic_error("invalid construction task-map domains");
  }
  std::vector<KFParticleGpuTaskTile> tiles;
  for (unsigned int local = 0; local < domains.size; ++local) {
    const unsigned int index = domains.offset + local;
    const auto& domain = transaction.Domains()[index];
    if (domain.statusIndex >= transaction.StatusCapacity()
        || !KFParticleGpuRangeFits(domain.taskReservation, taskCapacity)) {
      throw std::logic_error("invalid construction task-map reservation");
    }
    // Preserve the reference path's bounded prefix in an overflowing domain.
    const unsigned int count = std::min(domain.taskReservation.size,
      transaction.DomainStatuses()[domain.statusIndex].storedTasks);
    for (unsigned int offset = 0; offset < count;) {
      if (tiles.size() == std::numeric_limits<unsigned int>::max()) {
        throw std::overflow_error("construction task-map size overflows");
      }
      tiles.push_back({index, domain.taskReservation.offset + offset});
      if (count - offset <= KFParticleGpuPairTileSize) { break; }
      offset += KFParticleGpuPairTileSize;
    }
  }
  return tiles;
}

#endif
