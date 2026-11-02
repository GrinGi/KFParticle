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
 * @brief Host-side CPU finder channel catalogue, GPU plan assembly, and event capacity planning.
 *
 * The catalogue records stable channel identities, dependencies, input requirements,
 * and declared GPU support. Helpers translate these contracts into decay-plan
 * channels and report coverage. Capacity planners use event range sizes only;
 * they do not fit particles, allocate GPU buffers, or launch device work.
 */

#ifndef KFPARTICLEGPUCPUCHANNELCATALOGUE_H
#define KFPARTICLEGPUCPUCHANNELCATALOGUE_H

#include "planning/KFParticleGpuDecayGraph.h"

#include <array>
#include <memory>
#include <vector>

class KFParticleGpuDecayPlan;
struct KFParticleGpuEventDesc;

/** @brief CPU configuration/input availability, independent of GPU implementation status. */
enum KFParticleGpuCpuChannelActivation : unsigned int
{
  KFGpuCpuChannelActive = 0u,                 ///< Enabled in the catalogue's CPU configuration.
  KFGpuCpuChannelConditionalInput = 1u,       ///< Required when its conditional inputs are available.
  KFGpuCpuChannelConfigurationDisabled = 2u,  ///< Explicitly excluded from required coverage.
  KFGpuCpuChannelActivationCount = 3u         ///< Exclusive upper bound, not an activation state.
};

/** @brief Input-category bits describing a channel's requirements, not actual event availability. */
enum KFParticleGpuCpuChannelInput : unsigned int
{
  KFGpuCpuInputTracks = 1u << 0u,              ///< Reconstructed input tracks.
  KFGpuCpuInputPrimaryVertices = 1u << 1u,     ///< Event primary vertices.
  KFGpuCpuInputChiToPrimaryVertex = 1u << 2u,  ///< Track-to-primary-vertex chi values.
  KFGpuCpuInputNeutralCandidates = 1u << 3u    ///< Neutral-candidate input category.
};

/**
 * @brief Stable host-side description of one CPU finder channel and its declared GPU support.
 *
 * Channel IDs distinguish reconstruction paths even when mother PDGs coincide.
 * Parent IDs reference catalogue channels; source kinds describe how their inputs
 * are consumed. This is a contract, not an executable kernel descriptor: plan
 * builders supply numerical masses, cuts, and track-set choices separately.
 */
struct KFParticleGpuCpuChannelContract
{
  unsigned int channelId = 0u;                  ///< Nonzero stable channel ID, not a routing bit.
  unsigned int conjugateChannelId = 0u;         ///< Partner channel ID; may refer to this entry itself.
  unsigned int family = KFGpuCpuFamilyCount;     ///< KFParticleGpuCpuFinderFamily value.
  int motherPdg = 0;                            ///< Output hypothesis, including CPU-specific PDG aliases.
  int firstDaughterPdg = 0;                     ///< First input hypothesis; interpretation depends on topology.
  int secondDaughterPdg = 0;                    ///< Second input hypothesis; zero for unary operations.
  unsigned int topology = KFGpuGraphTopologyInvalid;  ///< Shape of the graph operation.
  unsigned int generation = 0u;                 ///< Dependency generation, greater than either parent's.
  unsigned int firstSourceKind = KFGpuGraphSourceNone;   ///< First KFParticleGpuGraphSourceKind value.
  unsigned int secondSourceKind = KFGpuGraphSourceNone;  ///< Second source kind; None for unary operations.
  unsigned int parentChannelId = 0u;            ///< First candidate dependency; zero means none declared.
  unsigned int secondParentChannelId = 0u;      ///< Second candidate dependency; zero means none declared.
  unsigned int operationMask = 0u;              ///< Required KFParticleGpuGraphOperation bits.
  unsigned int daughterHypothesisProfile = 0u;  ///< Stable hypothesis-profile ID, currently the channel ID.
  unsigned int selectionProfile = 0u;           ///< Stable selection-profile ID, currently the channel ID.
  unsigned int outputClass = KFGpuGraphOutputInvalid;  ///< Candidate category for subsequent stages.
  unsigned int requiredInputs = 0u;             ///< KFParticleGpuCpuChannelInput bit mask.
  unsigned int activation = KFGpuCpuChannelActive;  ///< CPU configuration/input activation state.
  unsigned int supportStatus = KFGpuGraphUnsupported;  ///< Declared GPU implementation status.
  unsigned int unsupportedReason = KFGpuGraphUnsupportedNotImplemented;  ///< Reason, or None when supported.
};

/** @brief Coverage of one catalogue channel by a particular decay plan; contains no event measurements. */
struct KFParticleGpuCpuChannelCoverage
{
  unsigned int channelId = 0u;  ///< Stable catalogue channel ID.
  unsigned int family = KFGpuCpuFamilyCount;  ///< CPU finder family.
  unsigned int activation = KFGpuCpuChannelActive;  ///< Catalogue activation state, unchanged by the plan.
  unsigned int supportStatus = KFGpuGraphUnsupported;  ///< Catalogue support if present; Unsupported if absent.
  unsigned int unsupportedReason = KFGpuGraphUnsupportedNotImplemented;  ///< Catalogue or missing-channel reason.
};

/**
 * @brief Channel accounting for a CPU finder plan assuming all conditional inputs are available.
 *
 * Active and conditional channels are required; configuration-disabled channels
 * are counted separately. These are channel counts, not candidate counts or a
 * proof of numerical equivalence between CPU and GPU reconstruction.
 */
struct KFParticleGpuCpuFinderCoverageSummary
{
  std::array<unsigned int, KFGpuCpuFamilyCount> requiredByFamily = {};     ///< Required channels by family.
  std::array<unsigned int, KFGpuCpuFamilyCount> implementedByFamily = {};  ///< Supported required channels by family.
  std::array<unsigned int, KFGpuCpuFamilyCount> disabledByFamily = {};     ///< Disabled channels by family.
  unsigned int catalogueChannels = 0u;    ///< Total catalogue entries.
  unsigned int requiredChannels = 0u;     ///< Active plus conditional entries.
  unsigned int implementedChannels = 0u;  ///< Required entries present and declared fully supported.
  unsigned int disabledChannels = 0u;     ///< Entries excluded by CPU configuration.
  unsigned int unsupportedChannels = 0u;  ///< Required entries absent or not fully supported.

  /// @brief Checks aggregate count consistency and required coverage; does not inspect per-family arrays.
  bool Complete() const
  {
    return requiredChannels == implementedChannels
      && unsupportedChannels == 0u
      && catalogueChannels == requiredChannels + disabledChannels;
  }
};

/**
 * @brief Returns the immutable catalogue, built and validated on its first use.
 * @return Shared reference valid for the lifetime of the function-local static catalogue.
 * @throws std::invalid_argument If the compiled catalogue fails its consistency checks.
 */
const std::vector<KFParticleGpuCpuChannelContract>&
KFParticleGpuCpuChannelCatalogue();

/**
 * @brief Checks IDs, contract fields, conjugate links, dependency generations, and family presence.
 * @throws std::invalid_argument If a catalogue consistency check fails.
 * @note Does not validate numerical fit settings or inspect event input availability.
 */
void ValidateKFParticleGpuCpuChannelCatalogue(
  const std::vector<KFParticleGpuCpuChannelContract>& catalogue);

/**
 * @brief Reports every catalogue entry in catalogue order against @p plan.
 * A channel matches by ID and mother PDG in any supported plan channel collection;
 * the result uses declared support, without comparing numerical cuts or running fits.
 */
std::vector<KFParticleGpuCpuChannelCoverage>
MakeCpuFinderChannelCoverage(const KFParticleGpuDecayPlan& plan);

/**
 * @brief Classifies family coverage from supplied implementation counts and catalogue requirements.
 * @param implementedCounts Channel counts indexed by CPU finder family; not verified against a plan here.
 * @return One entry per family; a family containing only disabled channels counts as supported.
 */
std::vector<KFParticleGpuGraphFamilyCoverage>
MakeCpuFinderFamilyCoverage(
  const std::array<unsigned int, KFGpuCpuFamilyCount>& implementedCounts);

/// @brief Appends active two-daughter catalogue channels; does not clear existing plan contents.
void AddCpuFinderTwoDaughterChannels(KFParticleGpuDecayPlan& plan);

/**
 * @brief Appends supported active composite-track and long-lived composite channels.
 * @note Channels requiring the pi0 parent (5019) are skipped unless it is already in the plan.
 * Other parent dependencies are not automatically added by this helper.
 */
void AddCpuFinderCompositeTrackChannels(KFParticleGpuDecayPlan& plan);

/// @brief Appends supported active composite-pair operations without automatically adding their parents.
void AddCpuFinderCompositeCompositeChannels(KFParticleGpuDecayPlan& plan);

/// @brief Appends supported active primary projections whose parent channels are already present.
void AddCpuFinderPrimaryProjectionChannels(KFParticleGpuDecayPlan& plan);

/// @brief Appends supported active missing-mass operations using mother-track and charged-daughter inputs.
void AddCpuFinderNeutralMissingMassChannels(KFParticleGpuDecayPlan& plan);

/// @brief Appends supported active kaon-matching operations whose parent channels are already present.
void AddCpuFinderKaonMatchingChannels(KFParticleGpuDecayPlan& plan);

/// @brief Appends supported active final-selection operations whose parent channels are already present.
void AddCpuFinderFinalSelectionChannels(KFParticleGpuDecayPlan& plan);

/**
 * @brief Appends the catalogue's complete active finder path in dependency-aware family order.
 * @note Does not clear or validate the resulting plan. Use ValidateCompleteCpuFinderPlan()
 * to check catalogue coverage; decay-graph validation remains a separate operation.
 */
void AddCompleteCpuFinderChannels(KFParticleGpuDecayPlan& plan);

/**
 * @brief Appends requested supported outputs and the transitive closure of their catalogue parents.
 * @param plan Destination plan; existing contents are retained.
 * @param requestedMotherPdgs Exact signed mother PDGs; all matching supported, enabled paths are selected.
 * Empty or unmatched requests add nothing; charge-conjugate outputs are not implicitly requested.
 * @throws std::logic_error If a selected channel refers to an unknown catalogue dependency.
 */
void AddRequestedCpuFinderChannels(
  KFParticleGpuDecayPlan& plan,
  const std::vector<int>& requestedMotherPdgs);

/// @brief Aggregates catalogue coverage, treating conditional-input channels as required.
KFParticleGpuCpuFinderCoverageSummary
MakeCompleteCpuFinderCoverageSummary(const KFParticleGpuDecayPlan& plan);

/**
 * @brief Requires complete aggregate and per-family catalogue coverage in @p plan.
 * @throws std::invalid_argument If any required channel or family is incompletely covered.
 * @note This is a manifest check, not numerical, input-data, or decay-graph validation.
 */
void ValidateCompleteCpuFinderPlan(const KFParticleGpuDecayPlan& plan);

/**
 * @brief Event-owned slot counts from combinatorial capacity planning, not observed output sizes.
 *
 * Counts use track-range products and propagated parent bounds without physical
 * selection. Graph operations are bounded by a per-channel task limit; task
 * storage is reused across generations. These limits do not guarantee storage
 * for every mathematically possible combination; runtime overflow checks remain necessary.
 * No host or device payload memory is allocated by this value type.
 */
struct KFParticleGpuEventCapacityPlan
{
  unsigned int twoDaughterCandidates = 0u;  ///< Sum of raw two-track channel bounds.
  unsigned int cascadeCandidates = 0u;     ///< Sum of propagated composite-plus-track bounds.
  unsigned int graphCandidates = 0u;       ///< Sum of bounded graph-operation outputs.
  unsigned int taskCapacity = 0u;          ///< Maximum task workspace requirement across planning phases.
  unsigned int candidates = 0u;            ///< Combined candidate slots from all three output categories.
  unsigned int daughterIds = 0u;           ///< Flattened lineage slots, using each channel's bounded lineage length.
  unsigned int selectedCandidates = 0u;    ///< Selection-index slots, currently equal to candidates.
};

/**
 * @brief Reuses host-side dependency indices and scratch storage to plan capacities for successive events.
 *
 * Construction maps stable channel IDs to dense slots once. Plan() reads event
 * range sizes, propagates candidate/lineage bounds, and reuses the same scratch
 * arrays; event stamps avoid clearing every slot on each call. No track states
 * are copied, fitted, or transferred, and no GPU buffer allocation is performed.
 *
 * The decay plan is borrowed and must outlive the planner without modification.
 * Plan() checks its revision before using cached indices. The planner owns its
 * CPU workspace through a private implementation and is non-copyable.
 *
 * @note Calls mutate shared scratch state and must be serialized. One CPU batch
 * builder uses one planner; separate planners may share an immutable decay plan.
 */
class KFParticleGpuCpuFinderCapacityPlanner
{
 public:
  /// @brief Borrows @p plan and allocates/compiles its host-side channel-slot workspace.
  explicit KFParticleGpuCpuFinderCapacityPlanner(
    const KFParticleGpuDecayPlan& plan);

  /// @brief Releases owned CPU workspace; does not destroy the borrowed plan or synchronize GPU work.
  ~KFParticleGpuCpuFinderCapacityPlanner();

  /// @brief Copying is disabled because the planner owns mutable per-event scratch state.
  KFParticleGpuCpuFinderCapacityPlanner(
    const KFParticleGpuCpuFinderCapacityPlanner&) = delete;

  /// @brief Copy assignment is disabled for the same workspace-ownership reason.
  KFParticleGpuCpuFinderCapacityPlanner& operator=(
    const KFParticleGpuCpuFinderCapacityPlanner&) = delete;

  /**
   * @brief Computes event slot counts using the compiled plan and reusable scratch arrays.
   * @param event Valid track-set/species ranges for one event; the event is not retained.
   * @return Counts under the bounded policy described by KFParticleGpuEventCapacityPlan.
   * @throws std::logic_error If the plan revision changed or a raw-track source encoding is invalid.
   * @throws std::overflow_error If an uncapped product or accumulated count exceeds the unsigned GPU ABI.
   * @note Follows plan iteration order; candidate sources not yet evaluated contribute zero.
   * Plan/input validity and runtime overflow handling remain the caller's responsibility.
   */
  KFParticleGpuEventCapacityPlan Plan(
    const KFParticleGpuEventDesc& event);

 private:
  struct Impl;                  ///< Private dependency indices and mutable host scratch storage.
  std::unique_ptr<Impl> fImpl;  ///< Sole owner of the implementation; its decay-plan pointer is borrowed.
};

/**
 * @brief Sums Cartesian track-pair counts for two-daughter and track/track graph-operation channels.
 * @note Uses declared range sizes, without per-track PID, overlap, or physical-cut filtering;
 * the same pair may contribute to multiple channel hypotheses. Composite inputs are excluded.
 * @throws std::overflow_error If a product or total exceeds the unsigned GPU ABI.
 * @throws std::logic_error If a graph raw-track source encoding is invalid.
 * @pre Event track-set/species ranges and the supplied plan must be valid.
 */
unsigned int KFParticleGpuCpuFinderRawTaskCapacity(
  const KFParticleGpuEventDesc& event,
  const KFParticleGpuDecayPlan& plan);

/**
 * @brief Computes event capacities with the same bounded policy using a temporary channel-ID map.
 * @note This one-shot variant allocates CPU scratch on each call; prefer the reusable
 * planner for repeated events. Candidate dependencies not yet evaluated contribute zero.
 * @throws std::overflow_error If an uncapped product or accumulated count exceeds the unsigned GPU ABI.
 * @throws std::logic_error If a graph raw-track source encoding is invalid.
 * @pre Event track-set/species ranges and plan iteration order must be suitable for capacity propagation.
 */
KFParticleGpuEventCapacityPlan KFParticleGpuCpuFinderEventCapacityPlan(
  const KFParticleGpuEventDesc& event,
  const KFParticleGpuDecayPlan& plan);

/**
 * @brief Returns a deterministic, order-sensitive hash of catalogue contract fields.
 * @note Not a cryptographic checksum or decay-plan revision. Numerical builder settings
 * outside the contracts are not included, and the hash is recomputed on each call.
 */
unsigned long long KFParticleGpuCpuChannelCatalogueRevision();

static_assert(KFGpuCpuFamilyCount == 9u,
              "the frozen CPU catalogue must be updated when a finder family is added");

#endif
