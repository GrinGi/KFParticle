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
 * @brief Flat host/device descriptions of decay-channel dependencies, execution groups, and coverage.
 *
 * These types describe a reconstruction plan, not reconstructed particle states.
 * The host-side graph planner validates and groups nodes; execution code resolves
 * their input roles for each event. Descriptors contain values and indices, while
 * KFParticleGpuDecayGraphView borrows arrays without allocating or transferring them.
 */

#ifndef KFPARTICLEGPUDECAYGRAPH_H
#define KFPARTICLEGPUDECAYGRAPH_H

#include "KFParticleGpuPlatform.h"

#include <type_traits>

/** @brief Input category determining how a graph source's generation and sourceId are interpreted. */
enum KFParticleGpuGraphSourceKind : unsigned int
{
  KFGpuGraphSourceNone = 0u,                 ///< No input, as for the second source of a unary operation.
  KFGpuGraphSourceTrackRange = 1u,           ///< Event track range identified by a compact track-set/species role.
  KFGpuGraphSourceCandidateGeneration = 2u,  ///< Candidates produced by a named channel in an earlier generation.
  KFGpuGraphSourceNeutralRange = 3u,         ///< Neutral input range identified by a host-defined role.
  KFGpuGraphSourceKindCount = 4u             ///< Exclusive upper bound, not an input category.
};

/** @brief Logical input arrangement of a channel; distinct from its operation bits and payload category. */
enum KFParticleGpuGraphTopology : unsigned int
{
  KFGpuGraphTopologyInvalid = 0u,             ///< Unconfigured topology.
  KFGpuGraphTopologyTrackTrack = 1u,          ///< Two raw-track inputs.
  KFGpuGraphTopologyTrackComposite = 2u,      ///< Raw track followed by a reconstructed composite.
  KFGpuGraphTopologyCompositeTrack = 3u,      ///< Reconstructed composite followed by a raw track.
  KFGpuGraphTopologyCompositeComposite = 4u,  ///< Two reconstructed composites.
  KFGpuGraphTopologyNeutralDaughter = 5u,     ///< Neutral-input or missing-mass reconstruction from two inputs.
  KFGpuGraphTopologyUnaryComposite = 6u,      ///< One composite input, for projection or final processing.
  KFGpuGraphTopologyCount = 7u                ///< Exclusive upper bound, not a topology.
};

/**
 * @brief Combinable operation requirements for a graph channel.
 * These bits describe work required by the channel, not a sequence of independent
 * kernel launches. Numerical settings and execution order belong to its payload
 * and implementation. KFGpuGraphKnownOperations is the mask used to reject unknown bits.
 */
enum KFParticleGpuGraphOperation : unsigned int
{
  KFGpuGraphConstruct = 1u << 0u,            ///< Construct a candidate from its inputs.
  KFGpuGraphTransport = 1u << 1u,            ///< Transport fit states as required by reconstruction.
  KFGpuGraphSetProductionVertex = 1u << 2u,  ///< Apply a production-vertex constraint.
  KFGpuGraphMassConstraint = 1u << 3u,       ///< Apply the configured mass constraint.
  KFGpuGraphMatch = 1u << 4u,                ///< Match a composite hypothesis to another input.
  KFGpuGraphSelect = 1u << 5u,               ///< Apply the channel's candidate selection.
  KFGpuGraphMissingMass = 1u << 6u,          ///< Reconstruct a missing neutral daughter.
  KFGpuGraphExtrapolate = 1u << 7u,          ///< Extrapolate an existing composite without binary construction.
  KFGpuGraphKnownOperations = (1u << 8u) - 1u  ///< Union of all recognized operation bits.
};

/**
 * @brief Output category used by reconstruction and later candidate processing.
 * Values are alternatives, not independent bits. A category describes the
 * channel's output role; it does not by itself mean that a candidate passed selection.
 */
enum KFParticleGpuGraphOutputClass : unsigned int
{
  KFGpuGraphOutputInvalid = 0u,              ///< Unconfigured output category.
  KFGpuGraphOutputTemporary = 1u,            ///< Intermediate reconstruction output.
  KFGpuGraphOutputPrimary = 2u,              ///< Primary-candidate output path.
  KFGpuGraphOutputSecondary = 3u,            ///< Secondary-candidate output path.
  KFGpuGraphOutputPrimaryAndSecondary = 4u,  ///< Output available to both primary and secondary paths.
  KFGpuGraphOutputFinal = 5u,                ///< Final-processing output category.
  KFGpuGraphOutputClassCount = 6u            ///< Exclusive upper bound, not an output category.
};

/** @brief Declared implementation coverage, not the success or failure of a particular event. */
enum KFParticleGpuGraphSupportStatus : unsigned int
{
  KFGpuGraphUnsupported = 0u,         ///< Required implementation or inputs are unavailable.
  KFGpuGraphPartiallySupported = 1u,  ///< Only part of the declared functionality is covered.
  KFGpuGraphSupported = 2u,           ///< Declared functionality is fully covered.
  KFGpuGraphSupportStatusCount = 3u   ///< Exclusive upper bound, not a coverage status.
};

/**
 * @brief Category of channel-specific execution settings referenced by a graph node.
 * Payload indices are local to a category, not node-array indices or stable IDs.
 * Generic graph operations additionally carry a separate operation-descriptor index.
 */
enum KFParticleGpuGraphPayloadKind : unsigned int
{
  KFGpuGraphPayloadNone = 0u,                ///< No executable payload, as for an unsupported node.
  KFGpuGraphPayloadTwoDaughter = 1u,         ///< Two-track channel settings.
  KFGpuGraphPayloadCompositeTrack = 2u,      ///< Composite-plus-track construction settings.
  KFGpuGraphPayloadCompositeComposite = 3u,  ///< Composite-pair graph-operation settings.
  KFGpuGraphPayloadNeutralDaughter = 4u,     ///< Neutral-input or missing-mass graph-operation settings.
  KFGpuGraphPayloadUnaryFinal = 5u,          ///< Unary processing settings, including projection and selection.
  KFGpuGraphPayloadBinaryFinal = 6u,         ///< Binary matching/final-processing settings.
  KFGpuGraphPayloadCount = 7u                ///< Exclusive upper bound, not a payload category.
};

/** @brief Reason paired with a support status; None is reserved for fully supported entries. */
enum KFParticleGpuGraphUnsupportedReason : unsigned int
{
  KFGpuGraphUnsupportedNone = 0u,                ///< No support restriction.
  KFGpuGraphUnsupportedNotImplemented = 1u,      ///< Required implementation is absent.
  KFGpuGraphUnsupportedMissingInput = 2u,        ///< Required input is unavailable.
  KFGpuGraphUnsupportedMissingMathematics = 3u,  ///< Required numerical operation is unavailable.
  KFGpuGraphUnsupportedValidationPending = 4u,   ///< Full validated coverage has not been established.
  KFGpuGraphUnsupportedReasonCount = 5u          ///< Exclusive upper bound, not a reason.
};

/**
 * @brief CPU finder families used to account for coverage separately from individual channels.
 * Family values also index coverage arrays. They identify reconstruction stages
 * or channel families, not PDG codes, graph generations, or execution-group indices.
 */
enum KFParticleGpuCpuFinderFamily : unsigned int
{
  KFGpuCpuFamilyTwoDaughter = 0u,              ///< Two-track candidate construction.
  KFGpuCpuFamilySameSignPrimaryResonance = 1u,  ///< Same-sign primary-track resonances.
  KFGpuCpuFamilyTrackComposite = 2u,           ///< Composite-plus-track channels.
  KFGpuCpuFamilyLongLivedComposite = 3u,       ///< Long-lived composite construction chains.
  KFGpuCpuFamilyCompositeComposite = 4u,      ///< Composite-pair channels.
  KFGpuCpuFamilyNeutralMissingMass = 5u,      ///< Neutral-daughter missing-mass reconstruction.
  KFGpuCpuFamilyKaonMatching = 6u,            ///< Reconstructed-kaon matching to charged tracks.
  KFGpuCpuFamilyPrimaryProjection = 7u,       ///< Projection of composites to their primary vertices.
  KFGpuCpuFamilyFinalSelection = 8u,          ///< Final constraint and selection stages.
  KFGpuCpuFamilyCount = 9u                    ///< Number of families, not a valid family value.
};

/**
 * @brief Logical input reference, resolved to event-specific storage by execution code.
 *
 * Candidate inputs name a stable parent channel and its positive generation;
 * the parent must precede the consuming node. Track and neutral inputs use a
 * nonzero range-role ID with generation zero. An absent input has all fields zero.
 * This descriptor contains neither a memory pointer nor a concrete offset/count.
 */
struct KFParticleGpuGraphSource
{
  unsigned int kind = KFGpuGraphSourceNone;  ///< KFParticleGpuGraphSourceKind value.
  unsigned int generation = 0u;             ///< Parent generation for candidate input; otherwise zero.
  unsigned int sourceId = 0u;               ///< Parent channel ID or encoded input-range role, according to kind.
};

/**
 * @brief One reconstruction channel and its dependencies in the compiled device graph.
 *
 * Source references connect channels, not individual candidate instances. Payload
 * indices refer to separately stored numerical settings; sorting graph nodes does
 * not change their stable channel IDs. Unsupported declarations can be retained
 * for coverage but have no executable payload. Default fields do not form a valid
 * executable node; validation is performed by the host-side planner.
 */
struct KFParticleGpuGraphNode
{
  static constexpr unsigned int InvalidDescriptorIndex = ~0u;  ///< No generic graph-operation descriptor.

  unsigned int channelId = 0u;  ///< Stable nonzero channel ID, not an array index or PDG code.
  unsigned int payloadKind = KFGpuGraphPayloadNone;  ///< Category of referenced execution settings.
  unsigned int payloadIndex = 0u;  ///< Index within the payload category; not the generic descriptor index.
  unsigned int operationDescriptorIndex = InvalidDescriptorIndex;  ///< Generic operation-table index, when applicable.
  int motherPdg = 0;  ///< Output hypothesis, including CPU finder aliases.
  unsigned int topology = KFGpuGraphTopologyInvalid;  ///< Logical input arrangement.
  unsigned int generation = 0u;  ///< Positive dependency generation, later than candidate-source generations.
  KFParticleGpuGraphSource firstSource;   ///< First required input reference.
  KFParticleGpuGraphSource secondSource;  ///< Second input, or None for a unary operation.
  unsigned int operationMask = 0u;  ///< KFParticleGpuGraphOperation bit mask.
  unsigned int selectionProfile = 0u;  ///< Selection-profile identifier, not selection results.
  unsigned int outputClass = KFGpuGraphOutputInvalid;  ///< Output processing category.
  unsigned int supportStatus = KFGpuGraphUnsupported;  ///< Declared implementation status.
  unsigned int unsupportedReason = KFGpuGraphUnsupportedNotImplemented;  ///< Reason, or None when fully supported.
};

/**
 * @brief Contiguous supported nodes sharing an execution-compatible graph signature.
 *
 * Grouping uses generation, topology, both source kinds, and operation mask.
 * Channels in a group may have different PDGs and source IDs. The node range
 * addresses the compiled node array, not event tracks or reconstructed candidates;
 * a group is a scheduling description, not a guarantee of one kernel launch.
 */
struct KFParticleGpuGraphExecutionGroup
{
  unsigned int groupIndex = 0u;  ///< Index in the compiled execution-group array.
  unsigned int generation = 0u;  ///< Dependency generation shared by the nodes.
  unsigned int topology = KFGpuGraphTopologyInvalid;  ///< Shared input topology.
  unsigned int firstSourceKind = KFGpuGraphSourceNone;   ///< Shared first-input category.
  unsigned int secondSourceKind = KFGpuGraphSourceNone;  ///< Shared second-input category.
  unsigned int operationMask = 0u;  ///< Shared operation requirements.
  unsigned int nodeOffset = 0u;  ///< First node's index in the compiled node array.
  unsigned int nodeCount = 0u;  ///< Number of consecutive nodes, not allocated capacity.
};

/**
 * @brief Declared coverage of one CPU finder family, independent of any event's candidate yield.
 * A family with no required channels may be covered with zero implementations
 * when its channels are explicitly configuration-disabled. This record alone
 * is not evidence of numerical CPU/GPU equivalence or production qualification.
 */
struct KFParticleGpuGraphFamilyCoverage
{
  unsigned int family = KFGpuCpuFamilyCount;  ///< KFParticleGpuCpuFinderFamily value.
  unsigned int supportStatus = KFGpuGraphUnsupported;  ///< Aggregate family coverage status.
  unsigned int unsupportedReason = KFGpuGraphUnsupportedNotImplemented;  ///< Reason for incomplete coverage, or None.
  unsigned int implementedChannelCount = 0u;  ///< Declared implemented channels, not processed candidates.
};

/**
 * @brief Read-only, non-owning host/device view of compiled graph and coverage arrays.
 *
 * Copies only pointers, counts, and a revision token. It does not allocate or
 * transfer arrays, validate their contents, launch kernels, or synchronize GPU
 * work. FindNode() and FindFamily() perform linear searches and return borrowed
 * pointers; no event data or particle states are accessed.
 *
 * @pre Bind a view before use. Arrays must remain alive and readable in the
 * current execution memory space, with at least the supplied element counts.
 * The owner must coordinate transfers/mutations; replacing or releasing storage
 * invalidates the view and any pointers obtained from it.
 */
class KFParticleGpuDecayGraphView
{
 public:
  /// @brief Defaulted descriptor constructor; assign a bound view before accessing its members.
  KFPARTICLE_GPU_HOST_DEVICE KFParticleGpuDecayGraphView() = default;

  /**
   * @brief Binds existing graph arrays without copying their contents or checking their consistency.
   * @param nodes Compiled channel-node array.
   * @param nodeCount Number of valid nodes, not allocated capacity.
   * @param groups Execution-group array referring to ranges in @p nodes.
   * @param groupCount Number of valid execution groups.
   * @param families CPU finder family coverage array.
   * @param familyCount Number of valid family coverage records.
   * @param revision Caller-supplied graph identity token, normally the compiled graph content hash.
   * @note A null array pointer is usable only with a zero corresponding count and no element access.
   */
  KFPARTICLE_GPU_HOST_DEVICE KFParticleGpuDecayGraphView(
    const KFParticleGpuGraphNode* nodes,
    unsigned int nodeCount,
    const KFParticleGpuGraphExecutionGroup* groups,
    unsigned int groupCount,
    const KFParticleGpuGraphFamilyCoverage* families,
    unsigned int familyCount,
    unsigned long long revision)
    : fNodes(nodes)
    , fNodeCount(nodeCount)
    , fGroups(groups)
    , fGroupCount(groupCount)
    , fFamilies(families)
    , fFamilyCount(familyCount)
    , fRevision(revision)
  {
  }

  /// @brief Returns the borrowed read-only node array; indices must be below NodeCount().
  KFPARTICLE_GPU_HOST_DEVICE const KFParticleGpuGraphNode* Nodes() const { return fNodes; }

  /// @brief Returns the bound number of graph nodes, not a particle count or storage capacity.
  KFPARTICLE_GPU_HOST_DEVICE unsigned int NodeCount() const { return fNodeCount; }

  /// @brief Returns the borrowed execution-group array; indices must be below GroupCount().
  KFPARTICLE_GPU_HOST_DEVICE const KFParticleGpuGraphExecutionGroup* Groups() const
  {
    return fGroups;
  }

  /// @brief Returns the bound number of execution groups, not a measured kernel-launch count.
  KFPARTICLE_GPU_HOST_DEVICE unsigned int GroupCount() const { return fGroupCount; }

  /// @brief Returns the bound number of coverage records, not the number of supported families.
  KFPARTICLE_GPU_HOST_DEVICE unsigned int FamilyCount() const { return fFamilyCount; }

  /// @brief Returns the stored graph identity token without recomputing a hash or checking freshness.
  KFPARTICLE_GPU_HOST_DEVICE unsigned long long Revision() const { return fRevision; }

  /**
   * @brief Finds the first node with the exact stable channel ID using a linear scan.
   * @return Borrowed node pointer, or nullptr if absent; support status is not filtered.
   */
  KFPARTICLE_GPU_HOST_DEVICE const KFParticleGpuGraphNode* FindNode(
    unsigned int channelId) const
  {
    for (unsigned int index = 0u; index < fNodeCount; ++index) {
      if (fNodes[index].channelId == channelId) {
        return &fNodes[index];
      }
    }
    return nullptr;
  }

  /**
   * @brief Finds the first coverage record with the exact family value using a linear scan.
   * @return Borrowed coverage pointer, or nullptr if absent; the family argument is not validated.
   */
  KFPARTICLE_GPU_HOST_DEVICE const KFParticleGpuGraphFamilyCoverage* FindFamily(
    unsigned int family) const
  {
    for (unsigned int index = 0u; index < fFamilyCount; ++index) {
      if (fFamilies[index].family == family) {
        return &fFamilies[index];
      }
    }
    return nullptr;
  }

 private:
  const KFParticleGpuGraphNode* fNodes;  ///< Borrowed compiled node storage.
  unsigned int fNodeCount;  ///< Valid elements in fNodes.
  const KFParticleGpuGraphExecutionGroup* fGroups;  ///< Borrowed execution-group storage.
  unsigned int fGroupCount;  ///< Valid elements in fGroups.
  const KFParticleGpuGraphFamilyCoverage* fFamilies;  ///< Borrowed family coverage storage.
  unsigned int fFamilyCount;  ///< Valid elements in fFamilies.
  unsigned long long fRevision;  ///< Graph identity token supplied when the view was bound.
};

static_assert(std::is_trivially_copyable<KFParticleGpuGraphSource>::value,
              "KFParticle GPU graph sources must remain flat device values");
static_assert(std::is_trivially_copyable<KFParticleGpuGraphNode>::value,
              "KFParticle GPU graph nodes must remain flat device values");
static_assert(std::is_trivially_copyable<KFParticleGpuGraphExecutionGroup>::value,
              "KFParticle GPU graph groups must remain flat device values");
static_assert(std::is_trivially_copyable<KFParticleGpuGraphFamilyCoverage>::value,
              "KFParticle GPU graph coverage must remain flat device values");
static_assert(std::is_trivially_copyable<KFParticleGpuDecayGraphView>::value,
              "KFParticle GPU decay graph view must remain a flat non-owning ABI");
static_assert(std::is_trivially_default_constructible<KFParticleGpuDecayGraphView>::value,
              "KFParticle GPU decay graph view must not require dynamic initialization");

#endif
