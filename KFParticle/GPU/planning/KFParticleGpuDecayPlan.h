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
 * @brief Reusable channel settings, channel-result ranges, and the host-owned reconstruction plan.
 *
 * Channels describe hypotheses independently of event data. Conversion helpers
 * bind settings to one event; routing and graph compilation are separate steps.
 * The plan owns configuration values only, not tracks, candidate pools, or GPU resources.
 */

#ifndef KFPARTICLEGPUDECAYPLAN_H
#define KFPARTICLEGPUDECAYPLAN_H

#include "algorithms/KFParticleGpuGraphOperations.h"
#include "algorithms/KFParticleGpuV0Track.h"

#include <cstddef>
#include <vector>

/**
 * @brief Event-independent settings for constructing and selecting a two-track candidate.
 *
 * Track-set/species roles are resolved against each packed event. Input source
 * PDGs identify eligible tracks independently of the daughter mass hypotheses;
 * alternate source PDGs allow reinterpretation without duplicating track data.
 * The selection configuration and secondary-candidate cuts serve distinct stages.
 * Default fields are placeholders, not a complete validated physical channel.
 */
struct KFParticleGpuTwoDaughterChannel
{
  unsigned int channelId = 0;  ///< Stable channel ID; zero is unconfigured.
  KFParticleGpuTrackSet firstTrackSet = SecondaryPositiveFirst;   ///< First event track-set role.
  KFParticleGpuTrackSet secondTrackSet = SecondaryNegativeFirst;  ///< Second event track-set role.
  KFParticleGpuTrackSpecies firstSpecies = Pion;   ///< First species role; NumberOfTrackSpecies requests the full set.
  KFParticleGpuTrackSpecies secondSpecies = Pion;  ///< Second species role; NumberOfTrackSpecies requests the full set.
  unsigned int flags = KFGpuTwoDaughterUseLineDca | KFGpuTwoDaughterUseEnergyFit;  ///< Construction-mode bits.
  unsigned int outputClass = KFGpuGraphOutputPrimaryAndSecondary;  ///< Output processing category.
  int motherPdg = 0;          ///< Reconstructed mother hypothesis.
  int firstDaughterPdg = 0;   ///< First daughter hypothesis used in the fit.
  int secondDaughterPdg = 0;  ///< Second daughter hypothesis used in the fit.
  int firstSourcePdg = 0;            ///< First input PID; zero falls back to firstDaughterPdg.
  int firstAlternateSourcePdg = 0;   ///< Optional additional first-input PID, e.g. pion PID for a proton hypothesis.
  int secondSourcePdg = 0;           ///< Second input PID; zero falls back to secondDaughterPdg.
  int secondAlternateSourcePdg = 0;  ///< Optional additional second-input PID.
  int primaryVertexIndex = -1;  ///< Output PV index, -1 for unassigned, or KFGpuPrimaryVertexFromDaughters.
  float firstMass = 0.f;        ///< First daughter mass hypothesis in GeV/c^2.
  float secondMass = 0.f;       ///< Second daughter mass hypothesis in GeV/c^2.
  float motherMass = 0.f;       ///< Expected mother mass in GeV/c^2.
  float motherMassSigma = 0.f;  ///< Mother mass resolution for secondary-candidate selection.
  float secondaryMassSigmaCut = -1.f;     ///< Secondary mass-window width in resolution units.
  float maxSecondaryTopoChi2PerNdf = -1.f;  ///< Secondary topology chi2/NDF upper limit.
  float minSecondaryLdL = -1.f;  ///< Secondary-list decay-length significance cut, separate from selection below.
  KFParticleGpuV0SelectionConfig selection;  ///< General candidate selection settings, not copied into task sources.
  int transportMode = KFGpuTransportStraightLine;  ///< Transport model for fitting.
  int firstCharge = 0;   ///< Expected first-track charge; source-cut helper treats zero as unrestricted.
  int secondCharge = 0;  ///< Expected second-track charge; compiled routing additionally validates nonzero charges.
  int minFirstPixelHits = 0;   ///< Minimum first-track pixel hits; nonpositive values omit the cut.
  int minSecondPixelHits = 0;  ///< Minimum second-track pixel hits; nonpositive values omit the cut.
  float maxFirstChiToPrimaryVertex = -1.f;   ///< First-track chi upper limit; negative disables it.
  float maxSecondChiToPrimaryVertex = -1.f;  ///< Second-track chi upper limit; negative disables it.
  float minFirstChiToPrimaryVertex = -1.f;   ///< First-track chi lower cut; negative leaves the stage default.
  float minSecondChiToPrimaryVertex = -1.f;  ///< Second-track chi lower cut; negative leaves the stage default.
  float minFirstPt = -1.f;   ///< First-track transverse-momentum minimum; negative disables it.
  float minSecondPt = -1.f;  ///< Second-track transverse-momentum minimum; negative disables it.
  float maxDaughterDistance = -1.f;  ///< Maximum daughter separation; negative disables the distance cut.
};

/**
 * @brief Copies channel settings into a two-track task-source descriptor for one event.
 * @param channel Reusable channel configuration.
 * @param eventIndex Index of the packed event descriptor, not an external event ID.
 * @return Value descriptor with the event index; no input arrays are copied or ranges resolved here.
 * @note Does not validate settings or launch work. The separate channel.selection
 * configuration is not part of the task source and remains a selection-stage input.
 */
KFPARTICLE_GPU_HOST_DEVICE inline KFParticleGpuTwoDaughterTaskSource
MakeTwoDaughterTaskSource(const KFParticleGpuTwoDaughterChannel& channel,
                          unsigned int eventIndex)
{
  KFParticleGpuTwoDaughterTaskSource source;
  source.channelId = channel.channelId;
  source.eventIndex = eventIndex;
  source.firstTrackSet = channel.firstTrackSet;
  source.secondTrackSet = channel.secondTrackSet;
  source.firstSpecies = channel.firstSpecies;
  source.secondSpecies = channel.secondSpecies;
  source.flags = channel.flags;
  source.outputClass = channel.outputClass;
  source.motherPdg = channel.motherPdg;
  source.firstDaughterPdg = channel.firstDaughterPdg;
  source.secondDaughterPdg = channel.secondDaughterPdg;
  source.firstSourcePdg = channel.firstSourcePdg;
  source.firstAlternateSourcePdg = channel.firstAlternateSourcePdg;
  source.secondSourcePdg = channel.secondSourcePdg;
  source.secondAlternateSourcePdg = channel.secondAlternateSourcePdg;
  source.primaryVertexIndex = channel.primaryVertexIndex;
  source.firstMass = channel.firstMass;
  source.secondMass = channel.secondMass;
  source.motherMass = channel.motherMass;
  source.motherMassSigma = channel.motherMassSigma;
  source.secondaryMassSigmaCut = channel.secondaryMassSigmaCut;
  source.maxSecondaryTopoChi2PerNdf = channel.maxSecondaryTopoChi2PerNdf;
  source.minSecondaryLdL = channel.minSecondaryLdL;
  source.transportMode = channel.transportMode;
  source.firstCharge = channel.firstCharge;
  source.secondCharge = channel.secondCharge;
  source.minFirstPixelHits = channel.minFirstPixelHits;
  source.minSecondPixelHits = channel.minSecondPixelHits;
  source.maxFirstChiToPrimaryVertex = channel.maxFirstChiToPrimaryVertex;
  source.maxSecondChiToPrimaryVertex = channel.maxSecondChiToPrimaryVertex;
  source.minFirstChiToPrimaryVertex = channel.minFirstChiToPrimaryVertex;
  source.minSecondChiToPrimaryVertex = channel.minSecondChiToPrimaryVertex;
  source.minFirstPt = channel.minFirstPt;
  source.minSecondPt = channel.minSecondPt;
  source.maxDaughterDistance = channel.maxDaughterDistance;
  return source;
}

/**
 * @brief Contiguous candidate and flattened daughter-ID intervals with producer-reported overflow bits.
 *
 * Stores offsets/counts only, with no ownership or pointer validation. Arithmetic
 * uses unsigned indices without overflow checks; the producer must supply bounded
 * ranges whose ends are representable. Local-index helpers require membership.
 */
struct KFParticleGpuCandidateRange
{
  unsigned int offset = 0;          ///< First candidate index in its pool.
  unsigned int size = 0;            ///< Candidate count, not allocation capacity.
  unsigned int daughterOffset = 0;  ///< First flattened source-ID index in daughter storage.
  unsigned int daughterSize = 0;    ///< Number of flattened source IDs, not direct daughter references.
  unsigned int overflowFlags = 0;   ///< Overflow bits supplied by the producer.

  /// @brief Tests only candidate count; ignores daughter count and overflow flags.
  KFPARTICLE_GPU_HOST_DEVICE bool Empty() const { return size == 0u; }

  /// @brief Returns the exclusive candidate end; unsigned addition is not checked for overflow.
  KFPARTICLE_GPU_HOST_DEVICE unsigned int End() const { return offset + size; }

  /// @brief Tests membership in the half-open candidate range; assumes a representable End().
  KFPARTICLE_GPU_HOST_DEVICE bool ContainsCandidate(unsigned int index) const
  {
    return index >= offset && index < End();
  }

  /// @brief Subtracts the candidate offset; caller must first establish ContainsCandidate(index).
  KFPARTICLE_GPU_HOST_DEVICE unsigned int LocalCandidateIndex(unsigned int index) const
  {
    return index - offset;
  }

  /// @brief Returns the exclusive daughter-ID end; unsigned addition is not checked for overflow.
  KFPARTICLE_GPU_HOST_DEVICE unsigned int DaughterEnd() const
  {
    return daughterOffset + daughterSize;
  }

  /// @brief Tests membership in flattened daughter storage; assumes a representable DaughterEnd().
  KFPARTICLE_GPU_HOST_DEVICE bool ContainsDaughter(unsigned int index) const
  {
    return index >= daughterOffset && index < DaughterEnd();
  }

  /// @brief Subtracts the daughter offset without checking bounds; requires ContainsDaughter(index).
  KFPARTICLE_GPU_HOST_DEVICE unsigned int LocalDaughterIndex(unsigned int index) const
  {
    return index - daughterOffset;
  }

  /// @brief Returns whether any requested overflow bit is set, not whether every bit in the mask is set.
  KFPARTICLE_GPU_HOST_DEVICE bool HasOverflow(unsigned int mask) const
  {
    return (overflowFlags & mask) != 0u;
  }

  /// @brief Tests whether the producer reported any overflow bit.
  KFPARTICLE_GPU_HOST_DEVICE bool HasAnyOverflow() const { return overflowFlags != 0u; }
};

/**
 * @brief Per-event, per-channel accounting for two-track generation and construction.
 *
 * Fused construction interleaves channels in a shared generation range. Counts
 * identify this channel's yield, while candidate metadata determines membership;
 * generationCandidates must not be treated as a contiguous per-channel subset.
 * The candidates range is retained for isolated explicit-channel execution.
 */
struct KFParticleGpuTwoDaughterChannelResult
{
  unsigned int channelId = 0;    ///< Stable channel identity.
  int motherPdg = 0;             ///< Reconstructed mother hypothesis.
  unsigned int eventIndex = 0;   ///< Packed event descriptor index.
  unsigned int totalPairs = 0;   ///< Producer's pair count: raw pairs or routing visits, depending on execution path.
  unsigned int acceptedTasks = 0;  ///< Tasks passing generation cuts, including those lost to task capacity.
  unsigned int storedTasks = 0;    ///< Accepted tasks retained in task storage.
  unsigned int constructedCandidates = 0;  ///< Candidates attributed to this channel, before final selection.
  unsigned int constructedDaughters = 0;   ///< Reported flattened daughter-ID count for this channel.
  KFParticleGpuCandidateRange generationCandidates;  ///< Shared event-generation range; filter channel metadata.
  KFParticleGpuCandidateRange candidates;  ///< Explicit-channel range; fused output uses its overflow bits only.

  /// @brief Tests task truncation only; does not detect every construction or candidate-pool overflow.
  KFPARTICLE_GPU_HOST_DEVICE bool Truncated() const { return acceptedTasks > storedTasks; }

  /// @brief Requires both the constructed count and explicit-channel range to be empty; ignores shared generation size.
  KFPARTICLE_GPU_HOST_DEVICE bool Empty() const
  {
    return constructedCandidates == 0u && candidates.Empty();
  }

  /// @brief Tests any requested bit in candidates.overflowFlags; does not inspect generationCandidates directly.
  KFPARTICLE_GPU_HOST_DEVICE bool HasOverflow(unsigned int mask) const
  {
    return candidates.HasOverflow(mask);
  }

  /// @brief Tests the overflow bits carried by the explicit-range field, also populated by fused steering.
  KFPARTICLE_GPU_HOST_DEVICE bool HasAnyOverflow() const
  {
    return candidates.HasAnyOverflow();
  }

  /// @brief Tests only the explicit-channel range; cannot determine channel membership in fused output.
  KFPARTICLE_GPU_HOST_DEVICE bool ContainsCandidate(unsigned int index) const
  {
    return candidates.ContainsCandidate(index);
  }
};

/**
 * @brief Event-independent settings for a selected composite paired with one bachelor track.
 *
 * The historical V0 name includes later-generation composites. A stable parent
 * channel ID distinguishes equal-PDG reconstruction paths; zero retains PDG-only
 * source matching. Event-local bachelor ranges are resolved when making tasks.
 * Default fields alone do not define a validated physical channel.
 */
struct KFParticleGpuV0TrackCascadeChannel
{
  unsigned int channelId = 0u;  ///< Stable output channel ID.
  unsigned int family = KFGpuCpuFamilyTrackComposite;  ///< CPU finder family used in coverage accounting.
  unsigned int generation = 2u;  ///< Dependency generation of the produced composite.
  unsigned int outputClass = KFGpuGraphOutputSecondary;  ///< Output processing category.
  unsigned int parentChannelId = 0u;  ///< Stable parent channel ID; zero omits the explicit ID requirement.
  int v0Pdg = 0;  ///< Required parent composite hypothesis.
  KFParticleGpuTrackSet bachelorTrackSet = SecondaryNegativeFirst;  ///< Event-local bachelor track-set role.
  KFParticleGpuTrackSpecies bachelorSpecies = Pion;  ///< Bachelor species role within that track set.
  int bachelorPdg = 0;  ///< Required bachelor input hypothesis.
  int motherPdg = 0;    ///< Produced mother hypothesis.
  int primaryVertexIndex = -1;  ///< Bachelor/output PV requirement or a routed-path PV selection sentinel.
  unsigned int flags = KFGpuV0TrackUseLineDca | KFGpuV0TrackUseEnergyFit;  ///< Construction-mode bits.
  int transportMode = KFGpuTransportFullField;  ///< Parent/bachelor transport model.
  float bachelorMass = 0.f;       ///< Bachelor fit mass in GeV/c^2.
  float motherMass = 0.f;         ///< Expected mother mass in GeV/c^2.
  float motherMassSigma = -1.f;   ///< Mother mass resolution for selection.
  float secondaryMassSigmaCut = -1.f;      ///< Secondary mass-window width in resolution units.
  float maxSecondaryTopoChi2PerNdf = -1.f;  ///< Secondary topology chi2/NDF upper limit.
  float minBachelorChiToPrimaryVertex = -1.f;  ///< Bachelor chi lower cut; negative omits it.
  float maxV0TrackDistance = -1.f;  ///< Composite-to-bachelor distance limit; negative omits it.
  float minBachelorPt = -1.f;      ///< Bachelor transverse-momentum minimum; negative omits it.
  int minBachelorPixelHits = -1;   ///< Minimum bachelor pixel hits; nonpositive values omit the cut.
  float parentMassConstraint = -1.f;      ///< Parent mass constraint in GeV/c^2; negative omits it.
  float parentMassConstraintSigma = 0.f;  ///< Constraint uncertainty; zero requests an exact mass constraint.
};

/**
 * @brief Binds cascade settings to one event and resolves its bachelor input range.
 * @param channel Reusable composite-plus-track configuration.
 * @param event Event descriptor containing valid track-set/species ranges.
 * @param eventIndex Packed event descriptor index corresponding to @p event.
 * @return Value descriptor containing the resolved range; no track or candidate data is copied.
 * @note A full-set species sentinel or empty species range falls back to the whole
 * bachelor track set. Family/generation remain plan metadata and are not copied into this descriptor.
 * Performs no consistency validation, GPU transfer, or kernel launch.
 */
KFPARTICLE_GPU_HOST_DEVICE inline KFParticleGpuV0TrackChannel
MakeV0TrackTaskChannel(const KFParticleGpuV0TrackCascadeChannel& channel,
                       const KFParticleGpuEventDesc& event,
                       unsigned int eventIndex)
{
  KFParticleGpuV0TrackChannel source = KFParticleGpuV0Track::MakeChannel(
    channel.channelId,
    channel.v0Pdg,
    channel.bachelorPdg,
    channel.motherPdg,
    eventIndex,
    ResolveTaskSourceRange(event, channel.bachelorTrackSet, channel.bachelorSpecies),
    channel.bachelorMass,
    channel.motherMass);
  source.parentChannelId = channel.parentChannelId;
  source.outputClass = channel.outputClass;
  source.primaryVertexIndex = channel.primaryVertexIndex;
  source.flags = channel.flags;
  source.transportMode = channel.transportMode;
  source.motherMassSigma = channel.motherMassSigma;
  source.secondaryMassSigmaCut = channel.secondaryMassSigmaCut;
  source.maxSecondaryTopoChi2PerNdf = channel.maxSecondaryTopoChi2PerNdf;
  source.minBachelorChiToPrimaryVertex = channel.minBachelorChiToPrimaryVertex;
  source.maxV0TrackDistance = channel.maxV0TrackDistance;
  source.minBachelorPt = channel.minBachelorPt;
  source.minBachelorPixelHits = channel.minBachelorPixelHits;
  source.parentMassConstraint = channel.parentMassConstraint;
  source.parentMassConstraintSigma = channel.parentMassConstraintSigma;
  return source;
}

/**
 * @brief Per-event accounting for one composite-plus-track channel.
 *
 * Fused output shares an unordered event/stage range across channels; membership
 * is identified by candidate metadata. Only explicit-channel execution supplies
 * a contiguous channel-specific candidates range. Neither range owns storage.
 */
struct KFParticleGpuV0TrackChannelResult
{
  unsigned int channelId = 0u;  ///< Stable channel identity.
  int motherPdg = 0;            ///< Produced mother hypothesis.
  unsigned int eventIndex = 0u;  ///< Packed event descriptor index.
  unsigned int totalPairs = 0u;  ///< Raw or routing-visited pairs, according to the producing execution path.
  unsigned int acceptedTasks = 0u;  ///< Tasks passing generation cuts, including task-capacity losses.
  unsigned int storedTasks = 0u;    ///< Accepted tasks retained in task storage.
  unsigned int constructedCandidates = 0u;  ///< Reported candidate count for this channel.
  unsigned int constructedDaughters = 0u;   ///< Reported lineage count; fused steering may infer it from generation.
  KFParticleGpuCandidateRange generationCandidates;  ///< Shared fused event/stage range, not a per-channel interval.
  KFParticleGpuCandidateRange candidates;  ///< Explicit-channel interval; fused output carries overflow bits here.

  /// @brief Tests only accepted-versus-stored task counts, not candidate-pool overflow.
  KFPARTICLE_GPU_HOST_DEVICE bool Truncated() const { return acceptedTasks > storedTasks; }

  /// @brief Uses a nonzero constructed count, otherwise the explicit range size; does not sum or use the shared range.
  KFPARTICLE_GPU_HOST_DEVICE unsigned int CandidateCount() const
  {
    return constructedCandidates != 0u ? constructedCandidates : candidates.size;
  }

  /// @brief Tests CandidateCount() only; overflow flags do not affect emptiness.
  KFPARTICLE_GPU_HOST_DEVICE bool Empty() const { return CandidateCount() == 0u; }
};

/** @brief Established stable IDs for the default neutral two-track channels; these are not PDG codes. */
enum KFParticleGpuDefaultChannelId
{
  KFGpuChannelK0ShortToPiPlusPiMinus = 1u,        ///< K0S to positive and negative pions.
  KFGpuChannelLambdaToProtonPiMinus = 2u,         ///< Lambda to proton and negative pion.
  KFGpuChannelAntiLambdaToAntiProtonPiPlus = 3u   ///< Anti-Lambda to antiproton and positive pion.
};

/** @brief Established stable IDs for default Xi/Omega cascade channels and their conjugates. */
enum KFParticleGpuDefaultCascadeChannelId
{
  KFGpuChannelXiMinusToLambdaPiMinus = 11u,       ///< Xi- from Lambda and pi-.
  KFGpuChannelAntiXiPlusToAntiLambdaPiPlus = 12u,  ///< Anti-Xi+ from anti-Lambda and pi+.
  KFGpuChannelOmegaMinusToLambdaKMinus = 13u,     ///< Omega- from Lambda and K-.
  KFGpuChannelAntiOmegaPlusToAntiLambdaKPlus = 14u  ///< Anti-Omega+ from anti-Lambda and K+.
};

/**
 * @brief IDs for representative channel/operation fixtures, separate from the full CPU catalogue's ID mapping.
 * Names describe test hypotheses or probes; this enum does not register them in a decay plan.
 */
enum KFParticleGpuRepresentativeChargedChannelId
{
  KFGpuChannelRho0ToPiPlusPiMinus = 21u,           ///< Opposite-sign pion-pair fixture.
  KFGpuChannelDeltaPlusPlusToProtonPiPlus = 22u,   ///< Same-sign proton-pion fixture.
  KFGpuChannelD0ToPiPlusKMinus = 23u,             ///< Pion-kaon two-track fixture.
  KFGpuChannelDPlusToD0PiPlus = 24u,              ///< Charm composite-plus-track fixture.
  KFGpuChannelHypertritonToDeuteronPiMinus = 25u,  ///< Nuclear daughter plus pion fixture.
  KFGpuChannelCompositeCompositeProbe = 26u,     ///< Composite-pair operation probe.
  KFGpuChannelNeutralMissingMassProbe = 27u,     ///< Missing-mass operation probe.
  KFGpuChannelUnaryFinalProbe = 28u              ///< Unary final-processing probe.
};

/**
 * @brief One graph-operation configuration combining dependency metadata and numerical settings.
 *
 * Source dependencies live in the node, while fit and selection parameters
 * live in the descriptor. Their channel IDs must agree and be nonzero when
 * appended to a plan. Other consistency checks belong to graph/descriptor validation.
 * The descriptor is a configuration value, not a candidate or an allocated device task.
 */
struct KFParticleGpuGraphOperationChannel
{
  KFParticleGpuGraphNode node;  ///< Channel identity, dependencies, topology, and scheduling metadata.
  KFParticleGpuGraphOperationDescriptor descriptor;  ///< Fit, constraint, matching, and selection settings.
};

/**
 * @brief Owns the host configuration of two-track, cascade, and generic graph-operation channels.
 *
 * Each category preserves insertion order. This container does not sort or
 * execute dependencies, generate tasks, allocate device buffers, or transfer
 * event data. Routing/graph compilers perform the deeper consistency checks.
 *
 * Add methods copy values and increment an object-local revision; Clear() also
 * increments it. The revision supports cache invalidation but is not a content
 * hash or a globally unique plan identity. Element references follow vector
 * lifetime rules: appending may invalidate them, and Clear() destroys them.
 *
 * Copying is disabled. Callers must serialize mutations and keep the plan unchanged
 * while consumers depend on its channel references or compiled indices.
 */
class KFParticleGpuDecayPlan
{
 public:
  /// @brief Creates an empty plan with revision one and no registered default channels.
  KFParticleGpuDecayPlan();

  /// @brief Destroys owned channel vectors; does not own or synchronize GPU resources.
  ~KFParticleGpuDecayPlan();

  /// @brief Clears all three channel lists, retaining vector capacity, and increments revision even if already empty.
  void Clear();

  /// @brief Returns true only when all three channel categories are empty.
  bool Empty() const;

  /// @brief Returns the local mutation counter; no content hashing or compiled-plan freshness check is performed.
  unsigned long long Revision() const;

  /// @brief Appends a two-track channel by value and increments revision; performs no validation or deduplication.
  void AddTwoDaughterChannel(const KFParticleGpuTwoDaughterChannel& channel);

  /// @brief Returns the number of declared two-track channels, not candidates or allocated capacity.
  std::size_t NumberOfTwoDaughterChannels() const;

  /**
   * @brief Returns a read-only reference to a two-track channel in insertion order.
   * @throws std::out_of_range If the index is outside the channel list.
   */
  const KFParticleGpuTwoDaughterChannel& TwoDaughterChannel(std::size_t index) const;

  /// @brief Appends a cascade by value and increments revision; does not validate or add parent dependencies.
  void AddV0TrackCascadeChannel(const KFParticleGpuV0TrackCascadeChannel& channel);

  /// @brief Returns the number of declared composite-plus-track channels.
  std::size_t NumberOfV0TrackCascadeChannels() const;

  /**
   * @brief Returns a read-only cascade reference in insertion order.
   * @throws std::out_of_range If the index is outside the cascade list.
   */
  const KFParticleGpuV0TrackCascadeChannel& V0TrackCascadeChannel(std::size_t index) const;

  /**
   * @brief Appends a graph operation by value and increments revision after checking shared channel identity.
   * @throws std::invalid_argument If the node ID is zero or differs from the descriptor ID.
   * @note Does not check duplicate IDs, dependencies, topology, or numerical settings.
   */
  void AddGraphOperationChannel(const KFParticleGpuGraphOperationChannel& channel);

  /// @brief Returns the number of declared generic graph operations.
  std::size_t NumberOfGraphOperationChannels() const;

  /**
   * @brief Returns a read-only graph-operation reference in insertion order.
   * @throws std::out_of_range If the index is outside the graph-operation list.
   */
  const KFParticleGpuGraphOperationChannel& GraphOperationChannel(
    std::size_t index) const;

 private:
  /// @brief Unimplemented private copy constructor prevents copying the plan.
  KFParticleGpuDecayPlan(const KFParticleGpuDecayPlan&);

  /// @brief Unimplemented private copy assignment prevents replacing the plan by copying.
  KFParticleGpuDecayPlan& operator=(const KFParticleGpuDecayPlan&);

  std::vector<KFParticleGpuTwoDaughterChannel> fTwoDaughterChannels;  ///< Owned raw two-track channel settings.
  std::vector<KFParticleGpuV0TrackCascadeChannel> fV0TrackCascadeChannels;  ///< Owned composite-plus-track settings.
  std::vector<KFParticleGpuGraphOperationChannel> fGraphOperationChannels;  ///< Owned generic graph operations.
  unsigned long long fRevision;  ///< Local unsigned mutation counter, initialized to one.
};

/**
 * @brief Creates the default secondary K0S to pi+ pi- channel with full-field transport and reversed fit order.
 * @param channelId Stable ID to assign; uniqueness and nonzero value are not checked here.
 * @note Mass resolution and secondary topology defaults depend on detector build macros.
 */
KFParticleGpuTwoDaughterChannel MakeK0ShortToPiPlusPiMinusChannel(
  unsigned int channelId = KFGpuChannelK0ShortToPiPlusPiMinus);

/**
 * @brief Creates the default secondary Lambda to p pi- channel with full-field transport and reversed fit order.
 * @param channelId Stable ID to assign without validation.
 * @note Positive input tracks may have proton or pion PID; both are fitted with the proton mass.
 * The first species role requests the full positive set to preserve this CPU-compatible fallback.
 */
KFParticleGpuTwoDaughterChannel MakeLambdaToProtonPiMinusChannel(
  unsigned int channelId = KFGpuChannelLambdaToProtonPiMinus);

/**
 * @brief Creates the default secondary anti-Lambda to antiproton pi+ channel with full-field transport.
 * @param channelId Stable ID to assign without validation.
 * @note Uses the negative proton species range; does not add the Lambda helper's pion-PID fallback.
 */
KFParticleGpuTwoDaughterChannel MakeAntiLambdaToAntiProtonPiPlusChannel(
  unsigned int channelId = KFGpuChannelAntiLambdaToAntiProtonPiPlus);

/**
 * @brief Appends default K0S, Lambda, and anti-Lambda channels in that order.
 * @note Does not clear or deduplicate the plan.
 */
void AddDefaultV0TwoDaughterChannels(KFParticleGpuDecayPlan& plan);

/**
 * @brief Appends default Xi-, anti-Xi+, Omega-, and anti-Omega+ cascade configurations in that order.
 * @note Retains existing channels and uses the default Lambda/anti-Lambda parent IDs.
 * Does not add or verify those parent channels, deduplicate IDs, or register the full CPU catalogue.
 */
void AddDefaultV0TrackCascadeChannels(KFParticleGpuDecayPlan& plan);

#endif
