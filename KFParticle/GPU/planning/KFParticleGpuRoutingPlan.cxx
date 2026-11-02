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
 * @brief Host-side compilation of decay channels into routing descriptors and compatibility tables.
 *
 * Implements the two routing-plan classes declared in planning/KFParticleGpuDecayGraphPlan.h.
 * Each source channel receives a transient mask bit. Channels sharing input geometry
 * are grouped so reconstruction can consider their hypotheses together. These tables
 * describe allowed combinations; they neither enumerate event tracks nor fit particles,
 * apply physical selection cuts, upload data, or launch kernels.
 */

#include "planning/KFParticleGpuDecayGraphPlan.h"

#include "planning/KFParticleGpuDecayPlan.h"

#include <cmath>
#include <stdexcept>

namespace
{
  /**
   * @brief Map a signed source PDG code to the supported track species, ignoring its sign.
   * Includes the CPU convention |PDG| = 19 for muons and the listed light nuclei.
   * Returns NumberOfTrackSpecies for an unrecognized code; this is not a general PDG decoder.
   * @pre pdg is not the minimum representable int, whose negation would overflow.
   */
  KFParticleGpuTrackSpecies ExpectedTrackSpecies(int pdg)
  {
    const int absolutePdg = pdg < 0 ? -pdg : pdg;
    switch (absolutePdg) {
      case 11: return Electron;
      case 13:
      case 19: return Muon;
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
      default: return NumberOfTrackSpecies;
    }
  }

  /**
   * @brief Identify the four positive-charge track sets, for primary/secondary and first/last states.
   * Returns false for every other value; validate the set separately before inferring a negative sign.
   */
  bool IsPositiveTrackSet(KFParticleGpuTrackSet set)
  {
    return set == SecondaryPositiveFirst || set == PrimaryPositiveFirst
           || set == SecondaryPositiveLast || set == PrimaryPositiveLast;
  }

  /**
   * @brief Infer charge sign using the supported charged-source PDG conventions.
   * Electrons and muons (including |PDG| = 19) have the opposite sign to their PDG code;
   * other codes use the PDG sign. Does not validate the code or determine charge magnitude.
   * @pre pdg is not the minimum representable int.
   */
  bool HasPositiveCharge(int pdg)
  {
    const int absolutePdg = pdg < 0 ? -pdg : pdg;
    if (absolutePdg == 11 || absolutePdg == 13 || absolutePdg == 19) {
      return pdg < 0;
    }
    return pdg > 0;
  }

  /**
   * @brief Copy a composite-track hypothesis into a flat descriptor with its assigned bit and roles.
   * Preserves source identity, fit settings, cuts, and parent constraints; performs no validation.
   */
  KFParticleGpuV0TrackRoutingDescriptor MakeRoutingDescriptor(
    const KFParticleGpuV0TrackCascadeChannel& channel,
    unsigned int channelBit,
    unsigned int selectedV0Role,
    unsigned int bachelorRole)
  {
    KFParticleGpuV0TrackRoutingDescriptor descriptor;
    descriptor.channelBit = channelBit;
    descriptor.channelId = channel.channelId;
    descriptor.parentChannelId = channel.parentChannelId;
    descriptor.family = channel.family;
    descriptor.generation = channel.generation;
    descriptor.outputClass = channel.outputClass;
    descriptor.selectedV0Role = selectedV0Role;
    descriptor.bachelorRole = bachelorRole;
    descriptor.bachelorTrackSet = channel.bachelorTrackSet;
    descriptor.bachelorSpecies = channel.bachelorSpecies;
    descriptor.v0Pdg = channel.v0Pdg;
    descriptor.bachelorPdg = channel.bachelorPdg;
    descriptor.motherPdg = channel.motherPdg;
    descriptor.primaryVertexIndex = channel.primaryVertexIndex;
    descriptor.flags = channel.flags;
    descriptor.transportMode = channel.transportMode;
    descriptor.bachelorMass = channel.bachelorMass;
    descriptor.motherMass = channel.motherMass;
    descriptor.motherMassSigma = channel.motherMassSigma;
    descriptor.secondaryMassSigmaCut = channel.secondaryMassSigmaCut;
    descriptor.maxSecondaryTopoChi2PerNdf = channel.maxSecondaryTopoChi2PerNdf;
    descriptor.minBachelorChiToPrimaryVertex = channel.minBachelorChiToPrimaryVertex;
    descriptor.maxV0TrackDistance = channel.maxV0TrackDistance;
    descriptor.minBachelorPt = channel.minBachelorPt;
    descriptor.minBachelorPixelHits = channel.minBachelorPixelHits;
    descriptor.parentMassConstraint = channel.parentMassConstraint;
    descriptor.parentMassConstraintSigma = channel.parentMassConstraintSigma;
    return descriptor;
  }

  /**
   * @brief Check that the track-set index is below NumberOfTrackSets, excluding the sentinel.
   */
  bool IsValidTrackSet(KFParticleGpuTrackSet set)
  {
    return static_cast<unsigned int>(set) < NumberOfTrackSets;
  }

  /**
   * @brief Accept a species index or the NumberOfTrackSpecies wildcard used by two-track channels.
   * Composite-track compilation additionally requires an exact recognized bachelor species.
   */
  bool IsValidSpecies(KFParticleGpuTrackSpecies species)
  {
    return static_cast<unsigned int>(species) <= NumberOfTrackSpecies;
  }

  /**
   * @brief Check finiteness of the eight scalar selection settings and a supported topology mode.
   * Accepts spatial or line topology. Does not require positive cuts or validate their physical meaning.
   */
  bool IsValidSelection(const KFParticleGpuV0SelectionConfig& selection)
  {
    return std::isfinite(selection.expectedMass)
           && std::isfinite(selection.expectedMassSigma)
           && std::isfinite(selection.massSigmaCut)
           && std::isfinite(selection.maxGeometricChi2PerNdf)
           && std::isfinite(selection.maxPrimaryVertexDistance)
           && std::isfinite(selection.minSecondaryLdL)
           && std::isfinite(selection.maxPrimaryTopologyChi2PerNdf)
           && std::isfinite(selection.maxSecondaryTopologyChi2PerNdf)
           && (selection.topologyMode == KFGpuV0TopologySpatial
               || selection.topologyMode == KFGpuV0TopologyLine);
  }

  /**
   * @brief Check finiteness of two-track masses, uncertainties, and scalar geometric/track cuts.
   * Does not check signs, integer settings, transport mode, or the embedded selection configuration;
   * Compile() performs the other supported checks separately.
   */
  bool HasFiniteTwoDaughterNumerics(const KFParticleGpuTwoDaughterChannel& channel)
  {
    return std::isfinite(channel.firstMass)
           && std::isfinite(channel.secondMass)
           && std::isfinite(channel.motherMass)
           && std::isfinite(channel.motherMassSigma)
           && std::isfinite(channel.secondaryMassSigmaCut)
           && std::isfinite(channel.maxSecondaryTopoChi2PerNdf)
           && std::isfinite(channel.minSecondaryLdL)
           && std::isfinite(channel.maxFirstChiToPrimaryVertex)
           && std::isfinite(channel.maxSecondChiToPrimaryVertex)
           && std::isfinite(channel.minFirstChiToPrimaryVertex)
           && std::isfinite(channel.minSecondChiToPrimaryVertex)
           && std::isfinite(channel.minFirstPt)
           && std::isfinite(channel.minSecondPt)
           && std::isfinite(channel.maxDaughterDistance);
  }

  /**
   * @brief Copy a two-track hypothesis into a flat descriptor with its assigned bit and source roles.
   * Retains both fit daughter PDGs and source/alternate PDGs. In particular, a zero source-PDG
   * fallback remains zero in the descriptor; the supplied roles already reflect its resolution.
   * Performs no validation or fitting.
   */
  KFParticleGpuTwoDaughterRoutingDescriptor MakeTwoDaughterRoutingDescriptor(
    const KFParticleGpuTwoDaughterChannel& channel,
    unsigned int channelBit,
    unsigned int firstRole,
    unsigned int secondRole)
  {
    KFParticleGpuTwoDaughterRoutingDescriptor descriptor;
    descriptor.channelBit = channelBit;
    descriptor.channelId = channel.channelId;
    descriptor.firstRole = firstRole;
    descriptor.secondRole = secondRole;
    descriptor.firstTrackSet = channel.firstTrackSet;
    descriptor.secondTrackSet = channel.secondTrackSet;
    descriptor.firstSpecies = channel.firstSpecies;
    descriptor.secondSpecies = channel.secondSpecies;
    descriptor.flags = channel.flags;
    descriptor.outputClass = channel.outputClass;
    descriptor.motherPdg = channel.motherPdg;
    descriptor.firstDaughterPdg = channel.firstDaughterPdg;
    descriptor.secondDaughterPdg = channel.secondDaughterPdg;
    descriptor.firstSourcePdg = channel.firstSourcePdg;
    descriptor.firstAlternateSourcePdg = channel.firstAlternateSourcePdg;
    descriptor.secondSourcePdg = channel.secondSourcePdg;
    descriptor.secondAlternateSourcePdg = channel.secondAlternateSourcePdg;
    descriptor.primaryVertexIndex = channel.primaryVertexIndex;
    descriptor.firstMass = channel.firstMass;
    descriptor.secondMass = channel.secondMass;
    descriptor.motherMass = channel.motherMass;
    descriptor.motherMassSigma = channel.motherMassSigma;
    descriptor.secondaryMassSigmaCut = channel.secondaryMassSigmaCut;
    descriptor.maxSecondaryTopoChi2PerNdf = channel.maxSecondaryTopoChi2PerNdf;
    descriptor.minSecondaryLdL = channel.minSecondaryLdL;
    descriptor.selection = channel.selection;
    descriptor.transportMode = channel.transportMode;
    descriptor.firstCharge = channel.firstCharge;
    descriptor.secondCharge = channel.secondCharge;
    descriptor.minFirstPixelHits = channel.minFirstPixelHits;
    descriptor.minSecondPixelHits = channel.minSecondPixelHits;
    descriptor.maxFirstChiToPrimaryVertex = channel.maxFirstChiToPrimaryVertex;
    descriptor.maxSecondChiToPrimaryVertex = channel.maxSecondChiToPrimaryVertex;
    descriptor.minFirstChiToPrimaryVertex = channel.minFirstChiToPrimaryVertex;
    descriptor.minSecondChiToPrimaryVertex = channel.minSecondChiToPrimaryVertex;
    descriptor.minFirstPt = channel.minFirstPt;
    descriptor.minSecondPt = channel.minSecondPt;
    descriptor.maxDaughterDistance = channel.maxDaughterDistance;
    return descriptor;
  }
}

/**
 * @brief Discard composite-track tables and enabled bits, resetting the source revision to zero.
 * Vector capacities are retained for subsequent compilation; existing element references become invalid.
 */
void KFParticleGpuV0TrackRoutingPlan::Clear()
{
  fSourceRevision = 0u;
  fDescriptors.clear();
  fCompatibility.clear();
  fGroups.clear();
  fEnabledChannels.Clear();
}

/**
 * @brief Rebuild composite-track routing from source channels, enabling all compiled hypotheses.
 *
 * Assigns channel bits in source order. Compatibility entries merge identical
 * (parent channel, selected PDG, bachelor PDG) triples; execution groups share a bachelor
 * track set and dependency generation. Group role/species masks summarize their channels.
 * Checks channel IDs and bachelor species/charge geometry, but not parent graph dependencies
 * or all numerical fit settings. The source revision is recorded only after successful completion.
 *
 * @throws std::length_error If the channel count exceeds the mask capacity.
 * @throws std::invalid_argument If a checked channel identity or source configuration is invalid.
 * @note Clears the old result first. Failure may leave partial tables with revision zero;
 * use only a successfully compiled plan. Allocation failures may also propagate.
 */
void KFParticleGpuV0TrackRoutingPlan::Compile(const KFParticleGpuDecayPlan& plan)
{
  Clear();
  const std::size_t channelCount = plan.NumberOfV0TrackCascadeChannels();
  if (channelCount > KFParticleGpuChannelMask::BitCapacity) {
    throw std::length_error("KFParticle GPU routing plan exceeds channel-mask capacity");
  }

  fDescriptors.reserve(channelCount);
  fCompatibility.reserve(channelCount);
  fGroups.reserve(channelCount);

  for (std::size_t index = 0u; index < channelCount; ++index) {
    const KFParticleGpuV0TrackCascadeChannel& channel = plan.V0TrackCascadeChannel(index);
    if (channel.channelId == 0u) {
      throw std::invalid_argument("KFParticle GPU routing channel ID must be nonzero");
    }
    for (const KFParticleGpuV0TrackRoutingDescriptor& existing : fDescriptors) {
      if (existing.channelId == channel.channelId) {
        throw std::invalid_argument("KFParticle GPU routing channel IDs must be unique");
      }
    }

    const unsigned int selectedV0Role =
      static_cast<unsigned int>(KFParticleGpuClassifySelectedV0Role(channel.v0Pdg));
    const unsigned int bachelorRole =
      static_cast<unsigned int>(KFParticleGpuClassifyBachelorRole(channel.bachelorPdg));
    if (!IsValidTrackSet(channel.bachelorTrackSet)
        || !IsValidSpecies(channel.bachelorSpecies)
        || channel.v0Pdg == 0 || channel.bachelorPdg == 0
        || ExpectedTrackSpecies(channel.bachelorPdg) == NumberOfTrackSpecies) {
      throw std::invalid_argument(
        "KFParticle GPU routing channel has invalid charged source geometry");
    }
    if (channel.bachelorSpecies != ExpectedTrackSpecies(channel.bachelorPdg)) {
      throw std::invalid_argument("KFParticle GPU routing bachelor species and PDG disagree");
    }
    if (IsPositiveTrackSet(channel.bachelorTrackSet)
        != HasPositiveCharge(channel.bachelorPdg)) {
      throw std::invalid_argument("KFParticle GPU routing bachelor track set has the wrong sign");
    }

    const unsigned int channelBit = static_cast<unsigned int>(index);
    fDescriptors.push_back(
      MakeRoutingDescriptor(channel, channelBit, selectedV0Role, bachelorRole));
    fEnabledChannels.Set(channelBit);

    KFParticleGpuV0TrackCompatibilityEntry* compatibility = nullptr;
    for (KFParticleGpuV0TrackCompatibilityEntry& entry : fCompatibility) {
      if (entry.parentChannelId == channel.parentChannelId
          && entry.selectedPdg == channel.v0Pdg
          && entry.bachelorPdg == channel.bachelorPdg) {
        compatibility = &entry;
        break;
      }
    }
    if (!compatibility) {
      KFParticleGpuV0TrackCompatibilityEntry entry;
      entry.parentChannelId = channel.parentChannelId;
      entry.selectedPdg = channel.v0Pdg;
      entry.bachelorPdg = channel.bachelorPdg;
      entry.selectedV0Role = selectedV0Role;
      entry.bachelorRole = bachelorRole;
      fCompatibility.push_back(entry);
      compatibility = &fCompatibility.back();
    }
    compatibility->channels.Set(channelBit);

    KFParticleGpuV0TrackExecutionGroup* group = nullptr;
    for (KFParticleGpuV0TrackExecutionGroup& entry : fGroups) {
      if (entry.bachelorTrackSet == channel.bachelorTrackSet
          && entry.generation == channel.generation) {
        group = &entry;
        break;
      }
    }
    if (!group) {
      KFParticleGpuV0TrackExecutionGroup entry;
      entry.groupIndex = static_cast<unsigned int>(fGroups.size());
      entry.generation = channel.generation;
      entry.bachelorTrackSet = channel.bachelorTrackSet;
      entry.selectedV0Roles = KFParticleGpuRoutingRoleBit(selectedV0Role);
      fGroups.push_back(entry);
      group = &fGroups.back();
    }
    if (selectedV0Role != KFGpuSelectedV0RoleInvalid) {
      group->selectedV0Roles |= KFParticleGpuRoutingRoleBit(selectedV0Role);
    }
    if (bachelorRole != KFGpuBachelorRoleInvalid) {
      group->bachelorRoles |= KFParticleGpuRoutingRoleBit(bachelorRole);
    }
    group->bachelorSpecies |= 1u << static_cast<unsigned int>(channel.bachelorSpecies);
    group->channels.Set(channelBit);
  }

  fSourceRevision = plan.Revision();
}

/**
 * @brief Union enabled composite-track channels whose stored coarse roles match both inputs.
 * Scans the compatibility table without checking exact PDGs, parent IDs, event membership,
 * or physical cuts. No matching entry yields an empty mask.
 */
KFParticleGpuChannelMask KFParticleGpuV0TrackRoutingPlan::CompatibleChannels(
  unsigned int selectedV0Role,
  unsigned int bachelorRole) const
{
  KFParticleGpuChannelMask result;
  for (const KFParticleGpuV0TrackCompatibilityEntry& entry : fCompatibility) {
    if (entry.selectedV0Role == selectedV0Role && entry.bachelorRole == bachelorRole) {
      result.UnionWith(entry.channels);
    }
  }
  return result.Intersected(fEnabledChannels);
}

/**
 * @brief Union enabled channels matching the selected/bachelor signed PDGs and parent requirement.
 * A stored parentChannelId of zero accepts any supplied parent ID; a supplied zero is not
 * itself a wildcard. Performs a linear table lookup, not event validation or physical selection.
 */
KFParticleGpuChannelMask KFParticleGpuV0TrackRoutingPlan::CompatibleChannelsByPdg(
  unsigned int parentChannelId,
  int selectedPdg,
  int bachelorPdg) const
{
  KFParticleGpuChannelMask result;
  for (const KFParticleGpuV0TrackCompatibilityEntry& entry : fCompatibility) {
    if ((entry.parentChannelId == 0u || entry.parentChannelId == parentChannelId)
        && entry.selectedPdg == selectedPdg
        && entry.bachelorPdg == bachelorPdg) {
      result.UnionWith(entry.channels);
    }
  }
  return result.Intersected(fEnabledChannels);
}

/**
 * @brief Discard two-track tables and enabled bits, resetting the source revision to zero.
 * Vector capacities are retained for subsequent compilation; existing element references become invalid.
 */
void KFParticleGpuTwoDaughterRoutingPlan::Clear()
{
  fSourceRevision = 0u;
  fDescriptors.clear();
  fCompatibility.clear();
  fGroups.clear();
  fEnabledChannels.Clear();
}

/**
 * @brief Rebuild two-track routing, including alternate source hypotheses and shared input-set groups.
 *
 * Resolves zero source PDGs to fit daughter PDGs for routing. Compatibility entries cover
 * the ordered combinations of each side's source and nonzero alternate PDG; repeated pairs
 * share a channel mask. Execution groups share the ordered pair of track sets and accumulate
 * role/species masks from the source hypotheses. Channel bits follow source-plan order.
 *
 * Validates IDs, track sets, output-class bounds, recognized main source species, configured
 * species compatibility, daughter charge signs, transport mode, and scalar finiteness.
 * These checks do not establish physical consistency of every setting or positivity of cuts.
 * All compiled bits are enabled; the source revision is recorded only on success.
 *
 * @throws std::length_error If the channel count exceeds the mask capacity.
 * @throws std::invalid_argument If a checked source, identity, or numerical setting is invalid.
 * @note Clears the old result first. Failure may leave partial tables with revision zero;
 * use only a successfully compiled plan. Allocation failures may also propagate.
 */
void KFParticleGpuTwoDaughterRoutingPlan::Compile(const KFParticleGpuDecayPlan& plan)
{
  Clear();
  const std::size_t channelCount = plan.NumberOfTwoDaughterChannels();
  if (channelCount > KFParticleGpuChannelMask::BitCapacity) {
    throw std::length_error(
      "KFParticle GPU two-daughter routing plan exceeds channel-mask capacity");
  }

  fDescriptors.reserve(channelCount);
  fCompatibility.reserve(channelCount);
  fGroups.reserve(channelCount);

  for (std::size_t index = 0u; index < channelCount; ++index) {
    const KFParticleGpuTwoDaughterChannel& channel = plan.TwoDaughterChannel(index);
    if (channel.channelId == 0u) {
      throw std::invalid_argument(
        "KFParticle GPU two-daughter routing channel ID must be nonzero");
    }
    for (const KFParticleGpuTwoDaughterRoutingDescriptor& existing : fDescriptors) {
      if (existing.channelId == channel.channelId) {
        throw std::invalid_argument(
          "KFParticle GPU two-daughter routing channel IDs must be unique");
      }
    }

    const int firstSourcePdg = channel.firstSourcePdg != 0
                                 ? channel.firstSourcePdg : channel.firstDaughterPdg;
    const int secondSourcePdg = channel.secondSourcePdg != 0
                                  ? channel.secondSourcePdg : channel.secondDaughterPdg;
    const unsigned int firstRole =
      static_cast<unsigned int>(KFParticleGpuClassifyTrackRole(firstSourcePdg));
    const unsigned int secondRole =
      static_cast<unsigned int>(KFParticleGpuClassifyTrackRole(secondSourcePdg));
    if (!IsValidTrackSet(channel.firstTrackSet)
        || !IsValidTrackSet(channel.secondTrackSet)
        || !IsValidSpecies(channel.firstSpecies)
        || !IsValidSpecies(channel.secondSpecies)
        || channel.outputClass >= KFGpuGraphOutputClassCount) {
      throw std::invalid_argument(
        "KFParticle GPU two-daughter routing channel has invalid source geometry");
    }
    const bool firstSpeciesMatches =
      channel.firstSpecies == NumberOfTrackSpecies
      || channel.firstSpecies == ExpectedTrackSpecies(firstSourcePdg)
      || (channel.firstAlternateSourcePdg != 0
          && channel.firstSpecies == ExpectedTrackSpecies(channel.firstAlternateSourcePdg));
    const bool secondSpeciesMatches =
      channel.secondSpecies == NumberOfTrackSpecies
      || channel.secondSpecies == ExpectedTrackSpecies(secondSourcePdg)
      || (channel.secondAlternateSourcePdg != 0
          && channel.secondSpecies == ExpectedTrackSpecies(channel.secondAlternateSourcePdg));
    if (ExpectedTrackSpecies(firstSourcePdg) == NumberOfTrackSpecies
        || ExpectedTrackSpecies(secondSourcePdg) == NumberOfTrackSpecies
        || !firstSpeciesMatches || !secondSpeciesMatches) {
      throw std::invalid_argument(
        "KFParticle GPU two-daughter routing species and daughter PDG disagree");
    }
    if (channel.firstCharge == 0 || channel.secondCharge == 0
        || (channel.firstCharge > 0)
             != HasPositiveCharge(channel.firstDaughterPdg)
        || (channel.secondCharge > 0)
             != HasPositiveCharge(channel.secondDaughterPdg)) {
      throw std::invalid_argument(
        "KFParticle GPU two-daughter routing charge and daughter PDG disagree");
    }
    if (IsPositiveTrackSet(channel.firstTrackSet) != (channel.firstCharge > 0)
        || IsPositiveTrackSet(channel.secondTrackSet) != (channel.secondCharge > 0)) {
      throw std::invalid_argument(
        "KFParticle GPU two-daughter routing track set has the wrong sign");
    }
    if (!IsSupportedTwoDaughterTransportMode(channel.transportMode)
        || !HasFiniteTwoDaughterNumerics(channel)
        || !IsValidSelection(channel.selection)) {
      throw std::invalid_argument(
        "KFParticle GPU two-daughter routing channel has invalid numerical configuration");
    }

    const unsigned int channelBit = static_cast<unsigned int>(index);
    fDescriptors.push_back(
      MakeTwoDaughterRoutingDescriptor(channel, channelBit, firstRole, secondRole));
    fEnabledChannels.Set(channelBit);

    const int firstPdgs[2] = {firstSourcePdg, channel.firstAlternateSourcePdg};
    const int secondPdgs[2] = {secondSourcePdg, channel.secondAlternateSourcePdg};
    for (unsigned int firstIndex = 0u; firstIndex < 2u; ++firstIndex) {
      if (firstPdgs[firstIndex] == 0) { continue; }
      for (unsigned int secondIndex = 0u; secondIndex < 2u; ++secondIndex) {
        if (secondPdgs[secondIndex] == 0) { continue; }
        KFParticleGpuTwoDaughterCompatibilityEntry* compatibility = nullptr;
        for (KFParticleGpuTwoDaughterCompatibilityEntry& entry : fCompatibility) {
          if (entry.firstPdg == firstPdgs[firstIndex]
              && entry.secondPdg == secondPdgs[secondIndex]) {
            compatibility = &entry;
            break;
          }
        }
        if (!compatibility) {
          KFParticleGpuTwoDaughterCompatibilityEntry entry;
          entry.firstPdg = firstPdgs[firstIndex];
          entry.secondPdg = secondPdgs[secondIndex];
          entry.firstRole = static_cast<unsigned int>(KFParticleGpuClassifyTrackRole(entry.firstPdg));
          entry.secondRole = static_cast<unsigned int>(KFParticleGpuClassifyTrackRole(entry.secondPdg));
          fCompatibility.push_back(entry);
          compatibility = &fCompatibility.back();
        }
        compatibility->channels.Set(channelBit);
      }
    }

    KFParticleGpuTwoDaughterExecutionGroup* group = nullptr;
    for (KFParticleGpuTwoDaughterExecutionGroup& entry : fGroups) {
      if (entry.firstTrackSet == channel.firstTrackSet
          && entry.secondTrackSet == channel.secondTrackSet) {
        group = &entry;
        break;
      }
    }
    if (!group) {
      KFParticleGpuTwoDaughterExecutionGroup entry;
      entry.groupIndex = static_cast<unsigned int>(fGroups.size());
      entry.firstTrackSet = channel.firstTrackSet;
      entry.secondTrackSet = channel.secondTrackSet;
      fGroups.push_back(entry);
      group = &fGroups.back();
    }
    if (firstRole != KFGpuTrackRoleInvalid) {
      group->firstRoles |= KFParticleGpuRoutingRoleBit(firstRole);
    }
    if (secondRole != KFGpuTrackRoleInvalid) {
      group->secondRoles |= KFParticleGpuRoutingRoleBit(secondRole);
    }
    const unsigned int firstAlternateRole = static_cast<unsigned int>(
      KFParticleGpuClassifyTrackRole(channel.firstAlternateSourcePdg));
    const unsigned int secondAlternateRole = static_cast<unsigned int>(
      KFParticleGpuClassifyTrackRole(channel.secondAlternateSourcePdg));
    if (firstAlternateRole != KFGpuTrackRoleInvalid) {
      group->firstRoles |= KFParticleGpuRoutingRoleBit(firstAlternateRole);
    }
    if (secondAlternateRole != KFGpuTrackRoleInvalid) {
      group->secondRoles |= KFParticleGpuRoutingRoleBit(secondAlternateRole);
    }
    const KFParticleGpuTrackSpecies firstSourceSpecies =
      ExpectedTrackSpecies(firstSourcePdg);
    const KFParticleGpuTrackSpecies secondSourceSpecies =
      ExpectedTrackSpecies(secondSourcePdg);
    const KFParticleGpuTrackSpecies firstAlternateSpecies =
      ExpectedTrackSpecies(channel.firstAlternateSourcePdg);
    const KFParticleGpuTrackSpecies secondAlternateSpecies =
      ExpectedTrackSpecies(channel.secondAlternateSourcePdg);
    group->firstSpecies |= 1u << static_cast<unsigned int>(firstSourceSpecies);
    group->secondSpecies |= 1u << static_cast<unsigned int>(secondSourceSpecies);
    if (firstAlternateSpecies != NumberOfTrackSpecies) {
      group->firstSpecies |= 1u << static_cast<unsigned int>(firstAlternateSpecies);
    }
    if (secondAlternateSpecies != NumberOfTrackSpecies) {
      group->secondSpecies |= 1u << static_cast<unsigned int>(secondAlternateSpecies);
    }
    group->channels.Set(channelBit);
  }

  fSourceRevision = plan.Revision();
}

/**
 * @brief Union enabled two-track channels matching the ordered pair of coarse source roles.
 * Includes alternate-source entries. Does not exchange daughters, check exact PDGs, or apply cuts.
 */
KFParticleGpuChannelMask KFParticleGpuTwoDaughterRoutingPlan::CompatibleChannels(
  unsigned int firstRole,
  unsigned int secondRole) const
{
  KFParticleGpuChannelMask result;
  for (const KFParticleGpuTwoDaughterCompatibilityEntry& entry : fCompatibility) {
    if (entry.firstRole == firstRole && entry.secondRole == secondRole) {
      result.UnionWith(entry.channels);
    }
  }
  return result.Intersected(fEnabledChannels);
}

/**
 * @brief Union enabled channels matching the exact ordered source-PDG pair, including alternate entries.
 * Scans the table without exchanging daughters or applying physical cuts; no match yields an empty mask.
 */
KFParticleGpuChannelMask KFParticleGpuTwoDaughterRoutingPlan::CompatibleChannelsByPdg(
  int firstPdg,
  int secondPdg) const
{
  KFParticleGpuChannelMask result;
  for (const KFParticleGpuTwoDaughterCompatibilityEntry& entry : fCompatibility) {
    if (entry.firstPdg == firstPdg && entry.secondPdg == secondPdg) {
      result.UnionWith(entry.channels);
    }
  }
  return result.Intersected(fEnabledChannels);
}
