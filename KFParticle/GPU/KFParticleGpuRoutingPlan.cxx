/*
 * This file is part of KFParticle package
 * Copyright (C) 2007-2026
 *
 * KFParticle is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 */

#include "KFParticleGpuRoutingPlan.h"

#include "KFParticleGpuDecayPlan.h"

#include <cmath>
#include <stdexcept>

namespace
{
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

  bool IsPositiveTrackSet(KFParticleGpuTrackSet set)
  {
    return set == SecondaryPositiveFirst || set == PrimaryPositiveFirst
           || set == SecondaryPositiveLast || set == PrimaryPositiveLast;
  }

  bool HasPositiveCharge(int pdg)
  {
    const int absolutePdg = pdg < 0 ? -pdg : pdg;
    if (absolutePdg == 11 || absolutePdg == 13 || absolutePdg == 19) {
      return pdg < 0;
    }
    return pdg > 0;
  }

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

  bool IsValidTrackSet(KFParticleGpuTrackSet set)
  {
    return static_cast<unsigned int>(set) < NumberOfTrackSets;
  }

  bool IsValidSpecies(KFParticleGpuTrackSpecies species)
  {
    return static_cast<unsigned int>(species) <= NumberOfTrackSpecies;
  }

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

void KFParticleGpuV0TrackRoutingPlan::Clear()
{
  fSourceRevision = 0u;
  fDescriptors.clear();
  fCompatibility.clear();
  fGroups.clear();
  fEnabledChannels.Clear();
}

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

void KFParticleGpuTwoDaughterRoutingPlan::Clear()
{
  fSourceRevision = 0u;
  fDescriptors.clear();
  fCompatibility.clear();
  fGroups.clear();
  fEnabledChannels.Clear();
}

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
