/*
 * This file is part of KFParticle package
 * Copyright (C) 2007-2026
 *
 * KFParticle is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 */

#include "KFParticleGpuDecayGraphPlan.h"

#include "KFParticleGpuCpuChannelCatalogue.h"
#include "KFParticleGpuDecayPlan.h"

#include <algorithm>
#include <stdexcept>

namespace
{
  bool IsValidSupport(unsigned int support)
  {
    return support < KFGpuGraphSupportStatusCount;
  }

  bool IsValidReason(unsigned int reason)
  {
    return reason < KFGpuGraphUnsupportedReasonCount;
  }

  bool IsValidSourceKind(unsigned int kind)
  {
    return kind < KFGpuGraphSourceKindCount;
  }

  bool IsValidSourceShape(const KFParticleGpuGraphSource& source)
  {
    if (!IsValidSourceKind(source.kind)) {
      return false;
    }
    if (source.kind == KFGpuGraphSourceNone) {
      return source.generation == 0u && source.sourceId == 0u;
    }
    if (source.kind == KFGpuGraphSourceCandidateGeneration) {
      return source.generation > 0u && source.sourceId != 0u;
    }
    return source.generation == 0u && source.sourceId != 0u;
  }

  bool IsValidSupportReason(unsigned int support, unsigned int reason)
  {
    if (!IsValidSupport(support) || !IsValidReason(reason)) {
      return false;
    }
    return support == KFGpuGraphSupported
             ? reason == KFGpuGraphUnsupportedNone
             : reason != KFGpuGraphUnsupportedNone;
  }

  bool IsBinaryTopology(unsigned int topology)
  {
    return topology == KFGpuGraphTopologyTrackTrack
           || topology == KFGpuGraphTopologyTrackComposite
           || topology == KFGpuGraphTopologyCompositeTrack
           || topology == KFGpuGraphTopologyCompositeComposite
           || topology == KFGpuGraphTopologyNeutralDaughter;
  }

  bool OperationsMatchTopology(const KFParticleGpuGraphNode& node)
  {
    if ((node.operationMask & KFGpuGraphMissingMass) != 0u
        && node.topology != KFGpuGraphTopologyNeutralDaughter) {
      return false;
    }
    if ((node.operationMask & KFGpuGraphExtrapolate) != 0u
        && node.topology != KFGpuGraphTopologyUnaryComposite) {
      return false;
    }
    if (node.topology == KFGpuGraphTopologyUnaryComposite
        && (node.operationMask & KFGpuGraphConstruct) != 0u) {
      return false;
    }
    return true;
  }

  bool PayloadMatchesTopology(const KFParticleGpuGraphNode& node)
  {
    if (node.supportStatus != KFGpuGraphSupported) {
      return node.payloadKind == KFGpuGraphPayloadNone;
    }
    if (node.topology == KFGpuGraphTopologyTrackTrack) {
      return node.payloadKind == KFGpuGraphPayloadTwoDaughter;
    }
    if (node.topology == KFGpuGraphTopologyTrackComposite
        || node.topology == KFGpuGraphTopologyCompositeTrack) {
      return (node.operationMask & KFGpuGraphMatch) != 0u
        ? node.payloadKind == KFGpuGraphPayloadBinaryFinal
        : node.payloadKind == KFGpuGraphPayloadCompositeTrack;
    }
    if (node.topology == KFGpuGraphTopologyCompositeComposite) {
      return node.payloadKind == KFGpuGraphPayloadCompositeComposite;
    }
    if (node.topology == KFGpuGraphTopologyNeutralDaughter) {
      return node.payloadKind == KFGpuGraphPayloadNeutralDaughter;
    }
    if (node.topology == KFGpuGraphTopologyUnaryComposite) {
      return node.payloadKind == KFGpuGraphPayloadUnaryFinal;
    }
    return false;
  }

  bool SourcesMatchTopology(const KFParticleGpuGraphNode& node)
  {
    const unsigned int first = node.firstSource.kind;
    const unsigned int second = node.secondSource.kind;
    switch (node.topology) {
      case KFGpuGraphTopologyTrackTrack:
        return first == KFGpuGraphSourceTrackRange
               && second == KFGpuGraphSourceTrackRange;
      case KFGpuGraphTopologyTrackComposite:
        return first == KFGpuGraphSourceTrackRange
               && second == KFGpuGraphSourceCandidateGeneration;
      case KFGpuGraphTopologyCompositeTrack:
        return first == KFGpuGraphSourceCandidateGeneration
               && second == KFGpuGraphSourceTrackRange;
      case KFGpuGraphTopologyCompositeComposite:
        return first == KFGpuGraphSourceCandidateGeneration
               && second == KFGpuGraphSourceCandidateGeneration;
      case KFGpuGraphTopologyNeutralDaughter:
        if (first == KFGpuGraphSourceNone || second == KFGpuGraphSourceNone) {
          return false;
        }
        // Missing-mass reconstruction consumes a measured mother and charged
        // daughter. Detector-neutral reconstruction instead names a neutral
        // source explicitly; both use the same bounded topology engine.
        return (node.operationMask & KFGpuGraphMissingMass) != 0u
               || first == KFGpuGraphSourceNeutralRange
               || second == KFGpuGraphSourceNeutralRange;
      case KFGpuGraphTopologyUnaryComposite:
        return first == KFGpuGraphSourceCandidateGeneration
               && second == KFGpuGraphSourceNone;
      default:
        return false;
    }
  }

  const KFParticleGpuGraphNode* FindNode(
    const std::vector<KFParticleGpuGraphNode>& nodes,
    unsigned int channelId)
  {
    for (const KFParticleGpuGraphNode& node : nodes) {
      if (node.channelId == channelId) {
        return &node;
      }
    }
    return nullptr;
  }

  void ValidateCandidateDependency(
    const KFParticleGpuGraphSource& source,
    const KFParticleGpuGraphNode& node,
    const std::vector<KFParticleGpuGraphNode>& nodes)
  {
    if (source.kind != KFGpuGraphSourceCandidateGeneration) {
      return;
    }
    if (source.generation >= node.generation) {
      throw std::invalid_argument(
        "KFParticle GPU graph candidate dependency must precede its consumer");
    }
    const KFParticleGpuGraphNode* parent = FindNode(nodes, source.sourceId);
    if (!parent || parent->generation != source.generation
        || parent->supportStatus != KFGpuGraphSupported) {
      throw std::invalid_argument(
        "KFParticle GPU graph candidate dependency does not name a supported parent");
    }
  }

  bool SameExecutionGroup(const KFParticleGpuGraphExecutionGroup& group,
                          const KFParticleGpuGraphNode& node)
  {
    return group.generation == node.generation
           && group.topology == node.topology
           && group.firstSourceKind == node.firstSource.kind
           && group.secondSourceKind == node.secondSource.kind
           && group.operationMask == node.operationMask;
  }

  void HashValue(unsigned long long& hash, unsigned int value)
  {
    for (unsigned int byte = 0u; byte < 4u; ++byte) {
      hash ^= static_cast<unsigned long long>((value >> (byte * 8u)) & 0xffu);
      hash *= 1099511628211ull;
    }
  }

  unsigned long long HashPlan(const std::vector<KFParticleGpuGraphNode>& nodes,
                              const std::vector<KFParticleGpuGraphExecutionGroup>& groups,
                              const std::vector<KFParticleGpuGraphFamilyCoverage>& families)
  {
    unsigned long long hash = 1469598103934665603ull;
    HashValue(hash, static_cast<unsigned int>(nodes.size()));
    for (const KFParticleGpuGraphNode& node : nodes) {
      HashValue(hash, node.channelId);
      HashValue(hash, node.payloadKind);
      HashValue(hash, node.payloadIndex);
      HashValue(hash, static_cast<unsigned int>(node.motherPdg));
      HashValue(hash, node.topology);
      HashValue(hash, node.generation);
      HashValue(hash, node.firstSource.kind);
      HashValue(hash, node.firstSource.generation);
      HashValue(hash, node.firstSource.sourceId);
      HashValue(hash, node.secondSource.kind);
      HashValue(hash, node.secondSource.generation);
      HashValue(hash, node.secondSource.sourceId);
      HashValue(hash, node.operationMask);
      HashValue(hash, node.selectionProfile);
      HashValue(hash, node.outputClass);
      HashValue(hash, node.supportStatus);
      HashValue(hash, node.unsupportedReason);
    }
    HashValue(hash, static_cast<unsigned int>(groups.size()));
    for (const KFParticleGpuGraphExecutionGroup& group : groups) {
      HashValue(hash, group.groupIndex);
      HashValue(hash, group.generation);
      HashValue(hash, group.topology);
      HashValue(hash, group.firstSourceKind);
      HashValue(hash, group.secondSourceKind);
      HashValue(hash, group.operationMask);
      HashValue(hash, group.nodeOffset);
      HashValue(hash, group.nodeCount);
    }
    HashValue(hash, static_cast<unsigned int>(families.size()));
    for (const KFParticleGpuGraphFamilyCoverage& family : families) {
      HashValue(hash, family.family);
      HashValue(hash, family.supportStatus);
      HashValue(hash, family.unsupportedReason);
      HashValue(hash, family.implementedChannelCount);
    }
    return hash == 0u ? 1u : hash;
  }

  unsigned int TrackSourceId(KFParticleGpuTrackSet set,
                             KFParticleGpuTrackSpecies species)
  {
    return KFParticleGpuGraphTrackSourceId(set, species);
  }

  KFParticleGpuGraphSource TrackSource(KFParticleGpuTrackSet set,
                                       KFParticleGpuTrackSpecies species)
  {
    KFParticleGpuGraphSource source;
    source.kind = KFGpuGraphSourceTrackRange;
    source.sourceId = TrackSourceId(set, species);
    return source;
  }

  KFParticleGpuGraphSource CandidateSource(unsigned int generation,
                                           unsigned int channelId)
  {
    KFParticleGpuGraphSource source;
    source.kind = KFGpuGraphSourceCandidateGeneration;
    source.generation = generation;
    source.sourceId = channelId;
    return source;
  }

  unsigned int FindChannelIdByMother(const KFParticleGpuDecayPlan& plan, int motherPdg)
  {
    for (std::size_t index = 0u; index < plan.NumberOfTwoDaughterChannels(); ++index) {
      const KFParticleGpuTwoDaughterChannel& channel = plan.TwoDaughterChannel(index);
      if (channel.motherPdg == motherPdg) {
        return channel.channelId;
      }
    }
    return 0u;
  }

  unsigned int PayloadForTopology(unsigned int topology,
                                  unsigned int operationMask)
  {
    if ((topology == KFGpuGraphTopologyTrackComposite
         || topology == KFGpuGraphTopologyCompositeTrack)
        && (operationMask & KFGpuGraphMatch) != 0u) {
      return KFGpuGraphPayloadBinaryFinal;
    }
    if (topology == KFGpuGraphTopologyCompositeComposite) {
      return KFGpuGraphPayloadCompositeComposite;
    }
    if (topology == KFGpuGraphTopologyNeutralDaughter) {
      return KFGpuGraphPayloadNeutralDaughter;
    }
    if (topology == KFGpuGraphTopologyUnaryComposite) {
      return KFGpuGraphPayloadUnaryFinal;
    }
    return KFGpuGraphPayloadNone;
  }
}

void KFParticleGpuDecayGraphManifest::Clear()
{
  fNodes.clear();
  fFamilyCoverage.clear();
}

void KFParticleGpuDecayGraphManifest::AddNode(const KFParticleGpuGraphNode& node)
{
  fNodes.push_back(node);
}

void KFParticleGpuDecayGraphManifest::AddFamilyCoverage(
  const KFParticleGpuGraphFamilyCoverage& coverage)
{
  fFamilyCoverage.push_back(coverage);
}

void KFParticleGpuDecayGraphPlan::Clear()
{
  fSourceRevision = 0u;
  fNodes.clear();
  fGroups.clear();
  fFamilyCoverage.clear();
}

void KFParticleGpuDecayGraphPlan::Compile(
  const KFParticleGpuDecayGraphManifest& manifest,
  const KFParticleGpuDecayGraphCompileLimits& limits)
{
  Clear();
  if (manifest.Nodes().size() > limits.nodes
      || manifest.FamilyCoverage().size() > limits.families) {
    throw std::length_error("KFParticle GPU decay graph exceeds compile capacity");
  }

  fNodes = manifest.Nodes();
  fFamilyCoverage = manifest.FamilyCoverage();

  bool covered[KFGpuCpuFamilyCount] = {};
  for (const KFParticleGpuGraphFamilyCoverage& family : fFamilyCoverage) {
    if (family.family >= KFGpuCpuFamilyCount || covered[family.family]
        || !IsValidSupportReason(family.supportStatus, family.unsupportedReason)
        || (family.supportStatus == KFGpuGraphUnsupported
            && family.implementedChannelCount != 0u)
        || (family.supportStatus != KFGpuGraphUnsupported
            && family.implementedChannelCount == 0u
            && family.family != KFGpuCpuFamilySameSignPrimaryResonance)) {
      throw std::invalid_argument(
        "KFParticle GPU decay graph has invalid CPU-family coverage");
    }
    covered[family.family] = true;
  }
  for (unsigned int family = 0u; family < KFGpuCpuFamilyCount; ++family) {
    if (!covered[family]) {
      throw std::invalid_argument(
        "KFParticle GPU decay graph omits an active CPU finder family");
    }
  }

  for (std::size_t index = 0u; index < fNodes.size(); ++index) {
    const KFParticleGpuGraphNode& node = fNodes[index];
    if (node.channelId == 0u || node.generation == 0u
        || node.topology <= KFGpuGraphTopologyInvalid
        || node.topology >= KFGpuGraphTopologyCount
        || node.outputClass <= KFGpuGraphOutputInvalid
        || node.outputClass >= KFGpuGraphOutputClassCount
        || !IsValidSourceShape(node.firstSource)
        || !IsValidSourceShape(node.secondSource)
        || !SourcesMatchTopology(node)
        || node.payloadKind >= KFGpuGraphPayloadCount
        || !PayloadMatchesTopology(node)
        || (node.operationMask & ~KFGpuGraphKnownOperations) != 0u
        || node.operationMask == 0u
        || !OperationsMatchTopology(node)
        || (IsBinaryTopology(node.topology)
            && (node.operationMask & KFGpuGraphConstruct) == 0u)
        || !IsValidSupportReason(node.supportStatus, node.unsupportedReason)
        || node.supportStatus == KFGpuGraphPartiallySupported) {
      throw std::invalid_argument("KFParticle GPU decay graph node is invalid");
    }
    for (std::size_t previous = 0u; previous < index; ++previous) {
      if (fNodes[previous].channelId == node.channelId) {
        throw std::invalid_argument(
          "KFParticle GPU decay graph channel IDs must be unique");
      }
    }
  }

  for (const KFParticleGpuGraphNode& node : fNodes) {
    if (node.supportStatus != KFGpuGraphSupported) {
      continue;
    }
    unsigned int payloadCount = 0u;
    unsigned int matchingIndexCount = 0u;
    for (const KFParticleGpuGraphNode& other : fNodes) {
      if (other.supportStatus == KFGpuGraphSupported
          && other.payloadKind == node.payloadKind) {
        ++payloadCount;
        if (other.payloadIndex == node.payloadIndex) {
          ++matchingIndexCount;
        }
      }
    }
    if (node.payloadIndex >= payloadCount || matchingIndexCount != 1u) {
      throw std::invalid_argument(
        "KFParticle GPU decay graph payload indices must be unique and dense");
    }
  }

  for (const KFParticleGpuGraphNode& node : fNodes) {
    if (node.supportStatus != KFGpuGraphSupported) {
      continue;
    }
    ValidateCandidateDependency(node.firstSource, node, fNodes);
    ValidateCandidateDependency(node.secondSource, node, fNodes);
  }

  std::stable_sort(
    fNodes.begin(), fNodes.end(),
    [](const KFParticleGpuGraphNode& left, const KFParticleGpuGraphNode& right) {
      if (left.supportStatus != right.supportStatus) {
        return left.supportStatus == KFGpuGraphSupported;
      }
      if (left.generation != right.generation) {
        return left.generation < right.generation;
      }
      if (left.topology != right.topology) {
        return left.topology < right.topology;
      }
      if (left.firstSource.kind != right.firstSource.kind) {
        return left.firstSource.kind < right.firstSource.kind;
      }
      if (left.secondSource.kind != right.secondSource.kind) {
        return left.secondSource.kind < right.secondSource.kind;
      }
      if (left.operationMask != right.operationMask) {
        return left.operationMask < right.operationMask;
      }
      return left.channelId < right.channelId;
    });

  for (std::size_t index = 0u; index < fNodes.size(); ++index) {
    const KFParticleGpuGraphNode& node = fNodes[index];
    if (node.supportStatus != KFGpuGraphSupported) {
      continue;
    }
    if (fGroups.empty() || !SameExecutionGroup(fGroups.back(), node)) {
      KFParticleGpuGraphExecutionGroup group;
      group.groupIndex = static_cast<unsigned int>(fGroups.size());
      group.generation = node.generation;
      group.topology = node.topology;
      group.firstSourceKind = node.firstSource.kind;
      group.secondSourceKind = node.secondSource.kind;
      group.operationMask = node.operationMask;
      group.nodeOffset = static_cast<unsigned int>(index);
      fGroups.push_back(group);
    }
    ++fGroups.back().nodeCount;
  }
  if (fGroups.size() > limits.groups) {
    Clear();
    throw std::length_error(
      "KFParticle GPU decay graph exceeds execution-group capacity");
  }

  std::sort(
    fFamilyCoverage.begin(), fFamilyCoverage.end(),
    [](const KFParticleGpuGraphFamilyCoverage& left,
       const KFParticleGpuGraphFamilyCoverage& right) {
      return left.family < right.family;
    });
  fSourceRevision = HashPlan(fNodes, fGroups, fFamilyCoverage);
}

KFParticleGpuDecayGraphManifest MakeDefaultCpuFinderDecayGraphManifest(
  const KFParticleGpuDecayPlan& plan)
{
  KFParticleGpuDecayGraphManifest manifest;
  const unsigned int twoDaughterCount =
    static_cast<unsigned int>(plan.NumberOfTwoDaughterChannels());
  unsigned int sameSignPrimaryCount = 0u;
  unsigned int compositeCompositeCount = 0u;
  unsigned int neutralMissingMassCount = 0u;
  unsigned int kaonMatchingCount = 0u;
  unsigned int unaryProjectionCount = 0u;
  unsigned int unarySelectionCount = 0u;
  unsigned int operationPayloadCounts[KFGpuGraphPayloadCount] = {};

  for (std::size_t index = 0u; index < plan.NumberOfTwoDaughterChannels(); ++index) {
    const KFParticleGpuTwoDaughterChannel& channel = plan.TwoDaughterChannel(index);
    KFParticleGpuGraphNode node;
    node.channelId = channel.channelId;
    node.payloadKind = KFGpuGraphPayloadTwoDaughter;
    node.payloadIndex = static_cast<unsigned int>(index);
    node.motherPdg = channel.motherPdg;
    node.topology = KFGpuGraphTopologyTrackTrack;
    node.generation = 1u;
    node.firstSource = TrackSource(channel.firstTrackSet, channel.firstSpecies);
    node.secondSource = TrackSource(channel.secondTrackSet, channel.secondSpecies);
    node.operationMask = KFGpuGraphConstruct | KFGpuGraphTransport | KFGpuGraphSelect;
    node.selectionProfile = channel.channelId;
    node.outputClass = channel.outputClass;
    node.supportStatus = KFGpuGraphSupported;
    node.unsupportedReason = KFGpuGraphUnsupportedNone;
    manifest.AddNode(node);
    const bool firstPrimary =
      channel.firstTrackSet == PrimaryPositiveFirst
      || channel.firstTrackSet == PrimaryNegativeFirst
      || channel.firstTrackSet == PrimaryPositiveLast
      || channel.firstTrackSet == PrimaryNegativeLast;
    if (firstPrimary
        && (channel.firstCharge > 0) == (channel.secondCharge > 0)) {
      ++sameSignPrimaryCount;
    }
  }

  for (std::size_t index = 0u; index < plan.NumberOfV0TrackCascadeChannels(); ++index) {
    const KFParticleGpuV0TrackCascadeChannel& channel =
      plan.V0TrackCascadeChannel(index);
    const unsigned int parentChannelId =
      channel.parentChannelId != 0u
        ? channel.parentChannelId : FindChannelIdByMother(plan, channel.v0Pdg);
    KFParticleGpuGraphNode node;
    node.channelId = channel.channelId;
    node.payloadKind = KFGpuGraphPayloadCompositeTrack;
    node.payloadIndex = static_cast<unsigned int>(index);
    node.motherPdg = channel.motherPdg;
    node.topology = KFGpuGraphTopologyCompositeTrack;
    node.generation = channel.generation;
    node.firstSource = CandidateSource(channel.generation - 1u, parentChannelId);
    node.secondSource = TrackSource(channel.bachelorTrackSet, channel.bachelorSpecies);
    node.operationMask = KFGpuGraphConstruct | KFGpuGraphTransport | KFGpuGraphSelect;
    if (channel.parentMassConstraint >= 0.f) {
      node.operationMask |= KFGpuGraphMassConstraint;
    }
    node.selectionProfile = channel.channelId;
    node.outputClass = channel.outputClass;
    node.supportStatus = KFGpuGraphSupported;
    node.unsupportedReason = KFGpuGraphUnsupportedNone;
    manifest.AddNode(node);
  }

  for (std::size_t index = 0u;
       index < plan.NumberOfGraphOperationChannels();
       ++index) {
    const KFParticleGpuGraphOperationChannel& channel =
      plan.GraphOperationChannel(index);
    if (!KFParticleGpuGraphOperations::ValidateDescriptor(channel.descriptor)) {
      throw std::invalid_argument(
        "KFParticle GPU decay plan contains an unsupported graph operation descriptor");
    }
    KFParticleGpuGraphNode node = channel.node;
    node.payloadKind = PayloadForTopology(node.topology,
                                          channel.descriptor.operationMask);
    if (node.payloadKind == KFGpuGraphPayloadNone) {
      throw std::invalid_argument(
        "KFParticle GPU decay plan graph operation has an unsupported topology");
    }
    node.payloadIndex = operationPayloadCounts[node.payloadKind]++;
    node.motherPdg = channel.descriptor.motherPdg;
    node.operationMask = channel.descriptor.operationMask;
    node.outputClass = channel.descriptor.outputClass;
    node.supportStatus = KFGpuGraphSupported;
    node.unsupportedReason = KFGpuGraphUnsupportedNone;
    manifest.AddNode(node);

    compositeCompositeCount +=
      node.topology == KFGpuGraphTopologyCompositeComposite ? 1u : 0u;
    neutralMissingMassCount +=
      node.topology == KFGpuGraphTopologyNeutralDaughter ? 1u : 0u;
    kaonMatchingCount +=
      (node.operationMask & KFGpuGraphMatch) != 0u ? 1u : 0u;
    unaryProjectionCount +=
      (node.operationMask
       & (KFGpuGraphExtrapolate | KFGpuGraphSetProductionVertex)) != 0u
        ? 1u : 0u;
    unarySelectionCount +=
      node.topology == KFGpuGraphTopologyUnaryComposite
      && node.outputClass == KFGpuGraphOutputFinal
      && (node.operationMask & KFGpuGraphSelect) != 0u ? 1u : 0u;
  }

  std::array<unsigned int, KFGpuCpuFamilyCount> implementedCounts = {};
  implementedCounts[KFGpuCpuFamilyTwoDaughter] = twoDaughterCount;
  implementedCounts[KFGpuCpuFamilySameSignPrimaryResonance] =
    sameSignPrimaryCount;
  for (std::size_t index = 0u; index < plan.NumberOfV0TrackCascadeChannels(); ++index) {
    const unsigned int family = plan.V0TrackCascadeChannel(index).family;
    if (family < KFGpuCpuFamilyCount) ++implementedCounts[family];
  }
  implementedCounts[KFGpuCpuFamilyCompositeComposite] =
    compositeCompositeCount;
  implementedCounts[KFGpuCpuFamilyNeutralMissingMass] =
    neutralMissingMassCount;
  implementedCounts[KFGpuCpuFamilyKaonMatching] = kaonMatchingCount;
  implementedCounts[KFGpuCpuFamilyPrimaryProjection] = unaryProjectionCount;
  implementedCounts[KFGpuCpuFamilyFinalSelection] = unarySelectionCount;
  for (const KFParticleGpuGraphFamilyCoverage& coverage :
       MakeCpuFinderFamilyCoverage(implementedCounts)) {
    manifest.AddFamilyCoverage(coverage);
  }
  return manifest;
}
