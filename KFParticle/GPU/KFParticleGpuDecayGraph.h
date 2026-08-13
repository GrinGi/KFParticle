/*
 * This file is part of KFParticle package
 * Copyright (C) 2007-2026
 *
 * KFParticle is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 */

#ifndef KFPARTICLEGPUDECAYGRAPH_H
#define KFPARTICLEGPUDECAYGRAPH_H

#include "KFParticleGpuPlatform.h"

#include <type_traits>

enum KFParticleGpuGraphSourceKind : unsigned int
{
  KFGpuGraphSourceNone = 0u,
  KFGpuGraphSourceTrackRange = 1u,
  KFGpuGraphSourceCandidateGeneration = 2u,
  KFGpuGraphSourceNeutralRange = 3u,
  KFGpuGraphSourceKindCount = 4u
};

enum KFParticleGpuGraphTopology : unsigned int
{
  KFGpuGraphTopologyInvalid = 0u,
  KFGpuGraphTopologyTrackTrack = 1u,
  KFGpuGraphTopologyTrackComposite = 2u,
  KFGpuGraphTopologyCompositeTrack = 3u,
  KFGpuGraphTopologyCompositeComposite = 4u,
  KFGpuGraphTopologyNeutralDaughter = 5u,
  KFGpuGraphTopologyUnaryComposite = 6u,
  KFGpuGraphTopologyCount = 7u
};

enum KFParticleGpuGraphOperation : unsigned int
{
  KFGpuGraphConstruct = 1u << 0u,
  KFGpuGraphTransport = 1u << 1u,
  KFGpuGraphSetProductionVertex = 1u << 2u,
  KFGpuGraphMassConstraint = 1u << 3u,
  KFGpuGraphMatch = 1u << 4u,
  KFGpuGraphSelect = 1u << 5u,
  KFGpuGraphMissingMass = 1u << 6u,
  KFGpuGraphExtrapolate = 1u << 7u,
  KFGpuGraphKnownOperations = (1u << 8u) - 1u
};

enum KFParticleGpuGraphOutputClass : unsigned int
{
  KFGpuGraphOutputInvalid = 0u,
  KFGpuGraphOutputTemporary = 1u,
  KFGpuGraphOutputPrimary = 2u,
  KFGpuGraphOutputSecondary = 3u,
  KFGpuGraphOutputPrimaryAndSecondary = 4u,
  KFGpuGraphOutputFinal = 5u,
  KFGpuGraphOutputClassCount = 6u
};

enum KFParticleGpuGraphSupportStatus : unsigned int
{
  KFGpuGraphUnsupported = 0u,
  KFGpuGraphPartiallySupported = 1u,
  KFGpuGraphSupported = 2u,
  KFGpuGraphSupportStatusCount = 3u
};

enum KFParticleGpuGraphPayloadKind : unsigned int
{
  KFGpuGraphPayloadNone = 0u,
  KFGpuGraphPayloadTwoDaughter = 1u,
  KFGpuGraphPayloadCompositeTrack = 2u,
  KFGpuGraphPayloadCompositeComposite = 3u,
  KFGpuGraphPayloadNeutralDaughter = 4u,
  KFGpuGraphPayloadUnaryFinal = 5u,
  KFGpuGraphPayloadBinaryFinal = 6u,
  KFGpuGraphPayloadCount = 7u
};

enum KFParticleGpuGraphUnsupportedReason : unsigned int
{
  KFGpuGraphUnsupportedNone = 0u,
  KFGpuGraphUnsupportedNotImplemented = 1u,
  KFGpuGraphUnsupportedMissingInput = 2u,
  KFGpuGraphUnsupportedMissingMathematics = 3u,
  KFGpuGraphUnsupportedValidationPending = 4u,
  KFGpuGraphUnsupportedReasonCount = 5u
};

/** CPU finder families audited independently from individual PDG channels. */
enum KFParticleGpuCpuFinderFamily : unsigned int
{
  KFGpuCpuFamilyTwoDaughter = 0u,
  KFGpuCpuFamilySameSignPrimaryResonance = 1u,
  KFGpuCpuFamilyTrackComposite = 2u,
  KFGpuCpuFamilyLongLivedComposite = 3u,
  KFGpuCpuFamilyCompositeComposite = 4u,
  KFGpuCpuFamilyNeutralMissingMass = 5u,
  KFGpuCpuFamilyKaonMatching = 6u,
  KFGpuCpuFamilyPrimaryProjection = 7u,
  KFGpuCpuFamilyFinalSelection = 8u,
  KFGpuCpuFamilyCount = 9u
};

struct KFParticleGpuGraphSource
{
  unsigned int kind = KFGpuGraphSourceNone;
  unsigned int generation = 0u;
  unsigned int sourceId = 0u;
};

/**
 * One physical channel in the device graph.
 *
 * Candidate dependencies reference a stable parent channel ID and generation.
 * Track and neutral source IDs are compact host-defined range roles.
 */
struct KFParticleGpuGraphNode
{
  unsigned int channelId = 0u;
  unsigned int payloadKind = KFGpuGraphPayloadNone;
  unsigned int payloadIndex = 0u;
  int motherPdg = 0;
  unsigned int topology = KFGpuGraphTopologyInvalid;
  unsigned int generation = 0u;
  KFParticleGpuGraphSource firstSource;
  KFParticleGpuGraphSource secondSource;
  unsigned int operationMask = 0u;
  unsigned int selectionProfile = 0u;
  unsigned int outputClass = KFGpuGraphOutputInvalid;
  unsigned int supportStatus = KFGpuGraphUnsupported;
  unsigned int unsupportedReason = KFGpuGraphUnsupportedNotImplemented;
};

/** Contiguous nodes with one launch-compatible topology and operation set. */
struct KFParticleGpuGraphExecutionGroup
{
  unsigned int groupIndex = 0u;
  unsigned int generation = 0u;
  unsigned int topology = KFGpuGraphTopologyInvalid;
  unsigned int firstSourceKind = KFGpuGraphSourceNone;
  unsigned int secondSourceKind = KFGpuGraphSourceNone;
  unsigned int operationMask = 0u;
  unsigned int nodeOffset = 0u;
  unsigned int nodeCount = 0u;
};

/** Machine-checkable coverage of one active KFParticleFinder CPU family. */
struct KFParticleGpuGraphFamilyCoverage
{
  unsigned int family = KFGpuCpuFamilyCount;
  unsigned int supportStatus = KFGpuGraphUnsupported;
  unsigned int unsupportedReason = KFGpuGraphUnsupportedNotImplemented;
  unsigned int implementedChannelCount = 0u;
};

class KFParticleGpuDecayGraphView
{
 public:
  KFPARTICLE_GPU_HOST_DEVICE KFParticleGpuDecayGraphView() = default;

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

  KFPARTICLE_GPU_HOST_DEVICE const KFParticleGpuGraphNode* Nodes() const { return fNodes; }
  KFPARTICLE_GPU_HOST_DEVICE unsigned int NodeCount() const { return fNodeCount; }
  KFPARTICLE_GPU_HOST_DEVICE const KFParticleGpuGraphExecutionGroup* Groups() const
  {
    return fGroups;
  }
  KFPARTICLE_GPU_HOST_DEVICE unsigned int GroupCount() const { return fGroupCount; }
  KFPARTICLE_GPU_HOST_DEVICE const KFParticleGpuGraphFamilyCoverage* Families() const
  {
    return fFamilies;
  }
  KFPARTICLE_GPU_HOST_DEVICE unsigned int FamilyCount() const { return fFamilyCount; }
  KFPARTICLE_GPU_HOST_DEVICE unsigned long long Revision() const { return fRevision; }

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
  const KFParticleGpuGraphNode* fNodes;
  unsigned int fNodeCount;
  const KFParticleGpuGraphExecutionGroup* fGroups;
  unsigned int fGroupCount;
  const KFParticleGpuGraphFamilyCoverage* fFamilies;
  unsigned int fFamilyCount;
  unsigned long long fRevision;
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
