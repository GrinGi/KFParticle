/*
 * This file is part of KFParticle package
 * Copyright (C) 2007-2026
 *
 * KFParticle is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 */

#ifndef KFPARTICLEGPUGRAPHOPERATIONS_H
#define KFPARTICLEGPUGRAPHOPERATIONS_H

#include "KFParticleGpuCandidateTransfer.h"
#include "KFParticleGpuDecayGraph.h"
#include "KFParticleGpuInputDataTransfer.h"
#include "KFParticleGpuMath.h"
#include "KFParticleGpuSelection.h"

#include <type_traits>

enum KFParticleGpuGraphTaskStatus : unsigned int
{
  KFGpuGraphTaskPending = 0u,
  KFGpuGraphTaskAccepted = 1u << 0u,
  KFGpuGraphTaskRejectDescriptor = 1u << 1u,
  KFGpuGraphTaskRejectSource = 1u << 2u,
  KFGpuGraphTaskRejectLineage = 1u << 3u,
  KFGpuGraphTaskRejectMathematics = 1u << 4u,
  KFGpuGraphTaskRejectSelection = 1u << 5u,
  KFGpuGraphTaskRejectCandidateCapacity = 1u << 6u,
  KFGpuGraphTaskRejectDaughterCapacity = 1u << 7u
};

enum KFParticleGpuGraphOperationOverflow : unsigned int
{
  KFGpuGraphTaskCapacityExceeded = 1u << 0u
};

enum KFParticleGpuGraphOperationFlags : unsigned int
{
  KFGpuGraphRequireSameEvent = 1u << 0u,
  KFGpuGraphRequireDistinctSources = 1u << 1u,
  KFGpuGraphApplyMassConstraint = 1u << 2u,
  KFGpuGraphRequirePositiveMissingEnergy = 1u << 3u,
  // MatchKaons validates the missing residual but publishes the matched input.
  KFGpuGraphStoreFirstOnMatch = 1u << 4u,
  // Same-source binary channels visit only one canonical candidate ordering.
  KFGpuGraphRequireOrderedCandidatePair = 1u << 5u,
  // Match the candidate against the primary vertex carried by the track.
  KFGpuGraphMatchTrackPrimaryVertex = 1u << 6u,
  // SelectParticles scans only the primary vertices of the task's event.
  KFGpuGraphSelectEventPrimaryVertices = 1u << 7u
};

enum KFParticleGpuMissingMassMode : unsigned int
{
  KFGpuGraphMissingMassLegacySubtract = 0u,
  KFGpuGraphMissingMassFilteredReconstruction = 1u
};

constexpr int KFGpuGraphPrimaryVertexFromCandidate = -2;

/** Bounded physical operation descriptor uploaded once with a graph revision. */
struct KFParticleGpuGraphOperationDescriptor
{
  unsigned int channelId = 0u;
  unsigned int topology = KFGpuGraphTopologyInvalid;
  unsigned int operationMask = 0u;
  unsigned int outputClass = KFGpuGraphOutputInvalid;
  unsigned int flags = KFGpuGraphRequireSameEvent | KFGpuGraphRequireDistinctSources;
  unsigned int missingMassMode = KFGpuGraphMissingMassLegacySubtract;
  int motherPdg = 0;
  int firstPdg = 0;
  int secondPdg = 0;
  int primaryVertexIndex = -1;
  float firstMass = 0.f;
  float secondMass = 0.f;
  float neutralMass = 0.f;
  float massConstraint = -1.f;
  float massConstraintSigma = 0.f;
  float maxGeometricChi2PerNdf = -1.f;
  float maxTopologyChi2PerNdf = -1.f;
  float maxMatchMomentum = -1.f;
  float maxMatchDistance = -1.f;
  float maxMatchMomentumSigma = -1.f;
  float maxMatchTopologyChi2PerNdf = -1.f;
  float matchMassWindowSigma = -1.f;
  float matchMassWindowCut = -1.f;
  float maxSelectionVertexDistance = -1.f;
  float minSelectionDecayLengthOverError = -1.f;
};

/** One generation-local work item. Source indices always refer to an earlier generation. */
struct KFParticleGpuGraphOperationTask
{
  unsigned int descriptorIndex = 0u;
  unsigned int firstCandidateIndex = 0u;
  unsigned int secondCandidateIndex = 0u;
  unsigned int firstSourceKind = KFGpuGraphSourceCandidateGeneration;
  unsigned int secondSourceKind = KFGpuGraphSourceCandidateGeneration;
  unsigned int eventIndex = 0u;
  unsigned int primaryVertexOffset = 0u;
  unsigned int primaryVertexCount = 0u;
};

struct KFParticleGpuGraphOperationResult
{
  unsigned int status = KFGpuGraphTaskPending;
  unsigned int channelId = 0u;
  unsigned int outputClass = KFGpuGraphOutputInvalid;
  unsigned int outputCandidateIndex = 0xffffffffu;
  unsigned int daughterCount = 0u;
  int primaryVertexIndex = -1;
  float mass = 0.f;
  float massError = 0.f;
  float topologyChi2PerNdf = 1.e8f;
};

/**
 * Flat non-owning view of the reusable later-generation graph workspace.
 *
 * The device-storage owner retains every allocation. Publishing this view once
 * per reconstruction transaction keeps stable pointers out of kernel argument
 * lists while preserving explicit capacities and revision identity.
 */
class KFParticleGpuGraphOperationStorageView
{
 public:
  KFPARTICLE_GPU_HOST_DEVICE KFParticleGpuGraphOperationStorageView() = default;

  KFPARTICLE_GPU_HOST_DEVICE KFParticleGpuGraphOperationStorageView(
    const KFParticleGpuGraphOperationDescriptor* descriptors,
    unsigned int descriptorCount,
    unsigned int descriptorCapacity,
    KFParticleGpuGraphOperationTask* tasks,
    unsigned int taskCapacity,
    KFParticleGpuGraphOperationResult* results,
    unsigned int* visited,
    unsigned int* accepted,
    unsigned int* stored,
    unsigned int* constructed,
    unsigned int* rejected,
    unsigned int* overflow,
    unsigned int* channelVisited,
    unsigned int* channelAccepted,
    unsigned int* channelStored,
    unsigned int* channelConstructed,
    unsigned int* channelRejected,
    unsigned long long revision)
    : fDescriptors(descriptors)
    , fDescriptorCount(descriptorCount)
    , fDescriptorCapacity(descriptorCapacity)
    , fTasks(tasks)
    , fTaskCapacity(taskCapacity)
    , fResults(results)
    , fVisited(visited)
    , fAccepted(accepted)
    , fStored(stored)
    , fConstructed(constructed)
    , fRejected(rejected)
    , fOverflow(overflow)
    , fChannelVisited(channelVisited)
    , fChannelAccepted(channelAccepted)
    , fChannelStored(channelStored)
    , fChannelConstructed(channelConstructed)
    , fChannelRejected(channelRejected)
    , fRevision(revision)
  {
  }

  KFPARTICLE_GPU_HOST_DEVICE const KFParticleGpuGraphOperationDescriptor*
  Descriptors() const
  {
    return fDescriptors;
  }
  KFPARTICLE_GPU_HOST_DEVICE unsigned int DescriptorCount() const
  {
    return fDescriptorCount;
  }
  KFPARTICLE_GPU_HOST_DEVICE unsigned int DescriptorCapacity() const
  {
    return fDescriptorCapacity;
  }
  KFPARTICLE_GPU_HOST_DEVICE KFParticleGpuGraphOperationTask* Tasks() const
  {
    return fTasks;
  }
  KFPARTICLE_GPU_HOST_DEVICE unsigned int TaskCapacity() const
  {
    return fTaskCapacity;
  }
  KFPARTICLE_GPU_HOST_DEVICE KFParticleGpuGraphOperationResult* Results() const
  {
    return fResults;
  }
  KFPARTICLE_GPU_HOST_DEVICE unsigned int* Visited() const { return fVisited; }
  KFPARTICLE_GPU_HOST_DEVICE unsigned int* Accepted() const { return fAccepted; }
  KFPARTICLE_GPU_HOST_DEVICE unsigned int* Stored() const { return fStored; }
  KFPARTICLE_GPU_HOST_DEVICE unsigned int* Constructed() const
  {
    return fConstructed;
  }
  KFPARTICLE_GPU_HOST_DEVICE unsigned int* Rejected() const { return fRejected; }
  KFPARTICLE_GPU_HOST_DEVICE unsigned int* Overflow() const { return fOverflow; }
  KFPARTICLE_GPU_HOST_DEVICE unsigned int* ChannelVisited() const
  {
    return fChannelVisited;
  }
  KFPARTICLE_GPU_HOST_DEVICE unsigned int* ChannelAccepted() const
  {
    return fChannelAccepted;
  }
  KFPARTICLE_GPU_HOST_DEVICE unsigned int* ChannelStored() const
  {
    return fChannelStored;
  }
  KFPARTICLE_GPU_HOST_DEVICE unsigned int* ChannelConstructed() const
  {
    return fChannelConstructed;
  }
  KFPARTICLE_GPU_HOST_DEVICE unsigned int* ChannelRejected() const
  {
    return fChannelRejected;
  }
  KFPARTICLE_GPU_HOST_DEVICE unsigned long long Revision() const
  {
    return fRevision;
  }

 private:
  const KFParticleGpuGraphOperationDescriptor* fDescriptors;
  unsigned int fDescriptorCount;
  unsigned int fDescriptorCapacity;
  KFParticleGpuGraphOperationTask* fTasks;
  unsigned int fTaskCapacity;
  KFParticleGpuGraphOperationResult* fResults;
  unsigned int* fVisited;
  unsigned int* fAccepted;
  unsigned int* fStored;
  unsigned int* fConstructed;
  unsigned int* fRejected;
  unsigned int* fOverflow;
  unsigned int* fChannelVisited;
  unsigned int* fChannelAccepted;
  unsigned int* fChannelStored;
  unsigned int* fChannelConstructed;
  unsigned int* fChannelRejected;
  unsigned long long fRevision;
};

namespace KFParticleGpuGraphOperations
{
  static const unsigned int MaximumLineageSize = 16u;

  KFPARTICLE_GPU_HOST_DEVICE inline bool IsFinite(float value)
  {
    return value == value && value > -1.e30f && value < 1.e30f;
  }

  KFPARTICLE_GPU_HOST_DEVICE inline bool IsSupportedTopology(unsigned int topology)
  {
    return topology == KFGpuGraphTopologyCompositeComposite
           || topology == KFGpuGraphTopologyNeutralDaughter
           || topology == KFGpuGraphTopologyCompositeTrack
           || topology == KFGpuGraphTopologyUnaryComposite;
  }

  KFPARTICLE_GPU_HOST_DEVICE inline bool ValidateDescriptor(
    const KFParticleGpuGraphOperationDescriptor& descriptor)
  {
    if (descriptor.channelId == 0u || !IsSupportedTopology(descriptor.topology)
        || descriptor.outputClass == KFGpuGraphOutputInvalid
        || descriptor.outputClass >= KFGpuGraphOutputClassCount
        || (descriptor.operationMask & ~KFGpuGraphKnownOperations) != 0u) {
      return false;
    }
    if (descriptor.topology == KFGpuGraphTopologyCompositeComposite) {
      return (descriptor.operationMask & KFGpuGraphConstruct) != 0u;
    }
    if (descriptor.topology == KFGpuGraphTopologyCompositeTrack) {
      return (descriptor.operationMask & KFGpuGraphMatch) != 0u
             && (descriptor.operationMask & KFGpuGraphConstruct) != 0u;
    }
    if (descriptor.topology == KFGpuGraphTopologyNeutralDaughter) {
      return (descriptor.operationMask & KFGpuGraphMissingMass) != 0u
             && (descriptor.missingMassMode == KFGpuGraphMissingMassLegacySubtract
                 || (descriptor.missingMassMode
                       == KFGpuGraphMissingMassFilteredReconstruction
                     && descriptor.firstMass >= 0.f
                     && descriptor.secondMass >= 0.f
                     && descriptor.neutralMass >= 0.f));
    }
    return (descriptor.operationMask
            & (KFGpuGraphExtrapolate | KFGpuGraphSetProductionVertex
               | KFGpuGraphMassConstraint | KFGpuGraphMatch | KFGpuGraphSelect))
           != 0u;
  }

  KFPARTICLE_GPU_HOST_DEVICE inline bool LoadLineage(
    const KFParticleGpuConstCandidatePoolView& candidates,
    unsigned int candidateIndex,
    int lineage[MaximumLineageSize],
    unsigned int& lineageSize)
  {
    if (candidateIndex >= candidates.Size()) {
      return false;
    }
    const unsigned int count = candidates.Metadata().DaughterCount(candidateIndex);
    const unsigned int offset = candidates.Metadata().DaughterOffset(candidateIndex);
    if (count == 0u || count > MaximumLineageSize
        || !candidates.Daughters().CanStore(offset, count)) {
      return false;
    }
    lineageSize = 0u;
    for (unsigned int daughter = 0u; daughter < count; ++daughter) {
      const int source = candidates.Daughters().SourceId(offset + daughter);
      unsigned int insertion = lineageSize;
      while (insertion > 0u && lineage[insertion - 1u] > source) {
        lineage[insertion] = lineage[insertion - 1u];
        --insertion;
      }
      if ((insertion > 0u && lineage[insertion - 1u] == source)
          || (insertion < lineageSize && lineage[insertion] == source)) {
        return false;
      }
      lineage[insertion] = source;
      ++lineageSize;
    }
    return true;
  }

  KFPARTICLE_GPU_HOST_DEVICE inline bool MergeLineage(
    const KFParticleGpuConstCandidatePoolView& candidates,
    unsigned int firstCandidateIndex,
    unsigned int secondCandidateIndex,
    int lineage[MaximumLineageSize],
    unsigned int& lineageSize)
  {
    int first[MaximumLineageSize];
    int second[MaximumLineageSize];
    unsigned int firstSize = 0u;
    unsigned int secondSize = 0u;
    if (!LoadLineage(candidates, firstCandidateIndex, first, firstSize)
        || !LoadLineage(candidates, secondCandidateIndex, second, secondSize)
        || firstSize + secondSize > MaximumLineageSize) {
      return false;
    }
    unsigned int firstIndex = 0u;
    unsigned int secondIndex = 0u;
    lineageSize = 0u;
    while (firstIndex < firstSize || secondIndex < secondSize) {
      const bool useFirst =
        secondIndex >= secondSize
        || (firstIndex < firstSize && first[firstIndex] < second[secondIndex]);
      const int source = useFirst ? first[firstIndex++] : second[secondIndex++];
      if (lineageSize > 0u && lineage[lineageSize - 1u] == source) {
        return false;
      }
      lineage[lineageSize++] = source;
    }
    return true;
  }

  KFPARTICLE_GPU_HOST_DEVICE inline bool LoadSource(
    const KFParticleGpuConstInputTrackSoAView& tracks,
    const KFParticleGpuConstCandidatePoolView& candidates,
    unsigned int sourceKind,
    unsigned int sourceIndex,
    int pdg,
    float mass,
    unsigned int eventIndex,
    KFParticleGpuFitState& fit,
    int lineage[MaximumLineageSize],
    unsigned int& lineageSize)
  {
    if (sourceKind == KFGpuGraphSourceCandidateGeneration) {
      if (sourceIndex >= candidates.Size()
          || candidates.Metadata().EventIndex(sourceIndex) != eventIndex
          || (pdg != 0 && candidates.Metadata().Pdg(sourceIndex) != pdg)) {
        return false;
      }
      LoadCandidateFit(candidates, sourceIndex, fit);
      return LoadLineage(candidates, sourceIndex, lineage, lineageSize);
    }
    if (sourceKind == KFGpuGraphSourceTrackRange) {
      if (sourceIndex >= tracks.Size()) {
        return false;
      }
      const int sourcePdg = tracks.Pdg(sourceIndex);
      const int absolutePdg = pdg < 0 ? -pdg : pdg;
      const int absoluteSourcePdg = sourcePdg < 0 ? -sourcePdg : sourcePdg;
      const bool matches = pdg == 0
        || (absolutePdg < 1000 && absoluteSourcePdg == absolutePdg)
        || (absolutePdg >= 1000 && absolutePdg < 10000
            && absoluteSourcePdg == 2000003112)
        || (absolutePdg >= 10000 && absoluteSourcePdg >= 1000020030);
      if (!matches) return false;
      KFParticleGpuTrackState track;
      LoadTrackState(tracks, sourceIndex, track);
      fit.Initialize(track, tracks.Charge(sourceIndex), mass);
      lineage[0] = tracks.SourceId(sourceIndex);
      lineageSize = 1u;
      return true;
    }
    return false;
  }

  KFPARTICLE_GPU_HOST_DEVICE inline bool MergeSourceLineage(
    const int first[MaximumLineageSize],
    unsigned int firstSize,
    const int second[MaximumLineageSize],
    unsigned int secondSize,
    int lineage[MaximumLineageSize],
    unsigned int& lineageSize)
  {
    if (firstSize + secondSize > MaximumLineageSize) return false;
    unsigned int firstIndex = 0u;
    unsigned int secondIndex = 0u;
    lineageSize = 0u;
    while (firstIndex < firstSize || secondIndex < secondSize) {
      const bool useFirst = secondIndex >= secondSize
        || (firstIndex < firstSize && first[firstIndex] < second[secondIndex]);
      const int source = useFirst ? first[firstIndex++] : second[secondIndex++];
      if (lineageSize > 0u && lineage[lineageSize - 1u] == source) return false;
      lineage[lineageSize++] = source;
    }
    return true;
  }

  KFPARTICLE_GPU_HOST_DEVICE inline bool ApplyFinalOperations(
    const KFParticleGpuGraphOperationDescriptor& descriptor,
    const KFParticleGpuGraphOperationTask& task,
    const KFParticleGpuConstVertexSoAView& primaryVertices,
    KFParticleGpuFitState& fit,
    KFParticleGpuGraphOperationResult& result)
  {
    if ((descriptor.operationMask & KFGpuGraphExtrapolate) != 0u) {
      if (descriptor.primaryVertexIndex < 0
          || static_cast<unsigned int>(descriptor.primaryVertexIndex)
               >= primaryVertices.Size()) {
        return false;
      }
      KFParticleGpuVertexState vertex;
      LoadVertexState(
        primaryVertices, static_cast<unsigned int>(descriptor.primaryVertexIndex), vertex);
      if (!KFParticleGpuMath::TransportLineToPoint(fit, vertex, fit)) {
        return false;
      }
    }
    if ((descriptor.operationMask & KFGpuGraphSetProductionVertex) != 0u) {
      if (descriptor.primaryVertexIndex < 0
          || static_cast<unsigned int>(descriptor.primaryVertexIndex)
               >= primaryVertices.Size()) {
        return false;
      }
      KFParticleGpuVertexState vertex;
      LoadVertexState(
        primaryVertices, static_cast<unsigned int>(descriptor.primaryVertexIndex), vertex);
      if (!KFParticleGpuMath::ConstrainToProductionVertex(fit, vertex)) {
        return false;
      }
    }
    if (((descriptor.operationMask & KFGpuGraphMassConstraint) != 0u
         || (descriptor.flags & KFGpuGraphApplyMassConstraint) != 0u)
        && (descriptor.flags & KFGpuGraphSelectEventPrimaryVertices) == 0u) {
      if (descriptor.massConstraint < 0.f
          || !KFParticleGpuMath::ApplyLinearMassConstraint(
            fit, descriptor.massConstraint, descriptor.massConstraintSigma)) {
        return false;
      }
    }
    if ((descriptor.operationMask & KFGpuGraphMatch) != 0u
        && descriptor.maxMatchMomentum >= 0.f) {
      const float momentum2 = KFParticleGpuMath::Momentum2(fit);
      if (!IsFinite(momentum2)
          || momentum2 > descriptor.maxMatchMomentum * descriptor.maxMatchMomentum) {
        result.status = KFGpuGraphTaskRejectSelection;
        return false;
      }
    }
    if ((descriptor.operationMask & KFGpuGraphMatch) != 0u
        && descriptor.maxMatchMomentumSigma >= 0.f) {
      for (int parameter = 3; parameter <= 5; ++parameter) {
        const float variance = fit.Covariance(parameter, parameter);
        const float error = variance > 0.f ? KFParticleGpuSqrt(variance) : -1.f;
        if (!IsFinite(error) || error < 0.f
            || fit.Parameter(parameter)
                 > descriptor.maxMatchMomentumSigma * error) {
          result.status = KFGpuGraphTaskRejectSelection;
          return false;
        }
      }
    }
    if ((descriptor.operationMask & KFGpuGraphSelect) != 0u) {
      const float geometricChi2 =
        fit.NDF() > 0 ? fit.Chi2() / static_cast<float>(fit.NDF()) : 1.e8f;
      if (fit.NDF() < 0 || fit.Chi2() < 0.f || !IsFinite(geometricChi2)
          || (descriptor.maxGeometricChi2PerNdf >= 0.f
              && geometricChi2 > descriptor.maxGeometricChi2PerNdf)) {
        result.status = KFGpuGraphTaskRejectSelection;
        return false;
      }
      if (descriptor.maxTopologyChi2PerNdf >= 0.f
          && (descriptor.flags & KFGpuGraphSelectEventPrimaryVertices) == 0u) {
        if (descriptor.primaryVertexIndex < 0
            || static_cast<unsigned int>(descriptor.primaryVertexIndex)
                 >= primaryVertices.Size()) {
          result.status = KFGpuGraphTaskRejectSelection;
          return false;
        }
        KFParticleGpuVertexState vertex;
        LoadVertexState(
          primaryVertices, static_cast<unsigned int>(descriptor.primaryVertexIndex), vertex);
        KFParticleGpuPrimaryVertexTopologyObservables topology;
        if (!KFParticleGpuSelection::BuildPrimaryVertexTopologyObservables(
              fit, vertex, topology)
            || topology.chi2PerNdf > descriptor.maxTopologyChi2PerNdf) {
          result.status = KFGpuGraphTaskRejectSelection;
          return false;
        }
        result.topologyChi2PerNdf = topology.chi2PerNdf;
      }
      if ((descriptor.flags & KFGpuGraphSelectEventPrimaryVertices) != 0u) {
        const KFParticleGpuRange vertexRange(
          task.primaryVertexOffset, task.primaryVertexCount);
        KFParticleGpuV0LineTopologyResult nearestLine;
        if (!KFParticleGpuSelection::FindNearestV0LineTopology(
              fit, primaryVertices, vertexRange, nearestLine)
            || !KFParticleGpuSelection::PointsFromAnyPrimaryVertex(
                 fit, primaryVertices, vertexRange)
            || (descriptor.maxSelectionVertexDistance >= 0.f
                && nearestLine.lineDistance
                     >= descriptor.maxSelectionVertexDistance)
            || (descriptor.minSelectionDecayLengthOverError >= 0.f
                && nearestLine.decayLdL
                     <= descriptor.minSelectionDecayLengthOverError)) {
          result.status = KFGpuGraphTaskRejectSelection;
          return false;
        }
        int bestVertex = -1;
        float bestChi2PerNdf = 1.e8f;
        for (unsigned int local = 0u; local < vertexRange.size; ++local) {
          const unsigned int vertexIndex = vertexRange.offset + local;
          if (vertexIndex >= primaryVertices.Size()) {
            result.status = KFGpuGraphTaskRejectSelection;
            return false;
          }
          KFParticleGpuVertexState vertex;
          LoadVertexState(primaryVertices, vertexIndex, vertex);
          KFParticleGpuFitState constrained = fit;
          if (!KFParticleGpuMath::ConstrainToProductionVertex(
                constrained, vertex)
              || constrained.NDF() <= 0 || constrained.Chi2() <= 0.f) {
            continue;
          }
          const float chi2PerNdf =
            constrained.Chi2() / static_cast<float>(constrained.NDF());
          if (IsFinite(chi2PerNdf) && chi2PerNdf < bestChi2PerNdf) {
            bestChi2PerNdf = chi2PerNdf;
            bestVertex = static_cast<int>(vertexIndex);
          }
        }
        if (bestVertex < 0
            || (descriptor.maxTopologyChi2PerNdf >= 0.f
                && bestChi2PerNdf > descriptor.maxTopologyChi2PerNdf)) {
          result.status = KFGpuGraphTaskRejectSelection;
          return false;
        }
        result.primaryVertexIndex = bestVertex;
        result.topologyChi2PerNdf = bestChi2PerNdf;
        if (descriptor.matchMassWindowSigma >= 0.f) {
          float mass = 0.f;
          float massError = 0.f;
          if (descriptor.matchMassWindowCut < 0.f
              || !KFParticleGpuMath::GetMass(fit, mass, massError)
              || !IsFinite(mass)
              || (mass > descriptor.massConstraint
                    ? mass - descriptor.massConstraint
                    : descriptor.massConstraint - mass)
                   > descriptor.matchMassWindowCut
                       * descriptor.matchMassWindowSigma) {
            result.status = KFGpuGraphTaskRejectSelection;
            return false;
          }
        }
        if ((descriptor.operationMask & KFGpuGraphMassConstraint) != 0u
            || (descriptor.flags & KFGpuGraphApplyMassConstraint) != 0u) {
          if (descriptor.massConstraint < 0.f
              || !KFParticleGpuMath::ApplyLinearMassConstraint(
                   fit, descriptor.massConstraint,
                   descriptor.massConstraintSigma)) {
            result.status = KFGpuGraphTaskRejectMathematics;
            return false;
          }
        }
      }
    }
    return true;
  }

  KFPARTICLE_GPU_HOST_DEVICE inline bool ExecuteTask(
    const KFParticleGpuConstInputTrackSoAView& tracks,
    const KFParticleGpuConstCandidatePoolView& candidates,
    const KFParticleGpuConstVertexSoAView& primaryVertices,
    const KFParticleGpuGraphOperationDescriptor& descriptor,
    const KFParticleGpuGraphOperationTask& task,
    KFParticleGpuFitState& output,
    int lineage[MaximumLineageSize],
    unsigned int& lineageSize,
    KFParticleGpuGraphOperationResult& result)
  {
    result = KFParticleGpuGraphOperationResult();
    result.channelId = descriptor.channelId;
    result.outputClass = descriptor.outputClass;
    if (!ValidateDescriptor(descriptor)) {
      result.status = KFGpuGraphTaskRejectDescriptor;
      return false;
    }
    KFParticleGpuFitState first;
    int firstLineage[MaximumLineageSize];
    unsigned int firstLineageSize = 0u;
    if (!LoadSource(tracks, candidates, task.firstSourceKind,
                    task.firstCandidateIndex, descriptor.firstPdg,
                    descriptor.firstMass, task.eventIndex, first,
                    firstLineage, firstLineageSize)) {
      result.status = KFGpuGraphTaskRejectSource;
      return false;
    }
    KFParticleGpuGraphOperationDescriptor effectiveDescriptor = descriptor;
    if (effectiveDescriptor.primaryVertexIndex
          == KFGpuGraphPrimaryVertexFromCandidate) {
      effectiveDescriptor.primaryVertexIndex =
        task.firstSourceKind == KFGpuGraphSourceCandidateGeneration
          ? candidates.Metadata().PrimaryVertexIndex(task.firstCandidateIndex)
          : tracks.PrimaryVertexIndex(task.firstCandidateIndex);
      if (effectiveDescriptor.primaryVertexIndex < 0) {
        result.status = KFGpuGraphTaskRejectSource;
        return false;
      }
    }
    result.primaryVertexIndex = effectiveDescriptor.primaryVertexIndex;
    if (descriptor.topology == KFGpuGraphTopologyUnaryComposite) {
      for (unsigned int i = 0u; i < firstLineageSize; ++i) lineage[i] = firstLineage[i];
      lineageSize = firstLineageSize;
      output = first;
    }
    else {
      if ((descriptor.flags & KFGpuGraphRequireOrderedCandidatePair) != 0u
          && task.firstSourceKind == task.secondSourceKind
          && task.secondCandidateIndex <= task.firstCandidateIndex) {
        result.status = KFGpuGraphTaskRejectLineage;
        return false;
      }

      KFParticleGpuFitState second;
      int secondLineage[MaximumLineageSize];
      unsigned int secondLineageSize = 0u;
      if (!LoadSource(tracks, candidates, task.secondSourceKind,
                      task.secondCandidateIndex, descriptor.secondPdg,
                      descriptor.secondMass, task.eventIndex, second,
                      secondLineage, secondLineageSize)) {
        result.status = KFGpuGraphTaskRejectSource;
        return false;
      }
      if (!MergeSourceLineage(firstLineage, firstLineageSize,
                              secondLineage, secondLineageSize,
                              lineage, lineageSize)) {
        result.status = KFGpuGraphTaskRejectLineage;
        return false;
      }
      if ((descriptor.operationMask & KFGpuGraphMatch) != 0u) {
        if ((descriptor.flags & KFGpuGraphMatchTrackPrimaryVertex) != 0u) {
          if (task.firstSourceKind != KFGpuGraphSourceCandidateGeneration
              || task.firstCandidateIndex >= candidates.Size()
              || task.secondSourceKind != KFGpuGraphSourceTrackRange
              || task.secondCandidateIndex >= tracks.Size()) {
            result.status = KFGpuGraphTaskRejectSource;
            return false;
          }
          const int primaryVertexIndex =
            tracks.PrimaryVertexIndex(task.secondCandidateIndex);
          if (primaryVertexIndex < 0
              || static_cast<unsigned int>(primaryVertexIndex)
                   >= primaryVertices.Size()) {
            result.status = KFGpuGraphTaskRejectSource;
            return false;
          }
          if (descriptor.maxMatchTopologyChi2PerNdf >= 0.f) {
            KFParticleGpuVertexState vertex;
            LoadVertexState(primaryVertices,
                            static_cast<unsigned int>(primaryVertexIndex),
                            vertex);
            KFParticleGpuPrimaryVertexTopologyObservables topology;
            if (!KFParticleGpuSelection::BuildPrimaryVertexTopologyObservables(
                  first, vertex, topology)
                || topology.chi2PerNdf
                     > descriptor.maxMatchTopologyChi2PerNdf) {
              result.status = KFGpuGraphTaskRejectSelection;
              return false;
            }
            result.topologyChi2PerNdf = topology.chi2PerNdf;
          }
          result.primaryVertexIndex = primaryVertexIndex;
        }
        if (descriptor.matchMassWindowSigma >= 0.f) {
          float mass = 0.f;
          float massError = 0.f;
          if (descriptor.matchMassWindowCut < 0.f
              || !KFParticleGpuMath::GetMass(first, mass, massError)
              || !IsFinite(mass)
              || (mass > descriptor.massConstraint
                    ? mass - descriptor.massConstraint
                    : descriptor.massConstraint - mass)
                   > descriptor.matchMassWindowCut
                       * descriptor.matchMassWindowSigma) {
            result.status = KFGpuGraphTaskRejectSelection;
            return false;
          }
        }
        if ((effectiveDescriptor.operationMask & KFGpuGraphMassConstraint) != 0u
            || (effectiveDescriptor.flags & KFGpuGraphApplyMassConstraint) != 0u) {
          if (effectiveDescriptor.massConstraint < 0.f
              || !KFParticleGpuMath::ApplyLinearMassConstraint(
                   first, effectiveDescriptor.massConstraint,
                   effectiveDescriptor.massConstraintSigma)) {
            result.status = KFGpuGraphTaskRejectMathematics;
            return false;
          }
          effectiveDescriptor.operationMask &= ~KFGpuGraphMassConstraint;
          effectiveDescriptor.flags &= ~KFGpuGraphApplyMassConstraint;
        }
        if (descriptor.maxMatchDistance >= 0.f) {
          const float dx = first.X() - second.X();
          const float dy = first.Y() - second.Y();
          const float dz = first.Z() - second.Z();
          const float distance2 = dx * dx + dy * dy + dz * dz;
          if (!IsFinite(distance2)
              || distance2 > descriptor.maxMatchDistance
                               * descriptor.maxMatchDistance) {
            result.status = KFGpuGraphTaskRejectSelection;
            return false;
          }
        }
      }
      if (descriptor.topology == KFGpuGraphTopologyCompositeComposite) {
        KFParticleGpuMeasurement measurement;
        if (!KFParticleGpuMath::BuildLineDcaMeasurementSeed(
              first, second, output, measurement)
            || !KFParticleGpuMath::AddDaughterWithEnergyFit(
              output, measurement, second.Q())) {
          result.status = KFGpuGraphTaskRejectMathematics;
          return false;
        }
      }
      else {
        if ((descriptor.flags & KFGpuGraphRequirePositiveMissingEnergy) != 0u
            && first.E() <= second.E()) {
          result.status = KFGpuGraphTaskRejectMathematics;
          return false;
        }
        KFParticleGpuMeasurement measurement;
        KFParticleGpuFitState firstAtDca;
        if (!KFParticleGpuMath::BuildLineDcaMeasurementSeed(
              first, second, firstAtDca, measurement)) {
          result.status = KFGpuGraphTaskRejectMathematics;
          return false;
        }
        if (descriptor.missingMassMode
              == KFGpuGraphMissingMassFilteredReconstruction) {
          KFParticleGpuFitState neutral;
          KFParticleGpuFitState daughterFiltered;
          if (!KFParticleGpuMath::ReconstructMissingMassFiltered(
                firstAtDca, measurement, second.Q(), descriptor.firstMass,
                descriptor.secondMass, descriptor.neutralMass, neutral,
                output, daughterFiltered)) {
            result.status = KFGpuGraphTaskRejectMathematics;
            return false;
          }
        }
        else {
          output = firstAtDca;
          output.NDF() = -1;
          output.Chi2() = 0.f;
          if (!KFParticleGpuMath::SubtractDaughterWithEnergyFit(
                output, measurement, second.Q())) {
            result.status = KFGpuGraphTaskRejectMathematics;
            return false;
          }
        }
      }
    }

    if (!ApplyFinalOperations(
          effectiveDescriptor, task, primaryVertices, output, result)) {
      if (result.status == KFGpuGraphTaskPending) {
        result.status = KFGpuGraphTaskRejectMathematics;
      }
      return false;
    }
    if ((descriptor.operationMask & KFGpuGraphMatch) != 0u
        && (descriptor.flags & KFGpuGraphStoreFirstOnMatch) != 0u) {
      output = first;
    }
    if (!KFParticleGpuMath::GetMass(output, result.mass, result.massError)
        || !IsFinite(result.mass)) {
      if (((descriptor.operationMask & KFGpuGraphMassConstraint) != 0u
           || (descriptor.flags & KFGpuGraphApplyMassConstraint) != 0u)
          && descriptor.massConstraint >= 0.f
          && KFParticleGpuMath::IsFiniteState(output)) {
        // Exact constraints can leave a tiny negative projected mass variance
        // after float roundoff. The constrained mass remains valid and exact.
        result.mass = descriptor.massConstraint;
        result.massError = 0.f;
      }
      else if (descriptor.topology == KFGpuGraphTopologyNeutralDaughter
          && descriptor.missingMassMode
               == KFGpuGraphMissingMassFilteredReconstruction
          && descriptor.firstMass >= 0.f
          && KFParticleGpuMath::IsFiniteState(output)) {
        result.mass = descriptor.firstMass;
        result.massError = -1.f;
      }
      else {
        result.status = KFGpuGraphTaskRejectMathematics;
        return false;
      }
    }
    result.status = KFGpuGraphTaskAccepted;
    result.daughterCount = lineageSize;
    return true;
  }

  KFPARTICLE_GPU_HOST_DEVICE inline bool ExecuteTask(
    const KFParticleGpuConstCandidatePoolView& candidates,
    const KFParticleGpuConstVertexSoAView& primaryVertices,
    const KFParticleGpuGraphOperationDescriptor& descriptor,
    const KFParticleGpuGraphOperationTask& task,
    KFParticleGpuFitState& output,
    int lineage[MaximumLineageSize],
    unsigned int& lineageSize,
    KFParticleGpuGraphOperationResult& result)
  {
    return ExecuteTask(KFParticleGpuConstInputTrackSoAView(), candidates,
                       primaryVertices, descriptor, task, output, lineage,
                       lineageSize, result);
  }

  KFPARTICLE_GPU_HOST_DEVICE inline bool StoreOutput(
    const KFParticleGpuCandidatePoolView& candidates,
    const KFParticleGpuGraphOperationDescriptor& descriptor,
    const KFParticleGpuGraphOperationTask& task,
    unsigned int eventIndex,
    int primaryVertexIndex,
    unsigned int candidateIndex,
    unsigned int daughterOffset,
    const KFParticleGpuFitState& fit,
    const int lineage[MaximumLineageSize],
    unsigned int lineageSize)
  {
    if (!candidates.CanStoreCandidate(candidateIndex)
        || !candidates.Daughters().CanStore(daughterOffset, lineageSize)) {
      return false;
    }
    StoreCandidateFit(fit, candidates, candidateIndex);
    candidates.Metadata().Pdg(candidateIndex) = descriptor.motherPdg;
    candidates.Metadata().PrimaryVertexIndex(candidateIndex) =
      primaryVertexIndex;
    candidates.Metadata().EventIndex(candidateIndex) = eventIndex;
    candidates.Metadata().DaughterOffset(candidateIndex) = daughterOffset;
    candidates.Metadata().DaughterCount(candidateIndex) = lineageSize;
    candidates.Metadata().Flags(candidateIndex) = KFGpuCandidateValid;
    candidates.Metadata().ChannelId(candidateIndex) = descriptor.channelId;
    candidates.Metadata().Topology(candidateIndex) = descriptor.topology;
    candidates.Metadata().OutputClass(candidateIndex) = descriptor.outputClass;
    candidates.Metadata().OperationStatus(candidateIndex) = KFGpuCandidateOperationAccepted;
    const bool unary = descriptor.topology == KFGpuGraphTopologyUnaryComposite;
    candidates.Metadata().DirectDaughterCount(candidateIndex) = unary ? 1u : 2u;
    SetCandidateDirectDaughter(
      candidates.Metadata(), candidateIndex, 0u,
      task.firstSourceKind == KFGpuGraphSourceTrackRange
        ? KFGpuDirectDaughterInputTrack : KFGpuDirectDaughterCandidate,
      task.firstCandidateIndex);
    if (unary) {
      candidates.Metadata().DirectSecondKind(candidateIndex) = KFGpuDirectDaughterNone;
      candidates.Metadata().DirectSecondIndex(candidateIndex) = 0u;
    }
    else {
      SetCandidateDirectDaughter(
        candidates.Metadata(), candidateIndex, 1u,
        task.secondSourceKind == KFGpuGraphSourceTrackRange
          ? KFGpuDirectDaughterInputTrack : KFGpuDirectDaughterCandidate,
        task.secondCandidateIndex);
    }
    for (unsigned int daughter = 0u; daughter < lineageSize; ++daughter) {
      candidates.Daughters().SourceId(daughterOffset + daughter) = lineage[daughter];
    }
    return true;
  }
}

static_assert(std::is_trivially_copyable<KFParticleGpuGraphOperationDescriptor>::value,
              "KFParticle GPU graph operation descriptors must remain flat device values");
static_assert(std::is_trivially_copyable<KFParticleGpuGraphOperationTask>::value,
              "KFParticle GPU graph operation tasks must remain flat device values");
static_assert(std::is_trivially_copyable<KFParticleGpuGraphOperationResult>::value,
              "KFParticle GPU graph operation results must remain flat device values");
static_assert(std::is_trivially_copyable<KFParticleGpuGraphOperationStorageView>::value,
              "KFParticle GPU graph operation storage must remain a flat device ABI");
static_assert(
  std::is_trivially_default_constructible<KFParticleGpuGraphOperationStorageView>::value,
  "KFParticle GPU graph operation storage must not require dynamic initialization");

#endif
