/*
 * This file is part of KFParticle package
 * Copyright (C) 2007-2026
 *
 * KFParticle is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 */

#ifndef KFPARTICLEGPUDECAYGRAPHPLAN_H
#define KFPARTICLEGPUDECAYGRAPHPLAN_H

#include "KFParticleGpuDecayGraph.h"

#include <limits>
#include <vector>

class KFParticleGpuDecayPlan;

class KFParticleGpuDecayGraphManifest
{
 public:
  void Clear();
  void AddNode(const KFParticleGpuGraphNode& node);
  void AddFamilyCoverage(const KFParticleGpuGraphFamilyCoverage& coverage);

  const std::vector<KFParticleGpuGraphNode>& Nodes() const { return fNodes; }
  const std::vector<KFParticleGpuGraphFamilyCoverage>& FamilyCoverage() const
  {
    return fFamilyCoverage;
  }

 private:
  std::vector<KFParticleGpuGraphNode> fNodes;
  std::vector<KFParticleGpuGraphFamilyCoverage> fFamilyCoverage;
};

struct KFParticleGpuDecayGraphCompileLimits
{
  unsigned int nodes = std::numeric_limits<unsigned int>::max();
  unsigned int groups = std::numeric_limits<unsigned int>::max();
  unsigned int families = std::numeric_limits<unsigned int>::max();
};

/** Host compiler for the flat generation-ordered device graph. */
class KFParticleGpuDecayGraphPlan
{
 public:
  void Compile(
    const KFParticleGpuDecayGraphManifest& manifest,
    const KFParticleGpuDecayGraphCompileLimits& limits =
      KFParticleGpuDecayGraphCompileLimits());
  void Clear();

  unsigned long long SourceRevision() const { return fSourceRevision; }
  const std::vector<KFParticleGpuGraphNode>& Nodes() const { return fNodes; }
  const std::vector<KFParticleGpuGraphExecutionGroup>& Groups() const { return fGroups; }
  const std::vector<KFParticleGpuGraphFamilyCoverage>& FamilyCoverage() const
  {
    return fFamilyCoverage;
  }

 private:
  unsigned long long fSourceRevision = 0u;
  std::vector<KFParticleGpuGraphNode> fNodes;
  std::vector<KFParticleGpuGraphExecutionGroup> fGroups;
  std::vector<KFParticleGpuGraphFamilyCoverage> fFamilyCoverage;
};

/**
 * CPU catalogue-backed inventory seeded from the currently implemented GPU channels.
 *
 * Remaining CPU families stay explicit as partial or unsupported coverage;
 * later stages replace those statuses as their generic engines are added.
 */
KFParticleGpuDecayGraphManifest MakeDefaultCpuFinderDecayGraphManifest(
  const KFParticleGpuDecayPlan& plan);

#endif
