/*
 * This file is part of KFParticle package
 * Copyright (C) 2007-2026
 *
 * KFParticle is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 */

#ifndef KFPARTICLEGPUROUTINGPLAN_H
#define KFPARTICLEGPUROUTINGPLAN_H

#include "KFParticleGpuChannelRouting.h"
#include "KFParticleGpuTwoDaughterRouting.h"

#include <vector>

class KFParticleGpuDecayPlan;

/** Host-owned compiled routing tables uploaded only when the decay plan changes. */
class KFParticleGpuV0TrackRoutingPlan
{
 public:
  void Compile(const KFParticleGpuDecayPlan& plan);
  void Clear();

  unsigned long long SourceRevision() const { return fSourceRevision; }
  const std::vector<KFParticleGpuV0TrackRoutingDescriptor>& Descriptors() const
  {
    return fDescriptors;
  }
  const std::vector<KFParticleGpuV0TrackCompatibilityEntry>& CompatibilityEntries() const
  {
    return fCompatibility;
  }
  const std::vector<KFParticleGpuV0TrackExecutionGroup>& Groups() const { return fGroups; }
  const KFParticleGpuChannelMask& EnabledChannels() const { return fEnabledChannels; }

  KFParticleGpuChannelMask CompatibleChannels(unsigned int selectedV0Role,
                                              unsigned int bachelorRole) const;
  KFParticleGpuChannelMask CompatibleChannelsByPdg(unsigned int parentChannelId,
                                                   int selectedPdg,
                                                   int bachelorPdg) const;

 private:
  unsigned long long fSourceRevision = 0u;
  std::vector<KFParticleGpuV0TrackRoutingDescriptor> fDescriptors;
  std::vector<KFParticleGpuV0TrackCompatibilityEntry> fCompatibility;
  std::vector<KFParticleGpuV0TrackExecutionGroup> fGroups;
  KFParticleGpuChannelMask fEnabledChannels;
};

/** Host-owned two-daughter routing tables compiled from one decay-plan revision. */
class KFParticleGpuTwoDaughterRoutingPlan
{
 public:
  void Compile(const KFParticleGpuDecayPlan& plan);
  void Clear();

  unsigned long long SourceRevision() const { return fSourceRevision; }
  const std::vector<KFParticleGpuTwoDaughterRoutingDescriptor>& Descriptors() const
  {
    return fDescriptors;
  }
  const std::vector<KFParticleGpuTwoDaughterCompatibilityEntry>& CompatibilityEntries() const
  {
    return fCompatibility;
  }
  const std::vector<KFParticleGpuTwoDaughterExecutionGroup>& Groups() const
  {
    return fGroups;
  }
  const KFParticleGpuChannelMask& EnabledChannels() const { return fEnabledChannels; }

  KFParticleGpuChannelMask CompatibleChannels(unsigned int firstRole,
                                              unsigned int secondRole) const;
  KFParticleGpuChannelMask CompatibleChannelsByPdg(int firstPdg,
                                                   int secondPdg) const;

 private:
  unsigned long long fSourceRevision = 0u;
  std::vector<KFParticleGpuTwoDaughterRoutingDescriptor> fDescriptors;
  std::vector<KFParticleGpuTwoDaughterCompatibilityEntry> fCompatibility;
  std::vector<KFParticleGpuTwoDaughterExecutionGroup> fGroups;
  KFParticleGpuChannelMask fEnabledChannels;
};

#endif
