/*
 * This file is part of KFParticle package
 * Copyright (C) 2007-2026
 *
 * KFParticle is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 */

#ifndef KFPARTICLEGPUBUFFERMANAGER_H
#define KFPARTICLEGPUBUFFERMANAGER_H

#include "KFParticleGpuCandidatePool.h"
#include "KFParticleGpuDeviceStorage.h"
#include "KFParticleGpuInputData.h"
#include "KFParticleGpuSelection.h"

#include <memory>

namespace xpu
{
  class queue;
}

struct KFParticleGpuBufferCapacities
{
  unsigned int tracks = 0;
  unsigned int vertices = 0;
  unsigned int events = 0;
  unsigned int candidates = 0;
  unsigned int daughterIds = 0;
  unsigned int selectedCandidates = 0;
  unsigned int twoDaughterTasks = 0;
  bool nonhomogeneousField = false;
};

struct KFParticleGpuTwoDaughterTaskStatus
{
  unsigned int accepted = 0;
  unsigned int totalPairs = 0;
  unsigned int overflowFlags = 0;
};

struct KFParticleGpuCandidatePoolStatus
{
  unsigned int candidates = 0;
  unsigned int daughters = 0;
  unsigned int overflowFlags = 0;
};

/**
 * Process-persistent owner of KFParticle XPU buffers.
 *
 * Host and device views are temporary non-owning descriptors. Capacity grows
 * on demand and is not reduced between events.
 */
class KFParticleGpuBufferManager
{
 public:
  explicit KFParticleGpuBufferManager(xpu::queue& queue);
  ~KFParticleGpuBufferManager();

  KFParticleGpuBufferManager(const KFParticleGpuBufferManager&) = delete;
  KFParticleGpuBufferManager& operator=(const KFParticleGpuBufferManager&) = delete;

  void EnsureCapacity(const KFParticleGpuBufferCapacities& requested);
  const KFParticleGpuBufferCapacities& Capacities() const;
  const KFParticleGpuDeviceStorage& Storage() const;

  void SetInputSizes(unsigned int tracks, unsigned int vertices, unsigned int events);
  void EnsureTwoDaughterTaskCapacity(unsigned int capacity);
  unsigned int TrackSize() const;
  unsigned int VertexSize() const;
  unsigned int EventSize() const;

  KFParticleGpuInputTrackSoAView HostInputTracks();
  KFParticleGpuInputTrackSoAView DeviceInputTracks();
  KFParticleGpuVertexSoAView HostPrimaryVertices();
  KFParticleGpuVertexSoAView DevicePrimaryVertices();
  KFParticleGpuEventDesc* HostEvents();
  KFParticleGpuEventDesc* DeviceEvents();

  KFParticleGpuCandidatePoolView HostCandidates();
  KFParticleGpuCandidatePoolView DeviceCandidates();
  KFParticleGpuV0SelectionResultView HostV0SelectionResults();
  KFParticleGpuV0SelectionResultView DeviceV0SelectionResults();
  KFParticleGpuSelectedCandidateIndexView HostSelectedCandidates();
  KFParticleGpuSelectedCandidateIndexView DeviceSelectedCandidates();

  void UploadInput();
  void UploadCandidates();
  void ResetCandidates();
  void ResetTwoDaughterTaskStatus();
  KFParticleGpuTwoDaughterTaskStatus DownloadTwoDaughterTaskStatus();
  void ResetSelectedCandidates();
  void ResetV0SelectionResults();
  void MarkCandidateOverflow(unsigned int flags);
  KFParticleGpuCandidatePoolStatus DownloadCandidateStatus();
  void DownloadCandidates();
  void DownloadV0SelectionResults();
  void DownloadSelectedCandidates();

 private:
  struct Impl;
  std::unique_ptr<Impl> fImpl;
};

#endif
