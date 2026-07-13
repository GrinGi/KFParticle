/*
 * This file is part of KFParticle package
 * Copyright (C) 2007-2026
 *
 * KFParticle is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 */

#ifndef KFPARTICLEGPUDEVICESTORAGE_H
#define KFPARTICLEGPUDEVICESTORAGE_H

#include "KFParticleGpuCandidatePool.h"
#include "KFParticleGpuInputData.h"
#include "KFParticleGpuTwoDaughter.h"

#ifdef KFPARTICLE_USE_XPU
#include <xpu/host.h>

/**
 * Visible owner layout for process-persistent KFParticle GPU buffers.
 *
 * Storage is grouped by its role in the reconstruction pipeline. The current
 * buffer-manager API is a transitional adapter; kernels must ultimately use a
 * KFParticleGpuKernelState rebuilt from this owner, never own event memory.
 */
class KFParticleGpuDeviceStorage
{
 public:
  KFParticleGpuDeviceStorage() = default;
  KFParticleGpuDeviceStorage(const KFParticleGpuDeviceStorage&) = delete;
  KFParticleGpuDeviceStorage& operator=(const KFParticleGpuDeviceStorage&) = delete;

  // Packed event input, uploaded once for a device-resident reconstruction chain.
  xpu::buffer<float> fTrackParameters;
  xpu::buffer<float> fTrackCovariances;
  xpu::buffer<float> fTrackField;
  xpu::buffer<float> fTrackChiToPrimaryVertex;
  xpu::buffer<int> fTrackIntegers;
  xpu::buffer<float> fVertexParameters;
  xpu::buffer<float> fVertexCovariances;
  xpu::buffer<float> fVertexChi2;
  xpu::buffer<int> fVertexIntegers;
  xpu::buffer<KFParticleGpuEventDesc> fEvents;

  // Reusable intermediate storage. Kernels will consume these pools directly.
  xpu::buffer<KFParticleGpuTwoDaughterTask> fTwoDaughterTasks;
  xpu::buffer<unsigned int> fTwoDaughterTaskCount;
  xpu::buffer<unsigned int> fTwoDaughterTotalPairCount;
  xpu::buffer<unsigned int> fTwoDaughterTaskOverflowFlags;

  // Diagnostic raw output retained for downstream device stages and debugging.
  xpu::buffer<float> fCandidateParameters;
  xpu::buffer<float> fCandidateCovariances;
  xpu::buffer<float> fCandidateFitScalars;
  xpu::buffer<int> fCandidateFitIntegers;
  xpu::buffer<int> fCandidateMetadataIntegers;
  xpu::buffer<unsigned int> fCandidateMetadataUnsigned;
  xpu::buffer<int> fDaughterSourceIds;
  xpu::buffer<unsigned int> fCandidateSize;
  xpu::buffer<unsigned int> fDaughterSize;
  xpu::buffer<unsigned int> fOverflowFlags;

  // One selection diagnostic per raw candidate. It preserves the decision
  // without duplicating the raw fit/covariance SoA.
  xpu::buffer<KFParticleGpuV0SelectionResult> fV0SelectionResults;

  // Compact V0 output. It references the raw pool instead of duplicating fit data.
  xpu::buffer<unsigned int> fSelectedCandidateIndices;
  xpu::buffer<unsigned int> fSelectedCandidateChannelIds;
  xpu::buffer<unsigned int> fSelectedCandidateSize;
  xpu::buffer<unsigned int> fSelectedCandidateOverflowFlags;
};

#else

class KFParticleGpuDeviceStorage
{
};

#endif

#endif
