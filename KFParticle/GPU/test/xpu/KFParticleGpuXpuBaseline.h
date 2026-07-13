/*
 * This file is part of KFParticle package
 * Copyright (C) 2007-2026
 *
 * KFParticle is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 */

#ifndef KFPARTICLEGPUXPUBASELINE_H
#define KFPARTICLEGPUXPUBASELINE_H

#include <xpu/device.h>

struct KFParticleGpuXpuBaselineImage : xpu::device_image
{
};

struct KFParticleGpuXpuBaselineMarker : xpu::kernel<KFParticleGpuXpuBaselineImage>
{
  using context = xpu::kernel_context<xpu::no_smem>;

  XPU_D void operator()(context& context, unsigned int* marker);
};

#endif
