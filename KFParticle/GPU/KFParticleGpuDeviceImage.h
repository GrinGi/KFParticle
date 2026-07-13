/*
 * This file is part of KFParticle package
 * Copyright (C) 2007-2026
 *
 * KFParticle is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 */

#ifndef KFPARTICLEGPUDEVICEIMAGE_H
#define KFPARTICLEGPUDEVICEIMAGE_H

#include "KFParticleGpuPlatform.h"

#ifdef KFPARTICLE_GPU_XPU_ENABLED
struct KFParticleGpuDeviceImage : xpu::device_image {};
#else
struct KFParticleGpuDeviceImage {};
#endif

#endif
