/*
 * This file is part of KFParticle package
 * Copyright (C) 2007-2026
 *
 * KFParticle is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 */

#include "KFParticleGpuXpuBaseline.h"

XPU_IMAGE(KFParticleGpuXpuBaselineImage);

XPU_EXPORT(KFParticleGpuXpuBaselineMarker);
XPU_D void KFParticleGpuXpuBaselineMarker::operator()(context& context, unsigned int* marker)
{
  const unsigned int thread =
    static_cast<unsigned int>(context.pos().block_idx_x() * context.pos().block_dim_x()
                              + context.pos().thread_idx_x());
  if (thread == 0 && marker) {
    marker[0] = 0x58505542u; // "XPUB": independent XPU launch baseline.
  }
}
