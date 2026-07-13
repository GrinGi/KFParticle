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

#include <xpu/host.h>

#include <cassert>
#include <iostream>

#ifndef KFPARTICLE_GPU_TEST_DEVICE
#define KFPARTICLE_GPU_TEST_DEVICE "cpu"
#endif

namespace
{
  void Pass(const char* name, const char* details)
  {
    static int checkNumber = 0;
    ++checkNumber;
    std::cerr << "PASS [" << checkNumber << "] " << name << " - " << details << '\n';
  }
}

int main()
{
  xpu::settings settings;
  settings.device = KFPARTICLE_GPU_TEST_DEVICE;
  settings.verbose = false;
  xpu::initialize(settings);

  xpu::preload<KFParticleGpuXpuBaselineImage>();

  xpu::queue queue;
  xpu::buffer<unsigned int> marker(1, xpu::buf_device);
  unsigned int markerHost = 0;
  queue.memcpy(marker.get(), &markerHost, sizeof(markerHost));

  queue.launch<KFParticleGpuXpuBaselineMarker>(xpu::n_threads(1), marker.get());
  queue.wait();

  queue.memcpy(&markerHost, marker.get(), sizeof(markerHost));
  queue.wait();
  assert(markerHost == 0x58505542u);
  Pass("baseline-kernel", "XPU image preload, queue launch, and device-to-host marker readback");
  return 0;
}
