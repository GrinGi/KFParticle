/* Copyright (C) 2026 GSI Helmholtzzentrum fuer Schwerionenforschung, Darmstadt
   SPDX-License-Identifier: GPL-3.0-only */
#ifndef KFPARTICLE_GPU_EXAMPLE_DATA_H
#define KFPARTICLE_GPU_EXAMPLE_DATA_H

#include "execution/KFParticleGpuFinder.h"

class KFTopoPerformance;

namespace kfp_example {
/** @brief Owned toy event; track IDs equal their row indices in this example. */
struct Event {
  std::uint64_t id = 0;
  KFPTrackVector first, last;
  std::vector<float> chiToPV;
  std::vector<KFVertex> vertices;
};

/// Generate a displaced pion pair and two primary tracks per event, without MC.
std::vector<Event> GenerateEvents(unsigned count);

/// Borrow the generated arrays for a single batch; events must outlive the call.
std::vector<KFParticleGpuFinder::Event> MakeBatch(const std::vector<Event>& events);

/// Convert one generated track for the legacy QA's scalar-particle input.
KFParticle InputParticle(const Event& event, int row);

/// Fill the existing QA through private input/PV views, without CPU reconstruction.
void FillQa(KFTopoPerformance& qa, const std::vector<Event>& events,
            const std::vector<KFParticleGpuFinder::Result>& results);
} // namespace kfp_example
#endif
