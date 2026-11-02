/* Copyright (C) 2026 GSI Helmholtzzentrum fuer Schwerionenforschung, Darmstadt
   SPDX-License-Identifier: GPL-3.0-only */
#include "ExampleData.h"
#include "KFParticleTopoReconstructor.h"
#include "KFTopoPerformance.h"

#include <stdexcept>
#include <utility>

namespace kfp_example {
void FillQa(KFTopoPerformance& qa, const std::vector<Event>& events,
            const std::vector<KFParticleGpuFinder::Result>& results)
{
  if (events.size() != results.size()) { throw std::runtime_error("QA event count mismatch"); }
  for (std::size_t e = 0; e < events.size(); ++e) {
    const Event& event = events[e];
    const std::vector<KFParticle>& source = results[e].particles;

    // Legacy QA expects all input tracks and dense particle/daughter indices.
    // These private copies do not change the Finder output or run the CPU Finder.
    std::vector<KFParticle> particles;
    for (int row = 0; row < event.first.Size(); ++row) {
      KFParticle track = InputParticle(event, row);
      track.SetId(row);
      track.CleanDaughtersId();
      track.AddDaughterId(row);
      particles.push_back(track);
    }
    std::vector<int> remap(source.size());
    for (std::size_t i = 0; i < source.size(); ++i) {
      if (source[i].NDaughters() == 1) {
        remap[i] = source[i].DaughterIds().front(); // Generated source IDs are row indices.
        if (remap[i] < 0 || remap[i] >= event.first.Size()) {
          throw std::runtime_error("QA track ID outside generated input");
        }
      }
      else {
        remap[i] = particles.size();
        particles.push_back(source[i]);
      }
    }
    for (std::size_t i = 0; i < source.size(); ++i) {
      if (source[i].NDaughters() == 1) { continue; }
      KFParticle& candidate = particles[remap[i]];
      candidate.SetId(remap[i]);
      candidate.CleanDaughtersId();
      for (int daughter : source[i].DaughterIds()) { candidate.AddDaughterId(remap.at(daughter)); }
    }

    KFPTrackVector first = event.first;
    KFPTrackVector last = event.last;
    KFParticleTopoReconstructor view;
    view.GetKFParticleFinder()->AddDecayToReconstructionList(310);
    view.Init(first, last);
    for (std::size_t pv = 0; pv < event.vertices.size(); ++pv) {
      std::vector<int> members;
      for (int row = 0; row < event.first.Size(); ++row) {
        if (event.first.PVIndex()[row] == static_cast<int>(pv)) { members.push_back(row); }
      }
      view.AddPV(event.vertices[pv], members);
    }
    view.ReplaceParticles(std::move(particles));
    qa.SetTopoReconstructor(&view);
    qa.FillHistos();
    qa.SetTopoReconstructor(nullptr);
  }
}
} // namespace kfp_example
