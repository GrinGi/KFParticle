/* Copyright (C) 2026 GSI Helmholtzzentrum fuer Schwerionenforschung, Darmstadt
   SPDX-License-Identifier: GPL-3.0-only */
#include "ExampleData.h"
#include "KFPTrack.h"
#include "KFPVertex.h"

namespace kfp_example {
std::vector<Event> GenerateEvents(unsigned count)
{
  // A fixed software fixture, not a detector simulation or physical decay generator.
  const float parameters[4][6] = {
    {0.f, 0.f, 5.f,  0.206f,  0.012f, 0.5f},
    {0.f, 0.f, 5.f, -0.206f, -0.012f, 0.5f},
    {0.f, 0.f, 0.f,  0.05f,   0.02f, 1.f},
    {0.f, 0.f, 0.f,  0.05f,  -0.02f, 1.f}
  };
  const int pdgs[4] = {211, -211, 2212, -2212};

  std::vector<Event> events(count);
  for (unsigned i = 0; i < count; ++i) {
    Event& event = events[i];
    event.id = i;
    event.first.Resize(4);
    event.chiToPV.resize(4);

    KFPVertex pv;
    pv.SetXYZ(0.f, 0.f, 0.f);
    pv.SetCovarianceMatrix(0.001f, 0.f, 0.001f, 0.f, 0.f, 0.001f);
    pv.SetChi2(1.f);
    pv.SetNDF(1);
    pv.SetNContributors(2);
    event.vertices.emplace_back(pv);

    for (int row = 0; row < 4; ++row) {
      KFPTrackVector& tracks = event.first;
      for (int j = 0; j < 6; ++j) { tracks.SetParameter(parameters[row][j], j, row); }
      for (int j = 0; j < 21; ++j) { tracks.SetCovariance(0.f, j, row); }
      for (int j : {0, 2, 5}) { tracks.SetCovariance(0.001f, j, row); }
      for (int j : {9, 14, 20}) { tracks.SetCovariance(0.0001f, j, row); }
      // Constant By = 0.1 kG in the ten-coefficient field representation.
      for (int j = 0; j < 10; ++j) { tracks.SetFieldCoefficient(j == 3 ? 0.1f : 0.f, j, row); }
      tracks.SetId(row, row);
      tracks.SetPDG(pdgs[row], row);
      tracks.SetQ(row % 2 ? -1 : 1, row);
      tracks.SetPVIndex(row < 2 ? -1 : 0, row);
      tracks.SetNPixelHits(4, row);
      event.chiToPV[row] = InputParticle(event, row).GetDeviationFromVertex(event.vertices.front());
    }
    event.last = event.first; // Identical first/last states suffice for this toy input.
  }
  return events;
}

std::vector<KFParticleGpuFinder::Event> MakeBatch(const std::vector<Event>& events)
{
  std::vector<KFParticleGpuFinder::Event> batch;
  batch.reserve(events.size());
  for (const Event& event : events) {
    batch.push_back({event.id, event.first, event.last, event.chiToPV, event.vertices});
  }
  return batch;
}

KFParticle InputParticle(const Event& event, int row)
{
  KFPTrack track;
  float parameters[6], covariance[21];
  for (int i = 0; i < 6; ++i) { parameters[i] = event.first.Parameter(i)[row]; }
  for (int i = 0; i < 21; ++i) { covariance[i] = event.first.Covariance(i)[row]; }
  track.SetParameters(parameters);
  track.SetCovarianceMatrix(covariance);
  for (int i = 0; i < 10; ++i) { track.SetFieldCoeff(event.first.FieldCoefficient(i)[row], i); }
  track.SetCharge(event.first.Q()[row]);
  track.SetID(event.first.Id()[row]);
  track.SetChi2(0.f);
  track.SetNDF(0); // KFPTrackVector has no track-fit chi2/NDF.
  return KFParticle(track, event.first.PDG()[row]);
}
} // namespace kfp_example
