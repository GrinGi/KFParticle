/* Copyright (C) 2026 GSI Helmholtzzentrum fuer Schwerionenforschung, Darmstadt
   SPDX-License-Identifier: GPL-3.0-only */
#include "ExampleData.h"
#include "KFTopoPerformance.h"

#include <TFile.h>

#include <iostream>
#include <stdexcept>
#include <vector>

int main(int argc, char** argv)
{
  try {
    // ---- INPUT DATA: OWNED ARRAYS -> BORROWED BATCH VIEWS -----------------
    // kfp_example::Event is an example-only container, not a Finder requirement:
    //   id         std::uint64_t         Event ID, unique within the batch.
    //   first/last KFPTrackVector        Fitted tracks: parameters, covariance,
    //                                   charge, PDG, field and PV assignment.
    //   chiToPV    std::vector<float>    One track-to-PV compatibility per track.
    //   vertices   std::vector<KFVertex> Primary vertices belonging to this event.
    // First/last track IDs must match; PVIndex indexes vertices locally (-1: none).
    //
    // data[i] owns arrays; batch[i] is a KFParticleGpuFinder::Event borrowing them.
    // No track arrays are copied or uploaded by MakeBatch: reconstruction does that.
    // For your own input, fill these arrays in your own container and construct
    // KFParticleGpuFinder::Event references to them, as MakeBatch does here.
    // Keep arrays alive and unchanged until ReconstructBatch returns; this example
    // also keeps them for the QA below. See ExampleData.h/.cxx for the layout.
    // --------------------------------------------------------------------

    // 1. Generate eight toy events and borrow their arrays as one input batch.
    // No input file or MC is used.
    const std::vector<kfp_example::Event> data = kfp_example::GenerateEvents(8);
    const std::vector<KFParticleGpuFinder::Event> batch = kfp_example::MakeBatch(data);

    // 2. Select the device and decay channel, then create the Finder.
    // The optional first argument is a device: hip0 (default), hip1, or cpu.
    KFParticleGpuFinder::Options options;
    options.device = argc > 1 ? argv[1] : "hip0";
    options.motherPdgs = {310}; // K0S -> pi+ pi-; other settings keep their defaults.
    KFParticleGpuFinder finder(options);

    // 3. Reconstruct the batch. This synchronous call returns owned CPU results.
    const std::vector<KFParticleGpuFinder::Result> results = finder.ReconstructBatch(batch);
    finder.ReleaseResources(); // Results stay valid after GPU buffers are released.

    // 4. Access the reconstructed particles independently for each event.
    if (results.size() != data.size()) { throw std::runtime_error("missing event result"); }
    for (const KFParticleGpuFinder::Result& event : results) {
      if (!event.Succeeded()) { throw std::runtime_error(event.message); }
      const std::vector<KFParticle>& particles = event.particles; // Use these objects in your analysis.
      std::cout << "Event " << event.sourceEventId << ": "
                << particles.size() << " particles, " << event.selectedCandidates
                << " selected candidates\n";
    }

    // 5. Create the existing KFParticle QA, using reconstructed distributions only.
    // CPU constraint-cache and MC histograms are not used by this example.
    TFile output("kfparticle-qa.root", "RECREATE");
    if (output.IsZombie()) { throw std::runtime_error("cannot create QA file"); }
    KFTopoPerformance qa;
    qa.DoNotStoreMCHistograms();
    qa.DoNotStorePrimSecHistograms();
    qa.DoNotStoreZRHistograms();
    qa.CreateHistos("GPU", &output, {{310, true}});
    kfp_example::FillQa(qa, data, results); // Adapt GPU particles to the existing QA input.

    // 6. Save the histograms. Open this file with ROOT to inspect them.
    if (output.Write() <= 0 || output.TestBit(TFile::kWriteError)) {
      throw std::runtime_error("cannot write QA file");
    }
    std::cout << "Wrote kfparticle-qa.root\n";
    return 0;
  }
  catch (const std::exception& error) {
    std::cerr << "FAIL example: " << error.what() << '\n';
    return 1;
  }
}
