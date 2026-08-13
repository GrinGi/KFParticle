/* Copyright (C) 2026 GSI Helmholtzzentrum fuer Schwerionenforschung, Darmstadt
   SPDX-License-Identifier: GPL-3.0-only */

/// Minimal offline STS reconstruction macro:
/// STS digis -> STS clusters/hits -> CA tracks.
///
/// GPU tracking is selected through the same environment variables as before:
///   CBM_CA_TRACKING_BACKEND=gpu-single|gpu-multi
///   CBM_CA_XPU_DEVICE=hip1
///   CBM_CA_GPU_MULTIWINDOW_LENGTH_NS=<window length in ns>  (optional)

#include <RtypesCore.h>

#if !defined(__CLING__)
#include "CbmDefs.h"
#include "CbmBuildEventsFromTracksReal.h"
#include "CbmCaParametersHandler.h"
#include "CbmFindPrimaryVertex.h"
#include "CbmKF.h"
#include "CbmL1.h"
#include "CbmL1StsTrackFinder.h"
#include "CbmMCDataManager.h"
#include "CbmMatchRecoToMC.h"
#include "CbmRecoSetupManager.h"
#include "CbmRecoSts.h"
#include "CbmSetup.h"
#include "CbmStsFindTracks.h"
#include "CbmPVFinderKF.h"
#include "CbmTaskKfpSelector.h"
#include "CbmTaskRecoEventConverter.h"

#include <FairFileSource.h>
#include <FairLogger.h>
#include <FairParRootFileIo.h>
#include <FairRootFileSink.h>
#include <FairRunAna.h>
#include <FairRuntimeDb.h>

#include <TStopwatch.h>
#include <TString.h>
#include <TSystem.h>

#include <iostream>
#endif

/// \param input       Input prefix without ".raw.root"
/// \param nEntries    Number of ROOT entries to process; negative means all remaining entries
/// \param firstEntry  First ROOT entry to process
/// \param output      Output prefix without ".reco.root"; defaults to input
/// \param setup       Geometry setup tag
/// \param paramFile   Parameter prefix without ".par.root"; defaults to input
/// \param enableMcQa  Enable MC friends, matching, and CA performance information
/// \param kfpConfig    MainConfig.yaml containing kfp, eventSelector, caTimeslice, and caEvent nodes
/// \param kfpMode      disabled, cpu-only, diagnostic, or qualified-gpu
/// \param eventBasedInput  Preserve input CbmEvent objects instead of rebuilding events
/// \param kfpDiagnosticSamplePeriod  Process every Nth event through the GPU diagnostic path
void run_reco_tracks(TString input, Int_t nEntries = -1, Int_t firstEntry = 0, TString output = "",
                     TString setup = "sis100_hadron", TString paramFile = "", Bool_t enableMcQa = kFALSE,
                     TString kfpConfig = "", TString kfpMode = "disabled", Bool_t eventBasedInput = kFALSE,
                     UInt_t kfpDiagnosticSamplePeriod = 1)
{
  if (input.IsNull()) {
    std::cerr << "-E- run_reco_tracks: input prefix must not be empty" << std::endl;
    return;
  }
  if (firstEntry < 0 || nEntries == 0) {
    std::cerr << "-E- run_reco_tracks: invalid entry range" << std::endl;
    return;
  }
  kfpMode.ToLower();
  const Bool_t runKfp = kfpMode != "disabled";
  if (kfpMode != "disabled" && kfpMode != "cpu-only" && kfpMode != "diagnostic"
      && kfpMode != "qualified-gpu" && kfpMode != "validated-gpu") {
    std::cerr << "-E- run_reco_tracks: KFP mode must be disabled, cpu-only, diagnostic, or qualified-gpu"
              << std::endl;
    return;
  }
  if (runKfp && (kfpConfig.IsNull() || gSystem->AccessPathName(kfpConfig))) {
    std::cerr << "-E- run_reco_tracks: KFP mode " << kfpMode
              << " requires an existing MainConfig.yaml" << std::endl;
    return;
  }
  if (runKfp && kfpDiagnosticSamplePeriod == 0) {
    std::cerr << "-E- run_reco_tracks: KFP diagnostic sample period must be positive" << std::endl;
    return;
  }

  if (output.IsNull()) output = input;
  if (paramFile.IsNull()) paramFile = input;

  const TString rawFile  = input + ".raw.root";
  const TString traFile  = input + ".tra.root";
  const TString outFile  = output + ".reco.root";
  const TString parFile  = paramFile + ".par.root";
  const Int_t finalEntry = (nEntries < 0) ? -1 : firstEntry + nEntries;

  FairLogger::GetLogger()->SetLogScreenLevel("INFO");
  FairLogger::GetLogger()->SetLogVerbosityLevel("LOW");

  std::cout << "-I- run_reco_tracks: input      " << rawFile << '\n'
            << "-I- run_reco_tracks: output     " << outFile << '\n'
            << "-I- run_reco_tracks: parameters " << parFile << '\n'
            << "-I- run_reco_tracks: entries    " << firstEntry << " .. ";
  if (finalEntry < 0) std::cout << "end";
  else
    std::cout << finalEntry - 1;
  std::cout << "\n-I- run_reco_tracks: KFP mode   " << kfpMode;
  if (runKfp) {
    std::cout << "\n-I- run_reco_tracks: KFP config " << kfpConfig
              << "\n-I- run_reco_tracks: input kind "
              << (eventBasedInput ? "event-based" : "time-based")
              << "\n-I- run_reco_tracks: CA config  "
              << (eventBasedInput ? "caEvent" : "caTimeslice");
  }
  std::cout << std::endl;

  auto* geo = CbmSetup::Instance();
  geo->LoadSetup(setup);
  if (!geo->IsActive(ECbmModuleId::kSts)) {
    std::cerr << "-E- run_reco_tracks: setup \"" << setup << "\" has no active STS" << std::endl;
    return;
  }

  TStopwatch timer;
  timer.Start();

  auto* source = new FairFileSource(rawFile);
  if (enableMcQa) source->AddFriend(traFile);

  auto* run = new FairRunAna();
  run->SetSource(source);
  run->SetSink(new FairRootFileSink(outFile));
  run->SetGenerateRunInfo(kFALSE);

  if (enableMcQa) {
    auto* mcManager = new CbmMCDataManager("MCDataManager", 0);
    mcManager->AddFile(traFile);
    run->AddTask(mcManager);
  }

  auto* stsReco = new CbmRecoSts(ECbmRecoMode::Timeslice);
  run->AddTask(stsReco);

  if (enableMcQa) run->AddTask(new CbmMatchRecoToMC());

  run->AddTask(cbm::RecoSetupManager::Instance());
  run->AddTask(new CbmKF());

  if (runKfp) {
    auto& caParameters = cbm::ca::ParametersHandler::Instance();
    caParameters.SetMainConfig(kfpConfig.Data(), eventBasedInput ? "caEvent" : "caTimeslice");
    run->AddTask(&caParameters);
  }

  auto* l1 = enableMcQa ? new CbmL1("CA", 2, 3) : new CbmL1("CA");

  if (const char* backend = gSystem->Getenv("CBM_CA_TRACKING_BACKEND")) {
    l1->SetTrackingBackend(backend);
    std::cout << "-I- run_reco_tracks: CA backend " << backend << std::endl;
  }

  if (const char* lengthText = gSystem->Getenv("CBM_CA_GPU_MULTIWINDOW_LENGTH_NS")) {
    const Double_t lengthNs = TString(lengthText).Atof();
    if (lengthNs > 0.) {
      l1->SetGpuMultiWindowLengthNs(lengthNs);
      std::cout << "-I- run_reco_tracks: GPU multi-window length " << lengthNs << " ns" << std::endl;
    }
    else {
      std::cerr << "-W- run_reco_tracks: ignoring invalid CBM_CA_GPU_MULTIWINDOW_LENGTH_NS=" << lengthText
                << std::endl;
    }
  }

  run->AddTask(l1);

  auto* stsTrackFinder = new CbmL1StsTrackFinder();
  run->AddTask(new CbmStsFindTracks(0, stsTrackFinder, kFALSE));

  if (enableMcQa) {
    auto* trackMatch = new CbmMatchRecoToMC();
    trackMatch->SuppressHitReMatching();
    run->AddTask(trackMatch);
  }

  if (runKfp) {
    // Time-based input has no event boundaries. Event-based input already
    // carries them and must not be clustered a second time.
    if (!eventBasedInput) { run->AddTask(new CbmBuildEventsFromTracksReal()); }

    run->AddTask(new CbmFindPrimaryVertex(new CbmPVFinderKF()));

    auto* converter = new cbm::TaskRecoEventConverter();
    converter->SetFlagAddStsHits(true);
    converter->SetFlagAddTracks(true);
    converter->SetFlagAddPv(true);
    converter->SetFlagAddPidParams(true);
    run->AddTask(converter);

    auto* selector = new cbm::TaskKfpSelector();
    selector->SetConfigPath(kfpConfig.Data());
    selector->SetExecutionMode(kfpMode.Data());
    selector->SetDiagnosticSamplePeriod(kfpDiagnosticSamplePeriod);
    selector->SetUseStsPionPid(true);
    run->AddTask(selector);
  }

  auto* parIo = new FairParRootFileIo();
  if (!parIo->open(parFile.Data(), "in")) {
    std::cerr << "-E- run_reco_tracks: cannot open parameter file " << parFile << std::endl;
    return;
  }
  FairRuntimeDb* runtimeDb = run->GetRuntimeDb();
  runtimeDb->setFirstInput(parIo);

  std::cout << "-I- run_reco_tracks: initializing" << std::endl;
  run->Init();

  std::cout << "-I- run_reco_tracks: starting reconstruction" << std::endl;
  run->Run(firstEntry, finalEntry);

  timer.Stop();
  std::cout << "-I- run_reco_tracks: finished\n"
            << "-I- run_reco_tracks: real time " << timer.RealTime() << " s, CPU time " << timer.CpuTime() << " s\n"
            << "-I- run_reco_tracks: output " << outFile << std::endl;
}
