  void TestTwoDaughterTaskKernel(KFParticleGpuRuntime& runtime, KFParticleGpuBufferManager& buffers)
  {
    buffers.SetInputSizes(2, 1, 1);
    buffers.UploadInput();
    buffers.ResetCandidates();

    xpu::buffer<KFParticleGpuTwoDaughterTask> taskBuffer(1, xpu::buf_io);
    KFParticleGpuTwoDaughterTask* taskHost = HostPointer(taskBuffer);
    taskHost[0] = KFParticleGpuTwoDaughterTask();
    taskHost[0].firstTrack = 0;
    taskHost[0].secondTrack = 1;
    taskHost[0].eventIndex = 0;
    taskHost[0].motherPdg = 310;
    taskHost[0].firstDaughterPdg = 211;
    taskHost[0].secondDaughterPdg = -211;
    taskHost[0].primaryVertexIndex = -1;
    taskHost[0].firstMass = 0.13957f;
    taskHost[0].secondMass = 0.13957f;
    runtime.GetQueue().copy(taskBuffer, xpu::h2d);
    runtime.GetQueue().wait();

    runtime.GetQueue().launch<KFParticleGpuTwoDaughterTaskKernel>(
      xpu::n_threads(1),
      MakeConstView(buffers.DeviceInputTracks()),
      taskBuffer.get(),
      1u,
      buffers.DeviceCandidates());
    runtime.GetQueue().wait();
    buffers.DownloadCandidates();

    const KFParticleGpuConstCandidatePoolView candidates =
      MakeConstView(buffers.HostCandidates());
    assert(candidates.Size() == 1);
    assert(candidates.Daughters().Size() == 2);
    assert(candidates.OverflowFlags() == 0);
    assert(candidates.Metadata().Pdg(0) == 310);
    assert(candidates.Metadata().PrimaryVertexIndex(0) == -1);
    assert(candidates.Metadata().EventIndex(0) == 0);
    assert(candidates.Metadata().DaughterOffset(0) == 0);
    assert(candidates.Metadata().DaughterCount(0) == 2);
    assert(candidates.Metadata().Flags(0) == KFGpuCandidateValid);
    assert(candidates.Metadata().Topology(0) == KFGpuGraphTopologyTrackTrack);
    assert(candidates.Metadata().OutputClass(0)
           == KFGpuGraphOutputPrimaryAndSecondary);
    assert(candidates.Metadata().OperationStatus(0)
           == KFGpuCandidateOperationAccepted);
    assert(candidates.Metadata().DirectDaughterCount(0) == 2u);
    assert(candidates.Metadata().DirectFirstKind(0) == KFGpuDirectDaughterInputTrack);
    assert(candidates.Metadata().DirectFirstIndex(0) == 0u);
    assert(candidates.Metadata().DirectSecondKind(0) == KFGpuDirectDaughterInputTrack);
    assert(candidates.Metadata().DirectSecondIndex(0) == 1u);
    assert(candidates.Daughters().SourceId(0) == 17);
    assert(candidates.Daughters().SourceId(1) == 19);

    KFParticleGpuFitState mother;
    LoadCandidateFit(candidates, 0, mother);
    assert(mother.Q() == 0);
    assert(mother.Px() == 5.f);
    assert(mother.Py() == 7.f);
    assert(mother.Pz() == 9.f);
    assert(mother.SumDaughterMass() == 2.f * taskHost[0].firstMass);
    assert(KFParticleGpuMath::Mass2(mother) > 0.f);

    Pass("two-daughter-task-kernel", "prepared pair task builds and stores a two-daughter candidate on device");

    buffers.ResetCandidates();
    taskHost[0].flags = KFGpuTwoDaughterUseLineDca;
    runtime.GetQueue().copy(taskBuffer, xpu::h2d);
    runtime.GetQueue().wait();

    KFParticleGpuFitState referenceMother;
    const bool referenceBuilt =
      BuildTwoDaughterKinematicCandidate(MakeConstView(buffers.HostInputTracks()), taskHost[0], referenceMother);
    assert(referenceBuilt);

    runtime.GetQueue().launch<KFParticleGpuTwoDaughterTaskKernel>(
      xpu::n_threads(1),
      MakeConstView(buffers.DeviceInputTracks()),
      taskBuffer.get(),
      1u,
      buffers.DeviceCandidates());
    runtime.GetQueue().wait();
    buffers.DownloadCandidates();

    const KFParticleGpuConstCandidatePoolView lineDcaCandidates =
      MakeConstView(buffers.HostCandidates());
    KFParticleGpuFitState lineDcaMother;
    LoadCandidateFit(lineDcaCandidates, 0, lineDcaMother);
    assert(lineDcaCandidates.Size() == 1);
    assert(lineDcaCandidates.Daughters().Size() == 2);
    assert(AlmostEqual(lineDcaMother.X(), referenceMother.X()));
    assert(AlmostEqual(lineDcaMother.Y(), referenceMother.Y()));
    assert(AlmostEqual(lineDcaMother.Z(), referenceMother.Z()));
    assert(AlmostEqual(lineDcaMother.S(), referenceMother.S()));
    assert(AlmostEqual(lineDcaMother.Px(), referenceMother.Px()));
    assert(AlmostEqual(lineDcaMother.E(), referenceMother.E()));

    Pass("two-daughter-line-dca", "prepared pair task transports daughters to straight-line DCA before storing candidate");

    buffers.ResetCandidates();
    taskHost[0].flags = KFGpuTwoDaughterUseLineDca | KFGpuTwoDaughterUseEnergyFit;
    runtime.GetQueue().copy(taskBuffer, xpu::h2d);
    runtime.GetQueue().wait();

    KFParticleGpuFitState referenceFitMother;
    const bool referenceFitBuilt =
      BuildTwoDaughterEnergyFitCandidate(MakeConstView(buffers.HostInputTracks()), taskHost[0], referenceFitMother);
    assert(referenceFitBuilt);

    runtime.GetQueue().launch<KFParticleGpuTwoDaughterTaskKernel>(
      xpu::n_threads(1),
      MakeConstView(buffers.DeviceInputTracks()),
      taskBuffer.get(),
      1u,
      buffers.DeviceCandidates());
    runtime.GetQueue().wait();
    buffers.DownloadCandidates();

    const KFParticleGpuConstCandidatePoolView energyFitCandidates =
      MakeConstView(buffers.HostCandidates());
    KFParticleGpuFitState energyFitMother;
    LoadCandidateFit(energyFitCandidates, 0, energyFitMother);
    assert(energyFitCandidates.Size() == 1);
    assert(energyFitCandidates.Daughters().Size() == 2);
    assert(energyFitMother.NDF() == 1);
    assert(energyFitMother.Q() == 0);
    assert(energyFitMother.Chi2() >= 0.f);
    assert(AlmostEqual(energyFitMother.X(), referenceFitMother.X()));
    assert(AlmostEqual(energyFitMother.Y(), referenceFitMother.Y()));
    assert(AlmostEqual(energyFitMother.Z(), referenceFitMother.Z()));
    assert(AlmostEqual(energyFitMother.Px(), referenceFitMother.Px()));
    assert(AlmostEqual(energyFitMother.Py(), referenceFitMother.Py()));
    assert(AlmostEqual(energyFitMother.Pz(), referenceFitMother.Pz()));
    assert(AlmostEqual(energyFitMother.E(), referenceFitMother.E()));
    assert(AlmostEqual(energyFitMother.Chi2(), referenceFitMother.Chi2()));
    assert(energyFitCandidates.Metadata().Pdg(0) == 310);
    assert(energyFitCandidates.Metadata().Flags(0)
           == static_cast<unsigned int>(
             KFGpuCandidateValid | KFGpuCandidateLineDca | KFGpuCandidateEnergyFit));
    assert(energyFitCandidates.Daughters().SourceId(0) == 17);
    assert(energyFitCandidates.Daughters().SourceId(1) == 19);

    Pass("two-daughter-energy-fit", "prepared pair task builds a two-daughter candidate with Kalman energy-fit update");

    buffers.ResetCandidates();
    KFParticleGpuInputTrackSoAView diagnosticTracks = buffers.HostInputTracks();
    StoreDiagnosticField(diagnosticTracks, 0);
    buffers.UploadInput();

    taskHost[0].flags = KFGpuTwoDaughterUseLineDca;
    taskHost[0].transportMode = KFGpuTransportFieldAware;
    taskHost[0].motherPdg = 997;
    runtime.GetQueue().copy(taskBuffer, xpu::h2d);
    runtime.GetQueue().wait();

    KFParticleGpuFitState referenceFieldMother;
    const bool referenceFieldBuilt =
      BuildTwoDaughterKinematicCandidate(MakeConstView(buffers.HostInputTracks()), taskHost[0], referenceFieldMother);
    assert(referenceFieldBuilt);
    KFParticleGpuTwoDaughterTask lineReferenceTask = taskHost[0];
    lineReferenceTask.transportMode = KFGpuTransportStraightLine;
    KFParticleGpuFitState straightReferenceMother;
    assert(BuildTwoDaughterKinematicCandidate(
      MakeConstView(buffers.HostInputTracks()), lineReferenceTask, straightReferenceMother));

    runtime.GetQueue().launch<KFParticleGpuTwoDaughterTaskKernel>(
      xpu::n_threads(1),
      MakeConstView(buffers.DeviceInputTracks()),
      taskBuffer.get(),
      1u,
      buffers.DeviceCandidates());
    runtime.GetQueue().wait();
    buffers.DownloadCandidates();

    const KFParticleGpuConstCandidatePoolView fieldCandidates =
      MakeConstView(buffers.HostCandidates());
    KFParticleGpuFitState fieldMother;
    LoadCandidateFit(fieldCandidates, 0, fieldMother);
    assert(fieldCandidates.Size() == 1);
    assert(fieldCandidates.Daughters().Size() == 2);
    assert(fieldCandidates.Metadata().Pdg(0) == 997);
    assert(fieldCandidates.Metadata().Flags(0)
           == static_cast<unsigned int>(KFGpuCandidateValid | KFGpuCandidateLineDca));
    ExpectFieldAwareSeedClose(fieldMother, referenceFieldMother);
    assert(!AlmostEqual(fieldMother.X(), straightReferenceMother.X(), 1.e-6f));
    assert(!AlmostEqual(fieldMother.Px(), straightReferenceMother.Px(), 1.e-6f));

    Pass("two-daughter-field-aware-kinematic",
         "explicit field-aware prepared task uses constant-By DCA seed without enabling default V0 channels");

    buffers.ResetCandidates();
    xpu::buffer<KFParticleGpuTwoDaughterTask> guardTaskBuffer(2, xpu::buf_io);
    KFParticleGpuTwoDaughterTask* guardTasks = HostPointer(guardTaskBuffer);
    KFParticleGpuTwoDaughterTask guardBaseTask = taskHost[0];
    guardBaseTask.motherPdg = 310;
    guardBaseTask.flags = KFGpuTwoDaughterUseLineDca | KFGpuTwoDaughterUseEnergyFit;
    guardBaseTask.transportMode = KFGpuTransportStraightLine;
    guardTasks[0] = guardBaseTask;
    guardTasks[0].motherPdg = 999;
    guardTasks[0].firstDaughterPdg = -211;
    assert(!ValidateTwoDaughterTask(MakeConstView(buffers.HostInputTracks()), guardTasks[0]));
    guardTasks[1] = guardBaseTask;
    guardTasks[1].motherPdg = 999;
    guardTasks[1].transportMode = KFGpuTransportFieldAware;
    runtime.GetQueue().copy(guardTaskBuffer, xpu::h2d);
    runtime.GetQueue().wait();

    runtime.GetQueue().launch<KFParticleGpuTwoDaughterTaskKernel>(
      xpu::n_threads(2),
      MakeConstView(buffers.DeviceInputTracks()),
      guardTaskBuffer.get(),
      2u,
      buffers.DeviceCandidates());
    runtime.GetQueue().wait();
    buffers.DownloadCandidates();

    const KFParticleGpuConstCandidatePoolView guardedCandidates =
      MakeConstView(buffers.HostCandidates());
    assert(guardedCandidates.Size() == 2);
    assert(guardedCandidates.Metadata().Pdg(0) == 999);
    assert(guardedCandidates.Metadata().DaughterCount(0) == 0);
    assert(guardedCandidates.Metadata().Flags(0) & KFGpuCandidateBuildFailed);
    assert(guardedCandidates.Metadata().OperationStatus(0)
           == KFGpuCandidateOperationRejected);
    assert(guardedCandidates.Metadata().Flags(0) & KFGpuCandidateEnergyFit);
    assert(guardedCandidates.Metadata().Pdg(1) == 999);
    assert(guardedCandidates.Metadata().DaughterCount(1) == 2);
    assert(guardedCandidates.Metadata().Flags(1)
           == static_cast<unsigned int>(
             KFGpuCandidateValid | KFGpuCandidateLineDca | KFGpuCandidateEnergyFit));
    KFParticleGpuFitState guardedFieldFitMother;
    LoadCandidateFit(guardedCandidates, 1, guardedFieldFitMother);
    assert(guardedFieldFitMother.NDF() == 1);
    assert(guardedFieldFitMother.Q() == 0);

    Pass("two-daughter-guards", "invalid tasks fail while field-aware energy-fit tasks build candidates");
  }

  void StoreSyntheticTrack(KFParticleGpuInputTrackSoAView& tracks,
                           unsigned int index,
                           float x,
                           float y,
                           float z,
                           float px,
                           float py,
                           float pz,
                           int pdg,
                           int charge,
                           int sourceId)
  {
    KFParticleGpuTrackState track;
    track.X() = x;
    track.Y() = y;
    track.Z() = z;
    track.Px() = px;
    track.Py() = py;
    track.Pz() = pz;

    const float scale = 1.f + 0.2f * static_cast<float>(index);
    track.Covariance(0, 0) = 0.40f * scale;
    track.Covariance(1, 1) = 0.35f * scale;
    track.Covariance(2, 2) = 0.30f * scale;
    track.Covariance(3, 3) = 0.08f * scale;
    track.Covariance(4, 4) = 0.07f * scale;
    track.Covariance(5, 5) = 0.06f * scale;
    track.Covariance(0, 3) = 0.010f * scale;
    track.Covariance(1, 4) = -0.008f * scale;
    track.Covariance(2, 5) = 0.006f * scale;
    track.Covariance(3, 4) = 0.004f * scale;
    track.Covariance(4, 5) = -0.003f * scale;

    StoreTrackState(track, tracks, index);
    tracks.SourceId(index) = sourceId;
    tracks.Pdg(index) = pdg;
    tracks.Charge(index) = charge;
    tracks.PrimaryVertexIndex(index) = -1;
    tracks.NumberOfPixelHits(index) = 5;
    tracks.ChiToPrimaryVertex(index) = 1.f + static_cast<float>(index);
  }

  void TestTwoDaughterTaskGeneration(KFParticleGpuRuntime& runtime,
                                     KFParticleGpuBufferManager& buffers)
  {
    buffers.SetInputSizes(4, 1, 1);
    KFParticleGpuInputTrackSoAView tracks = buffers.HostInputTracks();
    StoreSyntheticTrack(tracks, 0, -0.2f, 0.1f, 0.0f, 0.8f, 0.1f, 1.0f, 211, 1, 501);
    StoreSyntheticTrack(tracks, 1, 0.4f, -0.3f, 0.2f, 0.5f, 0.6f, 0.9f, 211, 1, 502);
    StoreSyntheticTrack(tracks, 2, 0.1f, 0.5f, -0.4f, -0.3f, 0.7f, 1.1f, -211, -1, 601);
    StoreSyntheticTrack(tracks, 3, -0.6f, -0.2f, 0.3f, -0.4f, 0.2f, 0.8f, -211, -1, 602);
    tracks.NumberOfPixelHits(1) = 3;
    tracks.ChiToPrimaryVertex(3) = 9.f;

    KFParticleGpuEventDesc* events = buffers.HostEvents();
    events[0] = KFParticleGpuEventDesc();
    events[0].eventId = 42;
    events[0].TrackSet(SecondaryPositiveFirst).tracks = KFParticleGpuRange(0, 2);
    events[0].TrackSet(SecondaryPositiveFirst).Species(Pion) = KFParticleGpuRange(0, 2);
    events[0].TrackSet(SecondaryNegativeFirst).tracks = KFParticleGpuRange(2, 2);
    events[0].TrackSet(SecondaryNegativeFirst).Species(Pion) = KFParticleGpuRange(2, 2);
    buffers.UploadInput();

    xpu::buffer<KFParticleGpuTwoDaughterTask> taskBuffer(3, xpu::buf_io);
    xpu::buffer<unsigned int> writtenTasks(1, xpu::buf_io);
    xpu::buffer<unsigned int> totalPairs(1, xpu::buf_io);

    KFParticleGpuTwoDaughterTaskSource source;
    source.eventIndex = 0;
    source.firstTrackSet = SecondaryPositiveFirst;
    source.secondTrackSet = SecondaryNegativeFirst;
    source.firstSpecies = Pion;
    source.secondSpecies = Pion;
    source.flags = KFGpuTwoDaughterUseLineDca | KFGpuTwoDaughterUseEnergyFit;
    source.motherPdg = 310;
    source.firstDaughterPdg = 211;
    source.secondDaughterPdg = -211;
    source.primaryVertexIndex = -1;
    source.firstMass = 0.13957f;
    source.secondMass = 0.13957f;

    runtime.GetQueue().launch<KFParticleGpuGenerateTwoDaughterTasks>(
      xpu::n_threads(4),
      buffers.DeviceEvents(),
      MakeConstView(buffers.DeviceInputTracks()),
      source,
      taskBuffer.get(),
      3u,
      writtenTasks.get(),
      totalPairs.get());
    runtime.GetQueue().wait();
    runtime.GetQueue().copy(taskBuffer, xpu::d2h);
    runtime.GetQueue().copy(writtenTasks, xpu::d2h);
    runtime.GetQueue().copy(totalPairs, xpu::d2h);
    runtime.GetQueue().wait();

    const KFParticleGpuTwoDaughterTask* tasks = HostPointer(taskBuffer);
    const unsigned int* written = HostPointer(writtenTasks);
    const unsigned int* total = HostPointer(totalPairs);
    assert(written[0] == 3);
    assert(total[0] == 4);
    assert(tasks[0].firstTrack == 0 && tasks[0].secondTrack == 2);
    assert(tasks[1].firstTrack == 0 && tasks[1].secondTrack == 3);
    assert(tasks[2].firstTrack == 1 && tasks[2].secondTrack == 2);
    for (unsigned int i = 0; i < written[0]; ++i) {
      assert(tasks[i].eventIndex == 0);
      assert(tasks[i].motherPdg == 310);
      assert(tasks[i].firstDaughterPdg == 211);
      assert(tasks[i].secondDaughterPdg == -211);
      assert(tasks[i].flags == source.flags);
      assert(tasks[i].transportMode == source.transportMode);
      assert(ValidateTwoDaughterTask(MakeConstView(buffers.HostInputTracks()), tasks[i]));
    }

    source.firstCharge = 1;
    source.secondCharge = -1;
    source.minFirstPixelHits = 5;
    source.minSecondPixelHits = 5;
    source.maxSecondChiToPrimaryVertex = 4.f;
    runtime.GetQueue().launch<KFParticleGpuGenerateTwoDaughterTasks>(
      xpu::n_threads(4),
      buffers.DeviceEvents(),
      MakeConstView(buffers.DeviceInputTracks()),
      source,
      taskBuffer.get(),
      3u,
      writtenTasks.get(),
      totalPairs.get());
    runtime.GetQueue().wait();
    runtime.GetQueue().copy(taskBuffer, xpu::d2h);
    runtime.GetQueue().copy(writtenTasks, xpu::d2h);
    runtime.GetQueue().copy(totalPairs, xpu::d2h);
    runtime.GetQueue().wait();

    tasks = HostPointer(taskBuffer);
    written = HostPointer(writtenTasks);
    total = HostPointer(totalPairs);
    assert(written[0] == 3);
    assert(total[0] == 4);
    assert(tasks[0].firstTrack == 0 && tasks[0].secondTrack == 2);
    assert(tasks[1].firstTrack == buffers.HostInputTracks().Size());
    assert(tasks[2].firstTrack == buffers.HostInputTracks().Size());
    assert(PassTwoDaughterTaskSourceCuts(MakeConstView(buffers.HostInputTracks()),
                                         source,
                                         tasks[0].firstTrack,
                                         tasks[0].secondTrack));
    assert(!ValidateTwoDaughterTask(MakeConstView(buffers.HostInputTracks()), tasks[1]));

    KFParticleGpuTwoDaughterTaskSource chiSource;
    chiSource.firstDaughterPdg = 211;
    chiSource.secondDaughterPdg = -211;
    assert(PassTwoDaughterTaskSourceCuts(
      MakeConstView(buffers.HostInputTracks()), chiSource, 1u, 2u, 1.5f));
    assert(!PassTwoDaughterTaskSourceCuts(
      MakeConstView(buffers.HostInputTracks()), chiSource, 1u, 2u, 2.f));

    Pass("two-daughter-task-generation", "event species ranges and pre-fit cuts generate fixed prepared task slots");
  }

  bool HasGeneratedTask(const KFParticleGpuTwoDaughterTask* tasks,
                        unsigned int size,
                        unsigned int firstTrack,
                        unsigned int secondTrack)
  {
    for (unsigned int i = 0; i < size; ++i) {
      if (tasks[i].firstTrack == firstTrack && tasks[i].secondTrack == secondTrack) {
        return true;
      }
    }
    return false;
  }

  bool HasCandidateDaughters(const KFParticleGpuConstCandidatePoolView& candidates,
                             int firstSourceId,
                             int secondSourceId)
  {
    for (unsigned int i = 0; i < candidates.Size(); ++i) {
      if (candidates.Metadata().DaughterCount(i) != 2u) {
        continue;
      }
      const unsigned int offset = candidates.Metadata().DaughterOffset(i);
      if (candidates.Daughters().SourceId(offset) == firstSourceId
          && candidates.Daughters().SourceId(offset + 1u) == secondSourceId) {
        return true;
      }
    }
    return false;
  }

  unsigned int FindCandidateByDaughters(const KFParticleGpuConstCandidatePoolView& candidates,
                                        int firstSourceId,
                                        int secondSourceId)
  {
    for (unsigned int i = 0; i < candidates.Size(); ++i) {
      if (candidates.Metadata().DaughterCount(i) != 2u) {
        continue;
      }
      const unsigned int offset = candidates.Metadata().DaughterOffset(i);
      if (candidates.Daughters().SourceId(offset) == firstSourceId
          && candidates.Daughters().SourceId(offset + 1u) == secondSourceId) {
        return i;
      }
    }
    return candidates.Size();
  }

  unsigned int FindChannelCandidateByDaughters(
    const KFParticleGpuConstCandidatePoolView& candidates,
    unsigned int channelId,
    int firstSourceId,
    int secondSourceId)
  {
    for (unsigned int i = 0; i < candidates.Size(); ++i) {
      if (candidates.Metadata().ChannelId(i) != channelId
          || candidates.Metadata().DaughterCount(i) != 2u) {
        continue;
      }
      const unsigned int offset = candidates.Metadata().DaughterOffset(i);
      if (candidates.Daughters().SourceId(offset) == firstSourceId
          && candidates.Daughters().SourceId(offset + 1u) == secondSourceId) {
        return i;
      }
    }
    return candidates.Size();
  }

  void ExpectChannelCandidatePdgs(const KFParticleGpuConstCandidatePoolView& candidates,
                                  const KFParticleGpuTwoDaughterChannelResult& result,
                                  int expectedPdg)
  {
    const KFParticleGpuCandidateRange& generation =
      result.generationCandidates.Empty()
        ? result.candidates : result.generationCandidates;
    unsigned int matched = 0u;
    for (unsigned int i = generation.offset; i < generation.End(); ++i) {
      if (candidates.Metadata().ChannelId(i) != result.channelId) {
        continue;
      }
      assert(candidates.Metadata().Pdg(i) == expectedPdg);
      ++matched;
    }
    assert(matched == (result.constructedCandidates > 0u
                         ? result.constructedCandidates : result.candidates.size));
  }

  void ExpectChannelDaughterRange(const KFParticleGpuConstCandidatePoolView& candidates,
                                  const KFParticleGpuTwoDaughterChannelResult& result)
  {
    const KFParticleGpuCandidateRange& generation =
      result.generationCandidates.Empty()
        ? result.candidates : result.generationCandidates;
    for (unsigned int i = generation.offset; i < generation.End(); ++i) {
      if (candidates.Metadata().ChannelId(i) != result.channelId) {
        continue;
      }
      const unsigned int daughterOffset = candidates.Metadata().DaughterOffset(i);
      const unsigned int daughterCount = candidates.Metadata().DaughterCount(i);
      for (unsigned int j = 0; j < daughterCount; ++j) {
        assert(generation.ContainsDaughter(daughterOffset + j));
      }
    }
  }

  void ExpectChannelSelectionFlags(const KFParticleGpuConstCandidatePoolView& candidates,
                                   const KFParticleGpuTwoDaughterChannelResult& result,
                                   const KFParticleGpuTwoDaughterChannel& channel)
  {
    const KFParticleGpuTwoDaughterTaskSource source =
      MakeTwoDaughterTaskSource(channel, result.eventIndex);
    const KFParticleGpuCandidateRange& generation =
      result.generationCandidates.Empty()
        ? result.candidates : result.generationCandidates;
    for (unsigned int i = generation.offset; i < generation.End(); ++i) {
      if (candidates.Metadata().ChannelId(i) != result.channelId) {
        continue;
      }
      KFParticleGpuFitState candidate;
      LoadCandidateFit(candidates, i, candidate);
      const bool expectedRejected = !PassTwoDaughterPostBuildSelection(source, candidate);
      const bool markedRejected =
        (candidates.Metadata().Flags(i) & KFGpuCandidateSelectionRejected) != 0u;
      assert(markedRejected == expectedRejected);
    }
  }

  unsigned int CountChannelSelectionRejected(
    const KFParticleGpuConstCandidatePoolView& candidates,
    const KFParticleGpuTwoDaughterChannelResult& result)
  {
    unsigned int rejected = 0u;
    const KFParticleGpuCandidateRange& generation =
      result.generationCandidates.Empty()
        ? result.candidates : result.generationCandidates;
    for (unsigned int i = generation.offset; i < generation.End(); ++i) {
      if (candidates.Metadata().ChannelId(i) != result.channelId) {
        continue;
      }
      if (candidates.Metadata().Flags(i) & KFGpuCandidateSelectionRejected) {
        ++rejected;
      }
    }
    return rejected;
  }

  unsigned int FindChannelCandidate(
    const KFParticleGpuConstCandidatePoolView& candidates,
    const KFParticleGpuCandidateRange& generation,
    unsigned int channelId)
  {
    for (unsigned int candidateIndex = generation.offset;
         candidateIndex < generation.End();
         ++candidateIndex) {
      if (candidates.Metadata().ChannelId(candidateIndex) == channelId) {
        return candidateIndex;
      }
    }
    return candidates.Size();
  }

  const KFParticleGpuTwoDaughterChannel& FindTwoDaughterChannel(
    const KFParticleGpuDecayPlan& plan,
    unsigned int channelId)
  {
    for (std::size_t channelIndex = 0u;
         channelIndex < plan.NumberOfTwoDaughterChannels();
         ++channelIndex) {
      const KFParticleGpuTwoDaughterChannel& channel =
        plan.TwoDaughterChannel(channelIndex);
      if (channel.channelId == channelId) {
        return channel;
      }
    }
    assert(false);
    return plan.TwoDaughterChannel(0u);
  }

  void ExpectDefaultV0ChannelRegression(
    const KFParticleGpuConstCandidatePoolView& candidates,
    const KFParticleGpuTwoDaughterChannelResult& result,
    const KFParticleGpuTwoDaughterChannel& channel,
    unsigned int expectedChannelId,
    int expectedMotherPdg,
    unsigned int expectedOffset,
    unsigned int expectedDaughterOffset,
    unsigned int expectedCount = 1u)
  {
    assert(result.channelId == expectedChannelId);
    assert(result.motherPdg == expectedMotherPdg);
    assert(result.eventIndex == 0u);
    assert(result.totalPairs == expectedCount);
    assert(result.acceptedTasks == expectedCount);
    assert(result.storedTasks == expectedCount);
    assert(!result.Truncated());
    assert(!result.HasAnyOverflow());
    assert(result.constructedCandidates == expectedCount);
    assert(result.constructedDaughters == 2u * expectedCount);
    unsigned int candidateIndex = candidates.Size();
    for (unsigned int index = result.generationCandidates.offset;
         index < result.generationCandidates.End();
         ++index) {
      if (candidates.Metadata().ChannelId(index) == expectedChannelId) {
        candidateIndex = index;
        break;
      }
    }
    assert(candidateIndex < candidates.Size());
    assert(candidates.Metadata().EventIndex(candidateIndex) == 0u);
    assert(candidates.Metadata().Pdg(candidateIndex) == expectedMotherPdg);
    assert(candidates.Metadata().DaughterCount(candidateIndex) == 2u);
    assert(candidates.Metadata().Flags(candidateIndex) & KFGpuCandidateValid);
    assert(candidates.Metadata().Flags(candidateIndex) & KFGpuCandidateLineDca);
    assert(candidates.Metadata().Flags(candidateIndex) & KFGpuCandidateEnergyFit);
    assert(!(candidates.Metadata().Flags(candidateIndex) & KFGpuCandidateBuildFailed));
    ExpectChannelCandidatePdgs(candidates, result, expectedMotherPdg);
    ExpectChannelDaughterRange(candidates, result);
    ExpectChannelSelectionFlags(candidates, result, channel);
    assert(CountChannelSelectionRejected(candidates, result)
           <= result.constructedCandidates);
    (void) expectedOffset;
    (void) expectedDaughterOffset;
  }

  void TestCompactTwoDaughterTaskGeneration(KFParticleGpuRuntime& runtime,
                                            KFParticleGpuBufferManager& buffers)
  {
    buffers.SetInputSizes(4, 1, 1);
    KFParticleGpuInputTrackSoAView tracks = buffers.HostInputTracks();
    StoreSyntheticTrack(tracks, 0, -0.2f, 0.1f, 0.0f, 0.8f, 0.1f, 1.0f, 211, 1, 1101);
    StoreSyntheticTrack(tracks, 1, 0.4f, -0.3f, 0.2f, 0.5f, 0.6f, 0.9f, 211, 1, 1102);
    StoreSyntheticTrack(tracks, 2, 0.1f, 0.5f, -0.4f, -0.3f, 0.7f, 1.1f, -211, -1, 1201);
    StoreSyntheticTrack(tracks, 3, -0.6f, -0.2f, 0.3f, -0.4f, 0.2f, 0.8f, -211, -1, 1202);
    tracks.ChiToPrimaryVertex(3) = 9.f;

    KFParticleGpuEventDesc* events = buffers.HostEvents();
    events[0] = KFParticleGpuEventDesc();
    events[0].eventId = 45;
    events[0].TrackSet(SecondaryPositiveFirst).tracks = KFParticleGpuRange(0, 2);
    events[0].TrackSet(SecondaryPositiveFirst).Species(Pion) = KFParticleGpuRange(0, 2);
    events[0].TrackSet(SecondaryNegativeFirst).tracks = KFParticleGpuRange(2, 2);
    events[0].TrackSet(SecondaryNegativeFirst).Species(Pion) = KFParticleGpuRange(2, 2);
    buffers.UploadInput();

    KFParticleGpuTwoDaughterTaskSource source;
    source.eventIndex = 0;
    source.firstTrackSet = SecondaryPositiveFirst;
    source.secondTrackSet = SecondaryNegativeFirst;
    source.firstSpecies = Pion;
    source.secondSpecies = Pion;
    source.flags = KFGpuTwoDaughterUseLineDca | KFGpuTwoDaughterUseEnergyFit;
    source.motherPdg = 310;
    source.firstDaughterPdg = 211;
    source.secondDaughterPdg = -211;
    source.primaryVertexIndex = -1;
    source.firstMass = 0.13957f;
    source.secondMass = 0.13957f;
    source.firstCharge = 1;
    source.secondCharge = -1;
    source.minSecondPixelHits = 5;
    source.minFirstPixelHits = 0;
    source.maxSecondChiToPrimaryVertex = 4.f;

    xpu::buffer<KFParticleGpuTwoDaughterTask> taskBuffer(2, xpu::buf_io);
    xpu::buffer<unsigned int> acceptedTasks(1, xpu::buf_io);
    xpu::buffer<unsigned int> totalPairs(1, xpu::buf_io);

    unsigned int* accepted = HostPointer(acceptedTasks);
    unsigned int* total = HostPointer(totalPairs);
    accepted[0] = 0;
    total[0] = 0;
    runtime.GetQueue().copy(acceptedTasks, xpu::h2d);
    runtime.GetQueue().copy(totalPairs, xpu::h2d);

    runtime.GetQueue().launch<KFParticleGpuGenerateTwoDaughterTasksCompact>(
      xpu::n_threads(4),
      buffers.DeviceEvents(),
      MakeConstView(buffers.DeviceInputTracks()),
      source,
      taskBuffer.get(),
      2u,
      acceptedTasks.get(),
      totalPairs.get());
    runtime.GetQueue().wait();
    runtime.GetQueue().copy(taskBuffer, xpu::d2h);
    runtime.GetQueue().copy(acceptedTasks, xpu::d2h);
    runtime.GetQueue().copy(totalPairs, xpu::d2h);
    runtime.GetQueue().wait();

    const KFParticleGpuTwoDaughterTask* tasks = HostPointer(taskBuffer);
    accepted = HostPointer(acceptedTasks);
    total = HostPointer(totalPairs);
    assert(accepted[0] == 2);
    assert(total[0] == 4);
    assert(HasGeneratedTask(tasks, 2, 0, 2));
    assert(HasGeneratedTask(tasks, 2, 1, 2));
    for (unsigned int i = 0; i < 2; ++i) {
      assert(ValidateTwoDaughterTask(MakeConstView(buffers.HostInputTracks()), tasks[i]));
      assert(PassTwoDaughterTaskSourceCuts(MakeConstView(buffers.HostInputTracks()),
                                           source,
                                           tasks[i].firstTrack,
                                           tasks[i].secondTrack));
    }

    xpu::buffer<KFParticleGpuTwoDaughterTask> truncatedTaskBuffer(1, xpu::buf_io);
    accepted = HostPointer(acceptedTasks);
    total = HostPointer(totalPairs);
    accepted[0] = 0;
    total[0] = 0;
    runtime.GetQueue().copy(acceptedTasks, xpu::h2d);
    runtime.GetQueue().copy(totalPairs, xpu::h2d);

    runtime.GetQueue().launch<KFParticleGpuGenerateTwoDaughterTasksCompact>(
      xpu::n_threads(4),
      buffers.DeviceEvents(),
      MakeConstView(buffers.DeviceInputTracks()),
      source,
      truncatedTaskBuffer.get(),
      1u,
      acceptedTasks.get(),
      totalPairs.get());
    runtime.GetQueue().wait();
    runtime.GetQueue().copy(truncatedTaskBuffer, xpu::d2h);
    runtime.GetQueue().copy(acceptedTasks, xpu::d2h);
    runtime.GetQueue().copy(totalPairs, xpu::d2h);
    runtime.GetQueue().wait();

    tasks = HostPointer(truncatedTaskBuffer);
    accepted = HostPointer(acceptedTasks);
    total = HostPointer(totalPairs);
    assert(accepted[0] == 2);
    assert(total[0] == 4);
    assert(ValidateTwoDaughterTask(MakeConstView(buffers.HostInputTracks()), tasks[0]));
    assert((tasks[0].firstTrack == 0 || tasks[0].firstTrack == 1) && tasks[0].secondTrack == 2);

    Pass("two-daughter-task-generation-compact",
         "atomic compact pair generation stores only accepted task slots and reports truncation");
  }

  void TestCompactTwoDaughterCandidateConstruction(KFParticleGpuRuntime& runtime,
                                                   KFParticleGpuBufferManager& buffers)
  {
    buffers.SetInputSizes(4, 1, 1);
    KFParticleGpuInputTrackSoAView tracks = buffers.HostInputTracks();
    StoreSyntheticTrack(tracks, 0, -0.2f, 0.1f, 0.0f, 0.8f, 0.1f, 1.0f, 211, 1, 1301);
    StoreSyntheticTrack(tracks, 1, 0.4f, -0.3f, 0.2f, 0.5f, 0.6f, 0.9f, 211, 1, 1302);
    StoreSyntheticTrack(tracks, 2, 0.1f, 0.5f, -0.4f, -0.3f, 0.7f, 1.1f, -211, -1, 1401);
    StoreSyntheticTrack(tracks, 3, -0.6f, -0.2f, 0.3f, -0.4f, 0.2f, 0.8f, -211, -1, 1402);
    buffers.UploadInput();
    buffers.ResetCandidates();

    xpu::buffer<KFParticleGpuTwoDaughterTask> taskBuffer(3, xpu::buf_io);
    KFParticleGpuTwoDaughterTask* tasks = HostPointer(taskBuffer);
    tasks[0] = KFParticleGpuTwoDaughterTask();
    tasks[0].firstTrack = 0;
    tasks[0].secondTrack = 2;
    tasks[0].eventIndex = 0;
    tasks[0].flags = KFGpuTwoDaughterUseLineDca | KFGpuTwoDaughterUseEnergyFit;
    tasks[0].motherPdg = 310;
    tasks[0].firstDaughterPdg = 211;
    tasks[0].secondDaughterPdg = -211;
    tasks[0].primaryVertexIndex = -1;
    tasks[0].firstMass = 0.13957f;
    tasks[0].secondMass = 0.13957f;

    tasks[1] = tasks[0];
    tasks[1].firstTrack = 1;
    tasks[1].secondTrack = 3;

    tasks[2] = tasks[0];
    tasks[2].secondTrack = 0; // Duplicate track makes this task fail validation.

    runtime.GetQueue().copy(taskBuffer, xpu::h2d);
    runtime.GetQueue().wait();
    runtime.GetQueue().launch<KFParticleGpuTwoDaughterCompactCandidateKernel>(
      xpu::n_threads(3),
      MakeConstView(buffers.DeviceInputTracks()),
      taskBuffer.get(),
      3u,
      buffers.DeviceCandidates());
    runtime.GetQueue().wait();
    buffers.DownloadCandidates();

    const KFParticleGpuConstCandidatePoolView candidates = MakeConstView(buffers.HostCandidates());
    assert(candidates.Size() == 2);
    assert(candidates.Daughters().Size() == 4);
    assert(candidates.OverflowFlags() == 0u);
    assert(HasCandidateDaughters(candidates, 1301, 1401));
    assert(HasCandidateDaughters(candidates, 1302, 1402));
    for (unsigned int i = 0; i < candidates.Size(); ++i) {
      assert(candidates.Metadata().Pdg(i) == 310);
      assert(candidates.Metadata().DaughterCount(i) == 2u);
      assert(candidates.Metadata().Flags(i)
             == static_cast<unsigned int>(
               KFGpuCandidateValid | KFGpuCandidateLineDca | KFGpuCandidateEnergyFit));
    }

    Pass("two-daughter-candidate-construction-compact",
         "compact construction stores successful candidates without failed task placeholders");
  }

  void TestCompactTwoDaughterCandidateOverflow(KFParticleGpuRuntime& runtime,
                                               KFParticleGpuBufferManager& buffers)
  {
    buffers.SetInputSizes(6, 1, 1);
    KFParticleGpuInputTrackSoAView tracks = buffers.HostInputTracks();
    StoreSyntheticTrack(tracks, 0, -0.2f, 0.1f, 0.0f, 0.8f, 0.1f, 1.0f, 211, 1, 1501);
    StoreSyntheticTrack(tracks, 1, 0.4f, -0.3f, 0.2f, 0.5f, 0.6f, 0.9f, 211, 1, 1502);
    StoreSyntheticTrack(tracks, 2, 0.1f, 0.5f, -0.4f, -0.3f, 0.7f, 1.1f, 211, 1, 1503);
    StoreSyntheticTrack(tracks, 3, -0.6f, -0.2f, 0.3f, -0.4f, 0.2f, 0.8f, -211, -1, 1601);
    StoreSyntheticTrack(tracks, 4, -0.1f, 0.4f, 0.2f, -0.2f, 0.8f, 0.7f, -211, -1, 1602);
    StoreSyntheticTrack(tracks, 5, 0.3f, -0.5f, 0.1f, -0.6f, 0.4f, 0.9f, -211, -1, 1603);
    buffers.UploadInput();
    buffers.ResetCandidates();

    xpu::buffer<KFParticleGpuTwoDaughterTask> taskBuffer(3, xpu::buf_io);
    KFParticleGpuTwoDaughterTask* tasks = HostPointer(taskBuffer);
    for (unsigned int i = 0; i < 3; ++i) {
      tasks[i] = KFParticleGpuTwoDaughterTask();
      tasks[i].firstTrack = i;
      tasks[i].secondTrack = 3u + i;
      tasks[i].eventIndex = 0;
      tasks[i].flags = KFGpuTwoDaughterUseLineDca | KFGpuTwoDaughterUseEnergyFit;
      tasks[i].motherPdg = 310;
      tasks[i].firstDaughterPdg = 211;
      tasks[i].secondDaughterPdg = -211;
      tasks[i].primaryVertexIndex = -1;
      tasks[i].firstMass = 0.13957f;
      tasks[i].secondMass = 0.13957f;
    }

    runtime.GetQueue().copy(taskBuffer, xpu::h2d);
    runtime.GetQueue().wait();
    runtime.GetQueue().launch<KFParticleGpuTwoDaughterCompactCandidateKernel>(
      xpu::n_threads(3),
      MakeConstView(buffers.DeviceInputTracks()),
      taskBuffer.get(),
      3u,
      buffers.DeviceCandidates());
    runtime.GetQueue().wait();
    buffers.DownloadCandidates();

    const KFParticleGpuConstCandidatePoolView candidates = MakeConstView(buffers.HostCandidates());
    assert(candidates.Size() == 3);
    assert(candidates.Daughters().Size() == 4);
    assert(candidates.OverflowFlags() == DaughterCapacityExceeded);

    unsigned int failedCandidates = 0;
    for (unsigned int i = 0; i < candidates.Size(); ++i) {
      if (candidates.Metadata().DaughterCount(i) == 0u) {
        ++failedCandidates;
        assert(candidates.Metadata().Flags(i) & KFGpuCandidateBuildFailed);
      }
    }
    assert(failedCandidates == 1);

    Pass("two-daughter-candidate-overflow-compact",
         "compact construction reports daughter storage overflow without corrupting counters");
  }

  void FillGeneratedPipelineFixture(KFParticleGpuBufferManager& buffers)
  {
    buffers.SetInputSizes(4, 1, 1);
    KFParticleGpuInputTrackSoAView tracks = buffers.HostInputTracks();
    StoreSyntheticTrack(tracks, 0, -0.2f, 0.1f, 0.0f, 0.8f, 0.1f, 1.0f, 211, 1, 901);
    StoreSyntheticTrack(tracks, 1, 0.4f, -0.3f, 0.2f, 0.5f, 0.6f, 0.9f, 211, 1, 902);
    StoreSyntheticTrack(tracks, 2, 0.1f, 0.5f, -0.4f, -0.3f, 0.7f, 1.1f, -211, -1, 1001);
    StoreSyntheticTrack(tracks, 3, -0.6f, -0.2f, 0.3f, -0.4f, 0.2f, 0.8f, -211, -1, 1002);
    tracks.NumberOfPixelHits(1) = 3;
    tracks.ChiToPrimaryVertex(3) = 9.f;

    KFParticleGpuEventDesc* events = buffers.HostEvents();
    events[0] = KFParticleGpuEventDesc();
    events[0].eventId = 44;
    events[0].TrackSet(SecondaryPositiveFirst).tracks = KFParticleGpuRange(0, 2);
    events[0].TrackSet(SecondaryPositiveFirst).Species(Pion) = KFParticleGpuRange(0, 2);
    events[0].TrackSet(SecondaryNegativeFirst).tracks = KFParticleGpuRange(2, 2);
    events[0].TrackSet(SecondaryNegativeFirst).Species(Pion) = KFParticleGpuRange(2, 2);
  }

  KFParticleGpuTwoDaughterTaskSource MakePionPairSource()
  {
    KFParticleGpuTwoDaughterTaskSource source;
    source.eventIndex = 0;
    source.firstTrackSet = SecondaryPositiveFirst;
    source.secondTrackSet = SecondaryNegativeFirst;
    source.firstSpecies = Pion;
    source.secondSpecies = Pion;
    source.flags = KFGpuTwoDaughterUseLineDca | KFGpuTwoDaughterUseEnergyFit;
    source.motherPdg = 310;
    source.firstDaughterPdg = 211;
    source.secondDaughterPdg = -211;
    source.primaryVertexIndex = -1;
    source.firstMass = 0.13957f;
    source.secondMass = 0.13957f;
    source.firstCharge = 1;
    source.secondCharge = -1;
    source.minFirstPixelHits = 5;
    source.minSecondPixelHits = 5;
    source.maxSecondChiToPrimaryVertex = 4.f;
    return source;
  }

  KFParticleGpuTwoDaughterChannel MakePionPairChannel(unsigned int channelId)
  {
    KFParticleGpuTwoDaughterChannel channel;
    channel.channelId = channelId;
    channel.firstTrackSet = SecondaryPositiveFirst;
    channel.secondTrackSet = SecondaryNegativeFirst;
    channel.firstSpecies = Pion;
    channel.secondSpecies = Pion;
    channel.flags = KFGpuTwoDaughterUseLineDca | KFGpuTwoDaughterUseEnergyFit;
    channel.motherPdg = 310;
    channel.firstDaughterPdg = 211;
    channel.secondDaughterPdg = -211;
    channel.primaryVertexIndex = -1;
    channel.firstMass = 0.13957f;
    channel.secondMass = 0.13957f;
    channel.firstCharge = 1;
    channel.secondCharge = -1;
    channel.minFirstPixelHits = 0;
    channel.minSecondPixelHits = 5;
    channel.maxSecondChiToPrimaryVertex = 4.f;
    return channel;
  }

  KFParticleGpuTwoDaughterChannel MakeFieldAwarePionPairChannel(unsigned int channelId)
  {
    KFParticleGpuTwoDaughterChannel channel = MakePionPairChannel(channelId);
    channel.flags = KFGpuTwoDaughterUseLineDca;
    channel.transportMode = KFGpuTransportFieldAware;
    channel.motherPdg = 997;
    return channel;
  }

  KFParticleGpuTwoDaughterChannel MakeKaonPairChannel(unsigned int channelId)
  {
    KFParticleGpuTwoDaughterChannel channel;
    channel.channelId = channelId;
    channel.firstTrackSet = SecondaryPositiveFirst;
    channel.secondTrackSet = SecondaryNegativeFirst;
    channel.firstSpecies = Kaon;
    channel.secondSpecies = Kaon;
    channel.flags = KFGpuTwoDaughterUseLineDca | KFGpuTwoDaughterUseEnergyFit;
    channel.motherPdg = 333;
    channel.firstDaughterPdg = 321;
    channel.secondDaughterPdg = -321;
    channel.primaryVertexIndex = -1;
    channel.firstMass = 0.493677f;
    channel.secondMass = 0.493677f;
    channel.firstCharge = 1;
    channel.secondCharge = -1;
    channel.minFirstPixelHits = 0;
    channel.minSecondPixelHits = 5;
    channel.maxSecondChiToPrimaryVertex = 4.f;
    return channel;
  }

  void FillDecayPlanMultiChannelFixture(KFParticleGpuBufferManager& buffers)
  {
    KFParticleGpuBufferCapacities requested = buffers.Capacities();
    requested.tracks = requested.tracks > 8u ? requested.tracks : 8u;
    requested.events = requested.events > 1u ? requested.events : 1u;
    requested.candidates = requested.candidates > 6u ? requested.candidates : 6u;
    requested.daughterIds = requested.daughterIds > 8u ? requested.daughterIds : 8u;
    buffers.EnsureCapacity(requested);
    buffers.SetInputSizes(8, 1, 1);

    KFParticleGpuInputTrackSoAView tracks = buffers.HostInputTracks();
    StoreSyntheticTrack(tracks, 0, -0.2f, 0.1f, 0.0f, 0.8f, 0.1f, 1.0f, 211, 1, 2101);
    StoreSyntheticTrack(tracks, 1, 0.4f, -0.3f, 0.2f, 0.5f, 0.6f, 0.9f, 211, 1, 2102);
    StoreSyntheticTrack(tracks, 2, 0.2f, 0.4f, -0.1f, 0.9f, -0.2f, 1.2f, 321, 1, 2201);
    StoreSyntheticTrack(tracks, 3, -0.5f, 0.2f, 0.3f, 0.7f, 0.4f, 1.0f, 321, 1, 2202);
    StoreSyntheticTrack(tracks, 4, 0.1f, 0.5f, -0.4f, -0.3f, 0.7f, 1.1f, -211, -1, 2301);
    StoreSyntheticTrack(tracks, 5, -0.6f, -0.2f, 0.3f, -0.4f, 0.2f, 0.8f, -211, -1, 2302);
    StoreSyntheticTrack(tracks, 6, -0.1f, 0.3f, -0.2f, -0.5f, 0.6f, 1.0f, -321, -1, 2401);
    StoreSyntheticTrack(tracks, 7, 0.5f, -0.4f, 0.1f, -0.2f, 0.3f, 0.9f, -321, -1, 2402);
    tracks.ChiToPrimaryVertex(4) = 1.f;
    tracks.ChiToPrimaryVertex(5) = 9.f;
    tracks.ChiToPrimaryVertex(6) = 1.f;
    tracks.ChiToPrimaryVertex(7) = 9.f;

    KFParticleGpuEventDesc* events = buffers.HostEvents();
    events[0] = KFParticleGpuEventDesc();
    events[0].eventId = 61;
    events[0].TrackSet(SecondaryPositiveFirst).tracks = KFParticleGpuRange(0, 4);
    events[0].TrackSet(SecondaryPositiveFirst).Species(Pion) = KFParticleGpuRange(0, 2);
    events[0].TrackSet(SecondaryPositiveFirst).Species(Kaon) = KFParticleGpuRange(2, 2);
    events[0].TrackSet(SecondaryNegativeFirst).tracks = KFParticleGpuRange(4, 4);
    events[0].TrackSet(SecondaryNegativeFirst).Species(Pion) = KFParticleGpuRange(4, 2);
    events[0].TrackSet(SecondaryNegativeFirst).Species(Kaon) = KFParticleGpuRange(6, 2);
  }

  void FillDefaultV0Fixture(KFParticleGpuBufferManager& buffers)
  {
    KFParticleGpuBufferCapacities requested = buffers.Capacities();
    requested.tracks = requested.tracks > 4u ? requested.tracks : 4u;
    requested.events = requested.events > 1u ? requested.events : 1u;
    requested.candidates = requested.candidates > 6u ? requested.candidates : 6u;
    requested.daughterIds = requested.daughterIds > 8u ? requested.daughterIds : 8u;
    buffers.EnsureCapacity(requested);
    buffers.SetInputSizes(4, 1, 1);

    KFParticleGpuInputTrackSoAView tracks = buffers.HostInputTracks();
    StoreSyntheticTrack(tracks, 0, -0.2f, 0.1f, 0.0f, 0.8f, 0.1f, 1.0f, 211, 1, 3101);
    StoreSyntheticTrack(tracks, 1, 0.4f, -0.3f, 0.2f, 0.5f, 0.6f, 0.9f, 2212, 1, 3201);
    StoreSyntheticTrack(tracks, 2, 0.1f, 0.5f, -0.4f, -0.3f, 0.7f, 1.1f, -211, -1, 3301);
    StoreSyntheticTrack(tracks, 3, -0.6f, -0.2f, 0.3f, -0.4f, 0.2f, 0.8f, -2212, -1, 3401);

    KFParticleGpuEventDesc* events = buffers.HostEvents();
    events[0] = KFParticleGpuEventDesc();
    events[0].eventId = 81;
    events[0].TrackSet(SecondaryPositiveFirst).tracks = KFParticleGpuRange(0, 2);
    events[0].TrackSet(SecondaryPositiveFirst).Species(Pion) = KFParticleGpuRange(0, 1);
    events[0].TrackSet(SecondaryPositiveFirst).Species(Proton) = KFParticleGpuRange(1, 1);
    events[0].TrackSet(SecondaryNegativeFirst).tracks = KFParticleGpuRange(2, 2);
    events[0].TrackSet(SecondaryNegativeFirst).Species(Pion) = KFParticleGpuRange(2, 1);
    events[0].TrackSet(SecondaryNegativeFirst).Species(Proton) = KFParticleGpuRange(3, 1);
  }

  void AddDefaultV0ChannelsWithoutCpuPairGate(KFParticleGpuDecayPlan& plan)
  {
    KFParticleGpuTwoDaughterChannel k0 = MakeK0ShortToPiPlusPiMinusChannel();
    KFParticleGpuTwoDaughterChannel lambda = MakeLambdaToProtonPiMinusChannel();
    KFParticleGpuTwoDaughterChannel antiLambda =
      MakeAntiLambdaToAntiProtonPiPlusChannel();
    k0.maxDaughterDistance = -1.f;
    lambda.maxDaughterDistance = -1.f;
    antiLambda.maxDaughterDistance = -1.f;
    plan.AddTwoDaughterChannel(k0);
    plan.AddTwoDaughterChannel(lambda);
    plan.AddTwoDaughterChannel(antiLambda);
  }

  void FillDecayPlanMultiEventFixture(KFParticleGpuBufferManager& buffers)
  {
    KFParticleGpuBufferCapacities requested = buffers.Capacities();
    requested.tracks = requested.tracks > 10u ? requested.tracks : 10u;
    requested.events = requested.events > 2u ? requested.events : 2u;
    requested.candidates = requested.candidates > 6u ? requested.candidates : 6u;
    requested.daughterIds = requested.daughterIds > 8u ? requested.daughterIds : 8u;
    buffers.EnsureCapacity(requested);
    buffers.SetInputSizes(10, 1, 2);

    KFParticleGpuInputTrackSoAView tracks = buffers.HostInputTracks();
    StoreSyntheticTrack(tracks, 0, -0.8f, 0.2f, 0.1f, 0.3f, 0.2f, 1.0f, 211, 1, 2501);
    StoreSyntheticTrack(tracks, 1, -0.2f, 0.3f, -0.1f, -0.4f, 0.5f, 0.8f, -211, -1, 2601);
    StoreSyntheticTrack(tracks, 2, -0.2f, 0.1f, 0.0f, 0.8f, 0.1f, 1.0f, 211, 1, 2701);
    StoreSyntheticTrack(tracks, 3, 0.4f, -0.3f, 0.2f, 0.5f, 0.6f, 0.9f, 211, 1, 2702);
    StoreSyntheticTrack(tracks, 4, 0.2f, 0.4f, -0.1f, 0.9f, -0.2f, 1.2f, 321, 1, 2801);
    StoreSyntheticTrack(tracks, 5, -0.5f, 0.2f, 0.3f, 0.7f, 0.4f, 1.0f, 321, 1, 2802);
    StoreSyntheticTrack(tracks, 6, 0.1f, 0.5f, -0.4f, -0.3f, 0.7f, 1.1f, -211, -1, 2901);
    StoreSyntheticTrack(tracks, 7, -0.6f, -0.2f, 0.3f, -0.4f, 0.2f, 0.8f, -211, -1, 2902);
    StoreSyntheticTrack(tracks, 8, -0.1f, 0.3f, -0.2f, -0.5f, 0.6f, 1.0f, -321, -1, 3001);
    StoreSyntheticTrack(tracks, 9, 0.5f, -0.4f, 0.1f, -0.2f, 0.3f, 0.9f, -321, -1, 3002);
    tracks.ChiToPrimaryVertex(6) = 1.f;
    tracks.ChiToPrimaryVertex(7) = 9.f;
    tracks.ChiToPrimaryVertex(8) = 1.f;
    tracks.ChiToPrimaryVertex(9) = 9.f;

    KFParticleGpuEventDesc* events = buffers.HostEvents();
    events[0] = KFParticleGpuEventDesc();
    events[0].eventId = 71;
    events[0].TrackSet(SecondaryPositiveFirst).tracks = KFParticleGpuRange(0, 1);
    events[0].TrackSet(SecondaryPositiveFirst).Species(Pion) = KFParticleGpuRange(0, 1);
    events[0].TrackSet(SecondaryNegativeFirst).tracks = KFParticleGpuRange(1, 1);
    events[0].TrackSet(SecondaryNegativeFirst).Species(Pion) = KFParticleGpuRange(1, 1);
    events[1] = KFParticleGpuEventDesc();
    events[1].eventId = 72;
    events[1].TrackSet(SecondaryPositiveFirst).tracks = KFParticleGpuRange(2, 4);
    events[1].TrackSet(SecondaryPositiveFirst).Species(Pion) = KFParticleGpuRange(2, 2);
    events[1].TrackSet(SecondaryPositiveFirst).Species(Kaon) = KFParticleGpuRange(4, 2);
    events[1].TrackSet(SecondaryNegativeFirst).tracks = KFParticleGpuRange(6, 4);
    events[1].TrackSet(SecondaryNegativeFirst).Species(Pion) = KFParticleGpuRange(6, 2);
    events[1].TrackSet(SecondaryNegativeFirst).Species(Kaon) = KFParticleGpuRange(8, 2);
  }

  void FillBatchSelectionFixture(KFParticleGpuBufferManager& buffers)
  {
    KFParticleGpuBufferCapacities requested = buffers.Capacities();
    requested.tracks = requested.tracks > 4u ? requested.tracks : 4u;
    requested.events = requested.events > 2u ? requested.events : 2u;
    requested.candidates = requested.candidates > 2u ? requested.candidates : 2u;
    requested.daughterIds = requested.daughterIds > 4u ? requested.daughterIds : 4u;
    requested.selectedCandidates = requested.selectedCandidates > 2u ? requested.selectedCandidates : 2u;
    buffers.EnsureCapacity(requested);
    buffers.SetInputSizes(4u, 0u, 2u);

    KFParticleGpuInputTrackSoAView tracks = buffers.HostInputTracks();
    StoreSyntheticTrack(tracks, 0u, -0.2f, 0.1f, 0.f, 0.8f, 0.1f, 1.f, 211, 1, 4101);
    StoreSyntheticTrack(tracks, 1u, 0.1f, 0.5f, -0.4f, -0.3f, 0.7f, 1.1f, -211, -1, 4201);
    StoreSyntheticTrack(tracks, 2u, -0.1f, -0.2f, 0.1f, 0.7f, 0.2f, 0.9f, 211, 1, 4301);
    StoreSyntheticTrack(tracks, 3u, 0.2f, 0.4f, -0.3f, -0.2f, 0.6f, 1.0f, -211, -1, 4401);
    if (tracks.FieldCoefficientsData()) {
      const KFParticleGpuFieldRegion zeroField;
      for (unsigned int track = 0u; track < 4u; ++track) {
        StoreFieldRegion(zeroField, tracks, track);
      }
    }

    KFParticleGpuEventDesc* events = buffers.HostEvents();
    for (unsigned int event = 0u; event < 2u; ++event) {
      const unsigned int offset = 2u * event;
      events[event] = KFParticleGpuEventDesc();
      events[event].eventId = 191u + event;
      events[event].TrackSet(SecondaryPositiveFirst).tracks = KFParticleGpuRange(offset, 1u);
      events[event].TrackSet(SecondaryPositiveFirst).Species(Pion) = KFParticleGpuRange(offset, 1u);
      events[event].TrackSet(SecondaryNegativeFirst).tracks = KFParticleGpuRange(offset + 1u, 1u);
      events[event].TrackSet(SecondaryNegativeFirst).Species(Pion) = KFParticleGpuRange(offset + 1u, 1u);
    }
  }

  void TestSteeringTwoDaughterStage(KFParticleGpuRuntime& runtime,
                                    KFParticleGpuBufferManager& buffers)
  {
    FillGeneratedPipelineFixture(buffers);
    const KFParticleGpuTwoDaughterTaskSource source = MakePionPairSource();

    KFParticleGpuRange firstRange(0, 2);
    KFParticleGpuRange secondRange(2, 2);
    KFParticleGpuTwoDaughterTask referenceTask;
    FillTwoDaughterTask(firstRange, secondRange, 0, source, referenceTask);
    KFParticleGpuFitState reference;
    assert(BuildTwoDaughterCandidate(MakeConstView(buffers.HostInputTracks()), referenceTask, reference));

    runtime.GetSteering().RunTwoDaughterStage(source, 3u);

    const KFParticleGpuConstCandidatePoolView candidates = MakeConstView(buffers.HostCandidates());
    assert(candidates.Size() == 3);
    assert(candidates.Daughters().Size() == 4);
    assert(candidates.OverflowFlags() == DaughterCapacityExceeded);

    KFParticleGpuFitState actual;
    LoadCandidateFit(candidates, 0, actual);
    ExpectFitStateClose(actual, reference, 2.e-5f);
    assert(candidates.Metadata().Pdg(0) == 310);
    assert(candidates.Metadata().DaughterCount(0) == 2);
    assert(candidates.Daughters().SourceId(0) == 901);
    assert(candidates.Daughters().SourceId(1) == 1001);
    assert(candidates.Metadata().Flags(0)
           == (KFGpuCandidateValid | KFGpuCandidateLineDca | KFGpuCandidateEnergyFit));
    for (unsigned int i = 1u; i < 3u; ++i) {
      assert(candidates.Metadata().DaughterCount(i) == 0u);
      assert(candidates.Metadata().Flags(i) & KFGpuCandidateBuildFailed);
    }

    bool zeroCapacityRejected = false;
    try {
      runtime.GetSteering().RunTwoDaughterStage(source, 0u);
    }
    catch (const std::invalid_argument&) {
      zeroCapacityRejected = true;
    }
    assert(zeroCapacityRejected);

    Pass("two-daughter-steering-stage", "steering runs upload, task generation, construction, and download");
  }

  void TestSteeringCompactTwoDaughterStage(KFParticleGpuRuntime& runtime,
                                           KFParticleGpuBufferManager& buffers)
  {
    FillGeneratedPipelineFixture(buffers);
    KFParticleGpuTwoDaughterTaskSource source = MakePionPairSource();
    source.minFirstPixelHits = 0;

    runtime.GetSteering().RunTwoDaughterCompactStage(source, 2u);

    const KFParticleGpuConstCandidatePoolView candidates = MakeConstView(buffers.HostCandidates());
    assert(candidates.Size() == 2);
    assert(candidates.Daughters().Size() == 4);
    assert(candidates.OverflowFlags() == 0u);

    assert(HasCandidateDaughters(candidates, 901, 1001));
    assert(HasCandidateDaughters(candidates, 902, 1001));

    for (unsigned int i = 0; i < candidates.Size(); ++i) {
      assert(candidates.Metadata().Pdg(i) == 310);
      assert(candidates.Metadata().DaughterCount(i) == 2);
      assert(candidates.Metadata().Flags(i)
             == static_cast<unsigned int>(
               KFGpuCandidateValid | KFGpuCandidateLineDca | KFGpuCandidateEnergyFit));
    }

    bool zeroCapacityRejected = false;
    try {
      runtime.GetSteering().RunTwoDaughterCompactStage(source, 0u);
    }
    catch (const std::invalid_argument&) {
      zeroCapacityRejected = true;
    }
    assert(zeroCapacityRejected);

    Pass("two-daughter-compact-steering-stage",
         "steering runs compact pair generation and constructs only accepted task slots");
  }

  void TestSteeringCompactTwoDaughterEdgeCases(KFParticleGpuRuntime& runtime,
                                               KFParticleGpuBufferManager& buffers)
  {
    FillGeneratedPipelineFixture(buffers);
    KFParticleGpuTwoDaughterTaskSource source = MakePionPairSource();
    source.minFirstPixelHits = 0;

    runtime.GetSteering().RunTwoDaughterCompactStage(source, 1u);
    KFParticleGpuConstCandidatePoolView candidates = MakeConstView(buffers.HostCandidates());
    assert(candidates.Size() == 1);
    assert(candidates.Daughters().Size() == 2);
    assert(candidates.OverflowFlags() == CandidateCapacityExceeded);
    assert(HasCandidateDaughters(candidates, 901, 1001)
           || HasCandidateDaughters(candidates, 902, 1001));

    FillGeneratedPipelineFixture(buffers);
    KFParticleGpuEventDesc* events = buffers.HostEvents();
    events[0].TrackSet(SecondaryNegativeFirst).tracks = KFParticleGpuRange(2, 0);
    events[0].TrackSet(SecondaryNegativeFirst).Species(Pion) = KFParticleGpuRange(2, 0);
    runtime.GetSteering().RunTwoDaughterCompactStage(source, 2u);
    candidates = MakeConstView(buffers.HostCandidates());
    assert(candidates.Size() == 0);
    assert(candidates.Daughters().Size() == 0);
    assert(candidates.OverflowFlags() == 0u);

    buffers.SetInputSizes(6, 1, 2);
    KFParticleGpuInputTrackSoAView tracks = buffers.HostInputTracks();
    StoreSyntheticTrack(tracks, 0, -0.8f, 0.2f, 0.1f, 0.3f, 0.2f, 1.0f, 211, 1, 1701);
    StoreSyntheticTrack(tracks, 1, -0.2f, 0.3f, -0.1f, -0.4f, 0.5f, 0.8f, -211, -1, 1801);
    StoreSyntheticTrack(tracks, 2, -0.2f, 0.1f, 0.0f, 0.8f, 0.1f, 1.0f, 211, 1, 1901);
    StoreSyntheticTrack(tracks, 3, 0.4f, -0.3f, 0.2f, 0.5f, 0.6f, 0.9f, 211, 1, 1902);
    StoreSyntheticTrack(tracks, 4, 0.1f, 0.5f, -0.4f, -0.3f, 0.7f, 1.1f, -211, -1, 2001);
    StoreSyntheticTrack(tracks, 5, -0.6f, -0.2f, 0.3f, -0.4f, 0.2f, 0.8f, -211, -1, 2002);
    tracks.ChiToPrimaryVertex(4) = 1.f;
    tracks.ChiToPrimaryVertex(5) = 9.f;

    events = buffers.HostEvents();
    events[0] = KFParticleGpuEventDesc();
    events[0].eventId = 51;
    events[0].TrackSet(SecondaryPositiveFirst).tracks = KFParticleGpuRange(0, 1);
    events[0].TrackSet(SecondaryPositiveFirst).Species(Pion) = KFParticleGpuRange(0, 1);
    events[0].TrackSet(SecondaryNegativeFirst).tracks = KFParticleGpuRange(1, 1);
    events[0].TrackSet(SecondaryNegativeFirst).Species(Pion) = KFParticleGpuRange(1, 1);
    events[1] = KFParticleGpuEventDesc();
    events[1].eventId = 52;
    events[1].TrackSet(SecondaryPositiveFirst).tracks = KFParticleGpuRange(2, 2);
    events[1].TrackSet(SecondaryPositiveFirst).Species(Pion) = KFParticleGpuRange(2, 2);
    events[1].TrackSet(SecondaryNegativeFirst).tracks = KFParticleGpuRange(4, 2);
    events[1].TrackSet(SecondaryNegativeFirst).Species(Pion) = KFParticleGpuRange(4, 2);

    source.eventIndex = 1;
    runtime.GetSteering().RunTwoDaughterCompactStage(source, 2u);
    candidates = MakeConstView(buffers.HostCandidates());
    assert(candidates.Size() == 2);
    assert(candidates.Daughters().Size() == 4);
    assert(candidates.OverflowFlags() == 0u);
    assert(HasCandidateDaughters(candidates, 1901, 2001));
    assert(HasCandidateDaughters(candidates, 1902, 2001));
    for (unsigned int i = 0; i < candidates.Size(); ++i) {
      assert(candidates.Metadata().EventIndex(i) == 1u);
    }

    Pass("two-daughter-compact-steering-edge-cases",
         "compact steering reports truncation, handles empty ranges, and selects event descriptors");
  }

  void TestDecayPlanOneChannelExecutor(KFParticleGpuRuntime& runtime,
                                       KFParticleGpuBufferManager& buffers)
  {
    KFParticleGpuSteering& steering = runtime.GetSteering();
    KFParticleGpuDecayPlan& plan = steering.GetDecayPlan();
    plan.Clear();

    FillGeneratedPipelineFixture(buffers);
    const std::vector<KFParticleGpuTwoDaughterChannelResult>& emptyResults =
      steering.RunDecayPlan(0u, 2u);
    assert(emptyResults.empty());
    KFParticleGpuConstCandidatePoolView candidates = MakeConstView(buffers.HostCandidates());
    assert(candidates.Size() == 0u);
    assert(candidates.Daughters().Size() == 0u);
    assert(candidates.OverflowFlags() == 0u);

    const KFParticleGpuTwoDaughterChannel channel = MakePionPairChannel(41u);
    plan.AddTwoDaughterChannel(channel);
    FillGeneratedPipelineFixture(buffers);
    const std::vector<KFParticleGpuTwoDaughterChannelResult>& results =
      steering.RunDecayPlan(0u, 2u);
    assert(results.size() == 1u);
    assert(&results == &steering.LastDecayPlanResults());
    const KFParticleGpuTwoDaughterChannelResult& result = results[0];
    assert(result.channelId == 41u);
    assert(result.motherPdg == 310);
    assert(result.eventIndex == 0u);
    assert(result.totalPairs == 4u);
    assert(result.acceptedTasks == 2u);
    assert(result.storedTasks == 2u);
    assert(!result.Truncated());
    assert(result.candidates.offset == 0u);
    assert(result.candidates.size == 2u);
    assert(result.candidates.daughterOffset == 0u);
    assert(result.candidates.daughterSize == 4u);
    assert(result.candidates.overflowFlags == 0u);

    candidates = MakeConstView(buffers.HostCandidates());
    assert(candidates.Size() == result.candidates.size);
    assert(candidates.Daughters().Size() == result.candidates.daughterSize);
    assert(HasCandidateDaughters(candidates, 901, 1001));
    assert(HasCandidateDaughters(candidates, 902, 1001));

    FillGeneratedPipelineFixture(buffers);
    steering.RunTwoDaughterCompactStage(MakeTwoDaughterTaskSource(channel, 0u), 2u);
    candidates = MakeConstView(buffers.HostCandidates());
    assert(candidates.Size() == result.candidates.size);
    assert(candidates.Daughters().Size() == result.candidates.daughterSize);
    assert(HasCandidateDaughters(candidates, 901, 1001));
    assert(HasCandidateDaughters(candidates, 902, 1001));

    plan.Clear();
    Pass("decay-plan-one-channel-executor",
         "decay plan executor runs empty and one-channel plans through compact steering");
  }

  void TestDecayPlanFieldAwareDiagnosticExecutor(KFParticleGpuRuntime& runtime,
                                                 KFParticleGpuBufferManager& buffers)
  {
    KFParticleGpuSteering& steering = runtime.GetSteering();
    KFParticleGpuDecayPlan& plan = steering.GetDecayPlan();
    plan.Clear();

    const KFParticleGpuTwoDaughterChannel channel =
      MakeFieldAwarePionPairChannel(77u);
    plan.AddTwoDaughterChannel(channel);

    FillGeneratedPipelineFixture(buffers);
    KFParticleGpuInputTrackSoAView tracks = buffers.HostInputTracks();
    StoreDiagnosticField(tracks, 0);

    const KFParticleGpuTwoDaughterTaskSource source =
      MakeTwoDaughterTaskSource(channel, 0u);
    const KFParticleGpuRange firstRange(0, 2);
    const KFParticleGpuRange secondRange(2, 2);
    KFParticleGpuTwoDaughterTask referenceTask;
    FillTwoDaughterTask(firstRange, secondRange, 0u, source, referenceTask);
    KFParticleGpuFitState referenceMother;
    assert(BuildTwoDaughterKinematicCandidate(
      MakeConstView(buffers.HostInputTracks()), referenceTask, referenceMother));

    KFParticleGpuTwoDaughterTask straightTask = referenceTask;
    straightTask.transportMode = KFGpuTransportStraightLine;
    KFParticleGpuFitState straightMother;
    assert(BuildTwoDaughterKinematicCandidate(
      MakeConstView(buffers.HostInputTracks()), straightTask, straightMother));

    const std::vector<KFParticleGpuTwoDaughterChannelResult>& results =
      steering.RunDecayPlan(0u, 2u);
    assert(results.size() == 1u);
    const KFParticleGpuTwoDaughterChannelResult& result = results[0];
    assert(result.channelId == 77u);
    assert(result.motherPdg == 997);
    assert(result.eventIndex == 0u);
    assert(result.totalPairs == 4u);
    assert(result.acceptedTasks == 2u);
    assert(result.storedTasks == 2u);
    assert(!result.Truncated());
    assert(result.candidates.offset == 0u);
    assert(result.candidates.size == 2u);
    assert(result.candidates.daughterOffset == 0u);
    assert(result.candidates.daughterSize == 4u);
    assert(result.candidates.overflowFlags == 0u);

    const KFParticleGpuConstCandidatePoolView candidates =
      MakeConstView(buffers.HostCandidates());
    assert(candidates.Size() == 2u);
    assert(candidates.Daughters().Size() == 4u);
    for (unsigned int candidate = 0u; candidate < candidates.Size(); ++candidate) {
      assert(candidates.Metadata().Pdg(candidate) == 997);
      assert(candidates.Metadata().Flags(candidate)
             == static_cast<unsigned int>(KFGpuCandidateValid
                                          | KFGpuCandidateLineDca));
    }
    assert(HasCandidateDaughters(candidates, 901, 1001));
    assert(HasCandidateDaughters(candidates, 902, 1001));

    const unsigned int candidateIndex = FindChannelCandidateByDaughters(
      candidates, channel.channelId, 901, 1001);
    assert(candidateIndex < candidates.Size());
    KFParticleGpuFitState fieldMother;
    LoadCandidateFit(candidates, candidateIndex, fieldMother);
    ExpectFieldAwareSeedClose(fieldMother, referenceMother);
    assert(!AlmostEqual(fieldMother.X(), straightMother.X(), 1.e-6f));
    assert(!AlmostEqual(fieldMother.Px(), straightMother.Px(), 1.e-6f));

    plan.Clear();
    Pass("decay-plan-field-aware-diagnostic",
         "decay plan executor routes an explicit field-aware diagnostic channel through generated tasks");
  }

  void TestDecayPlanMixedTransportExecutor(KFParticleGpuRuntime& runtime,
                                           KFParticleGpuBufferManager& buffers)
  {
    KFParticleGpuSteering& steering = runtime.GetSteering();
    KFParticleGpuDecayPlan& plan = steering.GetDecayPlan();
    plan.Clear();

    const KFParticleGpuTwoDaughterChannel fieldChannel =
      MakeFieldAwarePionPairChannel(78u);
    const KFParticleGpuTwoDaughterChannel lineChannel =
      MakePionPairChannel(79u);
    plan.AddTwoDaughterChannel(fieldChannel);
    plan.AddTwoDaughterChannel(lineChannel);

    KFParticleGpuBufferCapacities requested = buffers.Capacities();
    requested.candidates = requested.candidates > 4u ? requested.candidates : 4u;
    requested.daughterIds = requested.daughterIds > 8u ? requested.daughterIds : 8u;
    buffers.EnsureCapacity(requested);
    FillGeneratedPipelineFixture(buffers);
    KFParticleGpuInputTrackSoAView tracks = buffers.HostInputTracks();
    StoreDiagnosticField(tracks, 0);

    const std::vector<KFParticleGpuTwoDaughterChannelResult>& results =
      steering.RunDecayPlan(0u, 2u);
    assert(results.size() == 2u);

    assert(results[0].channelId == 78u);
    assert(results[0].motherPdg == 997);
    assert(results[0].constructedCandidates == 2u);
    assert(results[0].constructedDaughters == 4u);
    assert(results[0].candidates.overflowFlags == 0u);

    assert(results[1].channelId == 79u);
    assert(results[1].motherPdg == 310);
    assert(results[1].constructedCandidates == 2u);
    assert(results[1].constructedDaughters == 4u);
    assert(results[1].candidates.overflowFlags == 0u);
    assert(results[0].generationCandidates.offset == 0u);
    assert(results[0].generationCandidates.size == 4u);
    assert(results[1].generationCandidates.offset == 0u);
    assert(results[1].generationCandidates.size == 4u);

    const KFParticleGpuConstCandidatePoolView candidates =
      MakeConstView(buffers.HostCandidates());
    assert(candidates.Size() == 4u);
    assert(candidates.Daughters().Size() == 8u);
    assert(candidates.OverflowFlags() == 0u);
    ExpectChannelCandidatePdgs(candidates, results[0], 997);
    ExpectChannelCandidatePdgs(candidates, results[1], 310);
    ExpectChannelDaughterRange(candidates, results[0]);
    ExpectChannelDaughterRange(candidates, results[1]);

    KFParticleGpuFitState fieldMother;
    KFParticleGpuFitState lineMother;
    const unsigned int fieldCandidate = FindChannelCandidateByDaughters(
      candidates, fieldChannel.channelId, 901, 1001);
    const unsigned int lineCandidate = FindChannelCandidateByDaughters(
      candidates, lineChannel.channelId, 901, 1001);
    assert(fieldCandidate < candidates.Size());
    assert(lineCandidate < candidates.Size());
    LoadCandidateFit(candidates, fieldCandidate, fieldMother);
    LoadCandidateFit(candidates, lineCandidate, lineMother);
    assert(!AlmostEqual(fieldMother.X(), lineMother.X(), 1.e-6f));
    assert(!AlmostEqual(fieldMother.Px(), lineMother.Px(), 1.e-6f));
    assert(HasCandidateDaughters(candidates, 901, 1001));
    assert(HasCandidateDaughters(candidates, 902, 1001));

    plan.Clear();
    Pass("decay-plan-mixed-transport",
         "decay plan appends field-aware diagnostic and straight-line channels without range overlap");
  }

  void TestDecayPlanDefaultK0Executor(KFParticleGpuRuntime& runtime,
                                      KFParticleGpuBufferManager& buffers)
  {
    KFParticleGpuSteering& steering = runtime.GetSteering();
    KFParticleGpuDecayPlan& plan = steering.GetDecayPlan();
    plan.Clear();
    KFParticleGpuTwoDaughterChannel channel =
      MakeK0ShortToPiPlusPiMinusChannel();
    // This fixture validates executor range accounting with four deliberately
    // broad pairs; the CPU pre-construction pair gate has its own test.
    channel.maxDaughterDistance = -1.f;
    plan.AddTwoDaughterChannel(channel);

    KFParticleGpuBufferCapacities requested = buffers.Capacities();
    requested.candidates = requested.candidates > 4u ? requested.candidates : 4u;
    requested.daughterIds = requested.daughterIds > 8u ? requested.daughterIds : 8u;
    buffers.EnsureCapacity(requested);
    FillGeneratedPipelineFixture(buffers);
    const std::vector<KFParticleGpuTwoDaughterChannelResult>& results =
      steering.RunDecayPlan(0u, 4u);
    assert(results.size() == 1u);
    assert(results[0].channelId == KFGpuChannelK0ShortToPiPlusPiMinus);
    assert(results[0].motherPdg == 310);
    assert(results[0].acceptedTasks == 4u);
    assert(results[0].candidates.size == 4u);

    const KFParticleGpuConstCandidatePoolView candidates =
      MakeConstView(buffers.HostCandidates());
    assert(candidates.Size() == 4u);
    assert(HasCandidateDaughters(candidates, 901, 1001));
    assert(HasCandidateDaughters(candidates, 902, 1001));
    assert(HasCandidateDaughters(candidates, 901, 1002));
    assert(HasCandidateDaughters(candidates, 902, 1002));
    ExpectChannelCandidatePdgs(candidates, results[0], 310);
    ExpectChannelDaughterRange(candidates, results[0]);

    plan.Clear();
    Pass("decay-plan-default-k0-executor",
         "default K0S channel descriptor runs through the standalone decay-plan executor");
  }

  void TestCpuFinderTwoDaughterPairGate(KFParticleGpuRuntime& runtime,
                                        KFParticleGpuBufferManager& buffers)
  {
    KFParticleGpuSteering& steering = runtime.GetSteering();
    KFParticleGpuDecayPlan& plan = steering.GetDecayPlan();
    plan.Clear();
    const KFParticleGpuTwoDaughterChannel channel =
      MakeK0ShortToPiPlusPiMinusChannel();
    assert(AlmostEqual(channel.maxDaughterDistance, 1.f));
    plan.AddTwoDaughterChannel(channel);
    FillGeneratedPipelineFixture(buffers);

    const KFParticleGpuConstInputTrackSoAView tracks =
      MakeConstView(buffers.HostInputTracks());
    unsigned int expectedAccepted = 0u;
    for (unsigned int first = 0u; first < 2u; ++first) {
      for (unsigned int second = 2u; second < 4u; ++second) {
        if (PassCpuFinderTwoDaughterPairGate(
              tracks, first, second, channel.transportMode,
              channel.maxDaughterDistance)) {
          ++expectedAccepted;
        }
      }
    }
    assert(expectedAccepted > 0u && expectedAccepted < 4u);

    const auto& results = steering.RunDecayPlan(0u, 4u);
    assert(results.size() == 1u);
    assert(results[0].acceptedTasks == expectedAccepted);
    assert(steering.LastTwoDaughterRoutingMonitorData().acceptedTasks
           == expectedAccepted);

    plan.Clear();
    Pass("cpu-finder-two-daughter-pair-gate",
         "device routing mirrors the CPU fast-DCA distance and momentum gate before V0 construction");
  }

  bool BuildHostFullFieldTrace(
    const KFParticleGpuConstInputTrackSoAView& inputTracks,
    const KFParticleGpuTwoDaughterTask& task,
    KFParticleGpuFullFieldTwoDaughterTrace& trace)
  {
    trace.stage = 0u;
    const bool reverse = (task.flags & KFGpuTwoDaughterReverseFitOrder) != 0u;
    const unsigned int firstTrackIndex = reverse ? task.secondTrack : task.firstTrack;
    const unsigned int secondTrackIndex = reverse ? task.firstTrack : task.secondTrack;
    const float firstMass = reverse ? task.secondMass : task.firstMass;
    const float secondMass = reverse ? task.firstMass : task.secondMass;
    KFParticleGpuTrackState firstTrack;
    KFParticleGpuTrackState secondTrack;
    LoadTrackState(inputTracks, firstTrackIndex, firstTrack);
    LoadTrackState(inputTracks, secondTrackIndex, secondTrack);
    KFParticleGpuFitState first;
    KFParticleGpuFitState second;
    first.Initialize(firstTrack, inputTracks.Charge(firstTrackIndex), firstMass);
    second.Initialize(secondTrack, inputTracks.Charge(secondTrackIndex), secondMass);
    first.NDF() = -1;
    KFParticleGpuFieldRegion firstField;
    KFParticleGpuFieldRegion secondField;
    LoadFieldRegion(inputTracks, firstTrackIndex, firstField);
    LoadFieldRegion(inputTracks, secondTrackIndex, secondField);
    trace.by = firstField.Get(first.Z()).y;
    float unusedDs[2] = {};
    KFParticleGpuMath::GetDStoParticleByCpuCompatible(
      first, second, trace.by, unusedDs, false, trace.firstRoots, trace.secondRoots);
    trace.useMiddlePoint = KFParticleGpuMath::UseCpuCompatibleMiddleDcaPoint(
      first, second, firstField, secondField, trace.by) ? 1u : 0u;
    KFParticleGpuMath::GetDStoParticleByCpuCompatibleDerivatives(
      first,
      second,
      trace.by,
      trace.useMiddlePoint != 0u,
      trace.dS,
      trace.dsdr,
      &trace.dcaArithmetic);
    trace.stage = 1u;
    if (!KFParticleGpuMath::BuildFullFieldDcaIndependentTransportStates(
          first,
          second,
          firstField,
          secondField,
          trace.preliminaryFirst,
          trace.preliminarySecond)) {
      return false;
    }
    trace.stage = 2u;
    if (!KFParticleGpuMath::BuildFullFieldDcaMeasurementSeedAnalytic(
          trace.preliminaryFirst,
          trace.preliminarySecond,
          firstField,
          secondField,
          trace.secondPassCurrent,
          trace.secondPassMeasurement)) {
      return false;
    }
    trace.stage = 3u;
    trace.fittedMother = trace.secondPassCurrent;
    if (!KFParticleGpuMath::AddDaughterWithEnergyFit(
          trace.fittedMother, trace.secondPassMeasurement, second.Q())) {
      return false;
    }
    trace.fittedMother.SumDaughterMass() =
      first.SumDaughterMass() + second.SumDaughterMass();
    trace.fittedMother.MassHypo() = -1.f;
    trace.fittedMother.ConstructMethod() = 0;
    trace.stage = 4u;
    return true;
  }

  KFParticleGpuFitState RunDirectFullFieldTask(
    KFParticleGpuRuntime& runtime,
    KFParticleGpuBufferManager& buffers,
    const KFParticleGpuTwoDaughterTask& task,
    KFParticleGpuFullFieldTwoDaughterTrace& trace)
  {
    buffers.UploadInput();
    xpu::buffer<KFParticleGpuTwoDaughterTask> taskBuffer(1u, xpu::buf_io);
    HostPointer(taskBuffer)[0] = task;
    runtime.GetQueue().copy(taskBuffer, xpu::h2d);
    runtime.GetQueue().wait();

    xpu::buffer<KFParticleGpuFullFieldTwoDaughterTrace> traceBuffer(1u, xpu::buf_io);
    runtime.GetQueue().launch<KFParticleGpuFullFieldTwoDaughterProbe>(
      xpu::n_threads(1),
      MakeConstView(buffers.DeviceInputTracks()),
      taskBuffer.get(),
      traceBuffer.get());
    runtime.GetQueue().copy(traceBuffer, xpu::d2h);
    runtime.GetQueue().wait();
    trace = HostPointer(traceBuffer)[0];

    buffers.ResetCandidates();
    runtime.GetQueue().launch<KFParticleGpuTwoDaughterTaskKernel>(
      xpu::n_threads(1),
      MakeConstView(buffers.DeviceInputTracks()),
      taskBuffer.get(),
      1u,
      buffers.DeviceCandidates());
    runtime.GetQueue().wait();
    buffers.DownloadCandidates();
    const KFParticleGpuConstCandidatePoolView candidates =
      MakeConstView(buffers.HostCandidates());
    assert(candidates.Size() == 1u);
    KFParticleGpuFitState mother;
    LoadCandidateFit(candidates, 0u, mother);
    return mother;
  }

  void DumpFullFieldTraceState(const char* stage,
                               const KFParticleGpuFitState& gpu,
                               const KFParticleGpuFitState& host)
  {
    for (int parameter = 0; parameter < KFParticleGpuFitState::NumberOfParameters;
         ++parameter) {
      std::cerr << "TRACE full-field stage=" << stage << " parameter=" << parameter
                << " gpu=" << gpu.Parameter(parameter)
                << " host=" << host.Parameter(parameter)
                << " delta=" << (gpu.Parameter(parameter) - host.Parameter(parameter)) << '\n';
    }
    for (int covariance = 0;
         covariance < KFParticleGpuFitState::NumberOfCovarianceElements;
         ++covariance) {
      std::cerr << "TRACE full-field stage=" << stage << " covariance=" << covariance
                << " gpu=" << gpu.Covariance(covariance)
                << " host=" << host.Covariance(covariance)
                << " delta=" << (gpu.Covariance(covariance) - host.Covariance(covariance)) << '\n';
    }
    std::cerr << "TRACE full-field stage=" << stage
              << " chi2-gpu=" << gpu.Chi2() << " chi2-host=" << host.Chi2()
              << " chi2-delta=" << (gpu.Chi2() - host.Chi2()) << '\n';
  }

  void DumpFullFieldTrace(const KFParticleGpuFullFieldTwoDaughterTrace& gpu,
                          const KFParticleGpuFullFieldTwoDaughterTrace& host)
  {
    std::cerr << "TRACE full-field stage=" << gpu.stage << '/' << host.stage
              << " middle=" << gpu.useMiddlePoint << '/' << host.useMiddlePoint
              << " by=" << gpu.by << '/' << host.by << '\n';
    DumpFullFieldTraceState("preliminary-first", gpu.preliminaryFirst, host.preliminaryFirst);
    DumpFullFieldTraceState("preliminary-second", gpu.preliminarySecond, host.preliminarySecond);
    DumpFullFieldTraceState("second-pass", gpu.secondPassCurrent, host.secondPassCurrent);
    DumpFullFieldTraceState("fitted-mother", gpu.fittedMother, host.fittedMother);
  }

  void DumpFullFieldTask(const char* path, const KFParticleGpuTwoDaughterTask& task)
  {
    std::cerr << "TRACE full-field task path=" << path
              << " mother=" << task.motherPdg
              << " tracks=" << task.firstTrack << ',' << task.secondTrack
              << " daughter-pdgs=" << task.firstDaughterPdg << ',' << task.secondDaughterPdg
              << " masses=" << task.firstMass << ',' << task.secondMass
              << " flags=" << task.flags
              << " reverse="
              << ((task.flags & KFGpuTwoDaughterReverseFitOrder) != 0u ? 1 : 0)
              << " transport=" << task.transportMode << '\n';
  }

  bool NeedsFullFieldTrace(const KFParticleGpuFitState& gpu,
                           const KFParticleGpuFitState& host,
                           int motherPdg)
  {
    const FieldAwareEnergyFitTolerance tolerance =
      DefaultV0EnergyFitTolerance(motherPdg);
    for (int parameter = 0; parameter < KFParticleGpuFitState::NumberOfParameters;
         ++parameter) {
      if (!AlmostEqualScaled(gpu.Parameter(parameter),
                             host.Parameter(parameter),
                             tolerance.stateAbsolute,
                             tolerance.stateRelative)) {
        return true;
      }
    }
    if (!AlmostEqual(gpu.Chi2(), host.Chi2(), tolerance.chi2)
        || gpu.NDF() != host.NDF()
        || gpu.Q() != host.Q()
        || !AlmostEqual(gpu.SumDaughterMass(), host.SumDaughterMass(), tolerance.mass)
        || !AlmostEqual(gpu.MassHypo(), host.MassHypo(), tolerance.mass)) {
      return true;
    }

    float gpuMass = 0.f;
    float gpuMassError = 0.f;
    float hostMass = 0.f;
    float hostMassError = 0.f;
    const bool gpuMassValid = KFParticleGpuMath::GetMass(gpu, gpuMass, gpuMassError);
    const bool hostMassValid = KFParticleGpuMath::GetMass(host, hostMass, hostMassError);
    return gpuMassValid != hostMassValid
           || (gpuMassValid && !AlmostEqual(gpuMass, hostMass, tolerance.mass));
  }

  void TestDefaultV0FieldAwareFixtures(KFParticleGpuRuntime& runtime,
                                      KFParticleGpuBufferManager& buffers)
  {
    struct Case {
      const char* name;
      KFParticleGpuTwoDaughterChannel channel;
      KFParticleGpuRange first, second;
      unsigned int fieldTrack, count;
      int firstId, secondId;
    };
    const Case cases[] = {
      {"k0", MakeK0ShortToPiPlusPiMinusChannel(), {0, 2}, {2, 2}, 0, 4, 901, 1001},
      {"lambda", MakeLambdaToProtonPiMinusChannel(), {1, 1}, {2, 1}, 1, 2, 3201, 3301},
      {"anti-lambda", MakeAntiLambdaToAntiProtonPiPlusChannel(), {3, 1}, {0, 1}, 3, 1, 3401, 3101}
    };
    auto& steering = runtime.GetSteering();
    auto& plan = steering.GetDecayPlan();
    auto requested = buffers.Capacities();
    requested.candidates = std::max(requested.candidates, 4u);
    requested.daughterIds = std::max(requested.daughterIds, 8u);
    buffers.EnsureCapacity(requested);

    for (const auto& fixture : cases) {
      std::cerr << "CASE field-aware-v0 " << fixture.name << '\n';
      plan.Clear();
      auto channel = fixture.channel;
      assert(channel.transportMode == KFGpuTransportFullField);
      channel.maxDaughterDistance = -1.f;
      plan.AddTwoDaughterChannel(channel);
      if (channel.motherPdg == 310) FillGeneratedPipelineFixture(buffers);
      else FillDefaultV0Fixture(buffers);
      auto tracks = buffers.HostInputTracks();
      StoreDiagnosticField(tracks, fixture.fieldTrack);
      KFParticleGpuTwoDaughterTask task;
      FillTwoDaughterTask(fixture.first, fixture.second, 0u,
                         MakeTwoDaughterTaskSource(channel, 0u), task);
      KFParticleGpuFitState reference, straight;
      assert(BuildTwoDaughterCandidate(MakeConstView(tracks), task, reference));
      auto straightTask = task;
      straightTask.transportMode = KFGpuTransportStraightLine;
      assert(BuildTwoDaughterCandidate(MakeConstView(tracks), straightTask, straight));

      // Distinguish device arithmetic errors from executor/routing errors.
      KFParticleGpuFullFieldTwoDaughterTrace hostTrace, deviceTrace;
      assert(BuildHostFullFieldTrace(MakeConstView(tracks), task, hostTrace));
      const auto direct = RunDirectFullFieldTask(runtime, buffers, task, deviceTrace);
      if (VerboseDiagnosticTraceEnabled() || NeedsFullFieldTrace(direct, reference, channel.motherPdg))
        DumpFullFieldTrace(deviceTrace, hostTrace);
      assert(deviceTrace.stage == hostTrace.stage);
      ExpectDefaultV0EnergyFitClose(direct, reference, channel.motherPdg);

      const auto& results = steering.RunDecayPlan(0u, fixture.count);
      assert(results.size() == 1u);
      assert(results[0].channelId == channel.channelId);
      assert(results[0].motherPdg == channel.motherPdg);
      assert(results[0].acceptedTasks == fixture.count);
      assert(results[0].candidates.size == fixture.count);
      const auto candidates = MakeConstView(buffers.HostCandidates());
      assert(candidates.Size() == fixture.count);
      assert(candidates.OverflowFlags() == 0u);
      const auto index = FindCandidateByDaughters(candidates, fixture.firstId, fixture.secondId);
      assert(index < candidates.Size());
      assert(candidates.Metadata().Pdg(index) == channel.motherPdg);
      const auto flags = KFGpuCandidateValid | KFGpuCandidateLineDca | KFGpuCandidateEnergyFit;
      assert((candidates.Metadata().Flags(index) & flags) == flags);
      KFParticleGpuFitState actual;
      LoadCandidateFit(candidates, index, actual);
      if (VerboseDiagnosticTraceEnabled() || NeedsFullFieldTrace(actual, reference, channel.motherPdg))
        DumpFullFieldTraceState("executor-vs-direct", actual, direct);
      ExpectDefaultV0EnergyFitClose(actual, reference, channel.motherPdg);
      assert(!AlmostEqual(actual.X(), straight.X(), 1.e-6f));
      assert(!AlmostEqual(actual.Px(), straight.Px(), 1.e-6f));
      Pass(fixture.name, "full-field direct kernel and executor preserve V0 fit and lineage");
    }
    plan.Clear();
  }

  void TestDefaultV0DeviceReference(KFParticleGpuRuntime& runtime,
                                    KFParticleGpuBufferManager& buffers, bool energyFit)
  {
    FillDefaultV0Fixture(buffers);
    KFParticleGpuInputTrackSoAView tracks = buffers.HostInputTracks();
    // One field-bearing first daughter per channel covers the track-set and
    // charge ordering of the three default V0 descriptors.
    StoreDiagnosticField(tracks, 0);
    StoreDiagnosticField(tracks, 1);
    StoreDiagnosticField(tracks, 3);

    const KFParticleGpuConstInputTrackSoAView inputTracks = MakeConstView(tracks);
    const KFParticleGpuTwoDaughterChannel channels[] = {
      MakeK0ShortToPiPlusPiMinusChannel(),
      MakeLambdaToProtonPiMinusChannel(),
      MakeAntiLambdaToAntiProtonPiPlusChannel()
    };
    constexpr unsigned int numberOfChannels = sizeof(channels) / sizeof(channels[0]);
    KFParticleGpuTwoDaughterTask tasks[numberOfChannels];
    KFParticleGpuFitState references[numberOfChannels];

    const KFParticleGpuEventDesc& event = buffers.HostEvents()[0];
    for (unsigned int i = 0; i < numberOfChannels; ++i) {
      const KFParticleGpuTwoDaughterChannel& channel = channels[i];
      assert(channel.transportMode == KFGpuTransportFullField);
      const KFParticleGpuTwoDaughterTaskSource source =
        MakeTwoDaughterTaskSource(channel, 0u);
      const KFParticleGpuRange firstRange =
        event.TrackSet(channel.firstTrackSet).Species(
          channel.firstSpecies == NumberOfTrackSpecies ? Proton : channel.firstSpecies);
      const KFParticleGpuRange secondRange =
        event.TrackSet(channel.secondTrackSet).Species(channel.secondSpecies);
      FillTwoDaughterTask(firstRange, secondRange, 0u, source, tasks[i]);

      // Isolate field transport from the later covariance-aware energy fit.
      if (energyFit) {
        assert((tasks[i].flags & KFGpuTwoDaughterUseLineDca) != 0u);
        assert((tasks[i].flags & KFGpuTwoDaughterUseEnergyFit) != 0u);
        assert(BuildTwoDaughterCandidate(inputTracks, tasks[i], references[i]));
      }
      else {
        tasks[i].flags = KFGpuTwoDaughterUseLineDca;
        assert(BuildTwoDaughterKinematicCandidate(inputTracks, tasks[i], references[i]));
      }
    }

    xpu::buffer<KFParticleGpuTwoDaughterTask> taskBuffer(numberOfChannels, xpu::buf_io);
    KFParticleGpuTwoDaughterTask* hostTasks = HostPointer(taskBuffer);
    for (unsigned int i = 0; i < numberOfChannels; ++i) {
      hostTasks[i] = tasks[i];
    }

    buffers.UploadInput();
    buffers.ResetCandidates();
    runtime.GetQueue().copy(taskBuffer, xpu::h2d);
    runtime.GetQueue().wait();
    runtime.GetQueue().launch<KFParticleGpuTwoDaughterTaskKernel>(
      xpu::n_threads(numberOfChannels),
      MakeConstView(buffers.DeviceInputTracks()),
      taskBuffer.get(),
      numberOfChannels,
      buffers.DeviceCandidates());
    runtime.GetQueue().wait();
    buffers.DownloadCandidates();

    const KFParticleGpuConstCandidatePoolView candidates =
      MakeConstView(buffers.HostCandidates());
    assert(candidates.Size() == numberOfChannels);
    assert(candidates.Daughters().Size() == 2u * numberOfChannels);
    assert(candidates.OverflowFlags() == 0u);
    for (unsigned int i = 0; i < numberOfChannels; ++i) {
      KFParticleGpuFitState actual;
      LoadCandidateFit(candidates, i, actual);
      if (energyFit) ExpectDefaultV0EnergyFitClose(actual, references[i], tasks[i].motherPdg);
      else ExpectFieldAwareSeedClose(actual, references[i]);
      assert(candidates.Metadata().Pdg(i) == tasks[i].motherPdg);
      const unsigned int flags = candidates.Metadata().Flags(i);
      if (energyFit) {
        assert(flags == (TwoDaughterCandidateFlags(tasks[i])
          | (PassTwoDaughterPostBuildSelection(tasks[i], references[i])
               ? 0u : static_cast<unsigned int>(KFGpuCandidateSelectionRejected))));
      }
      else {
        // Transport-only fixtures need not pass physical V0 selection cuts.
        const unsigned int required = KFGpuCandidateValid | KFGpuCandidateLineDca;
        assert((flags & required) == required);
        assert((flags & ~(required | KFGpuCandidateSelectionRejected)) == 0u);
      }
      assert(candidates.Daughters().SourceId(2u * i) == inputTracks.SourceId(tasks[i].firstTrack));
      assert(candidates.Daughters().SourceId(2u * i + 1u)
             == inputTracks.SourceId(tasks[i].secondTrack));
    }

    Pass(energyFit ? "default-v0-field-energy-fit" : "default-v0-full-field-device-reference",
         "all three V0 channels preserve field-aware state, flags and daughter identities");
  }

  void TestDecayPlanDefaultV0Executor(KFParticleGpuRuntime& runtime,
                                      KFParticleGpuBufferManager& buffers)
  {
    KFParticleGpuSteering& steering = runtime.GetSteering();
    KFParticleGpuDecayPlan& plan = steering.GetDecayPlan();
    plan.Clear();
    AddDefaultV0ChannelsWithoutCpuPairGate(plan);

    FillDefaultV0Fixture(buffers);
    const std::vector<KFParticleGpuTwoDaughterChannelResult>& results =
      steering.RunDecayPlan(0u, 4u);
    assert(results.size() == 3u);

    const KFParticleGpuConstCandidatePoolView candidates =
      MakeConstView(buffers.HostCandidates());
    assert(candidates.Size() == 4u);
    assert(candidates.Daughters().Size() == 8u);
    assert(candidates.OverflowFlags() == 0u);
    struct ExpectedChannel { unsigned int id; int pdg; unsigned int offset, daughterOffset, count; };
    const ExpectedChannel expectedChannels[] = {
      {KFGpuChannelK0ShortToPiPlusPiMinus, 310, 0u, 0u, 1u},
      {KFGpuChannelLambdaToProtonPiMinus, 3122, 1u, 2u, 2u},
      {KFGpuChannelAntiLambdaToAntiProtonPiPlus, -3122, 3u, 6u, 1u}};
    for (unsigned int i = 0u; i < 3u; ++i) {
      const auto& e = expectedChannels[i];
      ExpectDefaultV0ChannelRegression(candidates, results[i], plan.TwoDaughterChannel(i),
                                       e.id, e.pdg, e.offset, e.daughterOffset, e.count);
    }
    assert(HasCandidateDaughters(candidates, 3101, 3301));
    assert(HasCandidateDaughters(candidates, 3201, 3301));
    assert(HasCandidateDaughters(candidates, 3401, 3101));
    const KFParticleGpuConstSelectedCandidateIndexView selected =
      MakeConstView(buffers.HostSelectedCandidates());
    const KFParticleGpuSelectedCandidateRange selectedRange =
      steering.LastDecayPlanSelectedCandidates();
    assert(selectedRange.size == selected.Size());
    assert(selectedRange.overflowFlags == selected.OverflowFlags());

    const std::vector<KFParticleGpuDecayPlanEventResult>& events =
      steering.LastDecayPlanEventResults();
    const KFParticleGpuTwoDaughterRoutingMonitorData& monitoring =
      steering.LastTwoDaughterRoutingMonitorData();
    assert(events.size() == 1u);
    assert(events[0].channelCount == 3u);
    assert(events[0].candidates.size == 4u);
    assert(events[0].candidates.daughterSize == 8u);
    assert(events[0].generationRouting.acceptedTasks == 4u);
    assert(events[0].generationRouting.storedTasks == 4u);
    assert(events[0].generationRouting.blockReservations > 0u);
    assert(events[0].overflowFlags == 0u);
    for (const KFParticleGpuTwoDaughterChannelResult& result : results) {
      assert(result.candidates.Empty());
      const unsigned int expectedCount =
        result.channelId == KFGpuChannelLambdaToProtonPiMinus ? 2u : 1u;
      assert(result.constructedCandidates == expectedCount);
      assert(result.constructedDaughters == 2u * expectedCount);
      assert(result.generationCandidates.offset == events[0].candidates.offset);
      assert(result.generationCandidates.size == events[0].candidates.size);
      assert(FindChannelCandidate(
               candidates, result.generationCandidates, result.channelId)
             < candidates.Size());
    }
    assert(monitoring.groupLaunches == 2u);
    assert(monitoring.descriptorCount == 3u);
    assert(monitoring.selectionLaunches == 1u);
    assert(monitoring.acceptedTasks == 4u);
    assert(monitoring.storedTasks == 4u);
    assert(monitoring.blockReservations > 0u);
    assert(monitoring.blockReservations <= monitoring.acceptedTasks);
    assert(monitoring.candidates == 4u);
    assert(monitoring.daughters == 8u);
    assert(monitoring.selectedCandidates == selected.Size());
    assert(monitoring.overflowFlags == 0u);

    Pass("decay-plan-default-v0-executor",
         "default V0 fits, lineage, selection ranges and generation-wide routing counters agree");

    const KFParticleGpuConstVertexSoAView primaryVertices =
      MakeConstView(buffers.HostPrimaryVertices());
    const KFParticleGpuEventDesc& event = buffers.HostEvents()[0];
    std::vector<bool> expected(candidates.Size(), false);
    std::vector<KFParticleGpuV0SelectionResult> expectedDecisions(candidates.Size());
    unsigned int expectedSize = 0u;

    assert(results.size() == plan.NumberOfTwoDaughterChannels());
    for (unsigned int candidateIndex = 0u;
         candidateIndex < candidates.Size();
         ++candidateIndex) {
      const unsigned int channelId =
        candidates.Metadata().ChannelId(candidateIndex);
      const KFParticleGpuTwoDaughterChannel& channel =
        FindTwoDaughterChannel(plan, channelId);
      KFParticleGpuFitState candidate;
      LoadCandidateFit(candidates, candidateIndex, candidate);
      KFParticleGpuV0SelectionResult decision;
      KFParticleGpuSelection::EvaluateV0Selection(candidate,
                                                   candidates.Metadata().Flags(candidateIndex),
                                                   candidateIndex,
                                                   channelId,
                                                   0u,
                                                   primaryVertices,
                                                   event.primaryVertices,
                                                   channel.selection,
                                                   decision);
      expectedDecisions[candidateIndex] = decision;
      expected[candidateIndex] = KFParticleGpuSelection::IsSelected(decision);
      expectedSize += expected[candidateIndex] ? 1u : 0u;
    }

    assert(selected.Size() == expectedSize);
    assert(selectedRange.size == expectedSize);
    assert(selectedRange.overflowFlags == selected.OverflowFlags());
    const KFParticleGpuConstV0SelectionResultView selectionResults =
      MakeConstView(buffers.HostV0SelectionResults());
    const KFParticleGpuSelectedV0View selectedV0s =
      MakeSelectedV0View(candidates, selected, selectionResults);
    assert(selectedV0s.Size() == selected.Size());
    assert(selectedV0s.OverflowFlags() == selected.OverflowFlags());
    assert(selectedV0s.HasSelectionResults());

    for (unsigned int candidateIndex = 0; candidateIndex < candidates.Size(); ++candidateIndex) {
      ExpectSelectionResultClose(selectionResults.Result(candidateIndex),
                                 expectedDecisions[candidateIndex]);
    }

    const std::vector<KFParticleGpuSelectedChannelRange>& selectedChannels =
      steering.LastDecayPlanSelectedChannels();
    assert(selectedChannels.size() == results.size());
    unsigned int rangeEnd = 0u;
    for (std::size_t channelIndex = 0; channelIndex < selectedChannels.size(); ++channelIndex) {
      const KFParticleGpuSelectedChannelRange& channelRange = selectedChannels[channelIndex];
      assert(channelRange.channelId == results[channelIndex].channelId);
      assert(channelRange.eventIndex == 0u);
      assert(channelRange.candidates.offset == rangeEnd);
      rangeEnd = channelRange.candidates.End();
      for (unsigned int selectedIndex = channelRange.candidates.offset;
           selectedIndex < channelRange.candidates.End();
           ++selectedIndex) {
        assert(selected.ChannelId(selectedIndex) == channelRange.channelId);
      }
    }
    assert(rangeEnd == selected.Size());

    std::vector<bool> observed(candidates.Size(), false);
    for (unsigned int selectedIndex = 0; selectedIndex < selected.Size(); ++selectedIndex) {
      const unsigned int candidateIndex = selected.Index(selectedIndex);
      assert(candidateIndex < candidates.Size());
      assert(expected[candidateIndex]);
      assert(!observed[candidateIndex]);
      assert(selectedV0s.CandidateIndex(selectedIndex) == candidateIndex);
      assert(selectedV0s.EventIndex(selectedIndex) == 0u);
      assert(selectedV0s.DaughterCount(selectedIndex) == 2u);
      assert(selectedV0s.Pdg(selectedIndex) == candidates.Metadata().Pdg(candidateIndex));
      assert(selectedV0s.ChannelId(selectedIndex) == expectedDecisions[candidateIndex].channelId);
      ExpectSelectionResultClose(selectedV0s.Selection(selectedIndex), expectedDecisions[candidateIndex]);
      KFParticleGpuFitState selectedFit;
      selectedV0s.LoadFitState(selectedIndex, selectedFit);
      KFParticleGpuFitState rawFit;
      LoadCandidateFit(candidates, candidateIndex, rawFit);
      ExpectFitStateClose(selectedFit, rawFit);
      assert(selectedV0s.DaughterSourceId(selectedIndex, 0u)
             == candidates.Daughters().SourceId(candidates.Metadata().DaughterOffset(candidateIndex)));
      observed[candidateIndex] = true;
    }
    assert(observed == expected);

    plan.Clear();
    Pass("default-v0-selection-regression",
         "device selection diagnostics, channel ranges, and compact membership match host decisions");
  }

  void TestDecayPlanSelectedOutputTruncation(KFParticleGpuRuntime& runtime,
                                             KFParticleGpuBufferManager& buffers)
  {
    assert(buffers.Capacities().selectedCandidates == 1u);
    KFParticleGpuSteering& steering = runtime.GetSteering();
    KFParticleGpuDecayPlan& plan = steering.GetDecayPlan();
    plan.Clear();

    KFParticleGpuTwoDaughterChannel first = MakeK0ShortToPiPlusPiMinusChannel(81u);
    KFParticleGpuTwoDaughterChannel second = first;
    second.channelId = 82u;
    for (KFParticleGpuTwoDaughterChannel* channel : {&first, &second}) {
      channel->secondaryMassSigmaCut = -1.f;
      channel->maxSecondaryTopoChi2PerNdf = -1.f;
      channel->minSecondaryLdL = -1.f;
      channel->selection.expectedMass = 1.f;
      channel->selection.expectedMassSigma = -1.f;
      channel->selection.massSigmaCut = -1.f;
      channel->selection.maxGeometricChi2PerNdf = -1.f;
      channel->selection.maxPrimaryVertexDistance = -1.f;
      channel->selection.minSecondaryLdL = -1.f;
      channel->selection.maxPrimaryTopologyChi2PerNdf = -1.f;
      channel->selection.maxSecondaryTopologyChi2PerNdf = -1.f;
      channel->selection.requirePrimaryVertex = 0u;
    }
    plan.AddTwoDaughterChannel(first);
    plan.AddTwoDaughterChannel(second);
    FillInput(buffers, buffers.Capacities());
    KFParticleGpuInputTrackSoAView tracks = buffers.HostInputTracks();
    StoreSyntheticTrack(tracks, 0u, -0.2f, 0.1f, 0.f, 0.8f, 0.1f, 1.f, 211, 1, 8101);
    StoreSyntheticTrack(tracks, 1u, 0.1f, 0.5f, -0.4f, -0.3f, 0.7f, 1.1f, -211, -1, 8201);
    KFParticleGpuEventDesc* events = buffers.HostEvents();
    events[0] = KFParticleGpuEventDesc();
    events[0].eventId = 181;
    events[0].TrackSet(SecondaryPositiveFirst).tracks = KFParticleGpuRange(0u, 1u);
    events[0].TrackSet(SecondaryPositiveFirst).Species(Pion) = KFParticleGpuRange(0u, 1u);
    events[0].TrackSet(SecondaryNegativeFirst).tracks = KFParticleGpuRange(1u, 1u);
    events[0].TrackSet(SecondaryNegativeFirst).Species(Pion) = KFParticleGpuRange(1u, 1u);

    const std::vector<KFParticleGpuTwoDaughterChannelResult>& results =
      steering.RunDecayPlan(0u, 4u);
    assert(results.size() == 2u);
    assert(results[0].constructedCandidates == 1u);
    assert(results[1].constructedCandidates == 1u);
    assert(results[0].generationCandidates.size == 2u);
    assert(results[1].generationCandidates.size == 2u);

    const KFParticleGpuConstSelectedCandidateIndexView selected =
      MakeConstView(buffers.HostSelectedCandidates());
    const KFParticleGpuConstV0SelectionResultView selectionResults =
      MakeConstView(buffers.HostV0SelectionResults());
    const KFParticleGpuConstCandidatePoolView candidates =
      MakeConstView(buffers.HostCandidates());
    assert(selected.Size() == 1u);
    assert(selected.OverflowFlags() == KFGpuSelectedCandidateCapacityExceeded);
    unsigned int firstCandidate = candidates.Size();
    unsigned int secondCandidate = candidates.Size();
    for (unsigned int candidate = 0u; candidate < candidates.Size(); ++candidate) {
      if (candidates.Metadata().ChannelId(candidate) == 81u) {
        firstCandidate = candidate;
      }
      if (candidates.Metadata().ChannelId(candidate) == 82u) {
        secondCandidate = candidate;
      }
    }
    assert(firstCandidate < candidates.Size());
    assert(secondCandidate < candidates.Size());
    assert(selectionResults.Result(firstCandidate).channelId == 81u);
    assert(selectionResults.Result(secondCandidate).channelId == 82u);
    assert(KFParticleGpuSelection::IsSelected(
      selectionResults.Result(firstCandidate)));
    assert(KFParticleGpuSelection::HasRejection(
      selectionResults.Result(secondCandidate),
      KFGpuV0SelectionRejectOutputOverflow));

    const std::vector<KFParticleGpuSelectedChannelRange>& ranges =
      steering.LastDecayPlanSelectedChannels();
    assert(ranges.size() == 2u);
    assert(ranges[0].channelId == 81u);
    assert(ranges[0].candidates.offset == 0u);
    assert(ranges[0].candidates.size == 1u);
    assert(ranges[1].channelId == 82u);
    assert(ranges[1].candidates.offset == 1u);
    assert(ranges[1].candidates.size == 0u);

    KFParticleGpuBufferCapacities requested = buffers.Capacities();
    requested.selectedCandidates = requested.candidates;
    buffers.EnsureCapacity(requested);
    plan.Clear();
    Pass("decay-plan-selected-output-truncation",
         "RunDecayPlan records compact-pool overflow and preserves per-channel ranges");
  }

  void TestDecayPlanSelectionMultiPvBoundary(KFParticleGpuRuntime& runtime,
                                             KFParticleGpuBufferManager& buffers)
  {
    KFParticleGpuSteering& steering = runtime.GetSteering();
    KFParticleGpuDecayPlan& plan = steering.GetDecayPlan();
    plan.Clear();
    AddDefaultV0ChannelsWithoutCpuPairGate(plan);
    FillDefaultV0Fixture(buffers);
    buffers.SetInputSizes(4u, 2u, 1u);

    KFParticleGpuVertexState tiedVertex;
    tiedVertex.X() = 0.f;
    tiedVertex.Y() = 0.f;
    tiedVertex.Z() = 0.f;
    tiedVertex.Covariance(0) = 1.f;
    tiedVertex.Covariance(2) = 1.f;
    tiedVertex.Covariance(5) = 1.f;
    KFParticleGpuVertexSoAView vertices = buffers.HostPrimaryVertices();
    StoreVertexState(tiedVertex, vertices, 0u);
    StoreVertexState(tiedVertex, vertices, 1u);
    buffers.HostEvents()[0].primaryVertices = KFParticleGpuRange(0u, 2u);

    const std::vector<KFParticleGpuTwoDaughterChannelResult>& initialResults =
      steering.RunDecayPlan(0u, 4u);
    const KFParticleGpuConstV0SelectionResultView initialSelections =
      MakeConstView(buffers.HostV0SelectionResults());
    assert(initialResults.size() == 3u);
    const KFParticleGpuConstCandidatePoolView initialCandidates =
      MakeConstView(buffers.HostCandidates());
    for (unsigned int candidateIndex = 0u;
         candidateIndex < initialCandidates.Size();
         ++candidateIndex) {
      const KFParticleGpuV0SelectionResult& selection =
        initialSelections.Result(candidateIndex);
      if ((selection.topologyStatus & KFGpuV0LineTopologyValid) != 0u) {
        assert(selection.bestPrimaryVertexIndex == 0);
      }
      else {
        assert(selection.bestPrimaryVertexIndex < 0);
      }
    }

    struct BoundaryReference
    {
      unsigned int channelId = 0u;
      int firstSourceId = 0;
      int secondSourceId = 0;
    };
    std::vector<KFParticleGpuTwoDaughterChannel> boundaryChannels;
    std::vector<BoundaryReference> boundaryReferences;
    for (std::size_t channelIndex = 0; channelIndex < initialResults.size(); ++channelIndex) {
      const KFParticleGpuTwoDaughterChannelResult& result = initialResults[channelIndex];
      const unsigned int expectedCount =
        result.channelId == KFGpuChannelLambdaToProtonPiMinus ? 2u : 1u;
      assert(result.constructedCandidates == expectedCount);
      const unsigned int candidateIndex =
        FindChannelCandidate(initialCandidates,
                             result.generationCandidates,
                             result.channelId);
      assert(candidateIndex < initialCandidates.Size());
      const float boundary =
        initialSelections.Result(candidateIndex).observables.nearestPrimaryVertexDistance;
      assert(initialCandidates.Metadata().DaughterCount(candidateIndex) == 2u);
      const unsigned int daughterOffset =
        initialCandidates.Metadata().DaughterOffset(candidateIndex);
      boundaryReferences.push_back({
        result.channelId,
        initialCandidates.Daughters().SourceId(daughterOffset),
        initialCandidates.Daughters().SourceId(daughterOffset + 1u)});
      KFParticleGpuTwoDaughterChannel channel = plan.TwoDaughterChannel(channelIndex);
      channel.selection.maxPrimaryVertexDistance = boundary;
      boundaryChannels.push_back(channel);
    }

    plan.Clear();
    for (const KFParticleGpuTwoDaughterChannel& channel : boundaryChannels) {
      plan.AddTwoDaughterChannel(channel);
    }

    const std::vector<KFParticleGpuTwoDaughterChannelResult>& boundaryResults =
      steering.RunDecayPlan(0u, 4u);
    const KFParticleGpuConstCandidatePoolView candidates = MakeConstView(buffers.HostCandidates());
    const KFParticleGpuConstVertexSoAView primaryVertices =
      MakeConstView(buffers.HostPrimaryVertices());
    const KFParticleGpuConstV0SelectionResultView deviceSelections =
      MakeConstView(buffers.HostV0SelectionResults());
    const KFParticleGpuEventDesc& event = buffers.HostEvents()[0];

    assert(boundaryResults.size() == plan.NumberOfTwoDaughterChannels());
    std::vector<unsigned int> matchedBoundaryReferences(
      boundaryReferences.size(), 0u);
    for (unsigned int candidateIndex = 0u;
         candidateIndex < candidates.Size();
         ++candidateIndex) {
      const unsigned int channelId =
        candidates.Metadata().ChannelId(candidateIndex);
      const KFParticleGpuTwoDaughterChannel& channel =
        FindTwoDaughterChannel(plan, channelId);
      KFParticleGpuFitState candidate;
      LoadCandidateFit(candidates, candidateIndex, candidate);
      KFParticleGpuV0SelectionResult expected;
      KFParticleGpuSelection::EvaluateV0Selection(candidate,
                                                   candidates.Metadata().Flags(candidateIndex),
                                                   candidateIndex,
                                                   channelId,
                                                   0u,
                                                   primaryVertices,
                                                   event.primaryVertices,
                                                   channel.selection,
                                                   expected);
      const KFParticleGpuV0SelectionResult& actual =
        deviceSelections.Result(candidateIndex);
      if (VerboseDiagnosticTraceEnabled()
          || !SelectionResultClose(actual, expected, 1.e-5f, 2.e-5f)) {
        DumpSelectionResultComparison(
          "multi-pv-boundary-absolute", actual, expected, 1.e-5f, 2.e-5f);
        std::cerr << std::setprecision(9)
                  << "TRACE selection-context candidate=" << candidateIndex
                  << " flags=" << candidates.Metadata().Flags(candidateIndex)
                  << " topology-mode=" << channel.selection.topologyMode
                  << " max-distance=" << channel.selection.maxPrimaryVertexDistance
                  << " min-secondary-ldl=" << channel.selection.minSecondaryLdL
                  << " max-primary-chi2="
                  << channel.selection.maxPrimaryTopologyChi2PerNdf
                  << " max-secondary-chi2="
                  << channel.selection.maxSecondaryTopologyChi2PerNdf << '\n';
        for (int parameter = 0;
             parameter < KFParticleGpuFitState::NumberOfParameters;
             ++parameter) {
          std::cerr << "TRACE selection-candidate parameter=" << parameter
                    << " value=" << candidate.Parameter(parameter) << '\n';
        }
        for (int covariance = 0;
             covariance < KFParticleGpuFitState::NumberOfCovarianceElements;
             ++covariance) {
          std::cerr << "TRACE selection-candidate covariance=" << covariance
                    << " value=" << candidate.Covariance(covariance) << '\n';
        }
        std::cerr << "TRACE selection-candidate chi2=" << candidate.Chi2()
                  << " ndf=" << candidate.NDF() << " q=" << candidate.Q() << '\n';
        for (unsigned int localVertex = 0u;
             localVertex < event.primaryVertices.size;
             ++localVertex) {
          const unsigned int vertexIndex = event.primaryVertices.offset + localVertex;
          KFParticleGpuVertexState vertex;
          LoadVertexState(primaryVertices, vertexIndex, vertex);
          std::cerr << "TRACE selection-vertex index=" << vertexIndex
                    << " position=" << vertex.X() << ',' << vertex.Y() << ',' << vertex.Z();
          for (int covariance = 0; covariance < 6; ++covariance) {
            std::cerr << " covariance-" << covariance << '=' << vertex.Covariance(covariance);
          }
          std::cerr << '\n';
          KFParticleGpuV0LineTopologyResult topology;
          const bool topologyBuilt =
            KFParticleGpuSelection::BuildV0LineTopology(candidate, vertex, topology);
          std::cerr << "TRACE selection-host-topology vertex=" << vertexIndex
                    << " built=" << (topologyBuilt ? 1 : 0)
                    << " status=" << topology.status
                    << " distance=" << topology.lineDistance
                    << " distance-error=" << topology.lineDistanceError
                    << " line-ldl=" << topology.lineLdL
                    << " path=" << topology.pathToPrimaryVertex
                    << " decay-length=" << topology.decayLength
                    << " decay-length-error=" << topology.decayLengthError
                    << " decay-ldl=" << topology.decayLdL
                    << " pointing=" << topology.pointingCosine
                    << " chi2-ndf=" << topology.lineChi2PerNdf << '\n';
        }
      }
      ExpectSelectionResultClose(actual, expected);
      if ((deviceSelections.Result(candidateIndex).topologyStatus
           & KFGpuV0LineTopologyValid) != 0u) {
        assert(deviceSelections.Result(candidateIndex).bestPrimaryVertexIndex == 0);
      }
      else {
        assert(deviceSelections.Result(candidateIndex).bestPrimaryVertexIndex < 0);
      }
      for (std::size_t referenceIndex = 0u;
           referenceIndex < boundaryReferences.size(); ++referenceIndex) {
        const BoundaryReference& reference = boundaryReferences[referenceIndex];
        if (reference.channelId != channelId
            || candidates.Metadata().DaughterCount(candidateIndex) != 2u) {
          continue;
        }
        const unsigned int daughterOffset =
          candidates.Metadata().DaughterOffset(candidateIndex);
        if (candidates.Daughters().SourceId(daughterOffset)
              != reference.firstSourceId
            || candidates.Daughters().SourceId(daughterOffset + 1u)
                 != reference.secondSourceId) {
          continue;
        }
        ++matchedBoundaryReferences[referenceIndex];
        assert(KFParticleGpuSelection::HasRejection(
          actual, KFGpuV0SelectionRejectDistance));
      }
    }
    for (const unsigned int matches : matchedBoundaryReferences) {
      assert(matches == 1u);
    }

    plan.Clear();
    Pass("decay-plan-selection-multi-pv-boundary",
         "RunDecayPlan preserves deterministic multi-PV ties and exact distance-cut rejection");
  }

  void TestDecayPlanMultiChannelExecutor(KFParticleGpuRuntime& runtime,
                                         KFParticleGpuBufferManager& buffers)
  {
    KFParticleGpuSteering& steering = runtime.GetSteering();
    KFParticleGpuDecayPlan& plan = steering.GetDecayPlan();
    plan.Clear();
    plan.AddTwoDaughterChannel(MakePionPairChannel(51u));
    plan.AddTwoDaughterChannel(MakeKaonPairChannel(52u));

    FillDecayPlanMultiChannelFixture(buffers);
    const std::vector<KFParticleGpuTwoDaughterChannelResult>& results =
      steering.RunDecayPlan(0u, 2u);
    assert(results.size() == 2u);

    assert(results[0].channelId == 51u);
    assert(results[0].motherPdg == 310);
    assert(results[0].totalPairs == 4u);
    assert(results[0].acceptedTasks == 2u);
    assert(results[0].storedTasks == 2u);
    assert(!results[0].Empty());
    assert(!results[0].HasAnyOverflow());
    assert(results[0].constructedCandidates == 2u);
    assert(results[0].constructedDaughters == 4u);
    assert(results[0].generationCandidates.offset == 0u);
    assert(results[0].generationCandidates.size == 4u);

    assert(results[1].channelId == 52u);
    assert(results[1].motherPdg == 333);
    assert(results[1].totalPairs == 4u);
    assert(results[1].acceptedTasks == 2u);
    assert(results[1].storedTasks == 2u);
    assert(!results[1].Empty());
    assert(!results[1].HasAnyOverflow());
    assert(results[1].constructedCandidates == 2u);
    assert(results[1].constructedDaughters == 4u);
    assert(results[1].generationCandidates.offset == 0u);
    assert(results[1].generationCandidates.size == 4u);

    const KFParticleGpuConstCandidatePoolView candidates =
      MakeConstView(buffers.HostCandidates());
    assert(candidates.Size() == 4u);
    assert(candidates.Daughters().Size() == 8u);
    assert(candidates.OverflowFlags() == 0u);
    assert(HasCandidateDaughters(candidates, 2101, 2301));
    assert(HasCandidateDaughters(candidates, 2102, 2301));
    assert(HasCandidateDaughters(candidates, 2201, 2401));
    assert(HasCandidateDaughters(candidates, 2202, 2401));
    ExpectChannelCandidatePdgs(candidates, results[0], 310);
    ExpectChannelCandidatePdgs(candidates, results[1], 333);
    ExpectChannelDaughterRange(candidates, results[0]);
    ExpectChannelDaughterRange(candidates, results[1]);

    plan.Clear();
    Pass("decay-plan-multi-channel-executor",
         "decay plan executor appends multiple channels with non-overlapping output ranges");
  }

