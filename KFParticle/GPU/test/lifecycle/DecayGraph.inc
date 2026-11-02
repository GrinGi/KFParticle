  void TestDecayPlanBatchExecutor(KFParticleGpuRuntime& runtime,
                                  KFParticleGpuBufferManager& buffers)
  {
    KFParticleGpuSteering& steering = runtime.GetSteering();
    KFParticleGpuDecayPlan& plan = steering.GetDecayPlan();
    plan.Clear();
    plan.AddTwoDaughterChannel(MakePionPairChannel(91u));
    plan.AddTwoDaughterChannel(MakeKaonPairChannel(92u));

    FillDecayPlanMultiEventFixture(buffers);
    KFParticleGpuBufferCapacities requested = buffers.Capacities();
    requested.candidates = requested.candidates > 10u ? requested.candidates : 10u;
    requested.daughterIds = requested.daughterIds > 18u ? requested.daughterIds : 18u;
    requested.selectedCandidates = requested.selectedCandidates > 10u ? requested.selectedCandidates : 10u;
    buffers.EnsureCapacity(requested);

    const std::vector<KFParticleGpuTwoDaughterChannelResult> serialFirst =
      steering.RunDecayPlan(0u, 4u);
    const std::vector<KFParticleGpuTwoDaughterChannelResult> serialSecond =
      steering.RunDecayPlan(1u, 4u);
    const unsigned int serialFirstCandidates =
      serialFirst[0u].generationCandidates.size;
    const unsigned int serialFirstDaughters =
      serialFirst[0u].generationCandidates.daughterSize;
    const unsigned int serialSecondCandidates =
      serialSecond[0u].generationCandidates.size;
    const unsigned int serialSecondDaughters =
      serialSecond[0u].generationCandidates.daughterSize;
    assert(!steering.LastPerformanceSnapshot().enabled);

    FillDecayPlanMultiEventFixture(buffers);
    steering.SetPerformanceMonitoringEnabled(true);
    assert(steering.AdministrativeResetStrategy()
           == KFGpuAdministrativeResetDeviceBatch);
    assert(steering.TwoDaughterTransactionStrategy()
           == KFGpuTwoDaughterTransactionFused);
    assert(steering.TwoDaughterSelectionStrategy()
           == KFGpuTwoDaughterSelectionDirect);
    const unsigned int firstAccepted = serialFirst[0u].acceptedTasks + serialFirst[1u].acceptedTasks;
    const unsigned int secondAccepted = serialSecond[0u].acceptedTasks + serialSecond[1u].acceptedTasks;
    requested = buffers.Capacities();
    requested.transactions = 1u;
    requested.domains = 2u;
    requested.workRanges = 2u;
    requested.domainStatuses = 2u;
    buffers.EnsureCapacity(requested);
    buffers.SetTransactionSizes(1u, 2u, 2u, 2u);
    KFParticleGpuTransactionView transaction = buffers.HostTransaction();
    transaction.Transaction()[0] = KFParticleGpuTransactionDesc{};
    transaction.Transaction()[0].transactionId = 7901u;
    transaction.Transaction()[0].decayPlanRevision = plan.Revision();
    transaction.Transaction()[0].domains = {0u, 2u};
    transaction.Transaction()[0].workRanges = {0u, 2u};
    transaction.Domains()[0] = KFParticleGpuDomainDesc{};
    transaction.Domains()[0].sourceId = buffers.HostEvents()[0].eventId;
    transaction.Domains()[0].inputDescriptorIndex = 0u;
    transaction.Domains()[0].sourceTracks = {0u, 2u};
    transaction.Domains()[0].primaryVertices = buffers.HostEvents()[0].primaryVertices;
    transaction.Domains()[0].taskReservation = {0u, firstAccepted};
    transaction.Domains()[0].candidateReservation = {0u, serialFirstCandidates};
    transaction.Domains()[0].daughterReservation = {0u, serialFirstDaughters};
    transaction.Domains()[0].selectedReservation = {0u, 5u};
    transaction.Domains()[0].statusIndex = 0u;
    transaction.Domains()[1] = KFParticleGpuDomainDesc{};
    transaction.Domains()[1].sourceId = buffers.HostEvents()[1].eventId;
    transaction.Domains()[1].inputDescriptorIndex = 1u;
    transaction.Domains()[1].sourceTracks = {2u, 8u};
    transaction.Domains()[1].primaryVertices = buffers.HostEvents()[1].primaryVertices;
    transaction.Domains()[1].taskReservation = {firstAccepted, secondAccepted};
    transaction.Domains()[1].candidateReservation = {serialFirstCandidates, serialSecondCandidates};
    transaction.Domains()[1].daughterReservation = {serialFirstDaughters, serialSecondDaughters};
    transaction.Domains()[1].selectedReservation = {5u, 5u};
    transaction.Domains()[1].statusIndex = 1u;
    const std::vector<KFParticleGpuTwoDaughterChannelResult>& batch =
      steering.RunDecayPlanBatch(0u, 2u, 4u);
    const std::vector<KFParticleGpuDecayPlanEventResult>& events =
      steering.LastDecayPlanEventResults();
    const KFParticleGpuTransactionView compiledTransaction =
      buffers.HostTransaction();

    assert(batch.size() == 4u);
    assert(events.size() == 2u);
    assert(compiledTransaction.Transaction()->workRanges.size == 2u);
    for (unsigned int workIndex = 0u;
         workIndex < compiledTransaction.Transaction()->workRanges.size;
         ++workIndex) {
      const KFParticleGpuWorkRange& work =
        compiledTransaction.WorkRanges()[workIndex];
      assert(work.family == KFParticleGpuTwoDaughterWork);
      assert(work.domainIndex < 2u);
      assert(work.sourceCandidates.size == 0u);
      assert(work.outputReservation.size > 0u);
    }
    assert(events[0].eventIndex == 0u);
    assert(events[0].channelOffset == 0u);
    assert(events[0].channelCount == 2u);
    assert(events[0].candidates.offset == 0u);
    assert(events[0].candidates.size == serialFirstCandidates);
    assert(events[0].candidates.daughterOffset == 0u);
    assert(events[0].candidates.daughterSize == serialFirstDaughters);
    assert(events[1].eventIndex == 1u);
    assert(events[1].channelOffset == 2u);
    assert(events[1].channelCount == 2u);
    assert(events[1].candidates.offset == serialFirstCandidates);
    assert(events[1].candidates.size == serialSecondCandidates);
    assert(events[1].candidates.daughterOffset == serialFirstDaughters);
    assert(events[1].candidates.daughterSize == serialSecondDaughters);

    for (unsigned int channel = 0u; channel < 2u; ++channel) {
      assert(batch[channel].eventIndex == 0u);
      assert(batch[channel + 2u].eventIndex == 1u);
      assert(batch[channel].totalPairs == serialFirst[channel].totalPairs);
      assert(batch[channel].acceptedTasks == serialFirst[channel].acceptedTasks);
      assert(batch[channel].constructedCandidates
             == serialFirst[channel].constructedCandidates);
      assert(batch[channel + 2u].totalPairs == serialSecond[channel].totalPairs);
      assert(batch[channel + 2u].acceptedTasks == serialSecond[channel].acceptedTasks);
      assert(batch[channel + 2u].constructedCandidates
             == serialSecond[channel].constructedCandidates);
    }

    const KFParticleGpuConstCandidatePoolView candidates = MakeConstView(buffers.HostCandidates());
    assert(candidates.Size() == serialFirstCandidates + serialSecondCandidates);
    assert(candidates.Daughters().Size() == serialFirstDaughters + serialSecondDaughters);
    for (unsigned int candidate = 0u; candidate < serialFirstCandidates; ++candidate) {
      assert(candidates.Metadata().EventIndex(candidate) == 0u);
    }
    for (unsigned int candidate = serialFirstCandidates; candidate < candidates.Size(); ++candidate) {
      assert(candidates.Metadata().EventIndex(candidate) == 1u);
    }
    assert(HasCandidateDaughters(candidates, 2501, 2601));
    assert(HasCandidateDaughters(candidates, 2701, 2901));
    assert(HasCandidateDaughters(candidates, 2801, 3001));
    assert(steering.LastDecayPlanSelectedChannels().size() == 4u);

    const KFParticleGpuPerformanceSnapshot& performance =
      steering.LastPerformanceSnapshot();
    assert(performance.enabled);
    assert(performance.events == 2u);
    assert(performance.tracks == buffers.TrackSize());
    assert(performance.vertices == buffers.VertexSize());
    assert(performance.descriptorGroups > 0u);
    assert(performance.visitedCombinations > 0u);
    assert(performance.storedTasks == candidates.Size());
    assert(performance.rawCandidates == candidates.Size());
    assert(performance.daughters == candidates.Daughters().Size());
    assert(performance.kernelLaunches == 5u);
    assert(performance.queueWaits >= 4u);
    assert(performance.queueWaits <= 6u);
    assert(performance.hostToDeviceBytes > 0u);
    assert(performance.deviceToHostBytes > 0u);
    assert(performance.allocatedBytesHighWater >= performance.hostToDeviceBytes);
    assert(performance.twoDaughterRealPairs == 17u);
    assert(performance.twoDaughterPairTiles == 2u);
    assert(performance.twoDaughterScheduledLanes == 128u);
    assert(performance.twoDaughterTailLanes == 111u);
    assert(performance.candidatePoolOccupancy > 0.);
    const unsigned int fusedSelectedSize =
      steering.LastDecayPlanSelectedCandidates().size;

    FillDecayPlanMultiEventFixture(buffers);
    const std::vector<KFParticleGpuTwoDaughterChannelResult> deviceResetBatch =
      steering.RunDecayPlanBatch(0u, 2u, 4u);
    const KFParticleGpuPerformanceSnapshot deviceResetPerformance =
      steering.LastPerformanceSnapshot();
    assert(deviceResetBatch.size() == 4u);
    assert(steering.LastDecayPlanSelectedCandidates().size == fusedSelectedSize);
    assert(deviceResetPerformance.kernelLaunches == 5u);
    assert(deviceResetPerformance.queueWaits == 4u);
    assert(deviceResetPerformance.monitoringReadbacks == 1u);
    assert(deviceResetPerformance.monitoringReadbackBuffers == 5u);

    steering.SetAdministrativeResetStrategy(
      KFGpuAdministrativeResetHostStaged);
    FillDecayPlanMultiEventFixture(buffers);
    const std::vector<KFParticleGpuTwoDaughterChannelResult> hostResetBatch =
      steering.RunDecayPlanBatch(0u, 2u, 4u);
    const KFParticleGpuPerformanceSnapshot hostResetPerformance =
      steering.LastPerformanceSnapshot();
    assert(hostResetBatch.size() == deviceResetBatch.size());
    for (unsigned int channel = 0u; channel < hostResetBatch.size(); ++channel) {
      assert(hostResetBatch[channel].totalPairs
             == deviceResetBatch[channel].totalPairs);
      assert(hostResetBatch[channel].acceptedTasks
             == deviceResetBatch[channel].acceptedTasks);
      assert(hostResetBatch[channel].constructedCandidates
             == deviceResetBatch[channel].constructedCandidates);
    }
    assert(steering.LastDecayPlanSelectedCandidates().size == fusedSelectedSize);
    assert(hostResetPerformance.kernelLaunches
           == deviceResetPerformance.kernelLaunches + 1u);
    assert(hostResetPerformance.queueWaits
           == deviceResetPerformance.queueWaits + 3u);
    steering.SetAdministrativeResetStrategy(
      KFGpuAdministrativeResetDeviceBatch);

    steering.SetTwoDaughterTransactionStrategy(KFGpuTwoDaughterTransactionStaged);
    steering.SetTwoDaughterSelectionStrategy(KFGpuTwoDaughterSelectionStaged);
    assert(steering.TwoDaughterTransactionStrategy()
           == KFGpuTwoDaughterTransactionStaged);
    assert(steering.TwoDaughterSelectionStrategy()
           == KFGpuTwoDaughterSelectionStaged);
    FillDecayPlanMultiEventFixture(buffers);
    const std::vector<KFParticleGpuTwoDaughterChannelResult>& stagedBatch =
      steering.RunDecayPlanBatch(0u, 2u, 4u);
    assert(stagedBatch.size() == 4u);
    for (unsigned int channel = 0u; channel < stagedBatch.size(); ++channel) {
      const std::vector<KFParticleGpuTwoDaughterChannelResult>& serial =
        channel < 2u ? serialFirst : serialSecond;
      const unsigned int serialChannel = channel % 2u;
      assert(stagedBatch[channel].eventIndex == channel / 2u);
      assert(stagedBatch[channel].totalPairs == serial[serialChannel].totalPairs);
      assert(stagedBatch[channel].acceptedTasks
             == serial[serialChannel].acceptedTasks);
      assert(stagedBatch[channel].constructedCandidates
             == serial[serialChannel].constructedCandidates);
    }
    const KFParticleGpuConstCandidatePoolView stagedCandidates =
      MakeConstView(buffers.HostCandidates());
    assert(stagedCandidates.Size() == serialFirstCandidates + serialSecondCandidates);
    assert(stagedCandidates.Daughters().Size()
           == serialFirstDaughters + serialSecondDaughters);
    assert(HasCandidateDaughters(stagedCandidates, 2501, 2601));
    assert(HasCandidateDaughters(stagedCandidates, 2701, 2901));
    assert(HasCandidateDaughters(stagedCandidates, 2801, 3001));
    assert(steering.LastDecayPlanSelectedCandidates().size == fusedSelectedSize);
    assert(steering.LastPerformanceSnapshot().kernelLaunches == 7u);
    steering.SetTwoDaughterTransactionStrategy(KFGpuTwoDaughterTransactionFused);
    steering.SetTwoDaughterSelectionStrategy(KFGpuTwoDaughterSelectionDirect);
    steering.SetPerformanceMonitoringEnabled(false);
    assert(!steering.LastPerformanceSnapshot().enabled);

    plan.Clear();
    Pass("decay-plan-batch-executor",
         "direct fused and fully staged two-daughter transactions preserve event plans, selection, and serial-equivalent lineage while fusion removes two launches");
    Pass("performance-monitoring-contract",
         "opt-in work, launch, wait, transfer, growth, and high-water counters show device-batch initialization removes three host queue waits and uses one compact status readback without changing output");
  }

  void TestDecayPlanBatchSelectedOutput(KFParticleGpuRuntime& runtime,
                                        KFParticleGpuBufferManager& buffers)
  {
    KFParticleGpuSteering& steering = runtime.GetSteering();
    KFParticleGpuDecayPlan& plan = steering.GetDecayPlan();
    plan.Clear();
    KFParticleGpuTwoDaughterChannel channel = MakeK0ShortToPiPlusPiMinusChannel(93u);
    // Isolate compact-output partitioning from physics cuts: the default
    // descriptor and daughter identity remain intact, while both events must
    // contribute one accepted compact index.
    channel.secondaryMassSigmaCut = -1.f;
    channel.maxSecondaryTopoChi2PerNdf = -1.f;
    channel.minSecondaryLdL = -1.f;
    channel.selection.expectedMass = 1.f;
    channel.selection.expectedMassSigma = -1.f;
    channel.selection.massSigmaCut = -1.f;
    channel.selection.maxGeometricChi2PerNdf = -1.f;
    channel.selection.maxPrimaryVertexDistance = -1.f;
    channel.selection.minSecondaryLdL = -1.f;
    channel.selection.maxPrimaryTopologyChi2PerNdf = -1.f;
    channel.selection.maxSecondaryTopologyChi2PerNdf = -1.f;
    channel.selection.requirePrimaryVertex = 0u;
    plan.AddTwoDaughterChannel(channel);

    FillBatchSelectionFixture(buffers);
    const std::vector<KFParticleGpuTwoDaughterChannelResult> serialFirst =
      steering.RunDecayPlan(0u, 1u);
    const unsigned int serialFirstSelected = steering.LastDecayPlanSelectedCandidates().size;
    FillBatchSelectionFixture(buffers);
    const std::vector<KFParticleGpuTwoDaughterChannelResult> serialSecond =
      steering.RunDecayPlan(1u, 1u);
    const unsigned int serialSecondSelected = steering.LastDecayPlanSelectedCandidates().size;

    FillBatchSelectionFixture(buffers);
    const std::vector<KFParticleGpuTwoDaughterChannelResult>& batch =
      steering.RunDecayPlanBatch(0u, 2u, 1u);
    const std::vector<KFParticleGpuDecayPlanEventResult>& events =
      steering.LastDecayPlanEventResults();
    const std::vector<KFParticleGpuSelectedChannelRange>& ranges =
      steering.LastDecayPlanSelectedChannels();
    const KFParticleGpuConstSelectedCandidateIndexView selected =
      MakeConstView(buffers.HostSelectedCandidates());
    const KFParticleGpuConstCandidatePoolView candidates = MakeConstView(buffers.HostCandidates());

    assert(serialFirst.size() == 1u && serialSecond.size() == 1u);
    assert(serialFirstSelected == 1u && serialSecondSelected == 1u);
    assert(batch.size() == 2u && events.size() == 2u && ranges.size() == 2u);
    assert(selected.Size() == 2u && selected.OverflowFlags() == 0u);
    for (unsigned int event = 0u; event < 2u; ++event) {
      assert(batch[event].eventIndex == event);
      assert(batch[event].constructedCandidates == 1u);
      assert(events[event].selectedCandidates.size == 1u);
      assert(ranges[event].eventIndex == event);
      assert(ranges[event].channelId == 93u);
      assert(ranges[event].candidates.size == 1u);
      const unsigned int selectedIndex = ranges[event].candidates.offset;
      const unsigned int candidateIndex = selected.Index(selectedIndex);
      assert(candidates.Metadata().EventIndex(candidateIndex) == event);
      assert(candidates.Metadata().ChannelId(candidateIndex) == 93u);
      assert(candidates.Daughters().SourceId(candidates.Metadata().DaughterOffset(candidateIndex))
             == (event == 0u ? 4101 : 4301));
    }

    plan.Clear();
    Pass("decay-plan-batch-selected-output",
         "two events retain non-empty compact selection ranges, event identity, and source lineage");
  }

  void TestDecayPlanHighOccupancyDomainIsolation(KFParticleGpuRuntime& runtime,
                                                 KFParticleGpuBufferManager& buffers)
  {
    constexpr unsigned int eventCount = 4u;
    constexpr unsigned int tracksPerCharge = 32u;
    constexpr unsigned int tracksPerEvent = 2u * tracksPerCharge;
    constexpr unsigned int candidatesPerEvent = tracksPerCharge * tracksPerCharge;
    constexpr unsigned int firstSelectedCapacity = 64u;
    constexpr unsigned int candidateCapacity = eventCount * candidatesPerEvent;
    constexpr unsigned int selectedCapacity = firstSelectedCapacity
      + (eventCount - 1u) * candidatesPerEvent;

    KFParticleGpuBufferCapacities requested = buffers.Capacities();
    requested.tracks = std::max(requested.tracks, eventCount * tracksPerEvent);
    requested.events = std::max(requested.events, eventCount);
    requested.candidates = std::max(requested.candidates, candidateCapacity);
    requested.daughterIds = std::max(requested.daughterIds, 2u * candidateCapacity);
    requested.selectedCandidates = std::max(requested.selectedCandidates, selectedCapacity);
    requested.twoDaughterTasks = std::max(requested.twoDaughterTasks, candidateCapacity);
    requested.transactions = std::max(requested.transactions, 1u);
    requested.domains = std::max(requested.domains, eventCount + 1u);
    requested.workRanges = std::max(requested.workRanges, eventCount);
    requested.domainStatuses = std::max(requested.domainStatuses, eventCount);
    buffers.EnsureCapacity(requested);
    buffers.SetInputSizes(eventCount * tracksPerEvent, 0u, eventCount);

    KFParticleGpuInputTrackSoAView tracks = buffers.HostInputTracks();
    KFParticleGpuEventDesc* events = buffers.HostEvents();
    for (unsigned int event = 0u; event < eventCount; ++event) {
      const unsigned int trackOffset = event * tracksPerEvent;
      for (unsigned int local = 0u; local < tracksPerCharge; ++local) {
        const float shift = 0.0005f * static_cast<float>(event * tracksPerCharge + local);
        StoreSyntheticTrack(tracks, trackOffset + local,
                            -0.2f + shift, 0.1f, 0.f,
                            0.8f, 0.1f + shift, 1.f,
                            211, 1, 20000 + static_cast<int>(trackOffset + local));
        StoreSyntheticTrack(tracks, trackOffset + tracksPerCharge + local,
                            0.1f - shift, 0.5f, -0.4f,
                            -0.3f, 0.7f - shift, 1.1f,
                            -211, -1,
                            20000 + static_cast<int>(trackOffset + tracksPerCharge + local));
      }
      if (tracks.FieldCoefficientsData()) {
        const KFParticleGpuFieldRegion zeroField;
        for (unsigned int local = 0u; local < tracksPerEvent; ++local) {
          StoreFieldRegion(zeroField, tracks, trackOffset + local);
        }
      }
      events[event] = KFParticleGpuEventDesc{};
      events[event].eventId = 21000u + event;
      events[event].TrackSet(SecondaryPositiveFirst).tracks =
        {trackOffset, tracksPerCharge};
      events[event].TrackSet(SecondaryPositiveFirst).Species(Pion) =
        {trackOffset, tracksPerCharge};
      events[event].TrackSet(SecondaryNegativeFirst).tracks =
        {trackOffset + tracksPerCharge, tracksPerCharge};
      events[event].TrackSet(SecondaryNegativeFirst).Species(Pion) =
        {trackOffset + tracksPerCharge, tracksPerCharge};
    }

    KFParticleGpuSteering& steering = runtime.GetSteering();
    KFParticleGpuDecayPlan& plan = steering.GetDecayPlan();
    plan.Clear();
    KFParticleGpuTwoDaughterChannel channel = MakeK0ShortToPiPlusPiMinusChannel(94u);
    channel.motherMassSigma = -1.f;
    channel.secondaryMassSigmaCut = -1.f;
    channel.maxSecondaryTopoChi2PerNdf = -1.f;
    channel.minSecondaryLdL = -1.f;
    channel.maxDaughterDistance = -1.f;
    channel.selection.expectedMassSigma = -1.f;
    channel.selection.massSigmaCut = -1.f;
    channel.selection.maxGeometricChi2PerNdf = -1.f;
    channel.selection.maxPrimaryVertexDistance = -1.f;
    channel.selection.minSecondaryLdL = -1.f;
    channel.selection.maxPrimaryTopologyChi2PerNdf = -1.f;
    channel.selection.maxSecondaryTopologyChi2PerNdf = -1.f;
    channel.selection.requirePrimaryVertex = 0u;
    plan.AddTwoDaughterChannel(channel);

    buffers.SetTransactionSizes(1u, eventCount, eventCount, eventCount);
    KFParticleGpuTransactionView transaction = buffers.HostTransaction();
    transaction.Transaction()[0] = KFParticleGpuTransactionDesc{};
    transaction.Transaction()[0].transactionId = 21001u;
    transaction.Transaction()[0].decayPlanRevision = plan.Revision();
    transaction.Transaction()[0].domains = {0u, eventCount};
    transaction.Transaction()[0].workRanges = {0u, eventCount};
    unsigned int selectedOffset = 0u;
    for (unsigned int event = 0u; event < eventCount; ++event) {
      KFParticleGpuDomainDesc& domain = transaction.Domains()[event];
      domain = KFParticleGpuDomainDesc{};
      domain.sourceId = events[event].eventId;
      domain.inputDescriptorIndex = event;
      domain.sourceTracks = {event * tracksPerEvent, tracksPerEvent};
      domain.workRanges = {event, 1u};
      domain.taskReservation = {event * candidatesPerEvent, candidatesPerEvent};
      domain.candidateReservation = {event * candidatesPerEvent, candidatesPerEvent};
      domain.daughterReservation = {2u * event * candidatesPerEvent,
                                    2u * candidatesPerEvent};
      const unsigned int eventSelectedCapacity = event == 0u
        ? firstSelectedCapacity : candidatesPerEvent;
      domain.selectedReservation = {selectedOffset, eventSelectedCapacity};
      domain.statusIndex = event;
      selectedOffset += eventSelectedCapacity;
      transaction.WorkRanges()[event] = KFParticleGpuWorkRange{};
      transaction.WorkRanges()[event].domainIndex = event;
      transaction.WorkRanges()[event].family = KFParticleGpuDomainInputWork;
      transaction.WorkRanges()[event].sourceTracks = domain.sourceTracks;
      transaction.WorkRanges()[event].outputReservation = domain.taskReservation;
    }

    const auto& results = steering.RunDecayPlanBatch(0u, eventCount, candidateCapacity);
    const auto& eventResults = steering.LastDecayPlanEventResults();
    buffers.DownloadTransactionStatus();
    transaction = buffers.HostTransaction();
    const auto selected = MakeConstView(buffers.HostSelectedCandidates());
    const auto candidates = MakeConstView(buffers.HostCandidates());
    assert(results.size() == eventCount);
    assert(eventResults.size() == eventCount);
    assert(selected.OverflowFlags() == KFGpuSelectedCandidateCapacityExceeded);
    assert(transaction.DomainStatuses()[0].selected == firstSelectedCapacity);
    assert((transaction.DomainStatuses()[0].overflowFlags
            & KFGpuSelectedCandidateCapacityExceeded) != 0u);
    for (unsigned int event = 0u; event < eventCount; ++event) {
      const auto& domain = transaction.Domains()[event];
      const auto& status = transaction.DomainStatuses()[domain.statusIndex];
      assert(status.visitedPairs == candidatesPerEvent);
      assert(status.storedTasks == candidatesPerEvent);
      assert(status.candidates > firstSelectedCapacity);
      assert(status.candidates <= domain.candidateReservation.size);
      assert(status.selected <= domain.selectedReservation.size);
      for (unsigned int local = 0u; local < status.selected; ++local) {
        const unsigned int selectedIndex = domain.selectedReservation.offset + local;
        assert(selectedIndex < domain.selectedReservation.End());
        assert(candidates.Metadata().EventIndex(selected.Index(selectedIndex)) == event);
      }
    }

    // Retained non-fused constructor: absolute domain indices, offset output,
    // an empty neighbour, and independent daughter/candidate exhaustion.
    const unsigned int channelStatusSize = transaction.DomainChannelStatusCapacity();
    buffers.SetTransactionSizes(1u, eventCount + 1u, eventCount, eventCount, channelStatusSize);
    transaction = buffers.HostTransaction();
    for (unsigned int i = eventCount; i > 0u; --i) {
      transaction.Domains()[i] = transaction.Domains()[i - 1u];
    }
    transaction.Domains()[0] = KFParticleGpuDomainDesc{};
    transaction.Transaction()->domains = {1u, eventCount};
    const auto& storage = buffers.Storage();
    xpu::buffer<KFParticleGpuTwoDaughterRoutedTask> routed(candidateCapacity, xpu::buf_io);
    auto* routedHost = HostPointer(routed);
    for (unsigned int event = 0u; event < eventCount; ++event) {
      auto& domain = transaction.Domains()[event + 1u];
      routedHost[domain.taskReservation.offset] =
        {0u, event * tracksPerEvent, event * tracksPerEvent + tracksPerCharge, event, event + 1u};
      transaction.WorkRanges()[event].domainIndex = event + 1u;
    }
    runtime.GetQueue().copy(HostPointer(routed), storage.fTwoDaughterRoutedTasks.get(), candidateCapacity);
    for (unsigned int compact = 0u; compact < 2u; ++compact)
    for (unsigned int shortage = 0u; shortage < 3u; ++shortage) {
      for (unsigned int event = 0u; event < eventCount; ++event) {
        auto& domain = transaction.Domains()[event + 1u];
        domain.candidateReservation.size = event == 0u && shortage == 2u ? 0u : candidatesPerEvent;
        domain.daughterReservation.size = event == 0u && shortage < 2u ? shortage : 2u * candidatesPerEvent;
        auto& status = transaction.DomainStatuses()[domain.statusIndex];
        status = KFParticleGpuDomainStatus{};
        status.storedTasks = event == 3u ? 0u : 1u;
      }
      auto outputPool = buffers.HostCandidates();
      outputPool.SizeData()[0] = 0u;
      outputPool.Daughters().SizeData()[0] = 0u;
      outputPool.OverflowFlagsData()[0] = 0u;
      buffers.UploadCandidates();
      const auto tiles = KFParticleGpuBuildTaskTiles(transaction, candidateCapacity);
      assert(tiles.size() == 3u && tiles.front().domainIndex == 1u && tiles.back().domainIndex == 3u);
      auto tileCapacity = buffers.Capacities();
      tileCapacity.twoDaughterTaskTiles = static_cast<unsigned int>(tiles.size());
      buffers.EnsureCapacity(tileCapacity);
      std::copy(tiles.begin(), tiles.end(), buffers.HostTwoDaughterTaskTiles());
      buffers.UploadTwoDaughterTaskTiles(tiles.size());
      buffers.UploadTransaction();
      const KFParticleGpuKernels state(
        MakeConstView(buffers.DeviceInputTracks()), MakeConstView(buffers.DevicePrimaryVertices()),
        buffers.DeviceEvents(), storage.fTwoDaughterTasks.get(), buffers.Capacities().twoDaughterTasks,
        buffers.DeviceCandidates(), buffers.DeviceV0SelectionResults(), buffers.DeviceSelectedCandidates(),
        buffers.DeviceV0TrackRouting(), buffers.DeviceTwoDaughterRouting(),
        buffers.DeviceCandidateDescriptorIndices(), buffers.DeviceDecayGraph(),
        buffers.DeviceGraphOperationStorage(), buffers.DeviceTwoDaughterGenerationStorage(),
        buffers.DeviceV0TrackGenerationStorage(), buffers.DeviceTransaction());
      if (compact) {
        runtime.GetQueue().launch<KFParticleGpuConstructTwoDaughterTaskTiles>(
          xpu::n_blocks(tiles.size()), state, buffers.DeviceTransaction(), 0u,
          buffers.DeviceTwoDaughterTaskTiles(), static_cast<unsigned int>(tiles.size()));
      }
      else {
        runtime.GetQueue().launch<KFParticleGpuConstructTwoDaughterBatch>(
          xpu::n_threads(candidateCapacity), state, buffers.DeviceTransaction(), 0u);
      }
      buffers.DownloadTransactionStatus();
      buffers.DownloadCandidates();
      transaction = buffers.HostTransaction();
      assert(transaction.DomainStatuses()[0].candidates == 0u);
      assert(transaction.DomainStatuses()[0].daughters == 0u);
      assert(transaction.DomainStatuses()[0].overflowFlags
             == (shortage == 2u ? CandidateCapacityExceeded : DaughterCapacityExceeded));
      for (unsigned int event = 0u; event < eventCount; ++event) {
        const auto& status = transaction.DomainStatuses()[event];
        assert(status.failureFlags == 0u);
        if (event == 0u || event == 3u) { continue; }
        assert(status.candidates == 1u && status.daughters == 2u && status.overflowFlags == 0u);
        const auto pool = MakeConstView(buffers.HostCandidates());
        const unsigned int index = transaction.Domains()[event + 1u].candidateReservation.offset;
        assert(pool.Metadata().EventIndex(index) == event);
        assert(pool.Metadata().DaughterCount(index) == 2u);
      }
      assert(transaction.DomainStatuses()[3].candidates == 0u);
    }

    plan.Clear();
    Pass("decay-plan-high-occupancy-domain-isolation",
         "four high-occupancy domains keep compact output inside event reservations and report local truncation without corrupting neighbours");
  }

  void TestDecayPlanExecutorEdgeCases(KFParticleGpuRuntime& runtime,
                                      KFParticleGpuBufferManager& buffers)
  {
    KFParticleGpuSteering& steering = runtime.GetSteering();
    KFParticleGpuDecayPlan& plan = steering.GetDecayPlan();

    plan.Clear();
    plan.AddTwoDaughterChannel(MakePionPairChannel(61u));
    FillDecayPlanMultiChannelFixture(buffers);
    const std::vector<KFParticleGpuTwoDaughterChannelResult>& truncated =
      steering.RunDecayPlan(0u, 1u);
    assert(truncated.size() == 1u);
    assert(truncated[0].acceptedTasks == 2u);
    assert(truncated[0].storedTasks == 1u);
    assert(truncated[0].Truncated());
    assert(truncated[0].HasOverflow(CandidateCapacityExceeded));
    assert(truncated[0].constructedCandidates == 1u);
    assert(truncated[0].constructedDaughters == 2u);
    assert(truncated[0].generationCandidates.overflowFlags
           == CandidateCapacityExceeded);

    plan.Clear();
    plan.AddTwoDaughterChannel(MakePionPairChannel(62u));
    plan.AddTwoDaughterChannel(MakeKaonPairChannel(63u));
    FillDecayPlanMultiChannelFixture(buffers);
    KFParticleGpuInputTrackSoAView overflowTracks = buffers.HostInputTracks();
    overflowTracks.ChiToPrimaryVertex(5) = 1.f;
    overflowTracks.ChiToPrimaryVertex(7) = 1.f;
    const std::vector<KFParticleGpuTwoDaughterChannelResult>& overflow =
      steering.RunDecayPlan(0u, 4u);
    assert(overflow.size() == 2u);
    assert(overflow[0].acceptedTasks == 4u);
    assert(overflow[1].acceptedTasks == 4u);
    assert(overflow[0].constructedCandidates
             + overflow[1].constructedCandidates == 4u);
    assert(overflow[0].constructedDaughters
             + overflow[1].constructedDaughters == 8u);
    assert(overflow[1].HasOverflow(CandidateCapacityExceeded));
    assert(overflow[1].HasOverflow(DaughterCapacityExceeded));
    const KFParticleGpuConstCandidatePoolView overflowCandidates =
      MakeConstView(buffers.HostCandidates());
    assert(overflowCandidates.Size() == overflowCandidates.Capacity());
    assert(overflowCandidates.Daughters().Size()
           == overflowCandidates.Daughters().Capacity());
    assert(overflowCandidates.OverflowFlags()
           == static_cast<unsigned int>(CandidateCapacityExceeded | DaughterCapacityExceeded));
    unsigned int constructedByChannel[2] = {0u, 0u};
    unsigned int rejectedByCapacity = 0u;
    for (unsigned int candidate = 0u;
         candidate < overflowCandidates.Size(); ++candidate) {
      const unsigned int status =
        overflowCandidates.Metadata().OperationStatus(candidate);
      if (status == KFGpuCandidateOperationAccepted) {
        assert(overflowCandidates.Metadata().DaughterCount(candidate) == 2u);
        const unsigned int channel =
          overflowCandidates.Metadata().ChannelId(candidate);
        assert(channel == 62u || channel == 63u);
        ++constructedByChannel[channel == 62u ? 0u : 1u];
      }
      else {
        assert(status == KFGpuCandidateOperationRejected);
        assert(overflowCandidates.Metadata().DaughterCount(candidate) == 0u);
        ++rejectedByCapacity;
      }
    }
    assert(constructedByChannel[0] == overflow[0].constructedCandidates);
    assert(constructedByChannel[1] == overflow[1].constructedCandidates);
    assert(constructedByChannel[0] + constructedByChannel[1] == 4u);
    assert(rejectedByCapacity == 2u);

    plan.Clear();
    plan.AddTwoDaughterChannel(MakePionPairChannel(64u));
    plan.AddTwoDaughterChannel(MakeKaonPairChannel(65u));
    FillDecayPlanMultiEventFixture(buffers);
    const std::vector<KFParticleGpuTwoDaughterChannelResult>& multiEvent =
      steering.RunDecayPlan(1u, 2u);
    assert(multiEvent.size() == 2u);
    assert(multiEvent[0].eventIndex == 1u);
    assert(multiEvent[1].eventIndex == 1u);
    assert(multiEvent[0].generationCandidates.offset == 0u);
    assert(multiEvent[1].generationCandidates.offset == 0u);
    const KFParticleGpuConstCandidatePoolView eventCandidates =
      MakeConstView(buffers.HostCandidates());
    assert(eventCandidates.Size() == 4u);
    assert(HasCandidateDaughters(eventCandidates, 2701, 2901));
    assert(HasCandidateDaughters(eventCandidates, 2702, 2901));
    assert(HasCandidateDaughters(eventCandidates, 2801, 3001));
    assert(HasCandidateDaughters(eventCandidates, 2802, 3001));
    ExpectChannelCandidatePdgs(eventCandidates, multiEvent[0], 310);
    ExpectChannelCandidatePdgs(eventCandidates, multiEvent[1], 333);
    ExpectChannelDaughterRange(eventCandidates, multiEvent[0]);
    ExpectChannelDaughterRange(eventCandidates, multiEvent[1]);
    for (unsigned int i = 0; i < eventCandidates.Size(); ++i) {
      assert(eventCandidates.Metadata().EventIndex(i) == 1u);
    }

    plan.Clear();
    Pass("decay-plan-executor-edge-cases",
         "decay plan executor reports truncation, overflow, and multi-event channel ranges");
  }

  void TestTwoDaughterSyntheticGrid(KFParticleGpuRuntime& runtime, KFParticleGpuBufferManager& buffers)
  {
    buffers.SetInputSizes(4, 1, 1);
    KFParticleGpuInputTrackSoAView tracks = buffers.HostInputTracks();
    StoreSyntheticTrack(tracks, 0, 0.0f, 0.1f, -0.2f, 0.8f, 0.3f, 1.2f, 211, 1, 301);
    StoreSyntheticTrack(tracks, 1, 0.3f, -0.4f, 0.5f, -0.2f, 0.9f, 0.7f, -211, -1, 302);
    StoreSyntheticTrack(tracks, 2, -0.5f, 0.2f, 0.4f, 1.1f, -0.6f, 0.5f, 321, 1, 401);
    StoreSyntheticTrack(tracks, 3, 0.7f, 0.9f, -0.3f, -0.4f, 0.2f, 1.0f, -321, -1, 402);
    buffers.UploadInput();

    xpu::buffer<KFParticleGpuTwoDaughterTask> taskBuffer(2, xpu::buf_io);
    KFParticleGpuTwoDaughterTask* tasks = HostPointer(taskBuffer);

    for (int pass = 0; pass < 2; ++pass) {
      buffers.ResetCandidates();

      tasks[0] = KFParticleGpuTwoDaughterTask();
      tasks[1] = KFParticleGpuTwoDaughterTask();
      if (pass == 0) {
        tasks[0].firstTrack = 0;
        tasks[0].secondTrack = 1;
        tasks[0].firstDaughterPdg = 211;
        tasks[0].secondDaughterPdg = -211;
        tasks[0].motherPdg = 310;
        tasks[0].flags = KFGpuTwoDaughterUseLineDca | KFGpuTwoDaughterUseEnergyFit;
        tasks[1].firstTrack = 2;
        tasks[1].secondTrack = 3;
        tasks[1].firstDaughterPdg = 321;
        tasks[1].secondDaughterPdg = -321;
        tasks[1].motherPdg = 333;
        tasks[1].flags = KFGpuTwoDaughterUseLineDca | KFGpuTwoDaughterUseEnergyFit;
      }
      else {
        tasks[0].firstTrack = 0;
        tasks[0].secondTrack = 3;
        tasks[0].firstDaughterPdg = 211;
        tasks[0].secondDaughterPdg = -321;
        tasks[0].motherPdg = 313;
        tasks[0].flags = KFGpuTwoDaughterUseEnergyFit;
        tasks[1].firstTrack = 2;
        tasks[1].secondTrack = 1;
        tasks[1].firstDaughterPdg = 321;
        tasks[1].secondDaughterPdg = -211;
        tasks[1].motherPdg = -313;
        tasks[1].flags = KFGpuTwoDaughterUseLineDca;
      }

      for (unsigned int i = 0; i < 2; ++i) {
        tasks[i].eventIndex = 0;
        tasks[i].primaryVertexIndex = -1;
        tasks[i].firstMass = 0.13957f + 0.05f * static_cast<float>(i + pass);
        tasks[i].secondMass = 0.13957f + 0.03f * static_cast<float>(i + 1);
      }

      KFParticleGpuFitState references[2];
      const KFParticleGpuConstInputTrackSoAView constTracks = MakeConstView(buffers.HostInputTracks());
      for (unsigned int i = 0; i < 2; ++i) {
        const bool built = BuildTwoDaughterCandidate(constTracks, tasks[i], references[i]);
        assert(built);
      }

      runtime.GetQueue().copy(taskBuffer, xpu::h2d);
      runtime.GetQueue().wait();
      runtime.GetQueue().launch<KFParticleGpuTwoDaughterTaskKernel>(
        xpu::n_threads(2),
        MakeConstView(buffers.DeviceInputTracks()),
        taskBuffer.get(),
        2u,
        buffers.DeviceCandidates());
      runtime.GetQueue().wait();
      buffers.DownloadCandidates();

      const KFParticleGpuConstCandidatePoolView candidates = MakeConstView(buffers.HostCandidates());
      assert(candidates.Size() == 2);
      assert(candidates.Daughters().Size() == 4);
      assert(candidates.OverflowFlags() == 0);
      for (unsigned int i = 0; i < 2; ++i) {
        KFParticleGpuFitState actual;
        LoadCandidateFit(candidates, i, actual);
        ExpectFitStateClose(actual, references[i], 2.e-5f);
        assert(candidates.Metadata().Pdg(i) == tasks[i].motherPdg);
        assert(candidates.Metadata().DaughterCount(i) == 2);
        assert(candidates.Metadata().Flags(i) & KFGpuCandidateValid);
        assert(candidates.Daughters().SourceId(candidates.Metadata().DaughterOffset(i))
               == tracks.SourceId(tasks[i].firstTrack));
        assert(candidates.Daughters().SourceId(candidates.Metadata().DaughterOffset(i) + 1u)
               == tracks.SourceId(tasks[i].secondTrack));
      }
    }

    Pass("two-daughter-synthetic-grid", "CPU-vs-GPU comparison over multiple pair geometries and covariance cross terms");
  }

  void TestSelectionHelpers(KFParticleGpuBufferManager& buffers)
  {
    KFParticleGpuCandidatePoolView mutableCandidates = buffers.HostCandidates();
    const KFParticleGpuConstCandidatePoolView candidates = MakeConstView(mutableCandidates);

    assert(candidates.Size() >= 1);
    assert(KFParticleGpuSelection::HasDistinctDaughterSourceIds(candidates, 0));

    KFParticleGpuFitState candidate;
    LoadCandidateFit(candidates, 0, candidate);
    candidate.Chi2() = 4.f;
    candidate.NDF() = 2;

    const float mass = KFParticleGpuMath::Mass(candidate);
    assert(KFParticleGpuSelection::PassMassWindow(candidate, mass, 1.e-5f));
    assert(!KFParticleGpuSelection::PassMassWindow(candidate, mass + 0.1f, 1.e-5f));
    assert(AlmostEqual(KFParticleGpuSelection::MassWindowHalfWidth(0.01f, 3.f), 0.03f));
    assert(KFParticleGpuSelection::MassWindowHalfWidth(-1.f, 3.f) < 0.f);
    assert(KFParticleGpuSelection::PassSigmaMassWindow(candidate, mass, 0.01f, 3.f));
    assert(!KFParticleGpuSelection::PassSigmaMassWindow(candidate, mass + 0.1f, 0.01f, 3.f));
    assert(KFParticleGpuSelection::PassChi2PerNdf(candidate, 2.1f));
    assert(!KFParticleGpuSelection::PassChi2PerNdf(candidate, 1.9f));

    KFParticleGpuTwoDaughterTaskSource source;
    source.motherMass = mass + 1.f;
    source.motherMassSigma = 0.f;
    source.secondaryMassSigmaCut = -1.f;
    source.maxSecondaryTopoChi2PerNdf = -1.f;
    assert(PassTwoDaughterPostBuildSelection(source, candidate));
    source.motherMass = mass;
    source.motherMassSigma = 0.01f;
    source.secondaryMassSigmaCut = 3.f;
    source.maxSecondaryTopoChi2PerNdf = 2.1f;
    assert(PassTwoDaughterPostBuildSelection(source, candidate));
    source.motherMass = mass + 0.1f;
    assert(!PassTwoDaughterPostBuildSelection(source, candidate));

    mutableCandidates.Daughters().SourceId(1) = mutableCandidates.Daughters().SourceId(0);
    assert(!KFParticleGpuSelection::HasDistinctDaughterSourceIds(MakeConstView(mutableCandidates), 0));
    Pass("selection-helpers", "mass window, chi2/ndf, post-build selection, and duplicate daughter-source predicates");
  }

  void TestV0TrackTaskContract(KFParticleGpuRuntime& runtime,
                               KFParticleGpuBufferManager& buffers)
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
    StoreSyntheticTrack(tracks, 0u, 0.f, 0.f, 0.f, 0.3f, 0.1f, 0.7f, -211, -1, 3001);
    StoreSyntheticTrack(tracks, 1u, 0.f, 0.f, 0.f, 0.2f, 0.2f, 0.6f, -211, -1, 1101);
    StoreSyntheticTrack(tracks, 2u, 0.f, 0.f, 0.f, 0.1f, 0.3f, 0.8f, 211, 1, 4001);
    StoreSyntheticTrack(tracks, 3u, 0.f, 0.f, 0.f, 0.2f, 0.1f, 0.9f, 211, 1, 4101);
    tracks.ChiToPrimaryVertex(0u) = 8.f;
    tracks.ChiToPrimaryVertex(1u) = 8.f;
    tracks.ChiToPrimaryVertex(2u) = 8.f;
    tracks.ChiToPrimaryVertex(3u) = 2.f;

    KFParticleGpuEventDesc* events = buffers.HostEvents();
    events[0] = KFParticleGpuEventDesc();
    events[0].eventId = 1401u;
    events[0].TrackSet(SecondaryNegativeFirst).tracks = KFParticleGpuRange(0u, 2u);
    events[1] = KFParticleGpuEventDesc();
    events[1].eventId = 1402u;
    events[1].TrackSet(SecondaryPositiveFirst).tracks = KFParticleGpuRange(2u, 2u);

    KFParticleGpuCandidatePoolView candidates = buffers.HostCandidates();
    for (unsigned int index = 0u; index < 2u; ++index) {
      candidates.Metadata().PrimaryVertexIndex(index) = -1;
      candidates.Metadata().Flags(index) = KFGpuCandidateValid;
      candidates.Metadata().ChannelId(index) = index;
    }
    candidates.Metadata().Pdg(0u) = 3122;
    candidates.Metadata().EventIndex(0u) = 0u;
    candidates.Metadata().DaughterOffset(0u) = 0u;
    candidates.Metadata().DaughterCount(0u) = 2u;
    candidates.Daughters().SourceId(0u) = 1101;
    candidates.Daughters().SourceId(1u) = 2101;
    candidates.Metadata().Pdg(1u) = -3122;
    candidates.Metadata().EventIndex(1u) = 1u;
    candidates.Metadata().DaughterOffset(1u) = 2u;
    candidates.Metadata().DaughterCount(1u) = 2u;
    candidates.Daughters().SourceId(2u) = 3101;
    candidates.Daughters().SourceId(3u) = 5101;
    candidates.SizeData()[0] = 2u;
    candidates.Daughters().SizeData()[0] = 4u;
    candidates.OverflowFlagsData()[0] = 0u;

    KFParticleGpuSelectedCandidateIndexView selected = buffers.HostSelectedCandidates();
    selected.Index(0u) = 0u;
    selected.ChannelId(0u) = 41u;
    selected.Index(1u) = 1u;
    selected.ChannelId(1u) = 42u;
    selected.SizeData()[0] = 2u;
    selected.OverflowFlagsData()[0] = 0u;
    const KFParticleGpuSelectedV0View selectedV0s = MakeSelectedV0View(
      MakeConstView(candidates), MakeConstView(selected));
    const KFParticleGpuConstInputTrackSoAView constTracks = MakeConstView(tracks);
    const KFParticleGpuV0TrackInputView v0TrackInput(selectedV0s, constTracks);

    KFParticleGpuV0TrackChannel lambdaPion;
    lambdaPion.channelId = 61u;
    lambdaPion.v0Pdg = 3122;
    lambdaPion.bachelorPdg = -211;
    lambdaPion.motherPdg = 3312;
    lambdaPion.primaryVertexIndex = -1;
    lambdaPion.eventIndex = 0u;
    lambdaPion.bachelorTracks = KFParticleGpuRange(0u, 2u);
    lambdaPion.minBachelorChiToPrimaryVertex = 5.f;

    KFParticleGpuV0TrackTask task;
    KFParticleGpuV0TrackLineage lineage;
    const KFParticleGpuSelectedCandidateRange firstRange{0u, 1u, 0u};
    unsigned int rejection = KFParticleGpuV0Track::ResolveTask(
      v0TrackInput, lambdaPion, firstRange, 0u, 0u, 0u, 1u, task, lineage);
    assert(KFParticleGpuV0Track::IsAccepted(rejection));
    assert(task.selectedV0Index == 0u);
    assert(task.v0CandidateIndex == 0u);
    assert(task.bachelorTrackIndex == 0u);
    assert(task.eventIndex == 0u);
    assert(task.channelId == 61u);
    assert(task.primaryVertexIndex == -1);
    assert(lineage.firstV0DaughterSourceId == 1101);
    assert(lineage.secondV0DaughterSourceId == 2101);
    assert(lineage.bachelorSourceId == 3001);
    assert(KFParticleGpuV0Track::IsCanonical(lineage));

    rejection = KFParticleGpuV0Track::ResolveTask(
      v0TrackInput, lambdaPion, firstRange, 0u, 1u, 0u, 1u, task, lineage);
    assert(rejection == KFGpuV0TrackTaskRejectDuplicateSource);
    rejection = KFParticleGpuV0Track::ResolveTask(
      v0TrackInput, lambdaPion, firstRange, 1u, 2u, 0u, 1u, task, lineage);
    assert(rejection == KFGpuV0TrackTaskRejectSelectedIndex);
    rejection = KFParticleGpuV0Track::ResolveTask(
      v0TrackInput, lambdaPion, firstRange, 0u, 0u, 1u, 1u, task, lineage);
    assert(rejection == KFGpuV0TrackTaskRejectCapacity);
    const KFParticleGpuSelectedCandidateRange emptyRange{1u, 0u, 0u};
    rejection = KFParticleGpuV0Track::ResolveTask(
      v0TrackInput, lambdaPion, emptyRange, 1u, 2u, 0u, 1u, task, lineage);
    assert(rejection == KFGpuV0TrackTaskRejectSelectedIndex);

    KFParticleGpuV0TrackChannel antiLambdaPion = lambdaPion;
    antiLambdaPion.channelId = 62u;
    antiLambdaPion.v0Pdg = -3122;
    antiLambdaPion.bachelorPdg = 211;
    antiLambdaPion.motherPdg = -3312;
    antiLambdaPion.eventIndex = 1u;
    antiLambdaPion.bachelorTracks = KFParticleGpuRange(2u, 2u);
    const KFParticleGpuSelectedCandidateRange secondRange{1u, 1u, 0u};
    rejection = KFParticleGpuV0Track::ResolveTask(
      v0TrackInput, antiLambdaPion, secondRange, 1u, 2u, 0u, 1u, task, lineage);
    assert(KFParticleGpuV0Track::IsAccepted(rejection));
    assert(task.v0CandidateIndex == 1u);
    assert(task.eventIndex == 1u);
    assert(lineage.firstV0DaughterSourceId == 3101);
    assert(lineage.secondV0DaughterSourceId == 5101);
    assert(lineage.bachelorSourceId == 4001);
    rejection = KFParticleGpuV0Track::ResolveTask(
      v0TrackInput, antiLambdaPion, secondRange, 1u, 3u, 0u, 1u, task, lineage);
    assert(rejection == KFGpuV0TrackTaskRejectBachelorChiToPrimaryVertex);

    buffers.UploadInput();
    buffers.UploadCandidates();
    buffers.UploadSelectedCandidates();
    xpu::buffer<unsigned int> deviceRejection(1u, xpu::buf_io);
    xpu::buffer<KFParticleGpuV0TrackTask> deviceTask(1u, xpu::buf_io);
    xpu::buffer<KFParticleGpuV0TrackLineage> deviceLineage(1u, xpu::buf_io);
    runtime.GetQueue().launch<KFParticleGpuV0TrackTaskProbe>(
      xpu::n_threads(1),
      MakeConstView(buffers.DeviceCandidates()),
      MakeConstView(buffers.DeviceSelectedCandidates()),
      MakeConstView(buffers.DeviceInputTracks()),
      lambdaPion,
      firstRange,
      0u,
      0u,
      deviceRejection.get(),
      deviceTask.get(),
      deviceLineage.get());
    runtime.GetQueue().wait();
    runtime.GetQueue().copy(deviceRejection, xpu::d2h);
    runtime.GetQueue().copy(deviceTask, xpu::d2h);
    runtime.GetQueue().copy(deviceLineage, xpu::d2h);
    runtime.GetQueue().wait();
    assert(HostPointer(deviceRejection)[0] == KFGpuV0TrackTaskAccept);
    assert(HostPointer(deviceTask)[0].v0CandidateIndex == 0u);
    assert(HostPointer(deviceTask)[0].bachelorTrackIndex == 0u);
    assert(HostPointer(deviceLineage)[0].firstV0DaughterSourceId == 1101);
    assert(HostPointer(deviceLineage)[0].secondV0DaughterSourceId == 2101);
    assert(HostPointer(deviceLineage)[0].bachelorSourceId == 3001);

    Pass("v0-track-task-contract",
         "host and device resolve selected Lambda and anti-Lambda V0 lineage and reject reuse, range, and capacity violations");
  }

  void TestV0TrackCompactCandidateConstruction(KFParticleGpuRuntime& runtime,
                                                KFParticleGpuBufferManager& buffers)
  {
    KFParticleGpuBufferCapacities requested = buffers.Capacities();
    requested.tracks = requested.tracks > 3u ? requested.tracks : 3u;
    requested.events = requested.events > 1u ? requested.events : 1u;
    requested.candidates = requested.candidates > 3u ? requested.candidates : 3u;
    requested.daughterIds = std::max(requested.daughterIds, 2u * KFParticleGpuV0TrackLineage::MaximumSize);
    requested.selectedCandidates = requested.selectedCandidates > 1u ? requested.selectedCandidates : 1u;
    requested.v0TrackTasks = requested.v0TrackTasks > 1u ? requested.v0TrackTasks : 1u;
    requested.nonhomogeneousField = true;
    buffers.EnsureCapacity(requested);
    buffers.SetInputSizes(3u, 0u, 1u);

    KFParticleGpuInputTrackSoAView tracks = buffers.HostInputTracks();
    StoreSyntheticTrack(tracks, 0u, 0.3f, -0.2f, 0.2f, 0.3f, 0.2f, 0.8f, -211, -1, 9301);
    StoreSyntheticTrack(tracks, 1u, -0.2f, 0.1f, -0.1f, 0.2f, 0.4f, 0.7f, -211, -1, 9302);
    StoreSyntheticTrack(tracks, 2u, 0.f, 0.f, 0.f, 0.2f, 0.2f, 0.6f, -211, -1, 9101);
    StoreDiagnosticField(tracks, 0u);
    StoreDiagnosticField(tracks, 1u);
    StoreDiagnosticField(tracks, 2u);

    KFParticleGpuEventDesc* events = buffers.HostEvents();
    events[0] = KFParticleGpuEventDesc();
    events[0].eventId = 1411u;
    events[0].TrackSet(SecondaryNegativeFirst).tracks = KFParticleGpuRange(0u, 3u);
    events[0].TrackSet(SecondaryNegativeFirst).Species(Pion) = KFParticleGpuRange(0u, 2u);

    KFParticleGpuFitState lambda;
    lambda.Initialize();
    lambda.X() = 0.f;
    lambda.Y() = 0.f;
    lambda.Z() = 0.f;
    lambda.Px() = 0.4f;
    lambda.Py() = 0.2f;
    lambda.Pz() = 1.1f;
    lambda.E() = 1.6f;
    lambda.Q() = 0;
    lambda.NDF() = 1;
    lambda.Chi2() = 0.5f;
    for (int component = 0; component < KFParticleGpuFitState::NumberOfCovarianceElements; ++component) {
      lambda.Covariance(component) = 0.f;
    }
    lambda.Covariance(0, 0) = 0.3f;
    lambda.Covariance(1, 1) = 0.3f;
    lambda.Covariance(2, 2) = 0.3f;
    lambda.Covariance(3, 3) = 0.1f;
    lambda.Covariance(4, 4) = 0.1f;
    lambda.Covariance(5, 5) = 0.1f;
    lambda.SumDaughterMass() = kCpuReferenceLambdaMass;

    KFParticleGpuCandidatePoolView candidates = buffers.HostCandidates();
    StoreCandidateFit(lambda, candidates, 0u);
    candidates.Metadata().Pdg(0u) = 3122;
    candidates.Metadata().PrimaryVertexIndex(0u) = -1;
    candidates.Metadata().EventIndex(0u) = 0u;
    candidates.Metadata().DaughterOffset(0u) = 0u;
    candidates.Metadata().DaughterCount(0u) = 2u;
    candidates.Metadata().Flags(0u) = KFGpuCandidateValid;
    candidates.Metadata().ChannelId(0u) = KFGpuChannelLambdaToProtonPiMinus;
    candidates.Daughters().SourceId(0u) = 9101;
    candidates.Daughters().SourceId(1u) = 9201;
    candidates.SizeData()[0] = 1u;
    candidates.Daughters().SizeData()[0] = 2u;
    candidates.OverflowFlagsData()[0] = 0u;

    KFParticleGpuSelectedCandidateIndexView selected = buffers.HostSelectedCandidates();
    selected.Index(0u) = 0u;
    selected.ChannelId(0u) = KFGpuChannelLambdaToProtonPiMinus;
    selected.SizeData()[0] = 1u;
    selected.OverflowFlagsData()[0] = 0u;

    KFParticleGpuV0TrackChannel channel;
    channel.channelId = 141u;
    channel.v0Pdg = 3122;
    channel.bachelorPdg = -211;
    channel.motherPdg = 3312;
    channel.primaryVertexIndex = -1;
    channel.eventIndex = 0u;
    channel.bachelorTracks = KFParticleGpuRange(0u, 2u);
    channel.bachelorMass = kCpuReferencePionMass;
    channel.motherMass = 1.32171f;
    channel.motherMassSigma = -1.f;
    channel.transportMode = KFGpuTransportFullField;

    buffers.UploadInput();
    buffers.UploadCandidates();
    buffers.UploadSelectedCandidates();
    buffers.ResetV0TrackTaskStatus();
    const KFParticleGpuDeviceStorage& storage = buffers.Storage();
    const KFParticleGpuSelectedCandidateRange selectedRange{0u, 1u, 0u};
    runtime.GetQueue().launch<KFParticleGpuGenerateV0TrackTasksCompact>(
      xpu::n_threads(2u),
      MakeConstView(buffers.DeviceCandidates()),
      MakeConstView(buffers.DeviceSelectedCandidates()),
      MakeConstView(buffers.DeviceInputTracks()),
      channel,
      selectedRange,
      storage.fV0TrackTasks.get(),
      1u,
      storage.fV0TrackTaskCount.get(),
      storage.fV0TrackTotalPairCount.get(),
      storage.fV0TrackTaskOverflowFlags.get());
    runtime.GetQueue().launch<KFParticleGpuV0TrackCompactCandidatePoolKernel>(
      xpu::n_threads(1u),
      MakeConstView(buffers.DeviceInputTracks()),
      MakeConstView(buffers.DeviceCandidates()),
      storage.fV0TrackTasks.get(),
      1u,
      storage.fV0TrackTaskCount.get(),
      buffers.DeviceCandidates());
    const KFParticleGpuV0TrackTaskStatus taskStatus = buffers.DownloadV0TrackTaskStatus();
    const KFParticleGpuCandidatePoolStatus candidateStatus = buffers.DownloadCandidateStatus();
    buffers.DownloadCandidates();

    assert(taskStatus.totalPairs == 2u);
    assert(taskStatus.accepted == 2u);
    assert(taskStatus.overflowFlags == CandidateCapacityExceeded);
    assert(candidateStatus.candidates == 2u);
    assert(candidateStatus.daughters == 5u);
    assert(candidateStatus.overflowFlags == 0u);
    const KFParticleGpuConstCandidatePoolView output = MakeConstView(buffers.HostCandidates());
    assert(output.Metadata().Pdg(0u) == 3122);
    assert(output.Metadata().Pdg(1u) == 3312);
    assert(output.Metadata().EventIndex(1u) == 0u);
    assert(output.Metadata().DaughterCount(1u) == 3u);
    assert(output.Metadata().Flags(1u) & KFGpuCandidateValid);
    assert(output.Metadata().Flags(1u) & KFGpuCandidateEnergyFit);
    assert(output.Metadata().Topology(1u) == KFGpuGraphTopologyCompositeTrack);
    assert(output.Metadata().OutputClass(1u) == KFGpuGraphOutputSecondary);
    assert(output.Metadata().OperationStatus(1u)
           == KFGpuCandidateOperationAccepted);
    assert(output.Metadata().DirectDaughterCount(1u) == 2u);
    assert(output.Metadata().DirectFirstKind(1u) == KFGpuDirectDaughterCandidate);
    assert(output.Metadata().DirectFirstIndex(1u) == 0u);
    assert(output.Metadata().DirectSecondKind(1u) == KFGpuDirectDaughterInputTrack);
    const unsigned int daughterOffset = output.Metadata().DaughterOffset(1u);
    assert(output.Daughters().SourceId(daughterOffset) == 9101);
    assert(output.Daughters().SourceId(daughterOffset + 1u) == 9201);
    const int bachelorSource = output.Daughters().SourceId(daughterOffset + 2u);
    assert(bachelorSource == 9301 || bachelorSource == 9302);
    KFParticleGpuFitState xi;
    LoadCandidateFit(output, 1u, xi);
    assert(std::isfinite(xi.X()));
    assert(std::isfinite(xi.Px()));
    assert(std::isfinite(xi.Chi2()));

    for (unsigned int parentSize : {3u, KFParticleGpuV0TrackLineage::MaximumSize - 1u}) {
      auto parent = buffers.HostCandidates();
      StoreCandidateFit(lambda, parent, 0u);
      parent.Metadata().DaughterCount(0u) = parentSize;
      for (unsigned int i = 2u; i < parentSize; ++i) {
        parent.Daughters().SourceId(i) = 100000 + static_cast<int>(i);
      }
      parent.SizeData()[0] = 1u;
      parent.Daughters().SizeData()[0] = parentSize;
      parent.OverflowFlagsData()[0] = 0u;
      buffers.UploadCandidates();
      runtime.GetQueue().launch<KFParticleGpuV0TrackCompactCandidatePoolKernel>(
        xpu::n_threads(1u), MakeConstView(buffers.DeviceInputTracks()),
        MakeConstView(buffers.DeviceCandidates()), storage.fV0TrackTasks.get(),
        1u, storage.fV0TrackTaskCount.get(), buffers.DeviceCandidates());
      buffers.DownloadCandidates();
      const auto extended = MakeConstView(buffers.HostCandidates());
      assert(extended.Size() == 2u);
      assert(extended.Daughters().Size() == 2u * parentSize + 1u);
      assert(extended.Metadata().DaughterCount(1u) == parentSize + 1u);
      assert(extended.OverflowFlags() == 0u);
    }

    // Poison an unused reusable slot. A failed daughter reservation must clear
    // the fit as well as the metadata before a later generation can observe it.
    auto poisoned = buffers.HostCandidates();
    poisoned.SizeData()[0] = 1u;
    poisoned.Daughters().SizeData()[0] = KFParticleGpuV0TrackLineage::MaximumSize - 1u;
    poisoned.OverflowFlagsData()[0] = 0u;
    StoreCandidateFit(xi, poisoned, 1u);
    poisoned.Metadata().Flags(1u) = KFGpuCandidateValid;
    poisoned.Metadata().DaughterCount(1u) = 3u;
    buffers.UploadCandidates();
    const auto devicePool = buffers.DeviceCandidates();
    const KFParticleGpuDaughterStorageView limitedDaughters(
      devicePool.Daughters().SourceIdsData(), devicePool.Daughters().SizeData(),
      KFParticleGpuV0TrackLineage::MaximumSize - 1u);
    const KFParticleGpuCandidatePoolView limitedPool(
      devicePool.Fit(), devicePool.Metadata(), limitedDaughters, devicePool.SizeData(),
      devicePool.OverflowFlagsData(), devicePool.Capacity());
    runtime.GetQueue().launch<KFParticleGpuV0TrackCompactCandidatePoolKernel>(
      xpu::n_threads(1u), MakeConstView(buffers.DeviceInputTracks()),
      MakeConstView(devicePool), storage.fV0TrackTasks.get(), 1u,
      storage.fV0TrackTaskCount.get(), limitedPool);
    buffers.DownloadCandidates();
    const auto rejected = MakeConstView(buffers.HostCandidates());
    assert(rejected.Size() == 2u);
    assert(rejected.OverflowFlags() == DaughterCapacityExceeded);
    assert(rejected.Metadata().Flags(1u) == KFGpuCandidateBuildFailed);
    assert(rejected.Metadata().OperationStatus(1u) == KFGpuCandidateOperationRejected);
    assert(rejected.Metadata().DaughterCount(1u) == 0u);
    KFParticleGpuFitState cleared;
    KFParticleGpuFitState empty;
    empty.Initialize();
    LoadCandidateFit(rejected, 1u, cleared);
    for (unsigned int i = 0u; i < 8u; ++i) { assert(cleared.Parameter(i) == empty.Parameter(i)); }
    for (unsigned int i = 0u; i < 36u; ++i) { assert(cleared.Covariance(i) == empty.Covariance(i)); }
    int invalidLineage[KFParticleGpuGraphOperations::MaximumLineageSize];
    unsigned int invalidSize = 0u;
    assert(!KFParticleGpuGraphOperations::LoadLineage(rejected, 1u, invalidLineage, invalidSize));

    Pass("v0-track-compact-candidate",
         "device V0-track task compaction appends a full-field Xi candidate with three-source lineage and bounded worklist overflow");
  }

  std::vector<KFParticleGpuTwoDaughterTask> MakeExplicitTwoDaughterRoutingTasks(
    const KFParticleGpuDecayPlan& plan,
    KFParticleGpuBufferManager& buffers,
    unsigned int eventIndex)
  {
    std::vector<KFParticleGpuTwoDaughterTask> tasks;
    const KFParticleGpuEventDesc& event = buffers.HostEvents()[eventIndex];
    const KFParticleGpuConstInputTrackSoAView tracks =
      MakeConstView(buffers.HostInputTracks());
    for (std::size_t channelIndex = 0u;
         channelIndex < plan.NumberOfTwoDaughterChannels();
         ++channelIndex) {
      const KFParticleGpuTwoDaughterTaskSource source =
        MakeTwoDaughterTaskSource(plan.TwoDaughterChannel(channelIndex), eventIndex);
      const KFParticleGpuRange firstRange =
        ResolveTaskSourceRange(event, source.firstTrackSet, source.firstSpecies);
      const KFParticleGpuRange secondRange =
        ResolveTaskSourceRange(event, source.secondTrackSet, source.secondSpecies);
      const unsigned int pairCount = firstRange.size * secondRange.size;
      for (unsigned int pair = 0u; pair < pairCount; ++pair) {
        KFParticleGpuTwoDaughterTask task;
        FillTwoDaughterTask(firstRange, secondRange, pair, source, task);
        if (PassTwoDaughterTaskSourceCuts(
              tracks, source, task.firstTrack, task.secondTrack)) {
          tasks.push_back(task);
        }
      }
    }
    return tasks;
  }

  bool SameTwoDaughterTaskKey(const KFParticleGpuTwoDaughterTask& first,
                              const KFParticleGpuTwoDaughterTask& second)
  {
    return first.channelId == second.channelId
           && first.firstTrack == second.firstTrack
           && first.secondTrack == second.secondTrack
           && first.eventIndex == second.eventIndex;
  }

  void ExpectTwoDaughterFitAndCovarianceClose(
    KFParticleGpuRuntime& runtime,
    KFParticleGpuBufferManager& buffers,
    const KFParticleGpuFitState& actual,
    const KFParticleGpuFitState& expected,
    const KFParticleGpuTwoDaughterTask& task,
    unsigned int candidateIndex,
    const char* scenario)
  {
    ExpectFullFieldEnergyFitClose(
      actual, expected, task.motherPdg, "fused-two-daughter");

    constexpr float covarianceAbsoluteTolerance = 5.e-3f;
    constexpr float covarianceRelativeTolerance = 2.e-5f;
    bool exceededLegacyAbsoluteTolerance = false;
    bool exceededScaledTolerance = false;
    for (int component = 0;
         component < KFParticleGpuFitState::NumberOfCovarianceElements;
         ++component) {
      const float gpu = actual.Covariance(component);
      const float host = expected.Covariance(component);
      exceededLegacyAbsoluteTolerance = exceededLegacyAbsoluteTolerance
                                        || !AlmostEqual(
                                          gpu, host, covarianceAbsoluteTolerance);
      exceededScaledTolerance = exceededScaledTolerance
                                || !AlmostEqualScaled(gpu,
                                                      host,
                                                      covarianceAbsoluteTolerance,
                                                      covarianceRelativeTolerance);
    }

    if (exceededLegacyAbsoluteTolerance || exceededScaledTolerance) {
      std::cerr << std::setprecision(9)
                << "DETAIL fused-two-daughter-covariance"
                << " scenario=" << scenario
                << " candidate=" << candidateIndex
                << " channel=" << task.channelId
                << " event=" << task.eventIndex
                << " first-track=" << task.firstTrack
                << " second-track=" << task.secondTrack
                << " mother=" << task.motherPdg
                << " legacy-absolute-close="
                << (exceededLegacyAbsoluteTolerance ? 0 : 1)
                << " scaled-close=" << (exceededScaledTolerance ? 0 : 1)
                << " absolute-tolerance=" << covarianceAbsoluteTolerance
                << " relative-tolerance=" << covarianceRelativeTolerance << '\n';
      for (int parameter = 0;
           parameter < KFParticleGpuFitState::NumberOfParameters;
           ++parameter) {
        const float gpu = actual.Parameter(parameter);
        const float host = expected.Parameter(parameter);
        const float scale = std::max(std::fabs(gpu), std::fabs(host));
        std::cerr << "DETAIL fused-two-daughter-parameter index=" << parameter
                  << " gpu=" << gpu
                  << " host=" << host
                  << " delta=" << (gpu - host)
                  << " relative-delta="
                  << (scale > 0.f ? std::fabs(gpu - host) / scale : 0.f)
                  << '\n';
      }
      for (int component = 0;
           component < KFParticleGpuFitState::NumberOfCovarianceElements;
           ++component) {
        const float gpu = actual.Covariance(component);
        const float host = expected.Covariance(component);
        const float scale = std::max(std::fabs(gpu), std::fabs(host));
        const float effectiveTolerance = covarianceAbsoluteTolerance
                                         + covarianceRelativeTolerance * scale;
        std::cerr << "DETAIL fused-two-daughter-covariance-component index="
                  << component
                  << " gpu=" << gpu
                  << " host=" << host
                  << " delta=" << (gpu - host)
                  << " relative-delta="
                  << (scale > 0.f ? std::fabs(gpu - host) / scale : 0.f)
                  << " effective-tolerance=" << effectiveTolerance
                  << " legacy-absolute-close="
                  << (AlmostEqual(gpu, host, covarianceAbsoluteTolerance) ? 1 : 0)
                  << " scaled-close="
                  << (AlmostEqualScaled(gpu,
                                        host,
                                        covarianceAbsoluteTolerance,
                                        covarianceRelativeTolerance)
                        ? 1 : 0)
                  << '\n';
      }
      std::cerr << "DETAIL fused-two-daughter-fit-metadata"
                << " chi2=" << actual.Chi2() << '/' << expected.Chi2()
                << " ndf=" << actual.NDF() << '/' << expected.NDF()
                << " q=" << actual.Q() << '/' << expected.Q()
                << " sum-mass=" << actual.SumDaughterMass() << '/'
                << expected.SumDaughterMass()
                << " mass-hypothesis=" << actual.MassHypo() << '/'
                << expected.MassHypo() << '\n';

      if (exceededScaledTolerance
          && task.transportMode == KFGpuTransportFullField) {
        KFParticleGpuFullFieldTwoDaughterTrace hostTrace;
        const bool hostTraceBuilt = BuildHostFullFieldTrace(
          MakeConstView(buffers.HostInputTracks()), task, hostTrace);
        KFParticleGpuFullFieldTwoDaughterTrace directDeviceTrace;
        const KFParticleGpuFitState directDevice = RunDirectFullFieldTask(
          runtime, buffers, task, directDeviceTrace);
        std::cerr << "DETAIL fused-two-daughter-trace-oracle"
                  << " scenario=" << scenario
                  << " host-built=" << (hostTraceBuilt ? 1 : 0)
                  << " fused-vs-direct-needs-trace="
                  << (NeedsFullFieldTrace(actual, directDevice, task.motherPdg) ? 1 : 0)
                  << " direct-vs-host-needs-trace="
                  << (NeedsFullFieldTrace(directDevice, expected, task.motherPdg) ? 1 : 0)
                  << '\n';
        DumpFullFieldTask("fused-failure", task);
        DumpFullFieldTraceState("fused-vs-direct-device", actual, directDevice);
        if (hostTraceBuilt) {
          DumpFullFieldTrace(directDeviceTrace, hostTrace);
        }
      }
    }
    assert(!exceededScaledTolerance);
  }

  void VerifyFusedTwoDaughterOutput(
    KFParticleGpuRuntime& runtime,
    KFParticleGpuBufferManager& buffers,
    const std::vector<KFParticleGpuTwoDaughterTask>& expectedTasks,
    const KFParticleGpuTwoDaughterFusedResult& result,
    const char* scenario)
  {
    assert(result.routing.acceptedTasks == expectedTasks.size());
    assert(result.routing.storedTasks == expectedTasks.size());
    assert(result.candidates.size == expectedTasks.size());
    assert(result.candidates.daughterSize == 2u * expectedTasks.size());
    assert(result.candidates.overflowFlags == 0u);

    const KFParticleGpuTwoDaughterRoutingView routing =
      buffers.HostTwoDaughterRouting();
    const KFParticleGpuTwoDaughterRoutedTask* routedTasks =
      buffers.HostTwoDaughterRoutedTasks();
    std::vector<bool> matchedTasks(expectedTasks.size(), false);
    for (unsigned int routedIndex = 0u;
         routedIndex < result.routing.storedTasks;
         ++routedIndex) {
      const KFParticleGpuTwoDaughterRoutedTask& routed = routedTasks[routedIndex];
      assert(routed.descriptorIndex < routing.DescriptorCount());
      KFParticleGpuTwoDaughterTask key;
      key.channelId = routing.Descriptors()[routed.descriptorIndex].channelId;
      key.firstTrack = routed.firstTrackIndex;
      key.secondTrack = routed.secondTrackIndex;
      key.eventIndex = routed.eventIndex;
      bool matched = false;
      for (std::size_t expected = 0u; expected < expectedTasks.size(); ++expected) {
        if (!matchedTasks[expected]
            && SameTwoDaughterTaskKey(key, expectedTasks[expected])) {
          matchedTasks[expected] = true;
          matched = true;
          break;
        }
      }
      assert(matched);
    }
    for (bool matched : matchedTasks) {
      assert(matched);
    }

    const KFParticleGpuConstInputTrackSoAView tracks =
      MakeConstView(buffers.HostInputTracks());
    const KFParticleGpuConstCandidatePoolView candidates =
      MakeConstView(buffers.HostCandidates());
    const KFParticleGpuCandidateDescriptorIndexView descriptorIndices =
      buffers.HostCandidateDescriptorIndices();
    std::vector<bool> matchedCandidates(expectedTasks.size(), false);
    for (unsigned int candidateIndex = result.candidates.offset;
         candidateIndex < result.candidates.End();
         ++candidateIndex) {
      const unsigned int daughterOffset =
        candidates.Metadata().DaughterOffset(candidateIndex);
      assert(candidates.Metadata().DaughterCount(candidateIndex) == 2u);
      assert(descriptorIndices.Index(candidateIndex) < routing.DescriptorCount());
      const KFParticleGpuTwoDaughterRoutingDescriptor& descriptor =
        routing.Descriptors()[descriptorIndices.Index(candidateIndex)];
      assert(descriptor.channelId == candidates.Metadata().ChannelId(candidateIndex));

      bool matched = false;
      for (std::size_t expected = 0u; expected < expectedTasks.size(); ++expected) {
        const KFParticleGpuTwoDaughterTask& task = expectedTasks[expected];
        if (matchedCandidates[expected]
            || task.channelId != candidates.Metadata().ChannelId(candidateIndex)
            || task.eventIndex != candidates.Metadata().EventIndex(candidateIndex)
            || tracks.SourceId(task.firstTrack)
                 != candidates.Daughters().SourceId(daughterOffset)
            || tracks.SourceId(task.secondTrack)
                 != candidates.Daughters().SourceId(daughterOffset + 1u)) {
          continue;
        }

        KFParticleGpuFitState reference;
        assert(BuildTwoDaughterCandidate(tracks, task, reference));
        KFParticleGpuFitState actual;
        LoadCandidateFit(candidates, candidateIndex, actual);
        ExpectTwoDaughterFitAndCovarianceClose(
          runtime, buffers, actual, reference, task, candidateIndex, scenario);
        unsigned int expectedFlags = TwoDaughterCandidateFlags(task);
        if (!PassTwoDaughterPostBuildSelection(task, reference)) {
          expectedFlags |= static_cast<unsigned int>(KFGpuCandidateSelectionRejected);
        }
        assert(candidates.Metadata().Pdg(candidateIndex) == task.motherPdg);
        assert(candidates.Metadata().PrimaryVertexIndex(candidateIndex)
               == task.primaryVertexIndex);
        assert(candidates.Metadata().Flags(candidateIndex) == expectedFlags);
        matchedCandidates[expected] = true;
        matched = true;
        break;
      }
      assert(matched);
    }
    for (bool matched : matchedCandidates) {
      assert(matched);
    }
  }

  void TestFusedTwoDaughterRoutingAndConstruction(
    KFParticleGpuRuntime& runtime,
    KFParticleGpuBufferManager& buffers)
  {
    KFParticleGpuSteering& steering = runtime.GetSteering();
    KFParticleGpuDecayPlan& plan = steering.GetDecayPlan();
    plan.Clear();
    AddDefaultV0ChannelsWithoutCpuPairGate(plan);

    const auto prepareDefaultFixture = [&]() {
      FillDefaultV0Fixture(buffers);
      KFParticleGpuInputTrackSoAView tracks = buffers.HostInputTracks();
      for (unsigned int track = 0u; track < buffers.TrackSize(); ++track) {
        StoreDiagnosticField(tracks, track);
      }
      buffers.UploadInput();
      buffers.ResetCandidates();
    };

    prepareDefaultFixture();
    const std::vector<KFParticleGpuTwoDaughterTask> expectedDefault =
      MakeExplicitTwoDaughterRoutingTasks(plan, buffers, 0u);
    assert(expectedDefault.size() == 4u);
    const KFParticleGpuTwoDaughterFusedResult atomic =
      steering.RunTwoDaughterFusedStage(
        0u, 8u, KFGpuTwoDaughterRoutingAtomic);
    assert(atomic.groupLaunches == 2u);
    assert(atomic.descriptorCount == 3u);
    assert(atomic.routing.visitedPairs == 3u);
    assert(atomic.routing.activeChannelBits == 4u);
    assert(atomic.routing.blockReservations == atomic.routing.acceptedTasks);
    VerifyFusedTwoDaughterOutput(
      runtime, buffers, expectedDefault, atomic, "default-atomic");
    {
      const KFParticleGpuTwoDaughterRoutingView routing =
        buffers.HostTwoDaughterRouting();
      for (unsigned int descriptor = 0u; descriptor < 3u; ++descriptor) {
        const unsigned int expectedCount = descriptor == 1u ? 2u : 1u;
        assert(routing.ChannelVisitedCounters()[descriptor] == expectedCount);
        assert(routing.ChannelAcceptedCounters()[descriptor] == expectedCount);
        assert(routing.ChannelStoredCounters()[descriptor] == expectedCount);
        assert(routing.ChannelConstructedCounters()[descriptor] == expectedCount);
      }
    }

    prepareDefaultFixture();
    const KFParticleGpuTwoDaughterFusedResult blockScan =
      steering.RunTwoDaughterFusedStage(
        0u, 8u, KFGpuTwoDaughterRoutingBlockScan);
    assert(blockScan.routing.visitedPairs == atomic.routing.visitedPairs);
    assert(blockScan.routing.activeChannelBits == atomic.routing.activeChannelBits);
    assert(blockScan.routing.acceptedTasks == atomic.routing.acceptedTasks);
    assert(blockScan.routing.storedTasks == atomic.routing.storedTasks);
    assert(blockScan.routing.blockReservations > 0u);
    assert(blockScan.routing.blockReservations <= atomic.routing.blockReservations);
    VerifyFusedTwoDaughterOutput(
      runtime, buffers, expectedDefault, blockScan, "default-block-scan");

    KFParticleGpuTwoDaughterRoutingPlan defaultRouting;
    defaultRouting.Compile(plan);
    KFParticleGpuChannelMask k0Only;
    k0Only.Set(0u);
    buffers.SetTwoDaughterRoutingEnabledChannels(k0Only);
    prepareDefaultFixture();
    const KFParticleGpuTwoDaughterFusedResult disabled =
      steering.RunTwoDaughterFusedStage(0u, 8u);
    assert(disabled.routing.visitedPairs == 3u);
    assert(disabled.routing.activeChannelBits == 1u);
    assert(disabled.routing.acceptedTasks == 1u);
    assert(disabled.candidates.size == 1u);
    assert(buffers.HostCandidates().Metadata().ChannelId(0u)
           == KFGpuChannelK0ShortToPiPlusPiMinus);
    buffers.SetTwoDaughterRoutingEnabledChannels(defaultRouting.EnabledChannels());

    KFParticleGpuTwoDaughterChannel alternateK0 =
      plan.TwoDaughterChannel(0u);
    alternateK0.channelId = 1601u;
    alternateK0.transportMode = KFGpuTransportStraightLine;
    plan.AddTwoDaughterChannel(alternateK0);
    prepareDefaultFixture();
    const std::vector<KFParticleGpuTwoDaughterTask> expectedMulti =
      MakeExplicitTwoDaughterRoutingTasks(plan, buffers, 0u);
    assert(expectedMulti.size() == 5u);
    const KFParticleGpuTwoDaughterFusedResult multiBit =
      steering.RunTwoDaughterFusedStage(0u, 8u);
    assert(multiBit.routing.visitedPairs == 3u);
    assert(multiBit.routing.activeChannelBits == 5u);
    VerifyFusedTwoDaughterOutput(
      runtime, buffers, expectedMulti, multiBit, "multi-channel");

    prepareDefaultFixture();
    const KFParticleGpuTwoDaughterFusedResult taskOverflow =
      steering.RunTwoDaughterFusedStage(0u, 2u);
    assert(taskOverflow.routing.acceptedTasks == 5u);
    assert(taskOverflow.routing.storedTasks == 2u);
    assert(taskOverflow.routing.Truncated());
    assert(taskOverflow.routing.overflowFlags == CandidateCapacityExceeded);
    assert(taskOverflow.candidates.size == 2u);
    assert(taskOverflow.candidates.HasOverflow(CandidateCapacityExceeded));

    plan.Clear();
    KFParticleGpuTwoDaughterChannel rejectedBySource =
      MakeK0ShortToPiPlusPiMinusChannel(1602u);
    rejectedBySource.minFirstPixelHits = 100;
    plan.AddTwoDaughterChannel(rejectedBySource);
    prepareDefaultFixture();
    const KFParticleGpuTwoDaughterFusedResult sourceRejected =
      steering.RunTwoDaughterFusedStage(0u, 4u);
    assert(sourceRejected.routing.visitedPairs == 1u);
    assert(sourceRejected.routing.activeChannelBits == 1u);
    assert(sourceRejected.routing.acceptedTasks == 0u);
    assert(sourceRejected.routing.storedTasks == 0u);
    assert(sourceRejected.candidates.Empty());
    assert(buffers.HostTwoDaughterRouting().ChannelVisitedCounters()[0] == 1u);
    assert(buffers.HostTwoDaughterRouting().ChannelAcceptedCounters()[0] == 0u);

    plan.Clear();
    KFParticleGpuTwoDaughterChannel overflowK0 =
      MakeK0ShortToPiPlusPiMinusChannel();
    overflowK0.maxDaughterDistance = -1.f;
    plan.AddTwoDaughterChannel(overflowK0);
    prepareDefaultFixture();
    {
      KFParticleGpuCandidatePoolView candidates = buffers.HostCandidates();
      candidates.SizeData()[0] = candidates.Capacity();
    }
    buffers.UploadCandidates();
    const KFParticleGpuTwoDaughterFusedResult candidateOverflow =
      steering.RunTwoDaughterFusedStage(0u, 4u);
    assert(candidateOverflow.routing.acceptedTasks == 1u);
    assert(candidateOverflow.routing.storedTasks == 1u);
    assert(candidateOverflow.candidates.size == 0u);
    assert(candidateOverflow.candidates.HasOverflow(CandidateCapacityExceeded));
    assert(buffers.HostTwoDaughterRouting().ChannelConstructedCounters()[0] == 0u);

    prepareDefaultFixture();
    {
      KFParticleGpuCandidatePoolView candidates = buffers.HostCandidates();
      candidates.Daughters().SizeData()[0] = candidates.Daughters().Capacity() - 1u;
    }
    buffers.UploadCandidates();
    const KFParticleGpuTwoDaughterFusedResult daughterOverflow =
      steering.RunTwoDaughterFusedStage(0u, 4u);
    assert(daughterOverflow.routing.acceptedTasks == 1u);
    assert(daughterOverflow.routing.storedTasks == 1u);
    assert(daughterOverflow.candidates.size == 1u);
    assert(daughterOverflow.candidates.daughterSize == 0u);
    assert(daughterOverflow.candidates.HasOverflow(DaughterCapacityExceeded));
    assert(buffers.HostCandidates().Metadata().Flags(0u) & KFGpuCandidateBuildFailed);
    assert(buffers.HostTwoDaughterRouting().ChannelConstructedCounters()[0] == 0u);

    prepareDefaultFixture();
    {
      KFParticleGpuInputTrackSoAView tracks = buffers.HostInputTracks();
      tracks.Numerical().Parameter(0, 0u) =
        std::numeric_limits<float>::quiet_NaN();
      tracks.Numerical().Covariance(0, 0u) =
        std::numeric_limits<float>::quiet_NaN();
    }
    const std::vector<KFParticleGpuTwoDaughterTask> buildRejectedTasks =
      MakeExplicitTwoDaughterRoutingTasks(plan, buffers, 0u);
    assert(buildRejectedTasks.size() == 1u);
    KFParticleGpuFitState rejectedReference;
    assert(!BuildTwoDaughterCandidate(
      MakeConstView(buffers.HostInputTracks()),
      buildRejectedTasks[0],
      rejectedReference));
    buffers.UploadInput();
    const KFParticleGpuTwoDaughterFusedResult buildRejected =
      steering.RunTwoDaughterFusedStage(0u, 4u);
    assert(buildRejected.routing.acceptedTasks == 1u);
    assert(buildRejected.routing.storedTasks == 1u);
    assert(buildRejected.candidates.Empty());
    assert(buffers.HostTwoDaughterRouting().ChannelConstructedCounters()[0] == 0u);

    plan.Clear();
    KFParticleGpuTwoDaughterChannel batchK0 =
      MakeK0ShortToPiPlusPiMinusChannel();
    batchK0.maxDaughterDistance = -1.f;
    plan.AddTwoDaughterChannel(batchK0);
    FillBatchSelectionFixture(buffers);
    buffers.UploadInput();
    for (unsigned int eventIndex = 0u; eventIndex < 2u; ++eventIndex) {
      buffers.ResetCandidates();
      const std::vector<KFParticleGpuTwoDaughterTask> expectedEvent =
        MakeExplicitTwoDaughterRoutingTasks(plan, buffers, eventIndex);
      assert(expectedEvent.size() == 1u);
      const KFParticleGpuTwoDaughterFusedResult eventResult =
        steering.RunTwoDaughterFusedStage(eventIndex, 2u);
      assert(eventResult.routing.visitedPairs == 1u);
      assert(eventResult.routing.activeChannelBits == 1u);
      VerifyFusedTwoDaughterOutput(
        runtime, buffers, expectedEvent, eventResult, "batch-event");
      assert(buffers.HostCandidates().Metadata().EventIndex(0u) == eventIndex);
    }

    bool invalidModeRejected = false;
    try {
      steering.RunTwoDaughterFusedStage(
        0u, 1u, static_cast<KFParticleGpuTwoDaughterRoutingMode>(99u));
    }
    catch (const std::invalid_argument&) {
      invalidModeRejected = true;
    }
    assert(invalidModeRejected);

    plan.Clear();
    Pass("two-daughter-fused-routing-construction",
         "Stage 16.2 atomic and block-scan routes match explicit default-V0 tasks, fits, covariance, lineage, sidecar, counters, and bounded failure semantics");
  }

  void TestFusedV0TrackRoutingAndConstruction(KFParticleGpuRuntime& runtime,
                                               KFParticleGpuBufferManager& buffers)
  {
    KFParticleGpuBufferCapacities requested = buffers.Capacities();
    requested.tracks = requested.tracks > 4u ? requested.tracks : 4u;
    requested.events = requested.events > 2u ? requested.events : 2u;
    requested.candidates = requested.candidates > 12u ? requested.candidates : 12u;
    requested.daughterIds = requested.daughterIds > 36u ? requested.daughterIds : 36u;
    requested.selectedCandidates = requested.selectedCandidates > 2u
                                     ? requested.selectedCandidates : 2u;
    requested.v0TrackRoutedTasks = requested.v0TrackRoutedTasks > 8u
                                     ? requested.v0TrackRoutedTasks : 8u;
    requested.nonhomogeneousField = true;
    buffers.EnsureCapacity(requested);
    buffers.SetInputSizes(4u, 0u, 2u);

    KFParticleGpuInputTrackSoAView tracks = buffers.HostInputTracks();
    StoreSyntheticTrack(
      tracks, 0u, 0.2f, -0.1f, 0.1f, -0.3f, 0.4f, 0.9f, -211, -1, 15101);
    StoreSyntheticTrack(
      tracks, 1u, -0.1f, 0.2f, -0.2f, -0.2f, 0.3f, 1.0f, -321, -1, 15102);
    StoreSyntheticTrack(
      tracks, 2u, 0.3f, 0.1f, -0.1f, 0.3f, -0.4f, 0.8f, 211, 1, 15201);
    StoreSyntheticTrack(
      tracks, 3u, -0.2f, -0.2f, 0.2f, 0.2f, -0.3f, 1.1f, 321, 1, 15202);
    for (unsigned int index = 0u; index < 4u; ++index) {
      StoreDiagnosticField(tracks, index);
    }

    KFParticleGpuEventDesc* events = buffers.HostEvents();
    events[0] = KFParticleGpuEventDesc();
    events[0].eventId = 1511u;
    events[0].TrackSet(SecondaryNegativeFirst).tracks = KFParticleGpuRange(0u, 2u);
    events[0].TrackSet(SecondaryNegativeFirst).Species(Pion) = KFParticleGpuRange(0u, 1u);
    events[0].TrackSet(SecondaryNegativeFirst).Species(Kaon) = KFParticleGpuRange(1u, 1u);
    events[1] = KFParticleGpuEventDesc();
    events[1].eventId = 1512u;
    events[1].TrackSet(SecondaryPositiveFirst).tracks = KFParticleGpuRange(2u, 2u);
    events[1].TrackSet(SecondaryPositiveFirst).Species(Pion) = KFParticleGpuRange(2u, 1u);
    events[1].TrackSet(SecondaryPositiveFirst).Species(Kaon) = KFParticleGpuRange(3u, 1u);

    KFParticleGpuFitState lambda;
    lambda.Initialize();
    lambda.X() = 0.f;
    lambda.Y() = 0.f;
    lambda.Z() = 0.f;
    lambda.Px() = 0.4f;
    lambda.Py() = 0.2f;
    lambda.Pz() = 1.1f;
    lambda.E() = 1.6f;
    lambda.Q() = 0;
    lambda.NDF() = 1;
    lambda.Chi2() = 0.5f;
    for (int component = 0;
         component < KFParticleGpuFitState::NumberOfCovarianceElements;
         ++component) {
      lambda.Covariance(component) = 0.f;
    }
    lambda.Covariance(0, 0) = 0.3f;
    lambda.Covariance(1, 1) = 0.3f;
    lambda.Covariance(2, 2) = 0.3f;
    lambda.Covariance(3, 3) = 0.1f;
    lambda.Covariance(4, 4) = 0.1f;
    lambda.Covariance(5, 5) = 0.1f;
    lambda.SumDaughterMass() = kCpuReferenceLambdaMass;
    KFParticleGpuFitState antiLambda = lambda;
    antiLambda.Px() = -lambda.Px();
    antiLambda.Py() = -lambda.Py();

    const auto storeRawV0s = [&]() {
      KFParticleGpuCandidatePoolView candidates = buffers.HostCandidates();
      StoreCandidateFit(lambda, candidates, 0u);
      StoreCandidateFit(antiLambda, candidates, 1u);
      for (unsigned int index = 0u; index < 2u; ++index) {
        candidates.Metadata().PrimaryVertexIndex(index) = -1;
        candidates.Metadata().DaughterOffset(index) = 2u * index;
        candidates.Metadata().DaughterCount(index) = 2u;
        candidates.Metadata().Flags(index) = KFGpuCandidateValid;
      }
      candidates.Metadata().Pdg(0u) = 3122;
      candidates.Metadata().EventIndex(0u) = 0u;
      candidates.Metadata().ChannelId(0u) = KFGpuChannelLambdaToProtonPiMinus;
      candidates.Daughters().SourceId(0u) = 15001;
      candidates.Daughters().SourceId(1u) = 15002;
      candidates.Metadata().Pdg(1u) = -3122;
      candidates.Metadata().EventIndex(1u) = 1u;
      candidates.Metadata().ChannelId(1u) = KFGpuChannelAntiLambdaToAntiProtonPiPlus;
      candidates.Daughters().SourceId(2u) = 15003;
      candidates.Daughters().SourceId(3u) = 15004;
      candidates.SizeData()[0] = 2u;
      candidates.Daughters().SizeData()[0] = 4u;
      candidates.OverflowFlagsData()[0] = 0u;
    };
    storeRawV0s();

    KFParticleGpuSelectedCandidateIndexView selected = buffers.HostSelectedCandidates();
    selected.Index(0u) = 0u;
    selected.ChannelId(0u) = KFGpuChannelLambdaToProtonPiMinus;
    selected.Index(1u) = 1u;
    selected.ChannelId(1u) = KFGpuChannelAntiLambdaToAntiProtonPiPlus;
    selected.SizeData()[0] = 2u;
    selected.OverflowFlagsData()[0] = 0u;

    KFParticleGpuSteering& steering = runtime.GetSteering();
    KFParticleGpuDecayPlan& plan = steering.GetDecayPlan();
    plan.Clear();
    AddDefaultV0TrackCascadeChannels(plan);
    KFParticleGpuV0TrackCascadeChannel alternateXi =
      plan.V0TrackCascadeChannel(0u);
    alternateXi.channelId = 1513u;
    plan.AddV0TrackCascadeChannel(alternateXi);

    buffers.UploadInput();
    buffers.UploadCandidates();
    buffers.UploadSelectedCandidates();
    const KFParticleGpuSelectedCandidateRange selectedRange{0u, 2u, 0u};

    const auto makeExplicitTasks = [&](unsigned int eventIndex) {
      std::vector<KFParticleGpuV0TrackTask> tasks;
      const KFParticleGpuSelectedV0View selectedV0s = MakeSelectedV0View(
        MakeConstView(buffers.HostCandidates()),
        MakeConstView(buffers.HostSelectedCandidates()));
      const KFParticleGpuV0TrackInputView input(
        selectedV0s, MakeConstView(buffers.HostInputTracks()));
      for (std::size_t channelIndex = 0u;
           channelIndex < plan.NumberOfV0TrackCascadeChannels();
           ++channelIndex) {
        const KFParticleGpuV0TrackChannel channel = MakeV0TrackTaskChannel(
          plan.V0TrackCascadeChannel(channelIndex), events[eventIndex], eventIndex);
        for (unsigned int selectedIndex = selectedRange.offset;
             selectedIndex < selectedRange.End();
             ++selectedIndex) {
          for (unsigned int bachelorIndex = channel.bachelorTracks.offset;
               bachelorIndex < channel.bachelorTracks.End();
               ++bachelorIndex) {
            KFParticleGpuV0TrackTask task;
            KFParticleGpuV0TrackLineage lineage;
            if (KFParticleGpuV0Track::IsAccepted(KFParticleGpuV0Track::ResolveTask(
                  input, channel, selectedRange, selectedIndex, bachelorIndex,
                  static_cast<unsigned int>(tasks.size()), 64u, task, lineage))) {
              tasks.push_back(task);
            }
          }
        }
      }
      return tasks;
    };

    const auto verifyFusedResult =
      [&](unsigned int eventIndex,
          const std::vector<KFParticleGpuV0TrackTask>& expectedTasks,
          const KFParticleGpuV0TrackFusedResult& result) {
        assert(result.eventIndex == eventIndex);
        assert(result.routing.acceptedTasks == expectedTasks.size());
        assert(result.routing.storedTasks == expectedTasks.size());
        assert(!result.routing.Truncated());
        assert(result.routing.overflowFlags == 0u);
        assert(result.candidates.size == expectedTasks.size());
        assert(result.candidates.daughterSize == 3u * expectedTasks.size());

        const KFParticleGpuV0TrackRoutingView routing = buffers.HostV0TrackRouting();
        KFParticleGpuV0TrackRoutedTask* routed = buffers.HostV0TrackRoutedTasks();
        std::vector<bool> matchedTasks(expectedTasks.size(), false);
        for (unsigned int index = 0u; index < result.routing.storedTasks; ++index) {
          assert(routed[index].descriptorIndex < routing.DescriptorCount());
          const unsigned int channelId =
            routing.Descriptors()[routed[index].descriptorIndex].channelId;
          bool matched = false;
          for (std::size_t expected = 0u; expected < expectedTasks.size(); ++expected) {
            if (!matchedTasks[expected]
                && expectedTasks[expected].channelId == channelId
                && expectedTasks[expected].selectedV0Index == routed[index].selectedV0Index
                && expectedTasks[expected].bachelorTrackIndex
                     == routed[index].bachelorTrackIndex
                && expectedTasks[expected].eventIndex == routed[index].eventIndex) {
              matchedTasks[expected] = true;
              matched = true;
              break;
            }
          }
          assert(matched);
        }
        for (bool matched : matchedTasks) {
          assert(matched);
        }

        const KFParticleGpuConstCandidatePoolView candidates =
          MakeConstView(buffers.HostCandidates());
        std::vector<bool> matchedCandidates(expectedTasks.size(), false);
        for (unsigned int candidateIndex = result.candidates.offset;
             candidateIndex < result.candidates.End();
             ++candidateIndex) {
          const unsigned int daughterOffset =
            candidates.Metadata().DaughterOffset(candidateIndex);
          bool matched = false;
          for (std::size_t expected = 0u; expected < expectedTasks.size(); ++expected) {
            KFParticleGpuFitState reference;
            KFParticleGpuV0TrackLineage lineage;
            if (matchedCandidates[expected]
                || !KFParticleGpuV0Track::BuildCandidate(
                  candidates, MakeConstView(buffers.HostInputTracks()),
                  expectedTasks[expected], reference, lineage)
                || candidates.Metadata().ChannelId(candidateIndex)
                     != expectedTasks[expected].channelId
                || candidates.Metadata().Pdg(candidateIndex)
                     != expectedTasks[expected].motherPdg
                || candidates.Metadata().EventIndex(candidateIndex)
                     != expectedTasks[expected].eventIndex
                || candidates.Daughters().SourceId(daughterOffset)
                     != lineage.firstV0DaughterSourceId
                || candidates.Daughters().SourceId(daughterOffset + 1u)
                     != lineage.secondV0DaughterSourceId
                || candidates.Daughters().SourceId(daughterOffset + 2u)
                     != lineage.bachelorSourceId) {
              continue;
            }
            KFParticleGpuFitState actual;
            LoadCandidateFit(candidates, candidateIndex, actual);
            ExpectFullFieldEnergyFitClose(
              actual, reference, expectedTasks[expected].motherPdg, "fused-v0-track");
            matchedCandidates[expected] = true;
            matched = true;
            break;
          }
          assert(matched);
        }
        for (bool matched : matchedCandidates) {
          assert(matched);
        }
      };

    const auto verifyChannelCounters =
      [&](const std::vector<unsigned int>& expected) {
        const KFParticleGpuV0TrackRoutingView routing =
          buffers.HostV0TrackRouting();
        assert(expected.size() == routing.DescriptorCount());
        for (unsigned int index = 0u; index < routing.DescriptorCount(); ++index) {
          assert(routing.ChannelVisitedCounters()[index] == expected[index]);
          assert(routing.ChannelAcceptedCounters()[index] == expected[index]);
          assert(routing.ChannelStoredCounters()[index] == expected[index]);
          assert(routing.ChannelConstructedCounters()[index] == expected[index]);
        }
      };

    const std::vector<KFParticleGpuV0TrackTask> event0Tasks = makeExplicitTasks(0u);
    assert(event0Tasks.size() == 3u);
    const std::vector<unsigned int> event0Channels{1u, 0u, 1u, 0u, 1u};
    const KFParticleGpuV0TrackFusedResult event0Atomic =
      steering.RunV0TrackFusedStage(
        0u, selectedRange, 8u, KFGpuV0TrackRoutingAtomic);
    assert(event0Atomic.routing.visitedPairs == 2u);
    assert(event0Atomic.routing.activeChannelBits == 3u);
    assert(event0Atomic.routing.blockReservations
           == event0Atomic.routing.acceptedTasks);
    verifyFusedResult(0u, event0Tasks, event0Atomic);
    verifyChannelCounters(event0Channels);

    storeRawV0s();
    buffers.UploadCandidates();
    const KFParticleGpuV0TrackFusedResult event0Scan =
      steering.RunV0TrackFusedStage(
        0u, selectedRange, 8u, KFGpuV0TrackRoutingBlockScan);
    assert(event0Scan.routing.visitedPairs == event0Atomic.routing.visitedPairs);
    assert(event0Scan.routing.activeChannelBits
           == event0Atomic.routing.activeChannelBits);
    assert(event0Scan.routing.acceptedTasks
           == event0Atomic.routing.acceptedTasks);
    assert(event0Scan.routing.storedTasks == event0Atomic.routing.storedTasks);
    assert(event0Scan.routing.blockReservations > 0u);
    assert(event0Scan.routing.blockReservations
           <= event0Atomic.routing.blockReservations);
    verifyFusedResult(0u, event0Tasks, event0Scan);
    verifyChannelCounters(event0Channels);

    const std::vector<KFParticleGpuV0TrackTask> event1Tasks = makeExplicitTasks(1u);
    assert(event1Tasks.size() == 2u);
    const std::vector<unsigned int> event1Channels{0u, 1u, 0u, 1u, 0u};
    storeRawV0s();
    buffers.UploadCandidates();
    const KFParticleGpuV0TrackFusedResult event1Atomic =
      steering.RunV0TrackFusedStage(
        1u, selectedRange, 8u, KFGpuV0TrackRoutingAtomic);
    assert(event1Atomic.routing.visitedPairs == 2u);
    assert(event1Atomic.routing.activeChannelBits == 2u);
    assert(event1Atomic.routing.blockReservations
           == event1Atomic.routing.acceptedTasks);
    verifyFusedResult(1u, event1Tasks, event1Atomic);
    verifyChannelCounters(event1Channels);

    storeRawV0s();
    buffers.UploadCandidates();
    const KFParticleGpuV0TrackFusedResult event1Scan =
      steering.RunV0TrackFusedStage(
        1u, selectedRange, 8u, KFGpuV0TrackRoutingBlockScan);
    assert(event1Scan.routing.visitedPairs == event1Atomic.routing.visitedPairs);
    assert(event1Scan.routing.activeChannelBits
           == event1Atomic.routing.activeChannelBits);
    assert(event1Scan.routing.acceptedTasks
           == event1Atomic.routing.acceptedTasks);
    assert(event1Scan.routing.storedTasks == event1Atomic.routing.storedTasks);
    assert(event1Scan.routing.blockReservations > 0u);
    assert(event1Scan.routing.blockReservations
           <= event1Atomic.routing.blockReservations);
    verifyFusedResult(1u, event1Tasks, event1Scan);
    verifyChannelCounters(event1Channels);

    storeRawV0s();
    buffers.UploadCandidates();
    KFParticleGpuV0TrackRoutingPlan routingPlan;
    routingPlan.Compile(plan);
    KFParticleGpuChannelMask limited;
    limited.Set(0u);
    limited.Set(4u);
    buffers.SetV0TrackRoutingEnabledChannels(limited);
    const KFParticleGpuV0TrackFusedResult truncated =
      steering.RunV0TrackFusedStage(0u, selectedRange, 1u);
    assert(truncated.routing.visitedPairs == 2u);
    assert(truncated.routing.activeChannelBits == 2u);
    assert(truncated.routing.acceptedTasks == 2u);
    assert(truncated.routing.storedTasks == 1u);
    assert(truncated.routing.Truncated());
    assert(truncated.routing.overflowFlags == CandidateCapacityExceeded);
    assert(truncated.candidates.size == 1u);
    assert(truncated.candidates.HasOverflow(CandidateCapacityExceeded));
    const KFParticleGpuV0TrackRoutingView limitedRouting = buffers.HostV0TrackRouting();
    assert(limitedRouting.ChannelAcceptedCounters()[0] == 1u);
    assert(limitedRouting.ChannelAcceptedCounters()[1] == 0u);
    assert(limitedRouting.ChannelAcceptedCounters()[2] == 0u);
    assert(limitedRouting.ChannelAcceptedCounters()[3] == 0u);
    assert(limitedRouting.ChannelAcceptedCounters()[4] == 1u);
    buffers.SetV0TrackRoutingEnabledChannels(routingPlan.EnabledChannels());

    storeRawV0s();
    {
      KFParticleGpuCandidatePoolView candidates = buffers.HostCandidates();
      candidates.SizeData()[0] = candidates.Capacity();
    }
    buffers.UploadCandidates();
    const KFParticleGpuV0TrackFusedResult candidateOverflow =
      steering.RunV0TrackFusedStage(0u, selectedRange, 8u);
    assert(candidateOverflow.routing.acceptedTasks == 3u);
    assert(candidateOverflow.routing.storedTasks == 3u);
    assert(candidateOverflow.candidates.size == 0u);
    assert(candidateOverflow.candidates.HasOverflow(CandidateCapacityExceeded));

    storeRawV0s();
    {
      KFParticleGpuCandidatePoolView candidates = buffers.HostCandidates();
      assert(candidates.Daughters().Capacity() >= 3u);
      candidates.Daughters().SizeData()[0] = candidates.Daughters().Capacity() - 2u;
    }
    buffers.UploadCandidates();
    const KFParticleGpuV0TrackFusedResult daughterOverflow =
      steering.RunV0TrackFusedStage(0u, selectedRange, 8u);
    assert(daughterOverflow.routing.acceptedTasks == 3u);
    assert(daughterOverflow.routing.storedTasks == 3u);
    assert(daughterOverflow.candidates.size == 3u);
    assert(daughterOverflow.candidates.daughterSize == 0u);
    assert(daughterOverflow.candidates.HasOverflow(DaughterCapacityExceeded));

    KFParticleGpuV0TrackCascadeChannel rejectedBuild =
      plan.V0TrackCascadeChannel(0u);
    rejectedBuild.channelId = 1514u;
    rejectedBuild.transportMode = 999;
    plan.AddV0TrackCascadeChannel(rejectedBuild);
    storeRawV0s();
    buffers.UploadCandidates();
    const KFParticleGpuV0TrackFusedResult buildRejected =
      steering.RunV0TrackFusedStage(0u, selectedRange, 8u);
    assert(buildRejected.routing.acceptedTasks == 4u);
    assert(buildRejected.routing.storedTasks == 4u);
    assert(buildRejected.candidates.size == 3u);
    assert(buildRejected.candidates.daughterSize == 9u);
    assert(buildRejected.candidates.overflowFlags == 0u);
    assert(buffers.HostV0TrackRouting().ChannelAcceptedCounters()[5] == 1u);
    assert(buffers.HostV0TrackRouting().ChannelStoredCounters()[5] == 1u);
    assert(buffers.HostV0TrackRouting().ChannelConstructedCounters()[5] == 0u);

    bool invalidRangeRejected = false;
    try {
      steering.RunV0TrackFusedStage(
        0u,
        KFParticleGpuSelectedCandidateRange{
          buffers.Capacities().selectedCandidates + 1u, 1u, 0u},
        1u);
    }
    catch (const std::out_of_range&) {
      invalidRangeRejected = true;
    }
    assert(invalidRangeRejected);

    plan.Clear();
    Pass("v0-track-fused-routing-construction",
         "Stage 15.3 atomic and block-scan routes match the explicit Xi/Omega task and candidate oracle, bounded counters, and failure semantics");
  }

  void TestV0TrackChannelBuilders()
  {
    const KFParticleGpuRange bachelors(7u, 3u);
    const KFParticleGpuV0TrackChannel xiMinus =
      KFParticleGpuV0Track::MakeXiMinusChannel(71u, 2u, bachelors);
    const KFParticleGpuV0TrackChannel antiXiPlus =
      KFParticleGpuV0Track::MakeAntiXiPlusChannel(72u, 2u, bachelors);
    const KFParticleGpuV0TrackChannel omegaMinus =
      KFParticleGpuV0Track::MakeOmegaMinusChannel(73u, 2u, bachelors);
    const KFParticleGpuV0TrackChannel antiOmegaPlus =
      KFParticleGpuV0Track::MakeAntiOmegaPlusChannel(74u, 2u, bachelors);
    assert(xiMinus.v0Pdg == 3122 && xiMinus.bachelorPdg == -211 && xiMinus.motherPdg == 3312);
    assert(antiXiPlus.v0Pdg == -3122 && antiXiPlus.bachelorPdg == 211
           && antiXiPlus.motherPdg == -3312);
    assert(omegaMinus.v0Pdg == 3122 && omegaMinus.bachelorPdg == -321
           && omegaMinus.motherPdg == 3334);
    assert(antiOmegaPlus.v0Pdg == -3122 && antiOmegaPlus.bachelorPdg == 321
           && antiOmegaPlus.motherPdg == -3334);
    assert(xiMinus.transportMode == KFGpuTransportFullField);
    assert(omegaMinus.bachelorMass > xiMinus.bachelorMass);
    assert(xiMinus.bachelorTracks.offset == bachelors.offset);
    assert(antiOmegaPlus.eventIndex == 2u);
    Pass("v0-track-channel-builders",
         "Xi and Omega charge-conjugate channels keep explicit V0, bachelor, mass, event, and full-field descriptors");
  }

  void TestGenericChargedDecayGraph(KFParticleGpuRuntime& runtime,
                                    KFParticleGpuBufferManager& buffers)
  {
    KFParticleGpuBufferCapacities requested = buffers.Capacities();
    requested.tracks = std::max(requested.tracks, 7u);
    requested.events = std::max(requested.events, 1u);
    // This fixture validates graph completeness, not bounded-pool behaviour.
    // Leave enough headroom that HIP atomic publication order cannot starve a
    // later channel; dedicated overflow tests exercise the tight limits.
    requested.candidates = std::max(requested.candidates, 128u);
    requested.daughterIds = std::max(requested.daughterIds, 512u);
    requested.selectedCandidates = std::max(requested.selectedCandidates, 128u);
    requested.twoDaughterRoutedTasks =
      std::max(requested.twoDaughterRoutedTasks, 128u);
    requested.v0TrackRoutedTasks =
      std::max(requested.v0TrackRoutedTasks, 128u);
    requested.nonhomogeneousField = true;
    buffers.EnsureCapacity(requested);
    buffers.SetInputSizes(7u, 0u, 1u);

    KFParticleGpuInputTrackSoAView tracks = buffers.HostInputTracks();
    StoreSyntheticTrack(
      tracks, 0u, -0.20f, 0.10f, 0.00f, 0.80f, 0.10f, 1.00f, 211, 1, 17001);
    StoreSyntheticTrack(
      tracks, 1u, 0.15f, -0.10f, 0.10f, 0.55f, 0.25f, 0.85f, 211, 1, 17002);
    StoreSyntheticTrack(
      tracks, 2u, 0.40f, -0.30f, 0.20f, 0.50f, 0.60f, 0.90f, 2212, 1, 17003);
    StoreSyntheticTrack(
      tracks, 3u, -0.25f, 0.20f, -0.10f, 0.45f, -0.35f, 1.10f, 2212, 1, 17004);
    StoreSyntheticTrack(
      tracks, 4u, 0.05f, 0.15f, 0.20f, 0.35f, 0.20f, 1.20f,
      1000010020, 1, 17005);
    StoreSyntheticTrack(
      tracks, 5u, 0.10f, 0.50f, -0.40f, -0.30f, 0.70f, 1.10f, -211, -1, 17006);
    StoreSyntheticTrack(
      tracks, 6u, -0.60f, -0.20f, 0.30f, -0.40f, 0.20f, 0.80f, -321, -1, 17007);
    for (unsigned int index = 0u; index < 7u; ++index) {
      StoreDiagnosticField(tracks, index);
    }

    KFParticleGpuEventDesc* events = buffers.HostEvents();
    events[0] = KFParticleGpuEventDesc();
    events[0].eventId = 1701u;
    events[0].TrackSet(SecondaryPositiveFirst).tracks = KFParticleGpuRange(0u, 5u);
    events[0].TrackSet(SecondaryPositiveFirst).Species(Pion) =
      KFParticleGpuRange(0u, 2u);
    events[0].TrackSet(SecondaryPositiveFirst).Species(Proton) =
      KFParticleGpuRange(2u, 2u);
    events[0].TrackSet(SecondaryPositiveFirst).Species(Deuteron) =
      KFParticleGpuRange(4u, 1u);
    events[0].TrackSet(SecondaryNegativeFirst).tracks = KFParticleGpuRange(5u, 2u);
    events[0].TrackSet(SecondaryNegativeFirst).Species(Pion) =
      KFParticleGpuRange(5u, 1u);
    events[0].TrackSet(SecondaryNegativeFirst).Species(Kaon) =
      KFParticleGpuRange(6u, 1u);
    events[0].TrackSet(PrimaryPositiveFirst) =
      events[0].TrackSet(SecondaryPositiveFirst);
    events[0].TrackSet(PrimaryNegativeFirst) =
      events[0].TrackSet(SecondaryNegativeFirst);

    const auto relax = [](KFParticleGpuTwoDaughterChannel& channel) {
      channel.flags = KFGpuTwoDaughterUseLineDca | KFGpuTwoDaughterUseEnergyFit;
      channel.transportMode = KFGpuTransportFullField;
      channel.motherMassSigma = -1.f;
      channel.secondaryMassSigmaCut = -1.f;
      channel.maxSecondaryTopoChi2PerNdf = -1.f;
      channel.minSecondaryLdL = -1.f;
      channel.selection.expectedMass = channel.motherMass;
      channel.selection.expectedMassSigma = -1.f;
      channel.selection.massSigmaCut = -1.f;
      channel.selection.maxGeometricChi2PerNdf = -1.f;
      channel.selection.maxPrimaryVertexDistance = -1.f;
      channel.selection.minSecondaryLdL = -1.f;
      channel.selection.maxPrimaryTopologyChi2PerNdf = -1.f;
      channel.selection.maxSecondaryTopologyChi2PerNdf = -1.f;
      channel.selection.requirePrimaryVertex = 0u;
      channel.selection.topologyMode = KFGpuV0TopologySpatial;
      channel.maxDaughterDistance = -1.f;
    };

    KFParticleGpuDecayPlan& plan = runtime.GetSteering().GetDecayPlan();
    plan.Clear();
    KFParticleGpuTwoDaughterChannel lambda =
      MakeLambdaToProtonPiMinusChannel(KFGpuChannelLambdaToProtonPiMinus);
    relax(lambda);
    plan.AddTwoDaughterChannel(lambda);

    KFParticleGpuTwoDaughterChannel rho =
      MakeK0ShortToPiPlusPiMinusChannel(KFGpuChannelRho0ToPiPlusPiMinus);
    rho.motherPdg = 113;
    rho.firstTrackSet = PrimaryPositiveFirst;
    rho.secondTrackSet = PrimaryNegativeFirst;
    rho.motherMass = 0.77526f;
    relax(rho);
    plan.AddTwoDaughterChannel(rho);

    KFParticleGpuTwoDaughterChannel delta =
      MakeLambdaToProtonPiMinusChannel(KFGpuChannelDeltaPlusPlusToProtonPiPlus);
    delta.motherPdg = 2224;
    delta.firstTrackSet = PrimaryPositiveFirst;
    delta.secondTrackSet = PrimaryPositiveFirst;
    delta.secondSpecies = Pion;
    delta.secondDaughterPdg = 211;
    delta.secondSourcePdg = 211;
    delta.secondAlternateSourcePdg = 0;
    delta.secondCharge = 1;
    delta.motherMass = 1.232f;
    relax(delta);
    plan.AddTwoDaughterChannel(delta);

    KFParticleGpuTwoDaughterChannel d0 =
      MakeK0ShortToPiPlusPiMinusChannel(KFGpuChannelD0ToPiPlusKMinus);
    d0.motherPdg = 421;
    d0.secondSpecies = Kaon;
    d0.secondDaughterPdg = -321;
    d0.secondSourcePdg = -321;
    d0.secondAlternateSourcePdg = 0;
    d0.secondMass = 0.493677f;
    d0.motherMass = 1.86484f;
    relax(d0);
    plan.AddTwoDaughterChannel(d0);

    KFParticleGpuTwoDaughterChannel hyper =
      MakeLambdaToProtonPiMinusChannel(
        KFGpuChannelHypertritonToDeuteronPiMinus);
    hyper.motherPdg = 3003;
    hyper.firstSpecies = Deuteron;
    hyper.firstDaughterPdg = 1000010020;
    hyper.firstSourcePdg = 1000010020;
    hyper.firstAlternateSourcePdg = 0;
    hyper.firstMass = 1.8756129f;
    hyper.motherMass = 2.99131f;
    relax(hyper);
    plan.AddTwoDaughterChannel(hyper);

    KFParticleGpuV0TrackCascadeChannel dPlus;
    dPlus.channelId = KFGpuChannelDPlusToD0PiPlus;
    dPlus.parentChannelId = KFGpuChannelD0ToPiPlusKMinus;
    dPlus.v0Pdg = 421;
    dPlus.bachelorTrackSet = SecondaryPositiveFirst;
    dPlus.bachelorSpecies = Pion;
    dPlus.bachelorPdg = 211;
    dPlus.motherPdg = 411;
    dPlus.flags = KFGpuV0TrackUseLineDca | KFGpuV0TrackUseEnergyFit;
    dPlus.transportMode = KFGpuTransportFullField;
    dPlus.bachelorMass = 0.13957039f;
    dPlus.motherMass = 1.86966f;
    plan.AddV0TrackCascadeChannel(dPlus);

    KFParticleGpuV0TrackCascadeChannel longLived = dPlus;
    longLived.channelId = 93001u;
    longLived.family = KFGpuCpuFamilyLongLivedComposite;
    longLived.generation = 3u;
    longLived.parentChannelId = KFGpuChannelDPlusToD0PiPlus;
    longLived.v0Pdg = 411;
    longLived.bachelorSpecies = Proton;
    longLived.bachelorPdg = 2212;
    longLived.motherPdg = 930011u;
    longLived.bachelorMass = 0.9382720813f;
    longLived.motherMass = 0.f;
    plan.AddV0TrackCascadeChannel(longLived);

    KFParticleGpuGraphOperationChannel composite;
    composite.node.channelId = KFGpuChannelCompositeCompositeProbe;
    composite.node.topology = KFGpuGraphTopologyCompositeComposite;
    composite.node.generation = 2u;
    composite.node.firstSource = {
      KFGpuGraphSourceCandidateGeneration, 1u,
      KFGpuChannelLambdaToProtonPiMinus};
    composite.node.secondSource = {
      KFGpuGraphSourceCandidateGeneration, 1u,
      KFGpuChannelD0ToPiPlusKMinus};
    composite.descriptor.channelId = composite.node.channelId;
    composite.descriptor.topology = composite.node.topology;
    composite.descriptor.operationMask = KFGpuGraphConstruct;
    composite.descriptor.outputClass = KFGpuGraphOutputSecondary;
    composite.descriptor.motherPdg = 9000026;
    composite.descriptor.firstPdg = 3122;
    composite.descriptor.secondPdg = 421;
    plan.AddGraphOperationChannel(composite);

    KFParticleGpuGraphOperationChannel neutral;
    neutral.node.channelId = KFGpuChannelNeutralMissingMassProbe;
    neutral.node.topology = KFGpuGraphTopologyNeutralDaughter;
    neutral.node.generation = 2u;
    neutral.node.firstSource = {
      KFGpuGraphSourceCandidateGeneration, 1u,
      KFGpuChannelD0ToPiPlusKMinus};
    neutral.node.secondSource = {
      KFGpuGraphSourceCandidateGeneration, 1u,
      KFGpuChannelLambdaToProtonPiMinus};
    neutral.descriptor.channelId = neutral.node.channelId;
    neutral.descriptor.topology = neutral.node.topology;
    neutral.descriptor.operationMask =
      KFGpuGraphConstruct | KFGpuGraphMissingMass;
    neutral.descriptor.outputClass = KFGpuGraphOutputTemporary;
    neutral.descriptor.motherPdg = 9000027;
    neutral.descriptor.firstPdg = 421;
    neutral.descriptor.secondPdg = 3122;
    plan.AddGraphOperationChannel(neutral);

    KFParticleGpuGraphOperationChannel finalState;
    finalState.node.channelId = KFGpuChannelUnaryFinalProbe;
    finalState.node.topology = KFGpuGraphTopologyUnaryComposite;
    finalState.node.generation = 3u;
    finalState.node.firstSource = {
      KFGpuGraphSourceCandidateGeneration, 2u,
      KFGpuChannelCompositeCompositeProbe};
    finalState.descriptor.channelId = finalState.node.channelId;
    finalState.descriptor.topology = finalState.node.topology;
    finalState.descriptor.operationMask = KFGpuGraphSelect;
    finalState.descriptor.outputClass = KFGpuGraphOutputFinal;
    finalState.descriptor.motherPdg = 9000028;
    finalState.descriptor.firstPdg = 9000026;
    plan.AddGraphOperationChannel(finalState);

    runtime.GetSteering().RunDecayPlan(0u, 128u);
    const KFParticleGpuConstCandidatePoolView candidates =
      MakeConstView(buffers.HostCandidates());
    unsigned int observed[10] = {};
    const unsigned int channelIds[10] = {
      KFGpuChannelLambdaToProtonPiMinus,
      KFGpuChannelRho0ToPiPlusPiMinus,
      KFGpuChannelDeltaPlusPlusToProtonPiPlus,
      KFGpuChannelD0ToPiPlusKMinus,
      KFGpuChannelHypertritonToDeuteronPiMinus,
      KFGpuChannelDPlusToD0PiPlus,
      93001u,
      KFGpuChannelCompositeCompositeProbe,
      KFGpuChannelNeutralMissingMassProbe,
      KFGpuChannelUnaryFinalProbe};
    for (unsigned int candidate = 0u; candidate < candidates.Size(); ++candidate) {
      for (unsigned int channel = 0u; channel < 10u; ++channel) {
        if (candidates.Metadata().ChannelId(candidate) == channelIds[channel]) {
          ++observed[channel];
          const unsigned int expectedDaughters =
            channel == 5u ? 3u : (channel >= 6u ? 4u : 2u);
          assert(candidates.Metadata().DaughterCount(candidate) == expectedDaughters);
          assert(candidates.Metadata().Flags(candidate) & KFGpuCandidateValid);
        }
      }
    }
    bool complete = true;
    for (unsigned int channel = 0u; channel < 10u; ++channel) {
      complete &= observed[channel] > 0u;
    }
    if (!complete) {
      std::cerr << "GENERIC_GRAPH_ERROR candidates=" << candidates.Size()
                << '/' << candidates.Capacity()
                << " daughters=" << candidates.Daughters().Size()
                << '/' << candidates.Daughters().Capacity()
                << " overflow=" << candidates.OverflowFlags() << '\n';
      for (unsigned int channel = 0u; channel < 10u; ++channel) {
        std::cerr << "GENERIC_GRAPH_CHANNEL slot/id/count=" << channel << '/'
                  << channelIds[channel] << '/' << observed[channel] << '\n';
      }
      for (const KFParticleGpuTwoDaughterChannelResult& result :
           runtime.GetSteering().LastDecayPlanResults()) {
        std::cerr << "GENERIC_GRAPH_TWO_DAUGHTER id/pairs/accepted/stored/constructed="
                  << result.channelId << '/' << result.totalPairs << '/'
                  << result.acceptedTasks << '/' << result.storedTasks << '/'
                  << result.constructedCandidates << '\n';
      }
      for (const KFParticleGpuV0TrackChannelResult& result :
           runtime.GetSteering().LastV0TrackCascadeResults()) {
        std::cerr << "GENERIC_GRAPH_CASCADE id/pairs/accepted/stored/constructed="
                  << result.channelId << '/' << result.totalPairs << '/'
                  << result.acceptedTasks << '/' << result.storedTasks << '/'
                  << result.constructedCandidates << '\n';
      }
      const KFParticleGpuGraphExecutionMonitorData& graphMonitor =
        runtime.GetSteering().LastGraphExecutionMonitorData();
      std::cerr << "GENERIC_GRAPH_MONITOR groups/descriptors/visited/accepted/stored/constructed/rejected/overflow="
                << graphMonitor.groupLaunches << '/'
                << graphMonitor.descriptorCount << '/'
                << graphMonitor.visitedCombinations << '/'
                << graphMonitor.acceptedTasks << '/'
                << graphMonitor.storedTasks << '/'
                << graphMonitor.constructedCandidates << '/'
                << graphMonitor.rejectedTasks << '/'
                << graphMonitor.overflowFlags << '\n';
      for (const KFParticleGpuGraphChannelMonitorData& monitor :
           runtime.GetSteering().LastGraphChannelMonitorData()) {
        std::cerr << "GENERIC_GRAPH_OPERATION id/generation/visited/accepted/stored/constructed/rejected="
                  << monitor.channelId << '/' << monitor.generation << '/'
                  << monitor.visitedCombinations << '/' << monitor.acceptedTasks
                  << '/' << monitor.storedTasks << '/'
                  << monitor.constructedCandidates << '/'
                  << monitor.rejectedTasks << '\n';
      }
      for (unsigned int candidate = 0u; candidate < candidates.Size(); ++candidate) {
        std::cerr << "GENERIC_GRAPH_CANDIDATE index/channel/pdg/event/daughters/flags/status="
                  << candidate << '/'
                  << candidates.Metadata().ChannelId(candidate) << '/'
                  << candidates.Metadata().Pdg(candidate) << '/'
                  << candidates.Metadata().EventIndex(candidate) << '/'
                  << candidates.Metadata().DaughterCount(candidate) << '/'
                  << candidates.Metadata().Flags(candidate) << '/'
                  << candidates.Metadata().OperationStatus(candidate) << '\n';
      }
    }
    assert(complete);
    const auto& compositeResults =
      runtime.GetSteering().LastV0TrackCascadeResults();
    assert(compositeResults.size() == 2u);
    assert(compositeResults[0].constructedDaughters
           == 3u * compositeResults[0].constructedCandidates);
    assert(compositeResults[1].constructedDaughters
           == 4u * compositeResults[1].constructedCandidates);

    const KFParticleGpuDecayGraphView graph = buffers.HostDecayGraph();
    assert(graph.NodeCount() == 10u);
    assert(graph.GroupCount() == 6u);
    assert(graph.FindNode(KFGpuChannelD0ToPiPlusKMinus));
    assert(graph.FindNode(KFGpuChannelD0ToPiPlusKMinus)->payloadKind
           == KFGpuGraphPayloadTwoDaughter);
    const KFParticleGpuGraphNode* continuation =
      graph.FindNode(KFGpuChannelDPlusToD0PiPlus);
    assert(continuation);
    assert(continuation->payloadKind == KFGpuGraphPayloadCompositeTrack);
    assert(continuation->firstSource.sourceId == KFGpuChannelD0ToPiPlusKMinus);
    assert(buffers.HostV0TrackRouting().CompatibleChannelsByPdg(
             KFGpuChannelD0ToPiPlusKMinus, 421, 211).Test(0u));
    assert(buffers.HostV0TrackRouting().CompatibleChannelsByPdg(
             KFGpuChannelLambdaToProtonPiMinus, 421, 211).Empty());
    const KFParticleGpuGraphExecutionMonitorData graphMonitorData =
      runtime.GetSteering().LastGraphExecutionMonitorData();
    assert(graphMonitorData.groupLaunches == 3u);
    assert(graphMonitorData.descriptorCount == 3u);
    assert(graphMonitorData.acceptedTasks >= 3u);
    assert(graphMonitorData.storedTasks == graphMonitorData.acceptedTasks);
    assert(graphMonitorData.constructedCandidates >= 3u);
    assert(graphMonitorData.constructedCandidates + graphMonitorData.rejectedTasks
           == graphMonitorData.storedTasks);
    assert(graphMonitorData.candidates == graphMonitorData.constructedCandidates);
    assert(graphMonitorData.overflowFlags == 0u);
    assert(runtime.GetSteering().LastGraphChannelMonitorData().size() == 3u);
    for (const KFParticleGpuGraphChannelMonitorData& channel :
         runtime.GetSteering().LastGraphChannelMonitorData()) {
      assert(channel.constructedCandidates > 0u);
      assert(channel.constructedCandidates + channel.rejectedTasks
             == channel.storedTasks);
    }
    assert(runtime.GetSteering().LastDecayPlanEventResults()[0]
             .graphCandidates.size == graphMonitorData.candidates);
    const KFParticleGpuDeviceStorage& storage = buffers.Storage();
    assert(storage.fGraphOperationDescriptors.get());
    assert(storage.fGraphOperationTasks.get());
    assert(storage.fGraphOperationResults.get());

    const std::vector<KFParticleGpuGraphChannelMonitorData> channelMonitorData =
      runtime.GetSteering().LastGraphChannelMonitorData();
    runtime.GetSteering().RunDecayPlan(0u, 32u);
    const KFParticleGpuGraphExecutionMonitorData& repeated =
      runtime.GetSteering().LastGraphExecutionMonitorData();
    assert(repeated.groupLaunches == graphMonitorData.groupLaunches);
    assert(repeated.descriptorCount == graphMonitorData.descriptorCount);
    assert(repeated.visitedCombinations == graphMonitorData.visitedCombinations);
    assert(repeated.acceptedTasks == graphMonitorData.acceptedTasks);
    assert(repeated.storedTasks == graphMonitorData.storedTasks);
    assert(repeated.constructedCandidates == graphMonitorData.constructedCandidates);
    assert(repeated.rejectedTasks == graphMonitorData.rejectedTasks);
    assert(repeated.candidates == graphMonitorData.candidates);
    assert(repeated.daughters == graphMonitorData.daughters);
    assert(repeated.overflowFlags == graphMonitorData.overflowFlags);
    const std::vector<KFParticleGpuGraphChannelMonitorData>& repeatedChannels =
      runtime.GetSteering().LastGraphChannelMonitorData();
    assert(repeatedChannels.size() == channelMonitorData.size());
    for (std::size_t index = 0u; index < channelMonitorData.size(); ++index) {
      assert(repeatedChannels[index].channelId == channelMonitorData[index].channelId);
      assert(repeatedChannels[index].eventIndex == channelMonitorData[index].eventIndex);
      assert(repeatedChannels[index].generation == channelMonitorData[index].generation);
      assert(repeatedChannels[index].visitedCombinations
             == channelMonitorData[index].visitedCombinations);
      assert(repeatedChannels[index].acceptedTasks
             == channelMonitorData[index].acceptedTasks);
      assert(repeatedChannels[index].storedTasks
             == channelMonitorData[index].storedTasks);
      assert(repeatedChannels[index].constructedCandidates
             == channelMonitorData[index].constructedCandidates);
      assert(repeatedChannels[index].rejectedTasks
             == channelMonitorData[index].rejectedTasks);
    }

    runtime.GetSteering().RunDecayPlan(0u, 1u);
    const KFParticleGpuGraphExecutionMonitorData& truncated =
      runtime.GetSteering().LastGraphExecutionMonitorData();
    assert(truncated.acceptedTasks > truncated.storedTasks);
    assert(truncated.overflowFlags & KFGpuGraphTaskCapacityExceeded);

    plan.Clear();
    Pass("generic-charged-decay-graph",
         "one nonzero-field device transaction executes representative charged, charm, hypernuclear, cascade, long-lived, composite-composite, neutral, and unary generations");
    Pass("device-graph-scheduler",
         "persistent graph-operation buffers preserve queue-ordered generations, repeatable monitoring, and bounded task overflow");
  }

  void TestCompositeNeutralAndFinalGraphOperations(
    KFParticleGpuRuntime& runtime,
    KFParticleGpuBufferManager& buffers)
  {
    KFParticleGpuBufferCapacities requested = buffers.Capacities();
    requested.vertices = std::max(requested.vertices, 2u);
    requested.events = std::max(requested.events, 2u);
    requested.candidates = std::max(requested.candidates, 24u);
    requested.daughterIds = std::max(requested.daughterIds, 64u);
    buffers.EnsureCapacity(requested);
    buffers.SetInputSizes(1u, 2u, 2u);

    KFParticleGpuInputTrackSoAView tracks = buffers.HostInputTracks();
    StoreSyntheticTrack(tracks, 0u, -0.03f, 0.01f, -0.02f,
                        0.60f, 0.20f, 0.10f, 321, 1, 17008);
    tracks.PrimaryVertexIndex(0u) = 0;

    KFParticleGpuVertexSoAView vertices = buffers.HostPrimaryVertices();
    vertices.Parameter(0u, 0u) = 0.25f;
    vertices.Parameter(1u, 0u) = -0.10f;
    vertices.Parameter(2u, 0u) = 0.15f;
    for (unsigned int component = 0u;
         component < KFParticleGpuVertexState::NumberOfCovarianceElements;
         ++component) {
      vertices.Covariance(component, 0u) = 0.f;
    }
    vertices.Covariance(0u, 0u) = 0.02f;
    vertices.Covariance(2u, 0u) = 0.02f;
    vertices.Covariance(5u, 0u) = 0.02f;
    vertices.Chi2(0u) = 1.f;
    vertices.NDF(0u) = 3;
    vertices.NContributors(0u) = 8;
    KFParticleGpuVertexState otherVertex;
    LoadVertexState(MakeConstView(vertices), 0u, otherVertex);
    otherVertex.X() = 10.f;
    otherVertex.Y() = 20.f;
    otherVertex.Z() = 30.f;
    StoreVertexState(otherVertex, vertices, 1u);
    buffers.HostEvents()[0] = KFParticleGpuEventDesc();
    buffers.HostEvents()[0].eventId = 1730u;
    buffers.HostEvents()[0].primaryVertices = KFParticleGpuRange(0u, 1u);
    buffers.HostEvents()[1] = KFParticleGpuEventDesc();
    buffers.HostEvents()[1].eventId = 1731u;
    buffers.HostEvents()[1].primaryVertices = KFParticleGpuRange(1u, 1u);

    buffers.ResetCandidates();
    KFParticleGpuCandidatePoolView candidates = buffers.HostCandidates();
    const auto makeFit = [](float x,
                            float y,
                            float z,
                            float px,
                            float py,
                            float pz,
                            float energy,
                            int charge) {
      KFParticleGpuFitState fit;
      fit.Initialize();
      fit.X() = x;
      fit.Y() = y;
      fit.Z() = z;
      fit.Px() = px;
      fit.Py() = py;
      fit.Pz() = pz;
      fit.E() = energy;
      fit.Q() = charge;
      fit.NDF() = 1;
      fit.Chi2() = 0.1f;
      for (int parameter = 0; parameter < 8; ++parameter) {
        fit.Covariance(parameter, parameter) =
          0.02f + 0.005f * static_cast<float>(parameter);
      }
      return fit;
    };
    const KFParticleGpuFitState sourceFits[6] = {
      makeFit(0.00f, 0.00f, 0.00f, 0.40f, 0.10f, 0.20f, 1.00f, 0),
      makeFit(0.03f, -0.02f, 0.01f, -0.10f, 0.20f, 0.05f, 0.50f, 0),
      makeFit(-0.05f, 0.01f, -0.02f, 0.60f, 0.20f, 0.10f, 1.20f, 1),
      makeFit(-0.04f, 0.02f, -0.01f, 0.20f, 0.05f, 0.02f, 0.35f, 1),
      makeFit(0.02f, 0.00f, 0.01f, 0.15f, -0.05f, 0.03f, 0.30f, 0),
      makeFit(-0.04f, 0.01f, -0.02f, 0.60f, 0.20f, 0.10f,
              0.810949f, 1)};
    const int pdgs[6] = {3122, 111, 321, 211, 111, 100321};
    const unsigned int daughterOffsets[6] = {0u, 2u, 4u, 5u, 6u, 7u};
    const unsigned int daughterCounts[6] = {2u, 2u, 1u, 1u, 1u, 1u};
    const int sourceIds[8] = {
      17001, 17002, 17003, 17004, 17005, 17006, 17002, 17007};
    for (unsigned int index = 0u; index < 6u; ++index) {
      StoreCandidateFit(sourceFits[index], candidates, index);
      candidates.Metadata().Pdg(index) = pdgs[index];
      candidates.Metadata().PrimaryVertexIndex(index) = index == 0u ? 0 : -1;
      candidates.Metadata().EventIndex(index) = 0u;
      candidates.Metadata().DaughterOffset(index) = daughterOffsets[index];
      candidates.Metadata().DaughterCount(index) = daughterCounts[index];
      candidates.Metadata().Flags(index) = KFGpuCandidateValid;
      candidates.Metadata().ChannelId(index) = 170u + index;
    }
    for (unsigned int source = 0u; source < 8u; ++source) {
      candidates.Daughters().SourceId(source) = sourceIds[source];
    }
    candidates.SizeData()[0] = 6u;
    candidates.Daughters().SizeData()[0] = 8u;
    candidates.OverflowFlagsData()[0] = 0u;

    KFParticleGpuGraphOperationDescriptor descriptors[5];
    descriptors[0].channelId = 1731u;
    descriptors[0].topology = KFGpuGraphTopologyCompositeComposite;
    descriptors[0].operationMask =
      KFGpuGraphConstruct | KFGpuGraphMassConstraint | KFGpuGraphSelect;
    descriptors[0].outputClass = KFGpuGraphOutputSecondary;
    descriptors[0].motherPdg = 3000;
    descriptors[0].firstPdg = 3122;
    descriptors[0].secondPdg = 111;
    descriptors[0].massConstraint = 1.15f;
    descriptors[0].massConstraintSigma = 0.01f;
    descriptors[0].maxGeometricChi2PerNdf = 100.f;

    descriptors[1].channelId = 1732u;
    descriptors[1].topology = KFGpuGraphTopologyNeutralDaughter;
    descriptors[1].operationMask =
      KFGpuGraphConstruct | KFGpuGraphMissingMass | KFGpuGraphMassConstraint;
    descriptors[1].outputClass = KFGpuGraphOutputTemporary;
    descriptors[1].flags |= KFGpuGraphRequirePositiveMissingEnergy
                            | KFGpuGraphApplyMassConstraint;
    descriptors[1].motherPdg = 7000111;
    descriptors[1].firstPdg = 321;
    descriptors[1].secondPdg = 211;
    descriptors[1].firstMass = 0.493677f;
    descriptors[1].secondMass = 0.13957039f;
    descriptors[1].neutralMass = 0.1349766f;
    descriptors[1].missingMassMode =
      KFGpuGraphMissingMassFilteredReconstruction;
    descriptors[1].massConstraint = 0.50f;
    descriptors[1].massConstraintSigma = 0.02f;

    descriptors[2].channelId = 1733u;
    descriptors[2].topology = KFGpuGraphTopologyUnaryComposite;
    descriptors[2].operationMask =
      KFGpuGraphExtrapolate | KFGpuGraphSetProductionVertex
      | KFGpuGraphSelect;
    descriptors[2].outputClass = KFGpuGraphOutputFinal;
    descriptors[2].motherPdg = 3122;
    descriptors[2].firstPdg = 3122;
    descriptors[2].primaryVertexIndex = KFGpuGraphPrimaryVertexFromCandidate;
    descriptors[2].maxGeometricChi2PerNdf = 100.f;
    descriptors[2].maxTopologyChi2PerNdf = 100.f;

    descriptors[3] = descriptors[0];
    descriptors[3].channelId = 1734u;

    descriptors[4] = descriptors[1];
    descriptors[4].channelId = 1735u;
    descriptors[4].topology = KFGpuGraphTopologyCompositeTrack;
    descriptors[4].operationMask =
      KFGpuGraphConstruct | KFGpuGraphMassConstraint
      | KFGpuGraphMatch | KFGpuGraphSelect;
    descriptors[4].outputClass = KFGpuGraphOutputPrimary;
    descriptors[4].flags = KFGpuGraphRequireSameEvent
                           | KFGpuGraphRequireDistinctSources
                           | KFGpuGraphStoreFirstOnMatch
                           | KFGpuGraphMatchTrackPrimaryVertex;
    descriptors[4].motherPdg = 200321;
    descriptors[4].firstPdg = 100321;
    descriptors[4].secondPdg = 321;
    descriptors[4].missingMassMode = KFGpuGraphMissingMassLegacySubtract;
    descriptors[4].massConstraint = 0.497614f;
    descriptors[4].massConstraintSigma = 0.f;
    descriptors[4].matchMassWindowSigma = 3.7e-3f;
    descriptors[4].matchMassWindowCut = 3.f;
    descriptors[4].maxMatchMomentum = -1.f;
    descriptors[4].maxMatchDistance = 20.f;
    descriptors[4].maxMatchMomentumSigma = 5.f;
    descriptors[4].maxMatchTopologyChi2PerNdf = 3.f;
    descriptors[4].maxGeometricChi2PerNdf = 100.f;

    KFParticleGpuGraphOperationTask tasks[5];
    tasks[0].descriptorIndex = 0u;
    tasks[0].firstCandidateIndex = 0u;
    tasks[0].secondCandidateIndex = 1u;
    tasks[1].descriptorIndex = 1u;
    tasks[1].firstCandidateIndex = 2u;
    tasks[1].secondCandidateIndex = 3u;
    tasks[2].descriptorIndex = 2u;
    tasks[2].firstCandidateIndex = 0u;
    tasks[3].descriptorIndex = 3u;
    tasks[3].firstCandidateIndex = 0u;
    tasks[3].secondCandidateIndex = 4u;
    tasks[4].descriptorIndex = 4u;
    tasks[4].firstCandidateIndex = 5u;
    tasks[4].secondCandidateIndex = 0u;
    tasks[4].secondSourceKind = KFGpuGraphSourceTrackRange;
    for (auto& task : tasks) {
      task.primaryVertexOffset = 0u;
      task.primaryVertexCount = 1u;
    }

    KFParticleGpuFitState references[4];
    KFParticleGpuGraphOperationResult referenceResults[4];
    const unsigned int acceptedTasks[4] = {0u, 1u, 2u, 4u};
    const KFParticleGpuConstCandidatePoolView sourceCandidates =
      MakeConstView(candidates);
    const KFParticleGpuConstInputTrackSoAView sourceTracks =
      MakeConstView(tracks);
    const KFParticleGpuConstVertexSoAView sourceVertices =
      MakeConstView(vertices);
    for (unsigned int reference = 0u; reference < 4u; ++reference) {
      const unsigned int task = acceptedTasks[reference];
      int lineage[KFParticleGpuGraphOperations::MaximumLineageSize];
      unsigned int lineageSize = 0u;
      assert(KFParticleGpuGraphOperations::ExecuteTask(
        sourceTracks,
        sourceCandidates,
        sourceVertices,
        descriptors[task],
        tasks[task],
        references[reference],
        lineage,
        lineageSize,
        referenceResults[reference]));
      assert(referenceResults[reference].status == KFGpuGraphTaskAccepted);
    }
    assert(referenceResults[2].primaryVertexIndex == 0);
    assert(referenceResults[3].daughterCount == 2u);
    {
      KFParticleGpuFitState rejectedFit;
      KFParticleGpuGraphOperationResult rejected;
      int lineage[KFParticleGpuGraphOperations::MaximumLineageSize];
      unsigned int lineageSize = 0u;
      KFParticleGpuGraphOperationDescriptor distanceRejected = descriptors[4];
      distanceRejected.maxMatchDistance = 0.005f;
      assert(!KFParticleGpuGraphOperations::ExecuteTask(
        sourceTracks, sourceCandidates, sourceVertices,
        distanceRejected, tasks[4],
        rejectedFit, lineage, lineageSize, rejected));
      assert(rejected.status == KFGpuGraphTaskRejectSelection);

      KFParticleGpuGraphOperationDescriptor momentumRejected = descriptors[4];
      momentumRejected.maxMatchMomentum = -1.f;
      momentumRejected.maxMatchMomentumSigma = 0.01f;
      tracks.Numerical().Parameter(3u, 0u) = 0.40f;
      assert(!KFParticleGpuGraphOperations::ExecuteTask(
        sourceTracks, sourceCandidates, sourceVertices,
        momentumRejected, tasks[4],
        rejectedFit, lineage, lineageSize, rejected));
      assert(rejected.status == KFGpuGraphTaskRejectSelection);
      tracks.Numerical().Parameter(3u, 0u) = 0.60f;

      KFParticleGpuGraphOperationDescriptor topologyRejected = descriptors[4];
      topologyRejected.maxMatchTopologyChi2PerNdf = 0.01f;
      assert(!KFParticleGpuGraphOperations::ExecuteTask(
        sourceTracks, sourceCandidates, sourceVertices,
        topologyRejected, tasks[4], rejectedFit, lineage, lineageSize,
        rejected));
      assert(rejected.status == KFGpuGraphTaskRejectSelection);

      KFParticleGpuGraphOperationDescriptor massRejected = descriptors[4];
      massRejected.massConstraint = 0.45f;
      assert(!KFParticleGpuGraphOperations::ExecuteTask(
        sourceTracks, sourceCandidates, sourceVertices,
        massRejected, tasks[4], rejectedFit, lineage, lineageSize,
        rejected));
      assert(rejected.status == KFGpuGraphTaskRejectSelection);

      auto foreignVertexTask = tasks[4];
      foreignVertexTask.primaryVertexOffset = 1u;
      assert(!KFParticleGpuGraphOperations::ExecuteTask(
        sourceTracks, sourceCandidates, sourceVertices, descriptors[4],
        foreignVertexTask, rejectedFit, lineage, lineageSize, rejected));
      assert(rejected.status == KFGpuGraphTaskRejectSource);
      foreignVertexTask = tasks[2];
      foreignVertexTask.primaryVertexOffset = 1u;
      for (unsigned int operation : {KFGpuGraphExtrapolate,
                                     KFGpuGraphSetProductionVertex,
                                     KFGpuGraphSelect}) {
        auto descriptor = descriptors[2];
        descriptor.operationMask = operation;
        assert(!KFParticleGpuGraphOperations::ExecuteTask(
          sourceTracks, sourceCandidates, sourceVertices, descriptor,
          foreignVertexTask, rejectedFit, lineage, lineageSize, rejected));
      }
    }
    {
      KFParticleGpuFitState rejectedFit;
      KFParticleGpuGraphOperationResult rejected;
      int lineage[KFParticleGpuGraphOperations::MaximumLineageSize];
      unsigned int lineageSize = 0u;
      assert(!KFParticleGpuGraphOperations::ExecuteTask(
        sourceCandidates,
        sourceVertices,
        descriptors[3],
        tasks[3],
        rejectedFit,
        lineage,
        lineageSize,
        rejected));
      assert(rejected.status == KFGpuGraphTaskRejectLineage);
    }

    xpu::buffer<KFParticleGpuGraphOperationDescriptor> descriptorBuffer(
      5u, xpu::buf_io);
    xpu::buffer<KFParticleGpuGraphOperationTask> taskBuffer(5u, xpu::buf_io);
    xpu::buffer<KFParticleGpuGraphOperationResult> resultBuffer(5u, xpu::buf_io);
    for (unsigned int index = 0u; index < 5u; ++index) {
      HostPointer(descriptorBuffer)[index] = descriptors[index];
      HostPointer(taskBuffer)[index] = tasks[index];
      HostPointer(resultBuffer)[index] = KFParticleGpuGraphOperationResult();
    }
    buffers.UploadInput();
    buffers.UploadCandidates();
    runtime.GetQueue().copy(descriptorBuffer, xpu::h2d);
    runtime.GetQueue().copy(taskBuffer, xpu::h2d);
    runtime.GetQueue().copy(resultBuffer, xpu::h2d);
    runtime.GetQueue().launch<KFParticleGpuExecuteGraphOperationsExplicit>(
      xpu::n_threads(5u),
      MakeConstView(buffers.DeviceInputTracks()),
      MakeConstView(buffers.DeviceCandidates()),
      MakeConstView(buffers.DevicePrimaryVertices()),
      descriptorBuffer.get(),
      5u,
      taskBuffer.get(),
      5u,
      buffers.DeviceCandidates(),
      resultBuffer.get());
    runtime.GetQueue().wait();
    runtime.GetQueue().copy(resultBuffer, xpu::d2h);
    buffers.DownloadCandidates();

    const KFParticleGpuConstCandidatePoolView output =
      MakeConstView(buffers.HostCandidates());
    assert(output.Size() == 10u);
    assert(output.Daughters().Size() == 18u);
    assert(output.OverflowFlags() == 0u);
    for (unsigned int reference = 0u; reference < 4u; ++reference) {
      const unsigned int task = acceptedTasks[reference];
      const KFParticleGpuGraphOperationResult& result =
        HostPointer(resultBuffer)[task];
      assert(result.status == KFGpuGraphTaskAccepted);
      assert(result.outputClass == descriptors[task].outputClass);
      assert(result.outputCandidateIndex >= 6u
             && result.outputCandidateIndex < output.Size());
      KFParticleGpuFitState actual;
      LoadCandidateFit(output, result.outputCandidateIndex, actual);
      ExpectFitStateClose(actual, references[reference], 5.e-4f);
      assert(output.Metadata().ChannelId(result.outputCandidateIndex)
             == descriptors[task].channelId);
      assert(output.Metadata().Topology(result.outputCandidateIndex)
             == descriptors[task].topology);
      assert(output.Metadata().OutputClass(result.outputCandidateIndex)
             == descriptors[task].outputClass);
      assert(output.Metadata().OperationStatus(result.outputCandidateIndex)
             == KFGpuCandidateOperationAccepted);
      if (task == 2u) {
        assert(output.Metadata().PrimaryVertexIndex(result.outputCandidateIndex) == 0);
      }
      if (task == 4u) {
        assert(output.Metadata().DirectDaughterCount(result.outputCandidateIndex) == 2u);
        assert(output.Metadata().DirectFirstKind(result.outputCandidateIndex)
               == KFGpuDirectDaughterCandidate);
        assert(output.Metadata().DirectSecondKind(result.outputCandidateIndex)
               == KFGpuDirectDaughterInputTrack);
      }
      assert(output.Metadata().DaughterCount(result.outputCandidateIndex)
             == referenceResults[reference].daughterCount);
      const unsigned int offset =
        output.Metadata().DaughterOffset(result.outputCandidateIndex);
      for (unsigned int daughter = 1u;
           daughter < output.Metadata().DaughterCount(result.outputCandidateIndex);
           ++daughter) {
        assert(output.Daughters().SourceId(offset + daughter - 1u)
               < output.Daughters().SourceId(offset + daughter));
      }
    }
    assert(HostPointer(resultBuffer)[3].status
           == KFGpuGraphTaskRejectLineage);
    assert(HostPointer(resultBuffer)[3].outputCandidateIndex == 0xffffffffu);

    candidates.SizeData()[0] = candidates.Capacity();
    candidates.OverflowFlagsData()[0] = 0u;
    HostPointer(resultBuffer)[0] = KFParticleGpuGraphOperationResult();
    buffers.UploadCandidates();
    runtime.GetQueue().copy(resultBuffer, xpu::h2d);
    runtime.GetQueue().launch<KFParticleGpuExecuteGraphOperationsExplicit>(
      xpu::n_threads(1u),
      MakeConstView(buffers.DeviceInputTracks()),
      MakeConstView(buffers.DeviceCandidates()),
      MakeConstView(buffers.DevicePrimaryVertices()),
      descriptorBuffer.get(),
      5u,
      taskBuffer.get(),
      1u,
      buffers.DeviceCandidates(),
      resultBuffer.get());
    runtime.GetQueue().wait();
    runtime.GetQueue().copy(resultBuffer, xpu::d2h);
    buffers.DownloadCandidates();
    assert(HostPointer(resultBuffer)[0].status
           == KFGpuGraphTaskRejectCandidateCapacity);
    assert(buffers.HostCandidates().OverflowFlags()
           & CandidateCapacityExceeded);

    KFParticleGpuGraphOperationDescriptor detectorNeutral = descriptors[1];
    detectorNeutral.operationMask = KFGpuGraphConstruct;
    assert(!KFParticleGpuGraphOperations::ValidateDescriptor(detectorNeutral));
    KFParticleGpuGraphOperationDescriptor filteredMissingMass = descriptors[1];
    assert(KFParticleGpuGraphOperations::ValidateDescriptor(filteredMissingMass));
    KFParticleGpuGraphOperationDescriptor invalidFiltered = filteredMissingMass;
    invalidFiltered.neutralMass = -1.f;
    assert(!KFParticleGpuGraphOperations::ValidateDescriptor(invalidFiltered));

    KFParticleGpuDecayPlan defaultPlan;
    AddDefaultV0TwoDaughterChannels(defaultPlan);
    AddDefaultV0TrackCascadeChannels(defaultPlan);
    const KFParticleGpuDecayGraphManifest baseManifest =
      MakeDefaultCpuFinderDecayGraphManifest(defaultPlan);
    KFParticleGpuDecayGraphManifest extendedManifest;
    for (const KFParticleGpuGraphNode& node : baseManifest.Nodes()) {
      extendedManifest.AddNode(node);
    }
    KFParticleGpuGraphNode compositeNode;
    compositeNode.channelId = descriptors[0].channelId;
    compositeNode.payloadKind = KFGpuGraphPayloadCompositeComposite;
    compositeNode.payloadIndex = 0u;
    compositeNode.motherPdg = descriptors[0].motherPdg;
    compositeNode.topology = KFGpuGraphTopologyCompositeComposite;
    compositeNode.generation = 2u;
    compositeNode.firstSource = {
      KFGpuGraphSourceCandidateGeneration, 1u,
      KFGpuChannelLambdaToProtonPiMinus};
    compositeNode.secondSource = {
      KFGpuGraphSourceCandidateGeneration, 1u,
      KFGpuChannelK0ShortToPiPlusPiMinus};
    compositeNode.operationMask = descriptors[0].operationMask;
    compositeNode.selectionProfile = descriptors[0].channelId;
    compositeNode.outputClass = descriptors[0].outputClass;
    compositeNode.supportStatus = KFGpuGraphSupported;
    compositeNode.unsupportedReason = KFGpuGraphUnsupportedNone;
    extendedManifest.AddNode(compositeNode);

    KFParticleGpuGraphNode neutralNode;
    neutralNode.channelId = descriptors[1].channelId;
    neutralNode.payloadKind = KFGpuGraphPayloadNeutralDaughter;
    neutralNode.payloadIndex = 0u;
    neutralNode.motherPdg = descriptors[1].motherPdg;
    neutralNode.topology = KFGpuGraphTopologyNeutralDaughter;
    neutralNode.generation = 1u;
    neutralNode.firstSource = {KFGpuGraphSourceTrackRange, 0u, 1u};
    neutralNode.secondSource = {KFGpuGraphSourceTrackRange, 0u, 2u};
    neutralNode.operationMask = descriptors[1].operationMask;
    neutralNode.selectionProfile = descriptors[1].channelId;
    neutralNode.outputClass = descriptors[1].outputClass;
    neutralNode.supportStatus = KFGpuGraphSupported;
    neutralNode.unsupportedReason = KFGpuGraphUnsupportedNone;
    extendedManifest.AddNode(neutralNode);

    KFParticleGpuGraphNode unaryNode;
    unaryNode.channelId = descriptors[2].channelId;
    unaryNode.payloadKind = KFGpuGraphPayloadUnaryFinal;
    unaryNode.payloadIndex = 0u;
    unaryNode.motherPdg = descriptors[2].motherPdg;
    unaryNode.topology = KFGpuGraphTopologyUnaryComposite;
    unaryNode.generation = 3u;
    unaryNode.firstSource = {
      KFGpuGraphSourceCandidateGeneration, 2u, descriptors[0].channelId};
    unaryNode.operationMask = descriptors[2].operationMask;
    unaryNode.selectionProfile = descriptors[2].channelId;
    unaryNode.outputClass = descriptors[2].outputClass;
    unaryNode.supportStatus = KFGpuGraphSupported;
    unaryNode.unsupportedReason = KFGpuGraphUnsupportedNone;
    extendedManifest.AddNode(unaryNode);
    for (const KFParticleGpuGraphFamilyCoverage& original :
         baseManifest.FamilyCoverage()) {
      KFParticleGpuGraphFamilyCoverage coverage = original;
      if (coverage.family == KFGpuCpuFamilyCompositeComposite
          || coverage.family == KFGpuCpuFamilyNeutralMissingMass) {
        coverage.supportStatus = KFGpuGraphPartiallySupported;
        coverage.unsupportedReason = KFGpuGraphUnsupportedValidationPending;
        coverage.implementedChannelCount = 1u;
      }
      extendedManifest.AddFamilyCoverage(coverage);
    }
    KFParticleGpuDecayGraphPlan graph;
    graph.Compile(extendedManifest);
    bool foundComposite = false;
    bool foundNeutral = false;
    bool foundUnary = false;
    for (const KFParticleGpuGraphNode& node : graph.Nodes()) {
      foundComposite |= node.channelId == descriptors[0].channelId
                        && node.payloadKind
                             == KFGpuGraphPayloadCompositeComposite;
      foundNeutral |= node.channelId == descriptors[1].channelId
                      && node.payloadKind == KFGpuGraphPayloadNeutralDaughter;
      foundUnary |= node.channelId == descriptors[2].channelId
                    && node.payloadKind == KFGpuGraphPayloadUnaryFinal;
    }
    assert(foundComposite && foundNeutral && foundUnary);

    // Two sources of event 1 differ only in PV association. Index zero is
    // globally addressable but belongs to event 0 and must not be dereferenced.
    candidates.SizeData()[0] = 2u;
    candidates.Daughters().SizeData()[0] = 4u;
    candidates.OverflowFlagsData()[0] = 0u;
    int ownVertex = -1;
    assert(KFParticleGpuPackPrimaryVertexIndex(
      0, buffers.HostEvents()[1].primaryVertices, ownVertex));
    for (unsigned int index = 0u; index < 2u; ++index) {
      StoreCandidateFit(sourceFits[0], candidates, index);
      candidates.Metadata().Pdg(index) = 3122;
      candidates.Metadata().PrimaryVertexIndex(index) = index == 0u ? 0 : ownVertex;
      candidates.Metadata().EventIndex(index) = 1u;
      candidates.Metadata().DaughterOffset(index) = 2u * index;
      candidates.Metadata().DaughterCount(index) = 2u;
      candidates.Metadata().Flags(index) = KFGpuCandidateValid;
      HostPointer(taskBuffer)[index] = tasks[2];
      HostPointer(taskBuffer)[index].descriptorIndex = 0u;
      HostPointer(taskBuffer)[index].firstCandidateIndex = index;
      HostPointer(taskBuffer)[index].eventIndex = 1u;
      HostPointer(taskBuffer)[index].primaryVertexOffset = 1u;
      HostPointer(taskBuffer)[index].primaryVertexCount = 1u;
      HostPointer(resultBuffer)[index] = KFParticleGpuGraphOperationResult();
    }
    auto projection = descriptors[2];
    projection.operationMask = KFGpuGraphExtrapolate;
    HostPointer(descriptorBuffer)[0] = projection;
    buffers.UploadInput();
    buffers.UploadCandidates();
    runtime.GetQueue().copy(descriptorBuffer, xpu::h2d);
    runtime.GetQueue().copy(taskBuffer, xpu::h2d);
    runtime.GetQueue().copy(resultBuffer, xpu::h2d);
    runtime.GetQueue().launch<KFParticleGpuExecuteGraphOperationsExplicit>(
      xpu::n_threads(2u), MakeConstView(buffers.DeviceInputTracks()),
      MakeConstView(buffers.DeviceCandidates()), MakeConstView(buffers.DevicePrimaryVertices()),
      descriptorBuffer.get(), 1u, taskBuffer.get(), 2u,
      buffers.DeviceCandidates(), resultBuffer.get());
    runtime.GetQueue().wait();
    runtime.GetQueue().copy(resultBuffer, xpu::d2h);
    buffers.DownloadCandidates();
    assert(HostPointer(resultBuffer)[0].status == KFGpuGraphTaskRejectMathematics);
    const auto& ownProjection = HostPointer(resultBuffer)[1];
    assert(ownProjection.status == KFGpuGraphTaskAccepted);
    assert(ownProjection.primaryVertexIndex == 1);
    assert(buffers.HostCandidates().Size() == 3u);
    assert(buffers.HostCandidates().OverflowFlags() == 0u);
    KFParticleGpuFitState expectedProjection = sourceFits[0];
    assert(KFParticleGpuMath::TransportLineToPoint(
      expectedProjection, otherVertex, expectedProjection));
    KFParticleGpuFitState actualProjection;
    LoadCandidateFit(MakeConstView(buffers.HostCandidates()),
                     ownProjection.outputCandidateIndex, actualProjection);
    ExpectFitStateClose(actualProjection, expectedProjection, 5.e-4f);

    buffers.ResetCandidates();
    Pass("composite-neutral-final-graph-operations",
         "graph fits agree on host/device; foreign-event PVs are rejected and packed own-event projections are preserved");
  }

  void TestCpuFinderFinalSelectionGraph(
    KFParticleGpuRuntime& runtime,
    KFParticleGpuBufferManager& buffers)
  {
    KFParticleGpuBufferCapacities requested = buffers.Capacities();
    requested.vertices = std::max(requested.vertices, 1u);
    requested.events = std::max(requested.events, 1u);
    requested.candidates = std::max(requested.candidates, 4u);
    requested.daughterIds = std::max(requested.daughterIds, 8u);
    buffers.EnsureCapacity(requested);
    buffers.SetInputSizes(0u, 2u, 2u);

    KFParticleGpuVertexSoAView vertices = buffers.HostPrimaryVertices();
    for (unsigned int vertex = 0u; vertex < 2u; ++vertex) {
      vertices.Parameter(0u, vertex) = 0.f;
      vertices.Parameter(1u, vertex) = vertex == 0u ? 0.f : 500.f;
      vertices.Parameter(2u, vertex) = 0.f;
      for (unsigned int component = 0u;
           component < KFParticleGpuVertexState::NumberOfCovarianceElements;
           ++component) {
        vertices.Covariance(component, vertex) = 0.f;
      }
      vertices.Covariance(0u, vertex) = 0.05f;
      vertices.Covariance(2u, vertex) = 0.05f;
      vertices.Covariance(5u, vertex) = 0.05f;
      vertices.Chi2(vertex) = 1.f;
      vertices.NDF(vertex) = 3;
      vertices.NContributors(vertex) = 10;
    }
    buffers.HostEvents()[0] = KFParticleGpuEventDesc();
    buffers.HostEvents()[0].eventId = 19001u;
    buffers.HostEvents()[0].primaryVertices = KFParticleGpuRange(0u, 1u);
    buffers.HostEvents()[1] = KFParticleGpuEventDesc();
    buffers.HostEvents()[1].eventId = 19002u;
    buffers.HostEvents()[1].primaryVertices = KFParticleGpuRange(1u, 1u);

    buffers.ResetCandidates();
    KFParticleGpuCandidatePoolView candidates = buffers.HostCandidates();
    KFParticleGpuFitState source;
    source.Initialize();
    source.X() = 1.f;
    source.Y() = 1.f;
    source.Z() = 0.f;
    source.Px() = 0.5f;
    source.Py() = 0.f;
    source.Pz() = 0.f;
    source.E() = std::sqrt(0.25f + 1.86484f * 1.86484f);
    source.Q() = 0;
    source.NDF() = 3;
    source.Chi2() = 1.f;
    for (int parameter = 0; parameter < 8; ++parameter) {
      source.Covariance(parameter, parameter) = 0.10f;
    }
    StoreCandidateFit(source, candidates, 0u);
    candidates.Metadata().Pdg(0u) = 421;
    candidates.Metadata().PrimaryVertexIndex(0u) = -1;
    candidates.Metadata().EventIndex(0u) = 0u;
    candidates.Metadata().DaughterOffset(0u) = 0u;
    candidates.Metadata().DaughterCount(0u) = 2u;
    candidates.Metadata().Flags(0u) = KFGpuCandidateValid;
    candidates.Metadata().ChannelId(0u) = 1007u;
    candidates.Daughters().SourceId(0u) = 19001;
    candidates.Daughters().SourceId(1u) = 19002;
    candidates.SizeData()[0] = 1u;
    candidates.Daughters().SizeData()[0] = 2u;

    KFParticleGpuGraphOperationDescriptor descriptor;
    descriptor.channelId = 9001u;
    descriptor.topology = KFGpuGraphTopologyUnaryComposite;
    descriptor.operationMask =
      KFGpuGraphMassConstraint | KFGpuGraphSelect;
    descriptor.outputClass = KFGpuGraphOutputFinal;
    descriptor.flags = KFGpuGraphRequireSameEvent
      | KFGpuGraphSelectEventPrimaryVertices;
    descriptor.motherPdg = 421;
    descriptor.firstPdg = 421;
    descriptor.massConstraint = 1.86484f;
    descriptor.massConstraintSigma = 0.f;
    descriptor.matchMassWindowSigma = 0.0145f;
    descriptor.matchMassWindowCut = 3.f;
    descriptor.maxTopologyChi2PerNdf = 1000.f;
    descriptor.maxSelectionVertexDistance = 200.f;
    descriptor.minSelectionDecayLengthOverError = -1.f;

    KFParticleGpuGraphOperationTask task;
    task.descriptorIndex = 0u;
    task.firstCandidateIndex = 0u;
    task.firstSourceKind = KFGpuGraphSourceCandidateGeneration;
    task.eventIndex = 0u;
    task.primaryVertexOffset = 0u;
    task.primaryVertexCount = 1u;
    KFParticleGpuFitState reference;
    KFParticleGpuGraphOperationResult referenceResult;
    int lineage[KFParticleGpuGraphOperations::MaximumLineageSize];
    unsigned int lineageSize = 0u;
    const bool referenceAccepted = KFParticleGpuGraphOperations::ExecuteTask(
      MakeConstView(candidates), MakeConstView(vertices), descriptor, task,
      reference, lineage, lineageSize, referenceResult);
    assert(referenceAccepted);
    assert(referenceResult.status == KFGpuGraphTaskAccepted);
    assert(referenceResult.primaryVertexIndex == 0);
    assert(lineageSize == 2u);

    KFParticleGpuFitState rejectedFit;
    KFParticleGpuGraphOperationResult rejectedResult;
    KFParticleGpuGraphOperationTask missingPv = task;
    missingPv.primaryVertexCount = 0u;
    assert(!KFParticleGpuGraphOperations::ExecuteTask(
      MakeConstView(candidates), MakeConstView(vertices), descriptor, missingPv,
      rejectedFit, lineage, lineageSize, rejectedResult));
    assert(rejectedResult.status == KFGpuGraphTaskRejectSelection);
    KFParticleGpuGraphOperationTask isolatedPv = task;
    isolatedPv.primaryVertexOffset = 1u;
    assert(!KFParticleGpuGraphOperations::ExecuteTask(
      MakeConstView(candidates), MakeConstView(vertices), descriptor, isolatedPv,
      rejectedFit, lineage, lineageSize, rejectedResult));
    assert(rejectedResult.status == KFGpuGraphTaskRejectSelection);
    KFParticleGpuGraphOperationDescriptor distanceRejected = descriptor;
    distanceRejected.maxSelectionVertexDistance = 0.5f;
    assert(!KFParticleGpuGraphOperations::ExecuteTask(
      MakeConstView(candidates), MakeConstView(vertices), distanceRejected, task,
      rejectedFit, lineage, lineageSize, rejectedResult));
    assert(rejectedResult.status == KFGpuGraphTaskRejectSelection);
    KFParticleGpuGraphOperationDescriptor significanceRejected = descriptor;
    significanceRejected.minSelectionDecayLengthOverError = 4.f;
    assert(!KFParticleGpuGraphOperations::ExecuteTask(
      MakeConstView(candidates), MakeConstView(vertices), significanceRejected,
      task, rejectedFit, lineage, lineageSize, rejectedResult));
    assert(rejectedResult.status == KFGpuGraphTaskRejectSelection);
    KFParticleGpuGraphOperationDescriptor topologyRejected = descriptor;
    topologyRejected.maxTopologyChi2PerNdf = 2.f;
    assert(!KFParticleGpuGraphOperations::ExecuteTask(
      MakeConstView(candidates), MakeConstView(vertices), topologyRejected, task,
      rejectedFit, lineage, lineageSize, rejectedResult));
    assert(rejectedResult.status == KFGpuGraphTaskRejectSelection);
    KFParticleGpuGraphOperationDescriptor massRejected = descriptor;
    massRejected.massConstraint = 1.70f;
    assert(!KFParticleGpuGraphOperations::ExecuteTask(
      MakeConstView(candidates), MakeConstView(vertices), massRejected, task,
      rejectedFit, lineage, lineageSize, rejectedResult));
    assert(rejectedResult.status == KFGpuGraphTaskRejectSelection);

    xpu::buffer<KFParticleGpuGraphOperationDescriptor> descriptorBuffer(
      1u, xpu::buf_io);
    xpu::buffer<KFParticleGpuGraphOperationTask> taskBuffer(1u, xpu::buf_io);
    xpu::buffer<KFParticleGpuGraphOperationResult> resultBuffer(
      1u, xpu::buf_io);
    HostPointer(descriptorBuffer)[0] = descriptor;
    HostPointer(taskBuffer)[0] = task;
    HostPointer(resultBuffer)[0] = KFParticleGpuGraphOperationResult();
    buffers.UploadInput();
    buffers.UploadCandidates();
    runtime.GetQueue().copy(descriptorBuffer, xpu::h2d);
    runtime.GetQueue().copy(taskBuffer, xpu::h2d);
    runtime.GetQueue().copy(resultBuffer, xpu::h2d);
    runtime.GetQueue().launch<KFParticleGpuExecuteGraphOperationsExplicit>(
      xpu::n_threads(1u),
      MakeConstView(buffers.DeviceInputTracks()),
      MakeConstView(buffers.DeviceCandidates()),
      MakeConstView(buffers.DevicePrimaryVertices()),
      descriptorBuffer.get(), 1u, taskBuffer.get(), 1u,
      buffers.DeviceCandidates(), resultBuffer.get());
    runtime.GetQueue().wait();
    runtime.GetQueue().copy(resultBuffer, xpu::d2h);
    buffers.DownloadCandidates();
    assert(HostPointer(resultBuffer)[0].status == KFGpuGraphTaskAccepted);
    assert(HostPointer(resultBuffer)[0].mass == descriptor.massConstraint);
    assert(HostPointer(resultBuffer)[0].massError == 0.f);
    const KFParticleGpuConstCandidatePoolView output =
      MakeConstView(buffers.HostCandidates());
    assert(output.Size() == 2u);
    KFParticleGpuFitState actual;
    LoadCandidateFit(output,
                     HostPointer(resultBuffer)[0].outputCandidateIndex,
                     actual);
    ExpectFitStateClose(actual, reference, 5.e-4f);
    assert(output.Metadata().OutputClass(1u) == KFGpuGraphOutputFinal);
    assert(output.Metadata().DaughterCount(1u) == 2u);
    Pass("cpu-finder-final-selection-graph",
         "event-scoped PV, line/topology, mass-window, constrained-output, lineage, and rejection contracts agree on host and device");
  }

  void TestFilteredMissingMassRawTrackGraph(KFParticleGpuRuntime& runtime,
                                            KFParticleGpuBufferManager& buffers)
  {
    KFParticleGpuBufferCapacities requested = buffers.Capacities();
    requested.tracks = requested.tracks > 2u ? requested.tracks : 2u;
    requested.events = requested.events > 1u ? requested.events : 1u;
    requested.candidates = requested.candidates > 8u ? requested.candidates : 8u;
    requested.daughterIds = requested.daughterIds > 16u ? requested.daughterIds : 16u;
    requested.graphOperationDescriptors =
      requested.graphOperationDescriptors > 1u
        ? requested.graphOperationDescriptors : 1u;
    requested.graphOperationTasks = requested.graphOperationTasks > 8u
                                      ? requested.graphOperationTasks : 8u;
    buffers.EnsureCapacity(requested);
    buffers.SetInputSizes(2u, 0u, 1u);

    KFParticleGpuInputTrackSoAView tracks = buffers.HostInputTracks();
    StoreSyntheticTrack(
      tracks, 0u, 0.f, 0.f, 0.f, 1.2f, 0.2f, 0.4f, 211, 1, 21001);
    StoreSyntheticTrack(
      tracks, 1u, 0.02f, -0.01f, 0.01f, 0.15f, 0.03f, 0.05f, -13, 1, 21002);
    KFParticleGpuEventDesc* events = buffers.HostEvents();
    events[0] = KFParticleGpuEventDesc();
    events[0].eventId = 2100u;
    events[0].TrackSet(PrimaryPositiveLast).tracks = KFParticleGpuRange(0u, 1u);
    events[0].TrackSet(SecondaryPositiveFirst).tracks = KFParticleGpuRange(1u, 1u);
    events[0].TrackSet(SecondaryPositiveFirst).Species(Muon) =
      KFParticleGpuRange(1u, 1u);

    KFParticleGpuDecayPlan neutralPlan;
    AddCpuFinderNeutralMissingMassChannels(neutralPlan);
    assert(neutralPlan.NumberOfGraphOperationChannels() == 22u);
    KFParticleGpuGraphOperationChannel pionMuon =
      neutralPlan.GraphOperationChannel(0u);
    assert(pionMuon.node.channelId == 6001u);
    pionMuon.descriptor.maxGeometricChi2PerNdf = -1.f;

    KFParticleGpuSteering& steering = runtime.GetSteering();
    KFParticleGpuDecayPlan& plan = steering.GetDecayPlan();
    plan.Clear();
    plan.AddGraphOperationChannel(pionMuon);
    steering.RunDecayPlan(0u, 8u);

    const KFParticleGpuConstCandidatePoolView candidates =
      MakeConstView(buffers.HostCandidates());
    assert(candidates.Size() == 1u);
    assert(candidates.Metadata().ChannelId(0u) == 6001u);
    assert(candidates.Metadata().EventIndex(0u) == 0u);
    assert(candidates.Metadata().Pdg(0u) == pionMuon.descriptor.motherPdg);
    assert(candidates.Metadata().DirectDaughterCount(0u) == 2u);
    assert(candidates.Metadata().DirectFirstKind(0u)
           == KFGpuDirectDaughterInputTrack);
    assert(candidates.Metadata().DirectSecondKind(0u)
           == KFGpuDirectDaughterInputTrack);
    assert(candidates.Metadata().DirectFirstIndex(0u) == 0u);
    assert(candidates.Metadata().DirectSecondIndex(0u) == 1u);
    assert(candidates.Metadata().DaughterCount(0u) == 2u);
    assert(candidates.Daughters().SourceId(0u) == 21001);
    assert(candidates.Daughters().SourceId(1u) == 21002);
    KFParticleGpuFitState filteredMother;
    LoadCandidateFit(candidates, 0u, filteredMother);
    assert(KFParticleGpuMath::IsFiniteState(filteredMother));
    assert(filteredMother.NDF() >= 0);
    plan.Clear();
    Pass("filtered-missing-mass-raw-track-graph",
         "GPU routing consumes primary-mother and secondary-daughter track ranges and publishes the filtered mother with two-track ancestry");
  }

  void TestDecayPlanV0TrackCascade(KFParticleGpuRuntime& runtime,
                                   KFParticleGpuBufferManager& buffers)
  {
    KFParticleGpuBufferCapacities requested = buffers.Capacities();
    requested.tracks = requested.tracks > 6u ? requested.tracks : 6u;
    requested.events = requested.events > 1u ? requested.events : 1u;
    requested.candidates = requested.candidates > 32u ? requested.candidates : 32u;
    requested.daughterIds = requested.daughterIds > 96u ? requested.daughterIds : 96u;
    requested.selectedCandidates = requested.selectedCandidates > 8u ? requested.selectedCandidates : 8u;
    requested.v0TrackTasks = requested.v0TrackTasks > 8u ? requested.v0TrackTasks : 8u;
    requested.nonhomogeneousField = true;
    buffers.EnsureCapacity(requested);
    buffers.SetInputSizes(6u, 0u, 1u);

    KFParticleGpuInputTrackSoAView tracks = buffers.HostInputTracks();
    StoreSyntheticTrack(tracks, 0u, -0.2f, 0.1f, 0.f, 0.8f, 0.1f, 1.f, 211, 1, 10101);
    StoreSyntheticTrack(tracks, 1u, 0.4f, -0.3f, 0.2f, 0.5f, 0.6f, 0.9f, 2212, 1, 10201);
    StoreSyntheticTrack(tracks, 2u, 0.1f, 0.5f, -0.4f, -0.3f, 0.7f, 1.1f, -211, -1, 10301);
    StoreSyntheticTrack(tracks, 3u, -0.6f, -0.2f, 0.3f, -0.4f, 0.2f, 0.8f, -211, -1, 10302);
    StoreSyntheticTrack(tracks, 4u, 0.3f, 0.1f, 0.1f, -0.1f, 0.5f, 0.7f, -211, -1, 10303);
    StoreSyntheticTrack(tracks, 5u, -0.1f, 0.2f, -0.2f, 0.2f, 0.3f, 0.9f, -321, -1, 10401);
    for (unsigned int index = 0u; index < 6u; ++index) {
      StoreDiagnosticField(tracks, index);
    }

    KFParticleGpuEventDesc* events = buffers.HostEvents();
    events[0] = KFParticleGpuEventDesc();
    events[0].eventId = 1414u;
    events[0].TrackSet(SecondaryPositiveFirst).tracks = KFParticleGpuRange(0u, 2u);
    events[0].TrackSet(SecondaryPositiveFirst).Species(Pion) = KFParticleGpuRange(0u, 1u);
    events[0].TrackSet(SecondaryPositiveFirst).Species(Proton) = KFParticleGpuRange(1u, 1u);
    events[0].TrackSet(SecondaryNegativeFirst).tracks = KFParticleGpuRange(2u, 4u);
    events[0].TrackSet(SecondaryNegativeFirst).Species(Pion) = KFParticleGpuRange(2u, 3u);
    events[0].TrackSet(SecondaryNegativeFirst).Species(Kaon) = KFParticleGpuRange(5u, 1u);

    KFParticleGpuSteering& steering = runtime.GetSteering();
    KFParticleGpuDecayPlan& plan = steering.GetDecayPlan();
    plan.Clear();
    const auto relaxSelection = [](KFParticleGpuTwoDaughterChannel& channel) {
      channel.motherMassSigma = -1.f;
      channel.secondaryMassSigmaCut = -1.f;
      channel.maxSecondaryTopoChi2PerNdf = -1.f;
      channel.minSecondaryLdL = -1.f;
      channel.selection.expectedMass = 1.f;
      channel.selection.expectedMassSigma = -1.f;
      channel.selection.massSigmaCut = -1.f;
      channel.selection.maxGeometricChi2PerNdf = -1.f;
      channel.selection.maxPrimaryVertexDistance = -1.f;
      channel.selection.minSecondaryLdL = -1.f;
      channel.selection.maxPrimaryTopologyChi2PerNdf = -1.f;
      channel.selection.maxSecondaryTopologyChi2PerNdf = -1.f;
      channel.selection.requirePrimaryVertex = 0u;
      channel.selection.topologyMode = KFGpuV0TopologySpatial;
      channel.maxDaughterDistance = -1.f;
    };
    KFParticleGpuTwoDaughterChannel k0 = MakeK0ShortToPiPlusPiMinusChannel();
    KFParticleGpuTwoDaughterChannel lambda = MakeLambdaToProtonPiMinusChannel();
    KFParticleGpuTwoDaughterChannel antiLambda = MakeAntiLambdaToAntiProtonPiPlusChannel();
    relaxSelection(k0);
    relaxSelection(lambda);
    relaxSelection(antiLambda);
    plan.AddTwoDaughterChannel(k0);
    plan.AddTwoDaughterChannel(lambda);
    plan.AddTwoDaughterChannel(antiLambda);
    AddDefaultV0TrackCascadeChannels(plan);

    const std::vector<KFParticleGpuTwoDaughterChannelResult>& v0Results =
      steering.RunDecayPlan(0u, 8u);
    const std::vector<KFParticleGpuV0TrackChannelResult>& cascadeResults =
      steering.LastV0TrackCascadeResults();
    const std::vector<KFParticleGpuDecayPlanEventResult>& eventResults =
      steering.LastDecayPlanEventResults();
    assert(v0Results.size() == 3u);
    assert(cascadeResults.size() == 4u);
    assert(eventResults.size() == 1u);
    assert(eventResults[0].cascadeChannelOffset == 0u);
    assert(eventResults[0].cascadeChannelCount == 4u);
    assert(eventResults[0].cascadeCandidates.size == 15u);
    assert(cascadeResults[0].channelId == KFGpuChannelXiMinusToLambdaPiMinus);
    assert(cascadeResults[0].acceptedTasks == 10u);
    assert(cascadeResults[0].storedTasks == 10u);
    assert(cascadeResults[0].constructedCandidates == 10u);
    assert(cascadeResults[0].CandidateCount() == 10u);
    assert(!cascadeResults[0].Empty());
    assert(cascadeResults[2].channelId == KFGpuChannelOmegaMinusToLambdaKMinus);
    assert(cascadeResults[2].acceptedTasks == 5u);
    assert(cascadeResults[2].constructedCandidates == 5u);
    assert(cascadeResults[1].constructedCandidates == 0u);
    assert(cascadeResults[3].constructedCandidates == 0u);
    assert(cascadeResults[1].Empty());
    assert(cascadeResults[3].Empty());
    for (const KFParticleGpuV0TrackChannelResult& result : cascadeResults) {
      assert(result.generationCandidates.offset
             == eventResults[0].cascadeCandidates.offset);
      assert(result.generationCandidates.size
             == eventResults[0].cascadeCandidates.size);
    }
    const KFParticleGpuV0TrackRoutingMonitorData& routingMonitorData =
      steering.LastV0TrackRoutingMonitorData();
    assert(routingMonitorData.descriptorCount == 4u);
    assert(routingMonitorData.groupLaunches == 2u);
    assert(routingMonitorData.acceptedTasks == 15u);
    assert(routingMonitorData.storedTasks == 15u);
    assert(routingMonitorData.candidates == 15u);
    assert(routingMonitorData.daughters == 45u);
    assert(routingMonitorData.blockReservations > 0u);
    assert(routingMonitorData.blockReservations <= routingMonitorData.acceptedTasks);
    assert(routingMonitorData.overflowFlags == 0u);
    assert(steering.LastDecayPlanTiming().cascadeConstructionMilliseconds >= 0.);

    const KFParticleGpuConstCandidatePoolView candidates = MakeConstView(buffers.HostCandidates());
    assert(candidates.Size() == 24u);
    const KFParticleGpuConstInputTrackSoAView constTracks = MakeConstView(tracks);
    const KFParticleGpuCandidateRange rawV0Candidates =
      eventResults[0].candidates;
    std::vector<unsigned int> observedPerChannel(cascadeResults.size(), 0u);
    for (unsigned int candidateIndex = eventResults[0].cascadeCandidates.offset;
         candidateIndex < eventResults[0].cascadeCandidates.End();
         ++candidateIndex) {
      std::size_t channelIndex = cascadeResults.size();
      for (std::size_t index = 0u; index < cascadeResults.size(); ++index) {
        if (cascadeResults[index].channelId
            == candidates.Metadata().ChannelId(candidateIndex)) {
          channelIndex = index;
          break;
        }
      }
      assert(channelIndex < cascadeResults.size());
      ++observedPerChannel[channelIndex];
      const KFParticleGpuV0TrackCascadeChannel& cascadeChannel =
        plan.V0TrackCascadeChannel(channelIndex);
      const KFParticleGpuV0TrackChannel source =
        MakeV0TrackTaskChannel(cascadeChannel, events[0], 0u);
      assert(candidates.Metadata().EventIndex(candidateIndex) == 0u);
      assert(candidates.Metadata().DaughterCount(candidateIndex) == 3u);
      assert(candidates.Metadata().Flags(candidateIndex) & KFGpuCandidateValid);

      assert(candidates.Metadata().DirectFirstKind(candidateIndex)
             == KFGpuDirectDaughterCandidate);
      assert(candidates.Metadata().DirectSecondKind(candidateIndex)
             == KFGpuDirectDaughterInputTrack);
      const unsigned int rawV0Index =
        candidates.Metadata().DirectFirstIndex(candidateIndex);
      const unsigned int bachelorTrackIndex =
        candidates.Metadata().DirectSecondIndex(candidateIndex);
      assert(rawV0Index >= rawV0Candidates.offset
             && rawV0Index < rawV0Candidates.End());
      assert(bachelorTrackIndex < constTracks.Size());
      KFParticleGpuV0TrackTask referenceTask;
      referenceTask.v0CandidateIndex = rawV0Index;
      referenceTask.bachelorTrackIndex = bachelorTrackIndex;
      referenceTask.eventIndex = 0u;
      referenceTask.channelId = source.channelId;
      referenceTask.flags = source.flags;
      referenceTask.motherPdg = source.motherPdg;
      referenceTask.bachelorPdg = source.bachelorPdg;
      referenceTask.primaryVertexIndex = source.primaryVertexIndex;
      referenceTask.transportMode = source.transportMode;
      referenceTask.bachelorMass = source.bachelorMass;
      referenceTask.motherMass = source.motherMass;
      referenceTask.motherMassSigma = source.motherMassSigma;
      referenceTask.secondaryMassSigmaCut = source.secondaryMassSigmaCut;
      referenceTask.maxSecondaryTopoChi2PerNdf =
        source.maxSecondaryTopoChi2PerNdf;
      KFParticleGpuFitState reference;
      KFParticleGpuV0TrackLineage referenceLineage;
      assert(KFParticleGpuV0Track::BuildCandidate(
        candidates, constTracks, referenceTask, reference, referenceLineage));
      KFParticleGpuFitState actual;
      LoadCandidateFit(candidates, candidateIndex, actual);
      ExpectFullFieldEnergyFitClose(
        actual, reference, source.motherPdg, "cascade");
    }
    for (std::size_t channelIndex = 0u;
         channelIndex < cascadeResults.size();
         ++channelIndex) {
      assert(observedPerChannel[channelIndex]
             == cascadeResults[channelIndex].constructedCandidates);
    }
    plan.Clear();
    Pass("decay-plan-v0-track-cascade",
         "ordered steering stages build selected V0s, append Xi/Omega cascades, and expose event/channel ranges");
  }

  void TestDecayPlanV0TrackCascadeBatch(KFParticleGpuRuntime& runtime,
                                        KFParticleGpuBufferManager& buffers)
  {
    KFParticleGpuBufferCapacities requested = buffers.Capacities();
    requested.tracks = requested.tracks > 12u ? requested.tracks : 12u;
    requested.events = requested.events > 2u ? requested.events : 2u;
    requested.candidates = requested.candidates > 96u ? requested.candidates : 96u;
    requested.daughterIds = requested.daughterIds > 240u ? requested.daughterIds : 240u;
    requested.selectedCandidates = requested.selectedCandidates > 96u
                                     ? requested.selectedCandidates : 96u;
    requested.twoDaughterTasks = requested.twoDaughterTasks > 64u
                                   ? requested.twoDaughterTasks : 64u;
    requested.v0TrackRoutedTasks = requested.v0TrackRoutedTasks > 256u
                                     ? requested.v0TrackRoutedTasks : 256u;
    requested.graphOperationTasks = requested.graphOperationTasks > 64u
                                      ? requested.graphOperationTasks : 64u;
    requested.transactions = requested.transactions > 1u ? requested.transactions : 1u;
    requested.domains = requested.domains > 2u ? requested.domains : 2u;
    requested.workRanges = requested.workRanges > 2u ? requested.workRanges : 2u;
    requested.domainStatuses = requested.domainStatuses > 2u
                                 ? requested.domainStatuses : 2u;
    requested.nonhomogeneousField = true;
    buffers.EnsureCapacity(requested);
    buffers.SetInputSizes(12u, 0u, 2u);

    KFParticleGpuInputTrackSoAView tracks = buffers.HostInputTracks();
    KFParticleGpuEventDesc* events = buffers.HostEvents();
    for (unsigned int eventIndex = 0u; eventIndex < 2u; ++eventIndex) {
      const unsigned int offset = 6u * eventIndex;
      const int source = 16000 + static_cast<int>(100u * eventIndex);
      const float shift = 0.03f * static_cast<float>(eventIndex);
      StoreSyntheticTrack(
        tracks, offset, -0.2f + shift, 0.1f, 0.f, 0.8f, 0.1f, 1.f,
        211, 1, source + 1);
      StoreSyntheticTrack(
        tracks, offset + 1u, 0.4f + shift, -0.3f, 0.2f, 0.5f, 0.6f, 0.9f,
        2212, 1, source + 2);
      StoreSyntheticTrack(
        tracks, offset + 2u, 0.1f + shift, 0.5f, -0.4f, -0.3f, 0.7f, 1.1f,
        -211, -1, source + 3);
      StoreSyntheticTrack(
        tracks, offset + 3u, -0.6f + shift, -0.2f, 0.3f, -0.4f, 0.2f, 0.8f,
        -211, -1, source + 4);
      StoreSyntheticTrack(
        tracks, offset + 4u, 0.3f + shift, 0.1f, 0.1f, -0.1f, 0.5f, 0.7f,
        -211, -1, source + 5);
      StoreSyntheticTrack(
        tracks, offset + 5u, -0.1f + shift, 0.2f, -0.2f, 0.2f, 0.3f, 0.9f,
        -321, -1, source + 6);
      for (unsigned int index = offset; index < offset + 6u; ++index) {
        StoreDiagnosticField(tracks, index);
      }

      events[eventIndex] = KFParticleGpuEventDesc();
      events[eventIndex].eventId = 16100u + eventIndex;
      events[eventIndex].TrackSet(SecondaryPositiveFirst).tracks =
        KFParticleGpuRange(offset, 2u);
      events[eventIndex].TrackSet(SecondaryPositiveFirst).Species(Pion) =
        KFParticleGpuRange(offset, 1u);
      events[eventIndex].TrackSet(SecondaryPositiveFirst).Species(Proton) =
        KFParticleGpuRange(offset + 1u, 1u);
      events[eventIndex].TrackSet(SecondaryNegativeFirst).tracks =
        KFParticleGpuRange(offset + 2u, 4u);
      events[eventIndex].TrackSet(SecondaryNegativeFirst).Species(Pion) =
        KFParticleGpuRange(offset + 2u, 3u);
      events[eventIndex].TrackSet(SecondaryNegativeFirst).Species(Kaon) =
        KFParticleGpuRange(offset + 5u, 1u);
    }

    KFParticleGpuSteering& steering = runtime.GetSteering();
    assert(steering.CascadeTransactionStrategy()
           == KFGpuCascadeTransactionFused);
    assert(steering.GraphTransactionStrategy()
           == KFGpuGraphTransactionFused);
    steering.SetPerformanceMonitoringEnabled(true);
    KFParticleGpuDecayPlan& plan = steering.GetDecayPlan();
    plan.Clear();
    const auto relaxSelection = [](KFParticleGpuTwoDaughterChannel& channel) {
      channel.motherMassSigma = -1.f;
      channel.secondaryMassSigmaCut = -1.f;
      channel.maxSecondaryTopoChi2PerNdf = -1.f;
      channel.minSecondaryLdL = -1.f;
      channel.selection.expectedMass = 1.f;
      channel.selection.expectedMassSigma = -1.f;
      channel.selection.massSigmaCut = -1.f;
      channel.selection.maxGeometricChi2PerNdf = -1.f;
      channel.selection.maxPrimaryVertexDistance = -1.f;
      channel.selection.minSecondaryLdL = -1.f;
      channel.selection.maxPrimaryTopologyChi2PerNdf = -1.f;
      channel.selection.maxSecondaryTopologyChi2PerNdf = -1.f;
      channel.selection.requirePrimaryVertex = 0u;
      channel.selection.topologyMode = KFGpuV0TopologySpatial;
      channel.maxDaughterDistance = -1.f;
    };
    KFParticleGpuTwoDaughterChannel k0 = MakeK0ShortToPiPlusPiMinusChannel();
    KFParticleGpuTwoDaughterChannel lambda = MakeLambdaToProtonPiMinusChannel();
    KFParticleGpuTwoDaughterChannel antiLambda =
      MakeAntiLambdaToAntiProtonPiPlusChannel();
    relaxSelection(k0);
    relaxSelection(lambda);
    relaxSelection(antiLambda);
    plan.AddTwoDaughterChannel(k0);
    plan.AddTwoDaughterChannel(lambda);
    plan.AddTwoDaughterChannel(antiLambda);
    AddDefaultV0TrackCascadeChannels(plan);
    KFParticleGpuGraphOperationChannel projectedLambda;
    projectedLambda.node.channelId = 98001u;
    projectedLambda.node.topology = KFGpuGraphTopologyUnaryComposite;
    projectedLambda.node.generation = 2u;
    projectedLambda.node.firstSource = {
      KFGpuGraphSourceCandidateGeneration, 1u,
      KFGpuChannelLambdaToProtonPiMinus};
    projectedLambda.descriptor.channelId = projectedLambda.node.channelId;
    projectedLambda.descriptor.topology = projectedLambda.node.topology;
    projectedLambda.descriptor.operationMask = KFGpuGraphMassConstraint;
    projectedLambda.descriptor.outputClass = KFGpuGraphOutputFinal;
    projectedLambda.descriptor.motherPdg = 3122;
    projectedLambda.descriptor.firstPdg = 3122;
    projectedLambda.descriptor.massConstraint = 1.115683f;
    projectedLambda.descriptor.massConstraintSigma = 0.001f;
    plan.AddGraphOperationChannel(projectedLambda);

    buffers.SetTransactionSizes(1u, 2u, 2u, 2u);
    KFParticleGpuTransactionView transaction = buffers.HostTransaction();
    transaction.Transaction()[0] = KFParticleGpuTransactionDesc{};
    transaction.Transaction()[0].transactionId = 16101u;
    transaction.Transaction()[0].decayPlanRevision = plan.Revision();
    transaction.Transaction()[0].domains = {0u, 2u};
    transaction.Transaction()[0].workRanges = {0u, 2u};
    for (unsigned int eventIndex = 0u; eventIndex < 2u; ++eventIndex) {
      KFParticleGpuDomainDesc& domain = transaction.Domains()[eventIndex];
      domain = KFParticleGpuDomainDesc{};
      domain.sourceId = events[eventIndex].eventId;
      domain.inputDescriptorIndex = eventIndex;
      domain.sourceTracks = {6u * eventIndex, 6u};
      domain.primaryVertices = events[eventIndex].primaryVertices;
      domain.workRanges = {eventIndex, 1u};
      domain.taskReservation = {32u * eventIndex, 32u};
      domain.flags = KFParticleGpuDomainCompactOutput;
      domain.candidateReservation = {};
      domain.daughterReservation = {};
      domain.selectedReservation = {};
      domain.statusIndex = eventIndex;
      transaction.WorkRanges()[eventIndex] = KFParticleGpuWorkRange{};
      transaction.WorkRanges()[eventIndex].domainIndex = eventIndex;
      transaction.WorkRanges()[eventIndex].family = KFParticleGpuDomainInputWork;
      transaction.WorkRanges()[eventIndex].sourceTracks = domain.sourceTracks;
      transaction.WorkRanges()[eventIndex].outputReservation =
        domain.taskReservation;
    }

    const std::vector<KFParticleGpuTwoDaughterChannelResult>& v0Results =
      steering.RunDecayPlanBatch(0u, 2u, 64u);
    const std::vector<KFParticleGpuV0TrackChannelResult>& cascadeResults =
      steering.LastV0TrackCascadeResults();
    const std::vector<KFParticleGpuDecayPlanEventResult>& eventResults =
      steering.LastDecayPlanEventResults();
    assert(v0Results.size() == 6u);
    assert(cascadeResults.size() == 8u);
    assert(eventResults.size() == 2u);
    buffers.DownloadTransactionStatus();
    transaction = buffers.HostTransaction();
    unsigned int twoDaughterWork = 0u;
    unsigned int cascadeWork = 0u;
    unsigned int graphWork = 0u;
    bool cascadeDomain[2] = {false, false};
    for (unsigned int workIndex = 0u;
         workIndex < transaction.Transaction()->workRanges.size;
         ++workIndex) {
      const KFParticleGpuWorkRange& work = transaction.WorkRanges()[workIndex];
      if (work.family == KFParticleGpuTwoDaughterWork) { ++twoDaughterWork; }
      if (work.family == KFParticleGpuCompositeTrackWork) {
        ++cascadeWork;
        assert(work.domainIndex < 2u);
        cascadeDomain[work.domainIndex] = true;
        assert(work.sourceCandidates.size
               == transaction.Domains()[work.domainIndex]
                    .selectedReservation.size);
        assert(work.sourceTracks.size > 0u);
        const KFParticleGpuWorkStatus& status =
          transaction.WorkStatuses()[workIndex];
        assert((status.flags & KFParticleGpuWorkSourceResolved) != 0u);
        assert(status.sourceCandidates.offset
               == eventResults[work.domainIndex].selectedCandidates.offset);
        assert(status.sourceCandidates.size
               == eventResults[work.domainIndex].selectedCandidates.size);
        assert(status.sourceTracks == work.sourceTracks.size);
      }
      if (work.family == KFParticleGpuGraphOperationWork) {
        ++graphWork;
        assert(work.domainIndex < 2u);
        const KFParticleGpuWorkStatus& status =
          transaction.WorkStatuses()[workIndex];
        assert((status.flags & KFParticleGpuWorkSourceResolved) != 0u);
        if (status.acceptedTasks == 0u) {
          for (unsigned int d = 0; d < 2u; ++d) {
            const auto& domain = transaction.Domains()[d];
            const auto& ds = transaction.DomainStatuses()[domain.statusIndex];
            std::cerr << "CONSTRUCTION_FAILURE domain=" << d << " tasks=" << ds.storedTasks
                      << " candidates=" << ds.candidates << " selected=" << ds.selected
                      << " failure=" << ds.failureFlags << " overflow=" << ds.overflowFlags
                      << " reservation=" << domain.taskReservation.offset << "," << domain.taskReservation.size
                      << " capacity=" << buffers.DeviceTwoDaughterGenerationStorage().TaskCapacity() << std::endl;
            const auto* tiles = buffers.HostTwoDaughterTaskTiles();
            if (tiles && d < buffers.Capacities().twoDaughterTaskTiles) {
              std::cerr << "TILE " << d << " domain=" << tiles[d].domainIndex
                        << " task=" << tiles[d].firstTask << std::endl;
            }
          }
        }
        assert(status.acceptedTasks > 0u);
        assert(status.storedTasks == status.acceptedTasks);
        assert(status.constructedCandidates == status.storedTasks);
        assert(status.overflowFlags == 0u);
      }
    }
    assert(twoDaughterWork > 0u);
    assert(cascadeWork > 0u);
    assert(graphWork == 2u);
    assert(cascadeDomain[0] && cascadeDomain[1]);
    assert(transaction.Domains()[0].candidateReservation.size
           <= 5u * transaction.DomainStatuses()[0].storedTasks + 8u);
    assert(transaction.Domains()[1].candidateReservation.size
           <= 5u * transaction.DomainStatuses()[1].storedTasks + 8u);
    assert(transaction.Domains()[0].candidateReservation.End()
           == transaction.Domains()[1].candidateReservation.offset);

    const unsigned int expectedPerChannel[4] = {10u, 0u, 5u, 0u};
    const KFParticleGpuConstCandidatePoolView candidates =
      MakeConstView(buffers.HostCandidates());
    for (unsigned int eventIndex = 0u; eventIndex < 2u; ++eventIndex) {
      const KFParticleGpuDecayPlanEventResult& event = eventResults[eventIndex];
      assert(event.eventIndex == eventIndex);
      assert(event.channelOffset == 3u * eventIndex);
      assert(event.channelCount == 3u);
      assert(event.candidates.size == 9u);
      const unsigned int expectedV0Pairs[] = {3u, 6u, 0u};
      for (unsigned int channel = 0u; channel < event.channelCount; ++channel) {
        const auto& result = v0Results[event.channelOffset + channel];
        assert(result.totalPairs == expectedV0Pairs[channel]);
        assert(result.storedTasks == expectedV0Pairs[channel]);
      }
      assert(event.cascadeChannelOffset == 4u * eventIndex);
      assert(event.cascadeChannelCount == 4u);
      assert(event.cascadeCandidates.offset == event.candidates.End());
      assert(event.cascadeCandidates.size == 15u);
      assert(event.graphCandidates.offset == event.cascadeCandidates.End());
      assert(event.graphCandidates.size > 0u);
      if (event.graphCandidates.overflowFlags != 0u) {
        const auto& domain = transaction.Domains()[eventIndex];
        const auto& status = transaction.DomainStatuses()[domain.statusIndex];
        std::cerr << "COMPACT_OUTPUT_ERROR event=" << eventIndex
                  << " candidate_range=" << domain.candidateReservation.offset
                  << ':' << domain.candidateReservation.size
                  << " daughter_range=" << domain.daughterReservation.offset
                  << ':' << domain.daughterReservation.size
                  << " selected_range=" << domain.selectedReservation.offset
                  << ':' << domain.selectedReservation.size
                  << " stored/candidates/daughters/selected/cascade/graph="
                  << status.storedTasks << '/' << status.candidates << '/'
                  << status.daughters << '/' << status.selected << '/'
                  << status.cascadeTasks << '/' << status.graphTasks
                  << " tasks/reservation=" << status.tasks << '/'
                  << domain.taskReservation.size
                  << " overflow/failure=" << status.overflowFlags << '/'
                  << status.failureFlags << '\n';
        for (unsigned int local = 0u; local < domain.workRanges.size; ++local) {
          const unsigned int workIndex = domain.workRanges.offset + local;
          const auto& work = transaction.WorkRanges()[workIndex];
          const auto& workStatus = transaction.WorkStatuses()[workIndex];
          if (workStatus.overflowFlags != 0u) {
            std::cerr << "COMPACT_OUTPUT_WORK index/family/generation/group="
                      << workIndex << '/' << work.family << '/'
                      << work.generation << '/' << work.executionGroup
                      << " accepted/stored/constructed/overflow="
                      << workStatus.acceptedTasks << '/'
                      << workStatus.storedTasks << '/'
                      << workStatus.constructedCandidates << '/'
                      << workStatus.overflowFlags << '\n';
          }
        }
      }
      assert(event.graphCandidates.overflowFlags == 0u);
      assert(event.cascadeRouting.acceptedTasks == 15u);
      assert(event.cascadeRouting.storedTasks == 15u);
      assert(event.cascadeRouting.blockReservations > 0u);
      assert(event.cascadeRouting.blockReservations <= 15u);
      assert(event.overflowFlags == 0u);
      for (unsigned int channel = 0u; channel < 4u; ++channel) {
        const KFParticleGpuV0TrackChannelResult& result =
          cascadeResults[event.cascadeChannelOffset + channel];
        assert(result.eventIndex == eventIndex);
        assert(result.acceptedTasks == expectedPerChannel[channel]);
        assert(result.storedTasks == expectedPerChannel[channel]);
        assert(result.constructedCandidates == expectedPerChannel[channel]);
        assert(result.generationCandidates.offset
               == event.cascadeCandidates.offset);
        assert(result.generationCandidates.size
               == event.cascadeCandidates.size);
      }
      for (unsigned int candidateIndex = event.cascadeCandidates.offset;
           candidateIndex < event.cascadeCandidates.End();
           ++candidateIndex) {
        assert(candidates.Metadata().EventIndex(candidateIndex) == eventIndex);
        assert(candidates.Metadata().DaughterCount(candidateIndex) == 3u);
      }
      for (unsigned int candidateIndex = event.graphCandidates.offset;
           candidateIndex < event.graphCandidates.End();
           ++candidateIndex) {
        assert(candidates.Metadata().EventIndex(candidateIndex) == eventIndex);
        assert(candidates.Metadata().ChannelId(candidateIndex) == 98001u);
        assert(candidates.Metadata().DaughterCount(candidateIndex) == 2u);
      }
    }
    assert(eventResults[0].graphCandidates.End()
           <= eventResults[1].candidates.offset);
    assert(candidates.Size() == eventResults[1].graphCandidates.End());
    assert(candidates.Daughters().Size()
           == eventResults[1].graphCandidates.daughterOffset
                + eventResults[1].graphCandidates.daughterSize);

    const KFParticleGpuV0TrackRoutingMonitorData monitoring =
      steering.LastV0TrackRoutingMonitorData();
    assert(monitoring.groupLaunches == 4u);
    assert(monitoring.descriptorCount == 4u);
    assert(monitoring.visitedPairs == 30u);
    assert(monitoring.activeChannelBits == 30u);
    assert(monitoring.acceptedTasks == 30u);
    assert(monitoring.storedTasks == 30u);
    assert(monitoring.blockReservations > 0u);
    assert(monitoring.blockReservations <= monitoring.acceptedTasks);
    assert(monitoring.candidates == 30u);
    assert(monitoring.daughters == 90u);
    assert(monitoring.overflowFlags == 0u);
    const KFParticleGpuGraphExecutionMonitorData graphMonitoring =
      steering.LastGraphExecutionMonitorData();
    assert(graphMonitoring.groupLaunches == 1u);
    assert(graphMonitoring.descriptorCount == 1u);
    assert(graphMonitoring.acceptedTasks > 0u);
    assert(graphMonitoring.storedTasks == graphMonitoring.acceptedTasks);
    assert(graphMonitoring.constructedCandidates
           == graphMonitoring.storedTasks);
    assert(graphMonitoring.overflowFlags == 0u);
    const unsigned int fusedKernelLaunches =
      steering.LastPerformanceSnapshot().kernelLaunches;

    steering.SetCascadeTransactionStrategy(KFGpuCascadeTransactionStaged);
    const std::vector<KFParticleGpuTwoDaughterChannelResult>& stagedV0 =
      steering.RunDecayPlanBatch(0u, 2u, 64u);
    const std::vector<KFParticleGpuV0TrackChannelResult>& stagedCascades =
      steering.LastV0TrackCascadeResults();
    const std::vector<KFParticleGpuDecayPlanEventResult>& stagedEvents =
      steering.LastDecayPlanEventResults();
    assert(stagedV0.size() == 6u);
    assert(stagedCascades.size() == 8u);
    assert(stagedEvents.size() == 2u);
    for (unsigned int eventIndex = 0u; eventIndex < 2u; ++eventIndex) {
      assert(stagedEvents[eventIndex].candidates.size == 9u);
      assert(stagedEvents[eventIndex].cascadeCandidates.size == 15u);
      assert(stagedEvents[eventIndex].graphCandidates.size > 0u);
      assert(stagedEvents[eventIndex].overflowFlags == 0u);
    }
    const KFParticleGpuV0TrackRoutingMonitorData& stagedMonitoring =
      steering.LastV0TrackRoutingMonitorData();
    assert(stagedMonitoring.visitedPairs == monitoring.visitedPairs);
    assert(stagedMonitoring.acceptedTasks == monitoring.acceptedTasks);
    assert(stagedMonitoring.storedTasks == monitoring.storedTasks);
    assert(stagedMonitoring.candidates == monitoring.candidates);
    assert(stagedMonitoring.daughters == monitoring.daughters);
    assert(stagedMonitoring.overflowFlags == monitoring.overflowFlags);
    assert(steering.LastPerformanceSnapshot().kernelLaunches
           == fusedKernelLaunches + 1u);
    steering.SetCascadeTransactionStrategy(KFGpuCascadeTransactionFused);

    steering.SetGraphTransactionStrategy(KFGpuGraphTransactionStaged);
    const std::vector<KFParticleGpuTwoDaughterChannelResult>& stagedGraphV0 =
      steering.RunDecayPlanBatch(0u, 2u, 64u);
    const std::vector<KFParticleGpuDecayPlanEventResult>& stagedGraphEvents =
      steering.LastDecayPlanEventResults();
    assert(stagedGraphV0.size() == 6u);
    assert(stagedGraphEvents.size() == 2u);
    for (unsigned int eventIndex = 0u; eventIndex < 2u; ++eventIndex) {
      assert(stagedGraphEvents[eventIndex].cascadeCandidates.size == 15u);
      assert(stagedGraphEvents[eventIndex].graphCandidates.size > 0u);
      assert(stagedGraphEvents[eventIndex].overflowFlags == 0u);
    }
    const KFParticleGpuGraphExecutionMonitorData stagedGraphMonitoring =
      steering.LastGraphExecutionMonitorData();
    assert(stagedGraphMonitoring.visitedCombinations
           == graphMonitoring.visitedCombinations);
    assert(stagedGraphMonitoring.acceptedTasks == graphMonitoring.acceptedTasks);
    assert(stagedGraphMonitoring.storedTasks == graphMonitoring.storedTasks);
    assert(stagedGraphMonitoring.constructedCandidates
           == graphMonitoring.constructedCandidates);
    assert(stagedGraphMonitoring.rejectedTasks == graphMonitoring.rejectedTasks);
    assert(stagedGraphMonitoring.overflowFlags == graphMonitoring.overflowFlags);
    assert(steering.LastPerformanceSnapshot().kernelLaunches
           == fusedKernelLaunches + 1u);
    steering.SetGraphTransactionStrategy(KFGpuGraphTransactionFused);
    steering.SetPerformanceMonitoringEnabled(false);

    for (bool graph : {false, true}) {
      steering.SetCascadeTransactionStrategy(graph ? KFGpuCascadeTransactionFused : KFGpuCascadeTransactionStaged);
      steering.SetGraphTransactionStrategy(graph ? KFGpuGraphTransactionStaged : KFGpuGraphTransactionFused);
      steering.RunDecayPlanBatch(0u, 2u, 64u);
      buffers.DownloadTransactionStatus();
      buffers.DownloadCandidates();
      transaction = buffers.HostTransaction();
      auto& failingDomain = transaction.Domains()[0u];
      auto& failingStatus = transaction.DomainStatuses()[failingDomain.statusIndex];
      assert((graph ? failingStatus.graphTasks : failingStatus.cascadeTasks) > 0u);
      assert(failingStatus.candidates > 0u);
      --failingStatus.candidates;
      const unsigned int failedSlot = failingDomain.candidateReservation.offset + failingStatus.candidates;
      failingDomain.candidateReservation.size = failingStatus.candidates + 1u;
      failingDomain.daughterReservation.size = failingStatus.daughters;
      if (graph) { failingStatus.graphTasks = 1u; }
      else { failingStatus.cascadeTasks = 1u; }
      const auto neighbourStatus = transaction.DomainStatuses()[1u];
      transaction.DomainStatuses()[1u].graphTasks = 0u;
      transaction.DomainStatuses()[1u].cascadeTasks = 0u;
      auto poison = buffers.HostCandidates();
      assert((poison.Metadata().Flags(failedSlot) & KFGpuCandidateValid) != 0u);
      buffers.UploadCandidates();
      buffers.UploadTransaction();
      const auto& storage = buffers.Storage();
      const KFParticleGpuKernels faultState(
        MakeConstView(buffers.DeviceInputTracks()), MakeConstView(buffers.DevicePrimaryVertices()),
        buffers.DeviceEvents(), storage.fTwoDaughterTasks.get(), buffers.Capacities().twoDaughterTasks,
        buffers.DeviceCandidates(), buffers.DeviceV0SelectionResults(), buffers.DeviceSelectedCandidates(),
        buffers.DeviceV0TrackRouting(), buffers.DeviceTwoDaughterRouting(),
        buffers.DeviceCandidateDescriptorIndices(), buffers.DeviceDecayGraph(),
        buffers.DeviceGraphOperationStorage(), buffers.DeviceTwoDaughterGenerationStorage(),
        buffers.DeviceV0TrackGenerationStorage(), buffers.DeviceTransaction());
      if (graph) {
        runtime.GetQueue().launch<KFParticleGpuExecuteGraphOperationBatch>(
          xpu::n_blocks(2u), faultState, buffers.DeviceTransaction(), 2u, 7u);
      }
      else {
        runtime.GetQueue().launch<KFParticleGpuConstructCascadeBatch>(
          xpu::n_blocks(2u), faultState, buffers.DeviceTransaction(), 2u, 3u);
      }
      buffers.DownloadTransactionStatus();
      buffers.DownloadCandidates();
      transaction = buffers.HostTransaction();
      const auto& failure = transaction.DomainStatuses()[0u];
      assert(failure.overflowFlags == DaughterCapacityExceeded);
      const auto rejectedGraph = MakeConstView(buffers.HostCandidates());
      assert(rejectedGraph.Metadata().Flags(failedSlot) == KFGpuCandidateBuildFailed);
      assert(rejectedGraph.Metadata().DaughterCount(failedSlot) == 0u);
      KFParticleGpuFitState rejectedFit;
      KFParticleGpuFitState emptyFit;
      emptyFit.Initialize();
      LoadCandidateFit(rejectedGraph, failedSlot, rejectedFit);
      assert(rejectedFit.NDF() == emptyFit.NDF());
      for (unsigned int i = 0u; i < 36u; ++i) {
        assert(rejectedFit.Covariance(i) == emptyFit.Covariance(i));
      }
      assert(transaction.DomainStatuses()[1u].candidates == neighbourStatus.candidates);
      assert(transaction.DomainStatuses()[1u].daughters == neighbourStatus.daughters);
      assert(transaction.DomainStatuses()[1u].overflowFlags == 0u);
    }
    steering.SetCascadeTransactionStrategy(KFGpuCascadeTransactionFused);
    steering.SetGraphTransactionStrategy(KFGpuGraphTransactionFused);
    plan.Clear();
    Pass("decay-plan-v0-track-cascade-batch",
         "independently fused and staged cascade/graph transactions preserve two-event descendants, monitoring, and launch accounting");
  }

