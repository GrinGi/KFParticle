  void TestRoundTrip(KFParticleGpuSteering& steering)
  {
    KFParticleGpuBufferManager& buffers = steering.GetBuffers();
    KFParticleGpuEventDesc* events = buffers.HostEvents();

    const float pionMass = 0.13957f;
    steering.RunRoundTrip(pionMass, 0);

    const KFParticleGpuConstCandidatePoolView hostCandidates =
      MakeConstView(buffers.HostCandidates());
    KFParticleGpuFitState loaded;
    LoadCandidateFit(hostCandidates, 0, loaded);
    assert(loaded.X() == 1.f);
    assert(loaded.Px() == 2.f);
    assert(AlmostEqual(loaded.E(), std::sqrt(29.f + pionMass * pionMass)));
    assert(loaded.Q() == 1);
    assert(loaded.NDF() == 0);
    assert(loaded.MassHypo() == pionMass);
    assert(hostCandidates.Size() == 2);
    assert(hostCandidates.Daughters().Size() == 2);
    assert(hostCandidates.OverflowFlags() == 0);
    assert(hostCandidates.Metadata().Pdg(0) == 211);
    assert(hostCandidates.Metadata().PrimaryVertexIndex(0) == -1);
    assert(hostCandidates.Metadata().EventIndex(0) == 0);
    assert(hostCandidates.Metadata().DaughterOffset(0) == 0);
    assert(hostCandidates.Metadata().DaughterCount(0) == 1);
    assert(hostCandidates.Daughters().SourceId(0) == 17);
    assert(hostCandidates.Daughters().SourceId(1) == 19);
    Pass("round-trip", "input tracks converted by the KFParticle GPU smoke kernel into candidate pool entries");

    buffers.SetInputSizes(8, 1, 1);
    events[0].TrackSet(SecondaryPositiveFirst).tracks = KFParticleGpuRange(0, 8);
    steering.RunRoundTrip(pionMass, 0);

    const KFParticleGpuConstCandidatePoolView overflowCandidates =
      MakeConstView(buffers.HostCandidates());
    const unsigned int expectedOverflowSize =
      buffers.TrackSize() < overflowCandidates.Capacity()
        ? buffers.TrackSize()
        : overflowCandidates.Capacity();
    const unsigned int expectedOutputSize =
      expectedOverflowSize < overflowCandidates.Daughters().Capacity()
        ? expectedOverflowSize
        : overflowCandidates.Daughters().Capacity();
    unsigned int expectedOverflowFlags = 0;
    if (buffers.TrackSize() > overflowCandidates.Capacity()) {
      expectedOverflowFlags |= CandidateCapacityExceeded;
    }
    if (buffers.TrackSize() > overflowCandidates.Daughters().Capacity()) {
      expectedOverflowFlags |= DaughterCapacityExceeded;
    }
    assert(overflowCandidates.Size() == expectedOutputSize);
    assert(overflowCandidates.Daughters().Size() == expectedOutputSize);
    assert(overflowCandidates.OverflowFlags() == expectedOverflowFlags);
    assert(overflowCandidates.Metadata().Pdg(expectedOutputSize - 1u)
           == ((expectedOutputSize - 1u) % 2u == 0u ? 211 : -211));
    assert(overflowCandidates.Daughters().SourceId(expectedOutputSize - 1u)
           == 17 + static_cast<int>(2u * (expectedOutputSize - 1u)));
    Pass("candidate-overflow", "candidate and daughter pool capacity limits set the expected overflow flags");
  }

  void TestGuardsAndEmptyRun(KFParticleGpuSteering& steering)
  {
    KFParticleGpuBufferManager& buffers = steering.GetBuffers();
    const float pionMass = 0.13957f;

    bool invalidMassRejected = false;
    try {
      steering.RunRoundTrip(-1.f, 0);
    }
    catch (const std::invalid_argument&) {
      invalidMassRejected = true;
    }
    assert(invalidMassRejected);

    bool invalidEventRejected = false;
    try {
      steering.RunRoundTrip(pionMass, 1);
    }
    catch (const std::out_of_range&) {
      invalidEventRejected = true;
    }
    assert(invalidEventRejected);

    buffers.SetInputSizes(0, 0, 0);
    steering.RunRoundTrip(pionMass, 0);
    assert(buffers.HostCandidates().Size() == 0);
    assert(buffers.HostCandidates().Daughters().Size() == 0);
    Pass("guards-empty-run", "invalid host inputs are rejected and empty events leave empty candidate pools");
  }

  struct MaterializedParticleProbe {
    float parameters[8] = {};
    float covariance[36] = {};
    float chi2 = 0.f;
    float sFromDecay = 0.f;
    float sumDaughterMass = 0.f;
    float massHypothesis = -1.f;
    int ndf = 0;
    int id = -1;
    int pdg = 0;
    int constructMethod = 0;
    char charge = 0;
    float field[10] = {};
    std::vector<int> daughters;

    float& Parameter(int component) { return parameters[component]; }
    float& Covariance(int component) { return covariance[component]; }
    float& Chi2() { return chi2; }
    int& NDF() { return ndf; }
    char& Q() { return charge; }
    void SetSFromDecay(float value) { sFromDecay = value; }
    void SetSumDaughterMass(float value) { sumDaughterMass = value; }
    void SetMassHypo(float value) { massHypothesis = value; }
    void SetConstructMethod(int value) { constructMethod = value; }
    void SetPDG(int value) { pdg = value; }
    void SetId(int value) { id = value; }
    void SetFieldCoeff(float value, int component) { field[component] = value; }
    void ReserveDaughterIds(unsigned int count) { daughters.reserve(count); }
    void AddDaughterId(int value) { daughters.push_back(value); }
  };

  void TestBoundedOutputMaterialization(KFParticleGpuBufferManager& buffers)
  {
    KFParticleGpuBufferCapacities requested = buffers.Capacities();
    requested.tracks = requested.tracks < 12u ? 12u : requested.tracks;
    requested.candidates = requested.candidates < 8u ? 8u : requested.candidates;
    requested.daughterIds = requested.daughterIds < 12u ? 12u : requested.daughterIds;
    buffers.EnsureCapacity(requested);
    buffers.SetInputSizes(8u, 0u, 1u);

    KFParticleGpuInputTrackSoAView tracks = buffers.HostInputTracks();
    StoreSyntheticTrack(tracks, 0u, 0.f, 0.f, 0.f, 0.3f, 0.1f, 0.5f, 211, 1, 101);
    StoreSyntheticTrack(tracks, 1u, 0.f, 0.f, 0.f, 0.2f, 0.2f, 0.4f, -211, -1, 102);
    StoreSyntheticTrack(tracks, 2u, 0.f, 0.f, 0.f, 0.1f, 0.3f, 0.6f, -211, -1, 103);
    StoreSyntheticTrack(tracks, 3u, 0.f, 0.f, 0.f, 0.4f, 0.1f, 0.7f, 321, 1, 104);
    StoreSyntheticTrack(tracks, 4u, 1.f, 0.f, 0.f, 0.3f, 0.1f, 0.5f, 211, 1, 101);
    StoreSyntheticTrack(tracks, 5u, 1.f, 0.f, 0.f, 0.2f, 0.2f, 0.4f, -211, -1, 102);
    StoreSyntheticTrack(tracks, 6u, 1.f, 0.f, 0.f, 0.1f, 0.3f, 0.6f, -211, -1, 103);
    StoreSyntheticTrack(tracks, 7u, 1.f, 0.f, 0.f, 0.4f, 0.1f, 0.7f, 321, 1, 104);

    KFParticleGpuCandidatePoolView candidates = buffers.HostCandidates();
    candidates.SizeData()[0] = 4u;
    candidates.Daughters().SizeData()[0] = 11u;
    candidates.OverflowFlagsData()[0] = 0u;
    for (unsigned int candidate = 0u; candidate < 4u; ++candidate) {
      KFParticleGpuFitState fit;
      fit.Parameter(0) = static_cast<float>(candidate + 1u);
      fit.Chi2() = static_cast<float>(candidate);
      fit.NDF() = static_cast<int>(candidate + 1u);
      StoreCandidateFit(fit, candidates, candidate);
      candidates.Metadata().EventIndex(candidate) = 0u;
      candidates.Metadata().Flags(candidate) = KFGpuCandidateValid;
      candidates.Metadata().ChannelId(candidate) = 1800u + candidate;
      candidates.Metadata().OperationStatus(candidate) = KFGpuCandidateOperationAccepted;
      candidates.Metadata().PrimaryVertexIndex(candidate) = -1;
    }

    candidates.Metadata().Pdg(0u) = 310;
    candidates.Metadata().Topology(0u) = KFGpuGraphTopologyTrackTrack;
    candidates.Metadata().OutputClass(0u) = KFGpuGraphOutputSecondary;
    candidates.Metadata().DaughterOffset(0u) = 0u;
    candidates.Metadata().DaughterCount(0u) = 2u;
    candidates.Daughters().SourceId(0u) = 101;
    candidates.Daughters().SourceId(1u) = 102;
    candidates.Metadata().DirectDaughterCount(0u) = 2u;
    SetCandidateDirectDaughter(
      candidates.Metadata(), 0u, 0u, KFGpuDirectDaughterInputTrack, 0u);
    SetCandidateDirectDaughter(
      candidates.Metadata(), 0u, 1u, KFGpuDirectDaughterInputTrack, 1u);

    candidates.Metadata().Pdg(1u) = 3312;
    candidates.Metadata().Topology(1u) = KFGpuGraphTopologyCompositeTrack;
    candidates.Metadata().OutputClass(1u) = KFGpuGraphOutputSecondary;
    candidates.Metadata().DaughterOffset(1u) = 2u;
    candidates.Metadata().DaughterCount(1u) = 3u;
    candidates.Daughters().SourceId(2u) = 101;
    candidates.Daughters().SourceId(3u) = 102;
    candidates.Daughters().SourceId(4u) = 103;
    candidates.Metadata().DirectDaughterCount(1u) = 2u;
    SetCandidateDirectDaughter(
      candidates.Metadata(), 1u, 0u, KFGpuDirectDaughterCandidate, 0u);
    SetCandidateDirectDaughter(
      candidates.Metadata(), 1u, 1u, KFGpuDirectDaughterInputTrack, 6u);

    candidates.Metadata().Pdg(2u) = 9000001;
    candidates.Metadata().Topology(2u) = KFGpuGraphTopologyCompositeComposite;
    candidates.Metadata().OutputClass(2u) = KFGpuGraphOutputTemporary;
    candidates.Metadata().DaughterOffset(2u) = 5u;
    candidates.Metadata().DaughterCount(2u) = 3u;
    candidates.Daughters().SourceId(5u) = 101;
    candidates.Daughters().SourceId(6u) = 102;
    candidates.Daughters().SourceId(7u) = 103;
    candidates.Metadata().DirectDaughterCount(2u) = 2u;
    SetCandidateDirectDaughter(
      candidates.Metadata(), 2u, 0u, KFGpuDirectDaughterCandidate, 1u);
    SetCandidateDirectDaughter(
      candidates.Metadata(), 2u, 1u, KFGpuDirectDaughterCandidate, 0u);

    candidates.Metadata().Pdg(3u) = 9000001;
    candidates.Metadata().Topology(3u) = KFGpuGraphTopologyUnaryComposite;
    candidates.Metadata().OutputClass(3u) = KFGpuGraphOutputFinal;
    candidates.Metadata().DaughterOffset(3u) = 8u;
    candidates.Metadata().DaughterCount(3u) = 3u;
    candidates.Daughters().SourceId(8u) = 101;
    candidates.Daughters().SourceId(9u) = 102;
    candidates.Daughters().SourceId(10u) = 103;
    candidates.Metadata().DirectDaughterCount(3u) = 1u;
    SetCandidateDirectDaughter(
      candidates.Metadata(), 3u, 0u, KFGpuDirectDaughterCandidate, 2u);
    candidates.Metadata().DirectSecondKind(3u) = KFGpuDirectDaughterNone;
    candidates.Metadata().DirectSecondIndex(3u) = 0u;

    const KFParticleGpuCandidateRange range{3u, 1u, 8u, 3u, 0u};
    KFParticleGpuMaterializationWorkspace workspace;
    assert(workspace.CapacityBytes() == 0u);
    KFParticleGpuMaterializationRequest request{
      MakeConstView(tracks), MakeConstView(candidates), &range, 1u, 0u};
    request.workspace = &workspace;
    const auto compareReference = [](KFParticleGpuMaterializationRequest optimized) {
      auto reference = optimized;
      reference.workspace = nullptr;
      const auto a = KFParticleGpuMaterializer::Build(optimized);
      const auto b = KFParticleGpuMaterializer::Build(reference);
      assert(a.status == b.status);
      assert(a.particles.size() == b.particles.size());
      for (std::size_t i = 0; i < a.particles.size(); ++i) {
        const auto& x = a.particles[i];
        const auto& y = b.particles[i];
        assert(x.daughterParticleIds == y.daughterParticleIds);
        assert(x.leafSourceIds == y.leafSourceIds);
        assert(x.gpuCandidateIndex == y.gpuCandidateIndex && x.channelId == y.channelId);
        assert(x.topology == y.topology && x.outputClass == y.outputClass);
        assert(x.operationStatus == y.operationStatus && x.pdg == y.pdg);
        assert(x.primaryVertexIndex == y.primaryVertexIndex && x.sourceId == y.sourceId);
        assert(x.inputTrack == y.inputTrack);
        for (int j = 0; j < 8; ++j) { assert(x.fit.Parameter(j) == y.fit.Parameter(j)); }
        for (int j = 0; j < 36; ++j) { assert(x.fit.Covariance(j) == y.fit.Covariance(j)); }
        for (int j = 0; j < 10; ++j) { assert(x.fieldCoefficients[j] == y.fieldCoefficients[j]); }
        assert(x.fit.Chi2() == y.fit.Chi2() && x.fit.NDF() == y.fit.NDF());
        assert(x.fit.Q() == y.fit.Q() && x.fit.SFromDecay() == y.fit.SFromDecay());
        assert(x.fit.SumDaughterMass() == y.fit.SumDaughterMass());
        assert(x.fit.MassHypo() == y.fit.MassHypo());
        assert(x.fit.ConstructMethod() == y.fit.ConstructMethod());
      }
    };
    compareReference(request);
    const KFParticleGpuMaterializedEvent materialized =
      KFParticleGpuMaterializer::Build(request);
    assert(materialized.Succeeded());
    assert(materialized.particles.size() == 7u);
    assert(materialized.particles[3].daughterParticleIds
           == (std::vector<int>{0, 1}));
    assert(materialized.particles[4].daughterParticleIds
           == (std::vector<int>{3, 2}));
    assert(materialized.particles[5].daughterParticleIds
           == (std::vector<int>{4, 3}));
    assert(materialized.particles[6].daughterParticleIds
           == (std::vector<int>{5}));
    assert(materialized.particles[6].leafSourceIds
           == (std::vector<int>{101, 102, 103}));

    std::vector<MaterializedParticleProbe> destination(1u);
    destination[0].id = 77;
    assert(KFParticleGpuMaterializer::CommitToParticles(request, destination));
    assert(destination.size() == 7u);
    assert(destination[6].id == 6);
    assert(destination[6].daughters == (std::vector<int>{5}));
    assert(destination[6].parameters[0] == 4.f);

    KFParticleGpuEventDesc event;
    event.trackSets[SecondaryPositiveFirst].tracks = KFParticleGpuRange(0u, 2u);
    event.trackSets[SecondaryNegativeFirst].tracks = KFParticleGpuRange(2u, 2u);
    event.trackSets[SecondaryPositiveLast].tracks = KFParticleGpuRange(4u, 2u);
    event.trackSets[SecondaryNegativeLast].tracks = KFParticleGpuRange(6u, 2u);
    KFParticleGpuMaterializationRequest cpuContractRequest = request;
    cpuContractRequest.event = &event;
    cpuContractRequest.includeAllInputTracks = true;
    const auto compareCommitted = [](KFParticleGpuMaterializationRequest optimized) {
      auto reference = optimized;
      reference.workspace = nullptr;
      std::vector<MaterializedParticleProbe> directParticles;
      std::vector<MaterializedParticleProbe> referenceParticles;
      KFParticleGpuMaterializationStatus directStatus = KFGpuMaterializationOverflow;
      KFParticleGpuMaterializationStatus referenceStatus = KFGpuMaterializationOverflow;
      assert(KFParticleGpuMaterializer::CommitToParticles(
        optimized, directParticles, nullptr, &directStatus));
      assert(KFParticleGpuMaterializer::CommitToParticles(
        reference, referenceParticles, nullptr, &referenceStatus));
      assert(directStatus == KFGpuMaterializationSucceeded);
      assert(referenceStatus == directStatus);
      assert(directParticles.size() == referenceParticles.size());
      for (std::size_t i = 0u; i < directParticles.size(); ++i) {
        const auto& x = directParticles[i];
        const auto& y = referenceParticles[i];
        for (int component = 0; component < 8; ++component) {
          assert(x.parameters[component] == y.parameters[component]);
        }
        for (int component = 0; component < 36; ++component) {
          assert(x.covariance[component] == y.covariance[component]);
        }
        for (int component = 0; component < 10; ++component) {
          assert(x.field[component] == y.field[component]);
        }
        assert(x.chi2 == y.chi2 && x.sFromDecay == y.sFromDecay);
        assert(x.sumDaughterMass == y.sumDaughterMass);
        assert(x.massHypothesis == y.massHypothesis);
        assert(x.ndf == y.ndf && x.id == y.id && x.pdg == y.pdg);
        assert(x.constructMethod == y.constructMethod && x.charge == y.charge);
        assert(x.daughters == y.daughters);
      }
    };
    const auto expectDirectFailure = [&](KFParticleGpuMaterializationStatus expected) {
      std::vector<MaterializedParticleProbe> unchanged(1u);
      unchanged.front().id = 314;
      KFParticleGpuMaterializationStatus directStatus = KFGpuMaterializationSucceeded;
      assert(!KFParticleGpuMaterializer::CommitToParticles(
        cpuContractRequest, unchanged, nullptr, &directStatus));
      assert(directStatus == expected);
      assert(unchanged.size() == 1u && unchanged.front().id == 314);
    };
    const KFParticleGpuMaterializedEvent cpuContract =
      KFParticleGpuMaterializer::Build(cpuContractRequest);
    assert(cpuContract.Succeeded());
    assert(cpuContract.particles.size() == 8u);
    for (unsigned int track = 0u; track < 4u; ++track) {
      assert(cpuContract.particles[track].inputTrack);
      assert(cpuContract.particles[track].sourceId == 101 + static_cast<int>(track));
    }
    assert(cpuContract.particles.back().pdg == 9000001);
    compareReference(cpuContractRequest);
    compareCommitted(cpuContractRequest);

    // Production consumes the complete GPU-authored candidate ranges in one
    // dependency-before-parent pass. It must remain exactly equivalent to the
    // independent DFS reference without creating one range per candidate.
    const KFParticleGpuCandidateRange publicationRange{0u, 4u, 0u, 11u, 0u};
    auto publicationRequest = cpuContractRequest;
    publicationRequest.ranges = &publicationRange;
    publicationRequest.numberOfRanges = 1u;
    publicationRequest.completePublicationPlan = true;
    compareReference(publicationRequest);
    compareCommitted(publicationRequest);

    // Scratch ownership must not affect either leaf dependency discovery or
    // atomic publication. Reuse one workspace across successful/failed modes.
    KFParticleGpuMaterializationWorkspace modeWorkspace;
    const auto checkMaterializationModes = [&](KFParticleGpuMaterializationStatus expected) {
      for (bool completePlan : {false, true}) {
        for (bool allTracks : {false, true}) {
          for (bool reuseWorkspace : {false, true}) {
            auto mode = publicationRequest;
            mode.completePublicationPlan = completePlan;
            mode.includeAllInputTracks = allTracks;
            mode.event = allTracks ? &event : nullptr;
            mode.workspace = reuseWorkspace ? &modeWorkspace : nullptr;
            const auto built = KFParticleGpuMaterializer::Build(mode);
            if (built.status != expected) {
              std::cerr << "FAIL materialization modes complete/all/workspace="
                        << completePlan << '/' << allTracks << '/' << reuseWorkspace
                        << " status/expected=" << built.status << '/' << expected << '\n';
            }
            assert(built.status == expected);
            if (built.Succeeded()) {
              const unsigned int leaves = allTracks ? 4u : 3u;
              assert(built.particles.size() == leaves + 4u);
              assert(built.particles.back().daughterParticleIds
                     == std::vector<int>{static_cast<int>(leaves + 2u)});
              if (reuseWorkspace) {
                compareReference(mode);
                compareCommitted(mode);
              }
            }
            for (bool keepMetadata : {false, true}) {
              std::vector<MaterializedParticleProbe> output(1u);
              output.front().id = 2718;
              std::vector<KFParticleGpuMaterializedParticle> metadata(1u);
              metadata.front().sourceId = 314;
              KFParticleGpuMaterializationStatus status = KFGpuMaterializationOverflow;
              const bool committed = KFParticleGpuMaterializer::CommitToParticles(
                mode, output, keepMetadata ? &metadata : nullptr, &status);
              assert(status == expected);
              assert(committed == (expected == KFGpuMaterializationSucceeded));
              if (committed) {
                assert(output.size() == built.particles.size());
                assert(output.back().daughters == built.particles.back().daughterParticleIds);
                if (keepMetadata) { assert(metadata.size() == built.particles.size()); }
              }
              else {
                assert(output.size() == 1u && output.front().id == 2718);
                assert(metadata.size() == 1u && metadata.front().sourceId == 314);
              }
            }
          }
        }
      }
    };
    checkMaterializationModes(KFGpuMaterializationSucceeded);
    candidates.Metadata().DirectFirstIndex(0u) = tracks.Size();
    checkMaterializationModes(KFGpuMaterializationInvalidDirectDaughter);
    candidates.Metadata().DirectFirstIndex(0u) = 0u;
    checkMaterializationModes(KFGpuMaterializationSucceeded);
    candidates.Metadata().DirectFirstIndex(3u) = 3u;
    checkMaterializationModes(KFGpuMaterializationCyclicAncestry);
    candidates.Metadata().DirectFirstIndex(3u) = 2u;
    checkMaterializationModes(KFGpuMaterializationSucceeded);
    candidates.Daughters().SourceId(10u) = 999;
    checkMaterializationModes(KFGpuMaterializationLineageMismatch);
    candidates.Daughters().SourceId(10u) = 103;
    checkMaterializationModes(KFGpuMaterializationSucceeded);
    const float modeParameter = candidates.Fit().Parameter(0u, 3u);
    candidates.Fit().Parameter(0u, 3u) = std::numeric_limits<float>::quiet_NaN();
    checkMaterializationModes(KFGpuMaterializationInvalidNumericalState);
    candidates.Fit().Parameter(0u, 3u) = modeParameter;
    checkMaterializationModes(KFGpuMaterializationSucceeded);

    // First/last representations of a physical track must share one leaf ID.
    candidates.Metadata().Topology(2u) = KFGpuGraphTopologyCompositeTrack;
    SetCandidateDirectDaughter(
      candidates.Metadata(), 2u, 1u, KFGpuDirectDaughterInputTrack, 4u);
    checkMaterializationModes(KFGpuMaterializationSucceeded);
    candidates.Metadata().Topology(2u) = KFGpuGraphTopologyCompositeComposite;
    SetCandidateDirectDaughter(
      candidates.Metadata(), 2u, 1u, KFGpuDirectDaughterCandidate, 0u);
    modeWorkspace.Begin(0u, 0u);
    checkMaterializationModes(KFGpuMaterializationSucceeded);
    Pass("materialization-request-modes",
         "workspace/plan/leaf modes preserve graph identity, deduplication, atomic failures and reuse after errors");

    // A sealed packing plan may bypass the second O(N log N) source-ID sort
    // while preserving byte-for-byte publication parity and workspace reset.
    KFParticleGpuMaterializationWorkspace trustedWorkspace;
    auto trustedPublicationRequest = publicationRequest;
    trustedPublicationRequest.validatedPublicationInput = true;
    trustedPublicationRequest.workspace = &trustedWorkspace;
    compareCommitted(trustedPublicationRequest);
    assert(trustedWorkspace.physicalSourceIds.empty());
    assert(std::all_of(trustedWorkspace.trackSeen.begin(),
                       trustedWorkspace.trackSeen.end(),
                       [](unsigned char value) { return value == 0u; }));
    compareCommitted(trustedPublicationRequest);
    assert(trustedWorkspace.resetEntries > 0u);

    auto invalidTrustedRequest = request;
    invalidTrustedRequest.validatedPublicationInput = true;
    invalidTrustedRequest.workspace = &trustedWorkspace;
    std::vector<MaterializedParticleProbe> invalidTrustedOutput;
    KFParticleGpuMaterializationStatus invalidTrustedStatus =
      KFGpuMaterializationSucceeded;
    assert(!KFParticleGpuMaterializer::CommitToParticles(
      invalidTrustedRequest, invalidTrustedOutput, nullptr,
      &invalidTrustedStatus));
    assert(invalidTrustedStatus == KFGpuMaterializationInvalidTrackLayout);

    // Production scratch is bounded by this event, not by unrelated storage
    // belonging to the rest of a batch.
    buffers.SetInputSizes(12u, 0u, 1u);
    candidates.SizeData()[0] = 8u;
    KFParticleGpuMaterializationWorkspace eventLocalWorkspace;
    auto paddedBatchRequest = publicationRequest;
    paddedBatchRequest.tracks = MakeConstView(buffers.HostInputTracks());
    paddedBatchRequest.workspace = &eventLocalWorkspace;
    compareCommitted(paddedBatchRequest);
    assert(eventLocalWorkspace.visit.size() == 4u);
    assert(eventLocalWorkspace.trackSeen.size() == 8u);
    assert(eventLocalWorkspace.CandidateMapCapacityEntries()
           < paddedBatchRequest.candidates.Size());
    assert(eventLocalWorkspace.TrackMapCapacityEntries()
           < paddedBatchRequest.tracks.Size());
    buffers.SetInputSizes(8u, 0u, 1u);
    candidates.SizeData()[0] = 4u;

    candidates.Metadata().OperationStatus(3u) = KFGpuCandidateOperationRejected;
    compareReference(publicationRequest);
    compareCommitted(publicationRequest);
    candidates.Metadata().OperationStatus(3u) = KFGpuCandidateOperationAccepted;

    const KFParticleGpuCandidateRange missingDependencyRange{1u, 3u, 2u, 9u, 0u};
    publicationRequest.ranges = &missingDependencyRange;
    std::vector<MaterializedParticleProbe> publicationUnchanged(1u);
    publicationUnchanged.front().id = 2718;
    KFParticleGpuMaterializationStatus publicationStatus =
      KFGpuMaterializationSucceeded;
    assert(!KFParticleGpuMaterializer::CommitToParticles(
      publicationRequest, publicationUnchanged, nullptr, &publicationStatus));
    assert(publicationStatus == KFGpuMaterializationInvalidDirectDaughter);
    assert(publicationUnchanged.size() == 1u
           && publicationUnchanged.front().id == 2718);
    publicationRequest.ranges = &publicationRange;

    event.trackSets[SecondaryPositiveLast].tracks.size = 1u;
    assert(KFParticleGpuMaterializer::Build(cpuContractRequest).status
           == KFGpuMaterializationInvalidTrackLayout);
    expectDirectFailure(KFGpuMaterializationInvalidTrackLayout);
    event.trackSets[SecondaryPositiveLast].tracks.size = 2u;
    event.trackSets[SecondaryPositiveLast].tracks.offset = 0u;
    assert(KFParticleGpuMaterializer::Build(cpuContractRequest).status
           == KFGpuMaterializationInvalidTrackLayout);
    expectDirectFailure(KFGpuMaterializationInvalidTrackLayout);
    event.trackSets[SecondaryPositiveLast].tracks.offset = 4u;
    const int savedLastSource = tracks.SourceId(4u);
    tracks.SourceId(4u) = 404;
    assert(KFParticleGpuMaterializer::Build(cpuContractRequest).status
           == KFGpuMaterializationInvalidTrackLayout);
    expectDirectFailure(KFGpuMaterializationInvalidTrackLayout);
    tracks.SourceId(4u) = savedLastSource;
    const int savedDistinctSource = tracks.SourceId(3u);
    tracks.SourceId(3u) = tracks.SourceId(0u);
    tracks.SourceId(7u) = tracks.SourceId(0u);
    assert(KFParticleGpuMaterializer::Build(cpuContractRequest).status
           == KFGpuMaterializationInvalidTrackLayout);
    expectDirectFailure(KFGpuMaterializationInvalidTrackLayout);
    tracks.SourceId(3u) = savedDistinctSource;
    tracks.SourceId(7u) = savedDistinctSource;
    compareReference(cpuContractRequest);
    compareCommitted(cpuContractRequest);
    const auto initialized = workspace.initializedEntries;
    const auto capacityBytes = workspace.CapacityBytes();
    assert(capacityBytes > 0u);
    for (unsigned int repeat = 0; repeat < 20; ++repeat) { compareReference(request); }
    assert(workspace.initializedEntries == initialized);
    assert(workspace.resetEntries > 0u);
    assert(workspace.CapacityBytes() == capacityBytes);

    auto emptyRequest = request;
    emptyRequest.numberOfRanges = 0u;
    assert(KFParticleGpuMaterializer::Build(emptyRequest).particles.empty());
    compareReference(request);
    auto otherEvent = request;
    otherEvent.eventIndex = 1u;
    assert(KFParticleGpuMaterializer::Build(otherEvent).status == KFGpuMaterializationInvalidCandidate);
    compareReference(request);
    workspace.Begin(0u, 0u);
    compareReference(request);

    candidates.Metadata().DirectFirstIndex(3u) = 3u;
    std::vector<MaterializedParticleProbe> unchanged(1u);
    unchanged[0].id = 91;
    KFParticleGpuMaterializationStatus status = KFGpuMaterializationSucceeded;
    assert(!KFParticleGpuMaterializer::CommitToParticles(
      request, unchanged, nullptr, &status));
    assert(status == KFGpuMaterializationCyclicAncestry);
    assert(unchanged.size() == 1u && unchanged[0].id == 91);
    expectDirectFailure(KFGpuMaterializationCyclicAncestry);
    candidates.Metadata().DirectFirstIndex(3u) = 2u;

    compareReference(request);  // A failed DFS must not poison the next event.

    candidates.OverflowFlagsData()[0] = CandidateCapacityExceeded;
    assert(KFParticleGpuMaterializer::Build(request).status
           == KFGpuMaterializationOverflow);
    candidates.OverflowFlagsData()[0] = 0u;
    const KFParticleGpuCandidateRange invalidRange{4u, 1u, 0u, 0u, 0u};
    const KFParticleGpuMaterializationRequest invalidRangeRequest{
      MakeConstView(tracks), MakeConstView(candidates), &invalidRange, 1u, 0u};
    assert(KFParticleGpuMaterializer::Build(invalidRangeRequest).status
           == KFGpuMaterializationInvalidRange);

    candidates.Metadata().DirectFirstIndex(3u) = 99u;
    assert(KFParticleGpuMaterializer::Build(request).status
           == KFGpuMaterializationInvalidCandidate);
    candidates.Metadata().DirectFirstIndex(3u) = 2u;
    candidates.Metadata().OutputClass(3u) = KFGpuGraphOutputClassCount;
    assert(KFParticleGpuMaterializer::Build(request).status
           == KFGpuMaterializationInvalidCandidate);
    candidates.Metadata().OutputClass(3u) = KFGpuGraphOutputFinal;
    candidates.Daughters().SourceId(10u) = 999;
    assert(KFParticleGpuMaterializer::Build(request).status
           == KFGpuMaterializationLineageMismatch);
    expectDirectFailure(KFGpuMaterializationLineageMismatch);
    candidates.Daughters().SourceId(10u) = 103;
    const float savedParameter = candidates.Fit().Parameter(0u, 3u);
    candidates.Fit().Parameter(0u, 3u) =
      std::numeric_limits<float>::quiet_NaN();
    assert(KFParticleGpuMaterializer::Build(request).status
           == KFGpuMaterializationInvalidNumericalState);
    expectDirectFailure(KFGpuMaterializationInvalidNumericalState);
    candidates.Fit().Parameter(0u, 3u) = savedParameter;

    KFParticleGpuDecayPlan completePlan;
    AddCompleteCpuFinderChannels(completePlan);
    const KFParticleGpuDecayGraphManifest completeManifest =
      MakeDefaultCpuFinderDecayGraphManifest(completePlan);
    assert(completeManifest.Nodes().size() == 252u);
    for (const KFParticleGpuGraphNode& node : completeManifest.Nodes()) {
      candidates.Metadata().ChannelId(3u) = node.channelId;
      candidates.Metadata().Pdg(3u) = node.motherPdg;
      candidates.Metadata().Topology(3u) = node.topology;
      candidates.Metadata().OutputClass(3u) = node.outputClass;

      KFParticleGpuParitySnapshot snapshot;
      assert(KFParticleGpuParity::BuildSnapshot(
        MakeConstView(candidates), 3u, 21u, nullptr, false, snapshot));
      assert(snapshot.key.channelId == node.channelId);
      assert(snapshot.key.pdg == node.motherPdg);
      assert(snapshot.topology == node.topology);
      assert(snapshot.outputClass == node.outputClass);

      const KFParticleGpuMaterializedEvent channelMaterialized =
        KFParticleGpuMaterializer::Build(request);
      assert(channelMaterialized.Succeeded());
      assert(!channelMaterialized.particles.empty());
      const KFParticleGpuMaterializedParticle& root =
        channelMaterialized.particles.back();
      compareCommitted(cpuContractRequest);
      assert(root.gpuCandidateIndex == 3u);
      assert(root.channelId == node.channelId);
      assert(root.pdg == node.motherPdg);
      assert(root.topology == node.topology);
      assert(root.outputClass == node.outputClass);
      compareReference(request);
    }
    Pass("bounded-output-materialization",
         "CPU-compatible input tracks, all 252 active channel identities, complete ancestry, bounded failures, full fit state, and atomic commit");
    Pass("materialization-workspace-reference",
         "exact reference identity for all 252 channels, sparse scratch reset, empty/error recovery and shrink/reuse");
    Pass("direct-final-materialization",
         "production layout constructs final particles once, matches the independent materialized reference, and preserves atomic failures");
    Pass("linear-publication-plan",
         "complete GPU-authored ranges publish in one bounded dependency-before-parent pass without per-candidate range allocation");
  }

  void TestCapacityPolicy(KFParticleGpuBufferManager& buffers)
  {
    const KFParticleGpuBufferCapacities oldCapacities = buffers.Capacities();
    float* previousParameters = buffers.HostInputTracks().Numerical().ParametersData();
    KFParticleGpuBufferCapacities smaller = oldCapacities;
    smaller.tracks = 4;
    buffers.EnsureCapacity(smaller);
    assert(buffers.HostInputTracks().Numerical().ParametersData() == previousParameters);

    KFParticleGpuBufferCapacities larger = oldCapacities;
    larger.tracks = 12;
    larger.nonhomogeneousField = false;
    buffers.EnsureCapacity(larger);
    assert(buffers.Capacities().tracks == std::max(oldCapacities.tracks, 12u));
    assert(buffers.Capacities().nonhomogeneousField);
    Pass("capacity-policy", "buffers reuse smaller requests and grow monotonically without dropping field storage");
  }
