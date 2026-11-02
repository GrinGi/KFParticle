  void TestV0SelectionResultDataTypes(KFParticleGpuRuntime& runtime)
  {
    KFParticleGpuV0SelectionResult defaultResult;
    assert(defaultResult.selectionClass == KFGpuV0SelectionNotEvaluated);
    assert(defaultResult.rejectionReasons == KFGpuV0SelectionRejectNone);
    assert(defaultResult.bestPrimaryVertexIndex == -1);
    assert(defaultResult.topologyStatus == KFGpuV0LineTopologyNone);
    assert(!KFParticleGpuSelection::IsSelected(defaultResult));

    KFParticleGpuV0SelectionResult result;
    result.candidateIndex = 17u;
    result.channelId = KFGpuChannelLambdaToProtonPiMinus;
    result.eventIndex = 4u;
    result.bestPrimaryVertexIndex = 2;
    result.selectionClass = KFGpuV0SelectionRejected;
    result.topologyStatus = KFGpuV0LineTopologyValid;
    result.rejectionReasons = static_cast<unsigned int>(
      KFGpuV0SelectionRejectMass | KFGpuV0SelectionRejectTopology);
    result.observables.mass = 1.12f;
    result.observables.massError = 0.004f;
    result.observables.geometricChi2PerNdf = 1.5f;
    result.observables.nearestPrimaryVertexDistance = 2.5f;
    result.observables.nearestPrimaryVertexDistanceError = 0.25f;
    result.observables.nearestPrimaryVertexLdL = 10.f;
    result.observables.bestPrimaryVertexTopoChi2PerNdf = 6.f;
    result.observables.bestPrimaryVertexDecayLength = 3.f;
    result.observables.bestPrimaryVertexDecayLengthError = 0.3f;
    result.observables.bestPrimaryVertexLdL = 10.f;

    assert(KFParticleGpuSelection::HasRejection(result, KFGpuV0SelectionRejectMass));
    assert(KFParticleGpuSelection::HasRejection(result, KFGpuV0SelectionRejectTopology));
    assert(!KFParticleGpuSelection::HasRejection(result, KFGpuV0SelectionRejectDistance));
    assert(!KFParticleGpuSelection::IsSelected(result));

    xpu::buffer<KFParticleGpuV0SelectionResult> transfer(1, xpu::buf_device);
    runtime.GetQueue().memcpy(transfer.get(), &result, sizeof(result));
    runtime.GetQueue().wait();
    KFParticleGpuV0SelectionResult copied;
    runtime.GetQueue().memcpy(&copied, transfer.get(), sizeof(copied));
    runtime.GetQueue().wait();

    assert(copied.candidateIndex == result.candidateIndex);
    assert(copied.channelId == result.channelId);
    assert(copied.eventIndex == result.eventIndex);
    assert(copied.bestPrimaryVertexIndex == result.bestPrimaryVertexIndex);
    assert(copied.selectionClass == result.selectionClass);
    assert(copied.topologyStatus == result.topologyStatus);
    assert(copied.rejectionReasons == result.rejectionReasons);
    assert(AlmostEqual(copied.observables.mass, result.observables.mass));
    assert(AlmostEqual(copied.observables.massError, result.observables.massError));
    assert(AlmostEqual(copied.observables.geometricChi2PerNdf,
                       result.observables.geometricChi2PerNdf));
    assert(AlmostEqual(copied.observables.nearestPrimaryVertexDistance,
                       result.observables.nearestPrimaryVertexDistance));
    assert(AlmostEqual(copied.observables.nearestPrimaryVertexDistanceError,
                       result.observables.nearestPrimaryVertexDistanceError));
    assert(AlmostEqual(copied.observables.nearestPrimaryVertexLdL,
                       result.observables.nearestPrimaryVertexLdL));
    assert(AlmostEqual(copied.observables.bestPrimaryVertexTopoChi2PerNdf,
                       result.observables.bestPrimaryVertexTopoChi2PerNdf));
    assert(AlmostEqual(copied.observables.bestPrimaryVertexDecayLength,
                       result.observables.bestPrimaryVertexDecayLength));
    assert(AlmostEqual(copied.observables.bestPrimaryVertexDecayLengthError,
                       result.observables.bestPrimaryVertexDecayLengthError));
    assert(AlmostEqual(copied.observables.bestPrimaryVertexLdL,
                       result.observables.bestPrimaryVertexLdL));

    result.selectionClass = KFGpuV0SelectionSecondary;
    result.rejectionReasons = KFGpuV0SelectionRejectNone;
    assert(KFParticleGpuSelection::IsSelected(result));
    result.selectionClass = KFGpuV0SelectionPrimary;
    assert(KFParticleGpuSelection::IsSelected(result));

    Pass("v0-selection-data-types",
         "flat V0 observables, classes, rejection masks, and host-device copies preserve selection diagnostics");
  }

  void TestPrimaryVertexTopologyObservables()
  {
    KFParticleGpuFitState candidate;
    candidate.Initialize();
    candidate.X() = 1.f;
    candidate.Y() = 2.f;
    candidate.Z() = 3.f;
    for (int i = 0; i < KFParticleGpuFitState::NumberOfCovarianceElements; ++i) {
      candidate.Covariance(i) = 0.f;
    }
    candidate.Covariance(0, 0) = 0.4f;
    candidate.Covariance(1, 1) = 0.5f;
    candidate.Covariance(2, 2) = 0.6f;

    KFParticleGpuVertexState primaryVertex;
    primaryVertex.X() = 4.f;
    primaryVertex.Y() = 6.f;
    primaryVertex.Z() = 3.f;
    primaryVertex.Covariance(0) = 0.6f;
    primaryVertex.Covariance(2) = 0.5f;
    primaryVertex.Covariance(5) = 0.4f;

    KFParticleGpuPrimaryVertexTopologyObservables observables;
    assert(KFParticleGpuSelection::BuildPrimaryVertexTopologyObservables(
      candidate, primaryVertex, observables));
    assert(observables.valid == 1u);
    assert(AlmostEqual(observables.distance, 5.f));
    assert(AlmostEqual(observables.distanceError, 1.f));
    assert(AlmostEqual(observables.ldL, 5.f));
    assert(AlmostEqual(observables.chi2PerNdf, 25.f / 3.f));

    float vertexParameters[2 * KFParticleGpuVertexState::NumberOfParameters] = {
      4.f, 2.f,
      6.f, 2.f,
      3.f, 3.f
    };
    float vertexCovariances[2 * KFParticleGpuVertexState::NumberOfCovarianceElements] = {};
    vertexCovariances[0] = 0.6f;
    vertexCovariances[1] = 0.6f;
    vertexCovariances[2 * 2] = 0.5f;
    vertexCovariances[2 * 2 + 1] = 0.5f;
    vertexCovariances[5 * 2] = 0.4f;
    vertexCovariances[5 * 2 + 1] = 0.4f;
    float vertexChi2[2] = {0.f, 0.f};
    int vertexIntegers[2 * KFParticleGpuVertexSoALayout::NumberOfIntegerComponents] = {};
    const KFParticleGpuVertexSoAView vertices(
      vertexParameters, vertexCovariances, vertexChi2, vertexIntegers, 2u, 2u);

    int bestPrimaryVertexIndex = -1;
    KFParticleGpuPrimaryVertexTopologyObservables bestObservables;
    assert(KFParticleGpuSelection::FindBestPrimaryVertexTopology(
      candidate,
      MakeConstView(vertices),
      KFParticleGpuRange(0u, 2u),
      bestPrimaryVertexIndex,
      bestObservables));
    assert(bestPrimaryVertexIndex == 1);
    assert(AlmostEqual(bestObservables.distance, 1.f));
    assert(AlmostEqual(bestObservables.distanceError, 1.f));
    assert(AlmostEqual(bestObservables.ldL, 1.f));
    assert(AlmostEqual(bestObservables.chi2PerNdf, 1.f / 3.f));

    bestPrimaryVertexIndex = 7;
    assert(!KFParticleGpuSelection::FindBestPrimaryVertexTopology(
      candidate,
      MakeConstView(vertices),
      KFParticleGpuRange(0u, 0u),
      bestPrimaryVertexIndex,
      bestObservables));
    assert(bestPrimaryVertexIndex == -1);
    assert(!KFParticleGpuSelection::FindBestPrimaryVertexTopology(
      candidate,
      MakeConstView(vertices),
      KFParticleGpuRange(2u, 1u),
      bestPrimaryVertexIndex,
      bestObservables));

    KFParticleGpuFitState invalidCandidate;
    invalidCandidate.Initialize();
    invalidCandidate.X() = 0.f;
    for (int i = 0; i < KFParticleGpuFitState::NumberOfCovarianceElements; ++i) {
      invalidCandidate.Covariance(i) = 0.f;
    }
    KFParticleGpuVertexState invalidVertex;
    invalidVertex.X() = 1.f;
    assert(!KFParticleGpuSelection::BuildPrimaryVertexTopologyObservables(
      invalidCandidate, invalidVertex, observables));
    assert(observables.valid == 0u);
    assert(observables.distanceError == 1.e8f);
    assert(observables.chi2PerNdf == 1.e8f);

    Pass("primary-vertex-topology-observables",
         "candidate-PV distance, uncertainty, L/dL, spatial chi2, and best-PV guards are deterministic");
  }

  void TestV0LineTopologyContract()
  {
    KFParticleGpuFitState candidate;
    candidate.Initialize();
    candidate.X() = 3.f;
    candidate.Y() = 4.f;
    candidate.Z() = 0.f;
    candidate.Px() = 1.f;
    candidate.Py() = 0.f;
    candidate.Pz() = 0.f;
    for (int i = 0; i < KFParticleGpuFitState::NumberOfCovarianceElements; ++i) {
      candidate.Covariance(i) = 0.f;
    }
    candidate.Covariance(0, 0) = 0.4f;
    candidate.Covariance(1, 0) = 0.05f;
    candidate.Covariance(1, 1) = 0.5f;
    candidate.Covariance(2, 2) = 0.6f;

    KFParticleGpuVertexState primaryVertex;
    primaryVertex.Initialize();
    primaryVertex.Covariance(0) = 0.6f;
    primaryVertex.Covariance(1) = 0.15f;
    primaryVertex.Covariance(2) = 0.5f;
    primaryVertex.Covariance(5) = 0.4f;

    KFParticleGpuV0LineTopologyResult result;
    assert(KFParticleGpuSelection::BuildV0LineTopology(candidate, primaryVertex, result));
    assert(result.status == KFGpuV0LineTopologyValid);
    // Closed-form CPU GetDistanceToVertexLine oracle for
    // C = {{1,.2,0},{.2,1,0},{0,0,1}}.
    assert(AlmostEqual(result.pathToPrimaryVertex, 3.f));
    assert(AlmostEqual(result.lineDistance, 5.f));
    assert(AlmostEqual(result.lineDistanceError, std::sqrt(29.8f) / 5.f));
    assert(AlmostEqual(result.lineLdL, 5.f / (std::sqrt(29.8f) / 5.f)));
    assert(AlmostEqual(result.decayLength, 2.69427f, 1.e-4f));
    assert(AlmostEqual(result.decayLengthError, 0.991419f, 1.e-4f));
    assert(AlmostEqual(result.decayLdL, 2.71759f, 1.e-4f));
    assert(AlmostEqual(result.pointingCosine, 0.6f));
    assert(AlmostEqual(result.lineChi2PerNdf, (20.2f / 0.96f) / 3.f));

    float vertexParameters[2 * KFParticleGpuVertexState::NumberOfParameters] = {
      0.f, 0.f,
      0.f, 0.f,
      0.f, 0.f};
    float vertexCovariances[2 * KFParticleGpuVertexState::NumberOfCovarianceElements] = {};
    vertexCovariances[0] = 0.6f;
    vertexCovariances[1] = 0.6f;
    vertexCovariances[2] = 0.15f;
    vertexCovariances[3] = 0.15f;
    vertexCovariances[4] = 0.5f;
    vertexCovariances[5] = 0.5f;
    vertexCovariances[10] = 0.4f;
    vertexCovariances[11] = 0.4f;
    float vertexChi2[2] = {0.f, 0.f};
    int vertexIntegers[2 * KFParticleGpuVertexSoALayout::NumberOfIntegerComponents] = {};
    const KFParticleGpuVertexSoAView vertices(
      vertexParameters, vertexCovariances, vertexChi2, vertexIntegers, 2u, 2u);

    KFParticleGpuV0LineTopologyResult best;
    assert(KFParticleGpuSelection::FindBestV0LineTopology(
      candidate, MakeConstView(vertices), KFParticleGpuRange(0u, 2u), best));
    assert(best.primaryVertexIndex == 0);
    assert(AlmostEqual(best.lineChi2PerNdf, result.lineChi2PerNdf));

    KFParticleGpuFitState zeroMomentum = candidate;
    zeroMomentum.Px() = 0.f;
    assert(!KFParticleGpuSelection::BuildV0LineTopology(zeroMomentum, primaryVertex, result));
    assert(result.status == KFGpuV0LineTopologyDegenerateMomentum);

    KFParticleGpuFitState singularCandidate = candidate;
    for (int i = 0; i < KFParticleGpuFitState::NumberOfCovarianceElements; ++i) {
      singularCandidate.Covariance(i) = 0.f;
    }
    KFParticleGpuVertexState singularVertex;
    singularVertex.Initialize();
    assert(KFParticleGpuSelection::BuildV0LineTopology(
      singularCandidate, singularVertex, result));
    assert(result.status == KFGpuV0LineTopologyValid);

    KFParticleGpuFitState negativeProjection = candidate;
    negativeProjection.Covariance(1, 0) = -2.2f;
    assert(KFParticleGpuSelection::BuildV0LineTopology(
      negativeProjection, primaryVertex, result));
    assert((result.status & KFGpuV0LineTopologyValid) != 0u);
    assert((result.status & KFGpuV0LineTopologyDegenerateCovariance) != 0u);
    assert(result.lineDistanceError == 1.e8f);
    assert(result.lineLdL == 0.f);
    assert(result.pointingCosine > 0.f);
    assert(result.decayLengthError < 1.e8f);

    KFParticleGpuFitState nonFinite = candidate;
    nonFinite.X() = std::numeric_limits<float>::quiet_NaN();
    assert(!KFParticleGpuSelection::BuildV0LineTopology(nonFinite, primaryVertex, result));
    assert(result.status == KFGpuV0LineTopologyNonFinite);

    Pass("v0-line-topology-contract",
         "correlated-covariance line-to-PV observables, CPU-compatible covariance fallback, stable PV ties, and bounded rejection states");
  }

  void TestV0SelectionDecision()
  {
    KFParticleGpuFitState candidate;
    candidate.Initialize();
    candidate.X() = 1.f;
    candidate.Y() = 2.f;
    candidate.Z() = 3.f;
    candidate.E() = 1.f;
    candidate.Chi2() = 3.f;
    candidate.NDF() = 2;
    for (int i = 0; i < KFParticleGpuFitState::NumberOfCovarianceElements; ++i) {
      candidate.Covariance(i) = 0.f;
    }
    candidate.Covariance(0, 0) = 0.4f;
    candidate.Covariance(1, 1) = 0.5f;
    candidate.Covariance(2, 2) = 0.6f;

    float vertexParameters[2 * KFParticleGpuVertexState::NumberOfParameters] = {
      4.f, 2.f,
      6.f, 2.f,
      3.f, 3.f
    };
    float vertexCovariances[2 * KFParticleGpuVertexState::NumberOfCovarianceElements] = {};
    vertexCovariances[0] = 0.6f;
    vertexCovariances[1] = 0.6f;
    vertexCovariances[2 * 2] = 0.5f;
    vertexCovariances[2 * 2 + 1] = 0.5f;
    vertexCovariances[5 * 2] = 0.4f;
    vertexCovariances[5 * 2 + 1] = 0.4f;
    float vertexChi2[2] = {0.f, 0.f};
    int vertexIntegers[2 * KFParticleGpuVertexSoALayout::NumberOfIntegerComponents] = {};
    const KFParticleGpuVertexSoAView vertices(
      vertexParameters, vertexCovariances, vertexChi2, vertexIntegers, 2u, 2u);

    KFParticleGpuV0SelectionConfig config;
    config.expectedMass = 1.f;
    config.expectedMassSigma = 0.1f;
    config.massSigmaCut = 3.f;
    config.maxGeometricChi2PerNdf = 2.f;
    config.maxPrimaryVertexDistance = 200.f;
    config.minSecondaryLdL = 0.5f;
    config.maxPrimaryTopologyChi2PerNdf = 1.f;
    config.maxSecondaryTopologyChi2PerNdf = 1.f;
    config.requirePrimaryVertex = 1u;
    const unsigned int validFlags =
      static_cast<unsigned int>(KFGpuCandidateValid | KFGpuCandidateLineDca | KFGpuCandidateEnergyFit);

    KFParticleGpuV0SelectionResult result;
    KFParticleGpuSelection::EvaluateV0Selection(
      candidate,
      validFlags,
      5u,
      KFGpuChannelK0ShortToPiPlusPiMinus,
      3u,
      MakeConstView(vertices),
      KFParticleGpuRange(0u, 2u),
      config,
      result);
    assert(KFParticleGpuSelection::IsSelected(result));
    assert(result.selectionClass == KFGpuV0SelectionPrimary);
    assert(result.bestPrimaryVertexIndex == 1);
    assert(result.rejectionReasons == KFGpuV0SelectionRejectNone);
    assert(AlmostEqual(result.observables.mass, 1.f));
    assert(AlmostEqual(result.observables.geometricChi2PerNdf, 1.5f));
    assert(AlmostEqual(result.observables.nearestPrimaryVertexDistance, 1.f));
    assert(AlmostEqual(result.observables.bestPrimaryVertexTopoChi2PerNdf, 1.f / 3.f));

    KFParticleGpuV0SelectionConfig secondaryConfig = config;
    secondaryConfig.maxPrimaryTopologyChi2PerNdf = 0.1f;
    KFParticleGpuSelection::EvaluateV0Selection(
      candidate, validFlags, 5u, 1u, 3u, MakeConstView(vertices), KFParticleGpuRange(0u, 2u),
      secondaryConfig, result);
    assert(KFParticleGpuSelection::IsSelected(result));
    assert(result.selectionClass == KFGpuV0SelectionSecondary);

    KFParticleGpuSelection::EvaluateV0Selection(
      candidate,
      KFGpuCandidateBuildFailed,
      5u,
      1u,
      3u,
      MakeConstView(vertices),
      KFParticleGpuRange(0u, 2u),
      config,
      result);
    assert(KFParticleGpuSelection::HasRejection(result, KFGpuV0SelectionRejectBuild));
    assert(!KFParticleGpuSelection::IsSelected(result));

    KFParticleGpuV0SelectionConfig massConfig = config;
    massConfig.expectedMass = 2.f;
    KFParticleGpuSelection::EvaluateV0Selection(
      candidate, validFlags, 5u, 1u, 3u, MakeConstView(vertices), KFParticleGpuRange(0u, 2u),
      massConfig, result);
    assert(KFParticleGpuSelection::HasRejection(result, KFGpuV0SelectionRejectMass));

    KFParticleGpuV0SelectionConfig geometryConfig = config;
    geometryConfig.maxGeometricChi2PerNdf = 1.f;
    KFParticleGpuSelection::EvaluateV0Selection(
      candidate, validFlags, 5u, 1u, 3u, MakeConstView(vertices), KFParticleGpuRange(0u, 2u),
      geometryConfig, result);
    assert(KFParticleGpuSelection::HasRejection(result, KFGpuV0SelectionRejectGeometricChi2));

    KFParticleGpuSelection::EvaluateV0Selection(
      candidate, validFlags, 5u, 1u, 3u, MakeConstView(vertices), KFParticleGpuRange(0u, 0u),
      config, result);
    assert(KFParticleGpuSelection::HasRejection(result, KFGpuV0SelectionRejectNoPrimaryVertex));

    KFParticleGpuV0SelectionConfig distanceConfig = config;
    distanceConfig.maxPrimaryVertexDistance = 0.5f;
    KFParticleGpuSelection::EvaluateV0Selection(
      candidate, validFlags, 5u, 1u, 3u, MakeConstView(vertices), KFParticleGpuRange(0u, 2u),
      distanceConfig, result);
    assert(KFParticleGpuSelection::HasRejection(result, KFGpuV0SelectionRejectDistance));

    KFParticleGpuV0SelectionConfig decayConfig = secondaryConfig;
    decayConfig.minSecondaryLdL = 2.f;
    KFParticleGpuSelection::EvaluateV0Selection(
      candidate, validFlags, 5u, 1u, 3u, MakeConstView(vertices), KFParticleGpuRange(0u, 2u),
      decayConfig, result);
    assert(KFParticleGpuSelection::HasRejection(result, KFGpuV0SelectionRejectDecayLength));

    KFParticleGpuV0SelectionConfig topologyConfig = secondaryConfig;
    topologyConfig.maxSecondaryTopologyChi2PerNdf = 0.2f;
    KFParticleGpuSelection::EvaluateV0Selection(
      candidate, validFlags, 5u, 1u, 3u, MakeConstView(vertices), KFParticleGpuRange(0u, 2u),
      topologyConfig, result);
    assert(KFParticleGpuSelection::HasRejection(result, KFGpuV0SelectionRejectTopology));

    KFParticleGpuFitState nonFiniteCandidate = candidate;
    nonFiniteCandidate.X() = std::numeric_limits<float>::quiet_NaN();
    KFParticleGpuSelection::EvaluateV0Selection(
      nonFiniteCandidate,
      validFlags,
      5u,
      1u,
      3u,
      MakeConstView(vertices),
      KFParticleGpuRange(0u, 2u),
      config,
      result);
    assert(KFParticleGpuSelection::HasRejection(result, KFGpuV0SelectionRejectNonFinite));

    Pass("v0-selection-decision",
         "configurable default-V0 decision classifies primary/secondary candidates and reports every rejection reason");
  }

  void TestV0LineTopologySelectionDecision()
  {
    KFParticleGpuFitState candidate;
    candidate.Initialize();
    candidate.X() = 3.f;
    candidate.Y() = 4.f;
    candidate.Px() = 1.f;
    candidate.E() = 2.f;
    candidate.Chi2() = 1.f;
    candidate.NDF() = 1;
    candidate.Covariance(0, 0) = 0.4f;
    candidate.Covariance(1, 0) = 0.05f;
    candidate.Covariance(1, 1) = 0.5f;
    candidate.Covariance(2, 2) = 0.6f;

    float vertexParameters[KFParticleGpuVertexState::NumberOfParameters] = {0.f, 0.f, 0.f};
    float vertexCovariances[KFParticleGpuVertexState::NumberOfCovarianceElements] = {
      0.6f, 0.15f, 0.5f, 0.f, 0.f, 0.4f};
    float vertexChi2[1] = {0.f};
    int vertexIntegers[KFParticleGpuVertexSoALayout::NumberOfIntegerComponents] = {};
    const KFParticleGpuVertexSoAView vertices(
      vertexParameters, vertexCovariances, vertexChi2, vertexIntegers, 1u, 1u);

    KFParticleGpuV0SelectionConfig config;
    config.requirePrimaryVertex = 1u;
    config.maxPrimaryVertexDistance = 10.f;
    config.topologyMode = KFGpuV0TopologyLine;
    const unsigned int validFlags = static_cast<unsigned int>(KFGpuCandidateValid);
    KFParticleGpuV0SelectionResult result;
    KFParticleGpuSelection::EvaluateV0Selection(
      candidate, validFlags, 7u, KFGpuChannelK0ShortToPiPlusPiMinus, 2u,
      MakeConstView(vertices), KFParticleGpuRange(0u, 1u), config, result);
    assert(KFParticleGpuSelection::IsSelected(result));
    assert(result.selectionClass == KFGpuV0SelectionSecondary);
    assert(result.bestPrimaryVertexIndex == 0);
    assert(result.topologyStatus == KFGpuV0LineTopologyValid);
    assert(AlmostEqual(result.observables.nearestPrimaryVertexDistance, 5.f));
    assert(AlmostEqual(
      result.observables.bestPrimaryVertexDecayLength, 2.69427f, 1.e-4f));

    config.maxPrimaryVertexDistance = 4.f;
    KFParticleGpuSelection::EvaluateV0Selection(
      candidate, validFlags, 7u, KFGpuChannelK0ShortToPiPlusPiMinus, 2u,
      MakeConstView(vertices), KFParticleGpuRange(0u, 1u), config, result);
    assert(KFParticleGpuSelection::HasRejection(result, KFGpuV0SelectionRejectDistance));

    config.maxPrimaryVertexDistance = 10.f;
    KFParticleGpuFitState backwardCandidate = candidate;
    backwardCandidate.Px() = -1.f;
    KFParticleGpuSelection::EvaluateV0Selection(
      backwardCandidate, validFlags, 7u, KFGpuChannelK0ShortToPiPlusPiMinus, 2u,
      MakeConstView(vertices), KFParticleGpuRange(0u, 1u), config, result);
    assert(KFParticleGpuSelection::HasRejection(
      result, KFGpuV0SelectionRejectPointing));

    config.topologyMode = 99;
    KFParticleGpuSelection::EvaluateV0Selection(
      candidate, validFlags, 7u, KFGpuChannelK0ShortToPiPlusPiMinus, 2u,
      MakeConstView(vertices), KFParticleGpuRange(0u, 1u), config, result);
    assert(KFParticleGpuSelection::HasRejection(result, KFGpuV0SelectionRejectTopology));

    Pass("v0-line-topology-selection",
         "line topology drives default-V0 distance, pointing, PV identity, and bounded selection rejection");
  }

  bool ContainsSelectedCandidate(const KFParticleGpuConstSelectedCandidateIndexView& selected,
                                 unsigned int candidateIndex)
  {
    for (unsigned int i = 0; i < selected.Size(); ++i) {
      if (selected.Index(i) == candidateIndex) {
        return true;
      }
    }
    return false;
  }

  void TestSelectedCandidateOutput(KFParticleGpuRuntime& runtime,
                                   KFParticleGpuBufferManager& buffers)
  {
    FillInput(buffers, buffers.Capacities());
    KFParticleGpuVertexSoAView vertices = buffers.HostPrimaryVertices();
    vertices.Parameter(0, 0) = 4.f;
    vertices.Parameter(1, 0) = 0.f;
    vertices.Parameter(2, 0) = 0.f;
    for (unsigned int i = 0; i < KFParticleGpuVertexState::NumberOfCovarianceElements; ++i) {
      vertices.Covariance(i, 0) = 0.f;
    }
    vertices.Covariance(0, 0) = 0.6f;
    vertices.Covariance(1, 0) = 0.5f;
    vertices.Covariance(2, 0) = 0.4f;

    KFParticleGpuFitState candidate;
    candidate.Initialize();
    candidate.X() = 3.f;
    candidate.Y() = 0.f;
    candidate.Z() = 0.f;
    candidate.E() = 1.f;
    candidate.Chi2() = 3.f;
    candidate.NDF() = 2;
    for (int i = 0; i < KFParticleGpuFitState::NumberOfCovarianceElements; ++i) {
      candidate.Covariance(i) = 0.f;
    }
    candidate.Covariance(0, 0) = 0.4f;
    candidate.Covariance(1, 1) = 0.5f;
    candidate.Covariance(2, 2) = 0.6f;

    KFParticleGpuCandidatePoolView rawCandidates = buffers.HostCandidates();
    for (unsigned int i = 0; i < 3u; ++i) {
      StoreCandidateFit(candidate, rawCandidates, i);
      rawCandidates.Metadata().Pdg(i) = 310;
      rawCandidates.Metadata().PrimaryVertexIndex(i) = -1;
      rawCandidates.Metadata().EventIndex(i) = 0u;
      rawCandidates.Metadata().DaughterOffset(i) = 0u;
      rawCandidates.Metadata().DaughterCount(i) = 0u;
      rawCandidates.Metadata().Flags(i) = static_cast<unsigned int>(
        KFGpuCandidateValid | KFGpuCandidateLineDca | KFGpuCandidateEnergyFit);
    }
    rawCandidates.SizeData()[0] = 3u;
    rawCandidates.Daughters().SizeData()[0] = 0u;
    rawCandidates.OverflowFlagsData()[0] = 0u;

    KFParticleGpuTwoDaughterChannel channel = MakeK0ShortToPiPlusPiMinusChannel();
    channel.selection.expectedMass = 1.f;
    channel.selection.expectedMassSigma = 0.1f;
    channel.selection.massSigmaCut = 3.f;
    channel.selection.maxGeometricChi2PerNdf = 2.f;
    channel.selection.maxPrimaryVertexDistance = 200.f;
    channel.selection.minSecondaryLdL = 0.1f;
    channel.selection.maxPrimaryTopologyChi2PerNdf = 1.f;
    channel.selection.maxSecondaryTopologyChi2PerNdf = 1.f;
    channel.selection.requirePrimaryVertex = 1u;
    // This fixture isolates compact-output ownership, so retain the legacy
    // spatial selection baseline rather than depending on line geometry.
    channel.selection.topologyMode = KFGpuV0TopologySpatial;

    buffers.UploadInput();
    buffers.UploadCandidates();
    const KFParticleGpuCandidateRange rawRange = {0u, 3u, 0u, 0u, 0u};
    const KFParticleGpuSelectedCandidateRange selectedRange =
      runtime.GetSteering().RunV0Selection(channel, rawRange, 0u);
    assert(selectedRange.offset == 0u);
    assert(selectedRange.size == 3u);
    assert(!selectedRange.Truncated());
    const KFParticleGpuConstSelectedCandidateIndexView selected =
      MakeConstView(buffers.HostSelectedCandidates());
    assert(selected.Size() == 3u);
    assert(selected.OverflowFlags() == 0u);
    assert(ContainsSelectedCandidate(selected, 0u));
    assert(ContainsSelectedCandidate(selected, 1u));
    assert(ContainsSelectedCandidate(selected, 2u));
    assert(MakeConstView(buffers.HostCandidates()).Size() == 3u);

    xpu::buffer<unsigned int> limitedIndices(2u, xpu::buf_io);
    xpu::buffer<unsigned int> limitedSize(1u, xpu::buf_io);
    xpu::buffer<unsigned int> limitedOverflow(1u, xpu::buf_io);
    HostPointer(limitedSize)[0] = 0u;
    HostPointer(limitedOverflow)[0] = 0u;
    runtime.GetQueue().copy(limitedSize, xpu::h2d);
    runtime.GetQueue().copy(limitedOverflow, xpu::h2d);
    runtime.GetQueue().wait();
    const KFParticleGpuSelectedCandidateIndexView limitedSelected(
      limitedIndices.get(), limitedSize.get(), limitedOverflow.get(), 2u);
    runtime.GetQueue().launch<KFParticleGpuSelectV0Candidates>(
      xpu::n_threads(rawRange.size),
      MakeConstView(buffers.DeviceCandidates()),
      MakeConstView(buffers.DevicePrimaryVertices()),
      buffers.DeviceEvents(),
      0u,
      rawRange.offset,
      rawRange.size,
      0u,
      channel.selection,
      KFParticleGpuV0SelectionResultView(),
      limitedSelected);
    runtime.GetQueue().wait();
    runtime.GetQueue().copy(limitedIndices, xpu::d2h);
    runtime.GetQueue().copy(limitedSize, xpu::d2h);
    runtime.GetQueue().copy(limitedOverflow, xpu::d2h);
    runtime.GetQueue().wait();
    const KFParticleGpuConstSelectedCandidateIndexView limitedHost(
      HostPointer(limitedIndices), HostPointer(limitedSize), HostPointer(limitedOverflow), 2u);
    assert(limitedHost.Size() == 2u);
    assert(limitedHost.OverflowFlags() == KFGpuSelectedCandidateCapacityExceeded);
    assert(limitedHost.Index(0) < rawRange.size);
    assert(limitedHost.Index(1) < rawRange.size);
    assert(limitedHost.Index(0) != limitedHost.Index(1));

    Pass("v0-selected-candidate-output",
         "selected candidate indices compact independently from raw candidates and report bounded-output overflow");
  }

  KFParticleGpuTwoDaughterChannel MakeTestChannel(unsigned int channelId,
                                                  int motherPdg,
                                                  KFParticleGpuTrackSpecies firstSpecies,
                                                  KFParticleGpuTrackSpecies secondSpecies)
  {
    KFParticleGpuTwoDaughterChannel channel;
    channel.channelId = channelId;
    channel.firstTrackSet = SecondaryPositiveFirst;
    channel.secondTrackSet = SecondaryNegativeFirst;
    channel.firstSpecies = firstSpecies;
    channel.secondSpecies = secondSpecies;
    channel.flags = KFGpuTwoDaughterUseLineDca | KFGpuTwoDaughterUseEnergyFit;
    channel.motherPdg = motherPdg;
    channel.firstDaughterPdg = 211;
    channel.secondDaughterPdg = -211;
    channel.primaryVertexIndex = -1;
    channel.firstMass = 0.13957f;
    channel.secondMass = 0.13957f;
    channel.firstCharge = 1;
    channel.secondCharge = -1;
    channel.minFirstPixelHits = 5;
    channel.minSecondPixelHits = 5;
    channel.maxSecondChiToPrimaryVertex = 4.f;
    return channel;
  }

  void TestDecayPlanChannelList()
  {
    KFParticleGpuDecayPlan plan;
    assert(plan.Empty());
    assert(plan.NumberOfTwoDaughterChannels() == 0);

    const KFParticleGpuTwoDaughterChannel first =
      MakeTestChannel(3u, 310, Pion, Pion);
    const KFParticleGpuTwoDaughterChannel second =
      MakeTestChannel(4u, 333, Kaon, Kaon);
    plan.AddTwoDaughterChannel(first);
    plan.AddTwoDaughterChannel(second);

    assert(!plan.Empty());
    assert(plan.NumberOfTwoDaughterChannels() == 2);
    assert(plan.TwoDaughterChannel(0).channelId == 3u);
    assert(plan.TwoDaughterChannel(0).motherPdg == 310);
    assert(plan.TwoDaughterChannel(0).firstSpecies == Pion);
    assert(plan.TwoDaughterChannel(1).channelId == 4u);
    assert(plan.TwoDaughterChannel(1).motherPdg == 333);
    assert(plan.TwoDaughterChannel(1).firstSpecies == Kaon);
    assert(plan.TwoDaughterChannel(1).secondSpecies == Kaon);

    const KFParticleGpuTwoDaughterTaskSource source =
      MakeTwoDaughterTaskSource(plan.TwoDaughterChannel(1), 2u);
    assert(source.eventIndex == 2u);
    assert(source.motherPdg == 333);
    assert(source.firstSpecies == Kaon);

    bool outOfRangeRejected = false;
    try {
      (void) plan.TwoDaughterChannel(2);
    }
    catch (const std::out_of_range&) {
      outOfRangeRejected = true;
    }
    assert(outOfRangeRejected);

    plan.Clear();
    assert(plan.Empty());
    assert(plan.NumberOfTwoDaughterChannels() == 0);

    Pass("decay-plan-channel-list",
         "decay plan preserves ordered two-daughter channels and clear/index guards");
  }

  void TestDefaultV0DecayPlanBuilders()
  {
    KFParticleGpuDecayPlan plan;
    AddDefaultV0TwoDaughterChannels(plan);
    assert(plan.NumberOfTwoDaughterChannels() == 3u);

    const KFParticleGpuTwoDaughterChannel& k0 = plan.TwoDaughterChannel(0);
    assert(k0.channelId == KFGpuChannelK0ShortToPiPlusPiMinus);
    assert(k0.motherPdg == 310);
    assert(k0.firstSpecies == Pion);
    assert(k0.secondSpecies == Pion);
    assert(k0.firstDaughterPdg == 211);
    assert(k0.secondDaughterPdg == -211);
    assert(AlmostEqual(k0.firstMass, kCpuReferencePionMass));
    assert(AlmostEqual(k0.secondMass, kCpuReferencePionMass));
    assert(AlmostEqual(k0.motherMass, kCpuReferenceK0ShortMass));
    assert(AlmostEqual(k0.motherMassSigma, kCpuReferenceK0ShortMassSigma));
    assert(AlmostEqual(k0.secondaryMassSigmaCut, 3.f));
    assert(AlmostEqual(k0.maxSecondaryTopoChi2PerNdf, kCpuReferenceSecondaryTopoChi2));
    assert(AlmostEqual(k0.minSecondaryLdL, 10.f));
    assert(k0.transportMode == KFGpuTransportFullField);
    assert(AlmostEqual(k0.selection.expectedMass, kCpuReferenceK0ShortMass));
    assert(AlmostEqual(k0.selection.expectedMassSigma, kCpuReferenceK0ShortMassSigma));
    assert(AlmostEqual(k0.selection.maxGeometricChi2PerNdf, 3.f));
    assert(AlmostEqual(k0.selection.maxPrimaryVertexDistance, 200.f));
    assert(AlmostEqual(k0.selection.minSecondaryLdL, 5.f));
    assert(k0.selection.requirePrimaryVertex == 1u);
    assert(k0.selection.topologyMode == KFGpuV0TopologyLine);

    const KFParticleGpuTwoDaughterChannel& lambda = plan.TwoDaughterChannel(1);
    assert(lambda.channelId == KFGpuChannelLambdaToProtonPiMinus);
    assert(lambda.motherPdg == 3122);
    assert(lambda.firstSpecies == NumberOfTrackSpecies);
    assert(lambda.secondSpecies == Pion);
    assert(lambda.firstDaughterPdg == 2212);
    assert(lambda.secondDaughterPdg == -211);
    assert(lambda.firstSourcePdg == 2212);
    assert(lambda.firstAlternateSourcePdg == 211);
    assert(AlmostEqual(lambda.firstMass, kCpuReferenceProtonMass));
    assert(AlmostEqual(lambda.secondMass, kCpuReferencePionMass));
    assert(AlmostEqual(lambda.motherMass, kCpuReferenceLambdaMass));
    assert(AlmostEqual(lambda.motherMassSigma, kCpuReferenceLambdaMassSigma));
    assert(AlmostEqual(lambda.secondaryMassSigmaCut, 3.f));
    assert(AlmostEqual(lambda.maxSecondaryTopoChi2PerNdf, kCpuReferenceSecondaryTopoChi2));
    assert(AlmostEqual(lambda.minSecondaryLdL, 10.f));
    assert(lambda.transportMode == KFGpuTransportFullField);
    assert(AlmostEqual(lambda.selection.expectedMass, kCpuReferenceLambdaMass));
    assert(AlmostEqual(lambda.selection.expectedMassSigma, kCpuReferenceLambdaMassSigma));
    assert(AlmostEqual(lambda.selection.minSecondaryLdL, 5.f));
    assert(lambda.selection.topologyMode == KFGpuV0TopologyLine);

    const KFParticleGpuTwoDaughterChannel& antiLambda = plan.TwoDaughterChannel(2);
    assert(antiLambda.channelId == KFGpuChannelAntiLambdaToAntiProtonPiPlus);
    assert(antiLambda.motherPdg == -3122);
    assert(antiLambda.firstTrackSet == SecondaryNegativeFirst);
    assert(antiLambda.secondTrackSet == SecondaryPositiveFirst);
    assert(antiLambda.firstSpecies == Proton);
    assert(antiLambda.secondSpecies == Pion);
    assert(antiLambda.firstDaughterPdg == -2212);
    assert(antiLambda.secondDaughterPdg == 211);
    assert(AlmostEqual(antiLambda.firstMass, kCpuReferenceProtonMass));
    assert(AlmostEqual(antiLambda.secondMass, kCpuReferencePionMass));
    assert(AlmostEqual(antiLambda.motherMass, kCpuReferenceLambdaMass));
    assert(AlmostEqual(antiLambda.motherMassSigma, kCpuReferenceLambdaMassSigma));
    assert(AlmostEqual(antiLambda.secondaryMassSigmaCut, 3.f));
    assert(AlmostEqual(antiLambda.maxSecondaryTopoChi2PerNdf, kCpuReferenceSecondaryTopoChi2));
    assert(AlmostEqual(antiLambda.minSecondaryLdL, 10.f));
    assert(antiLambda.transportMode == KFGpuTransportFullField);
    assert(antiLambda.firstCharge == -1);
    assert(antiLambda.secondCharge == 1);
    assert(AlmostEqual(antiLambda.selection.minSecondaryLdL, 5.f));
    assert(antiLambda.selection.topologyMode == KFGpuV0TopologyLine);

    Pass("decay-plan-default-v0-builders",
         "default V0 channel builders provide ordered full-field K0S, Lambda, and anti-Lambda descriptors");
  }

  void TestKinematicMathSeed()
  {
    KFParticleGpuTrackState firstTrack;
    firstTrack.Px() = 3.f;
    firstTrack.Py() = 0.f;
    firstTrack.Pz() = 4.f;
    firstTrack.Covariance(3, 3) = 0.04f;
    firstTrack.Covariance(4, 4) = 0.05f;
    firstTrack.Covariance(5, 5) = 0.06f;
    KFParticleGpuFitState first;
    first.Initialize(firstTrack, 1, 0.5f);

    KFParticleGpuTrackState secondTrack;
    secondTrack.Px() = -1.f;
    secondTrack.Py() = 2.f;
    secondTrack.Pz() = 2.f;
    secondTrack.Covariance(3, 3) = 0.03f;
    secondTrack.Covariance(4, 4) = 0.02f;
    secondTrack.Covariance(5, 5) = 0.01f;
    KFParticleGpuFitState second;
    second.Initialize(secondTrack, -1, 0.25f);

    KFParticleGpuFitState mother;
    KFParticleGpuMath::BuildKinematicMother(first, second, mother);

    assert(mother.Q() == 0);
    assert(mother.Px() == 2.f);
    assert(mother.Py() == 2.f);
    assert(mother.Pz() == 6.f);
    assert(AlmostEqual(mother.E(), first.E() + second.E()));
    assert(AlmostEqual(KFParticleGpuMath::Momentum2(mother), 44.f));
    assert(AlmostEqual(KFParticleGpuMath::TransverseMomentum2(mother), 8.f));
    assert(KFParticleGpuMath::Mass2(mother) > 0.f);

    float mass = 0.f;
    float massError = 0.f;
    const bool validMass = KFParticleGpuMath::GetMass(mother, mass, massError);
    assert(validMass);
    assert(AlmostEqual(mass, KFParticleGpuMath::Mass(mother)));
    assert(massError > 0.f);
    assert(massError < 1.e8f);
    Pass("math-mass-error", "device-safe invariant mass and covariance-based mass error helpers");
    Pass("math-kinematic-seed", "device-safe two-daughter kinematic seed helpers");
  }

  void TestMeasurementSeed()
  {
    KFParticleGpuFitState first;
    first.X() = 0.f;
    first.Y() = 0.f;
    first.Z() = 0.f;
    first.Px() = 1.f;
    first.Py() = 0.f;
    first.Pz() = 0.f;
    first.E() = 1.5f;
    first.Q() = 1;
    first.Covariance(0, 0) = 2.f;
    first.Covariance(3, 3) = 0.2f;

    KFParticleGpuFitState second;
    second.X() = 0.f;
    second.Y() = 1.f;
    second.Z() = 0.f;
    second.Px() = 0.f;
    second.Py() = -1.f;
    second.Pz() = 0.f;
    second.E() = 1.25f;
    second.Q() = -1;
    second.Covariance(1, 1) = 3.f;
    second.Covariance(4, 4) = 0.3f;

    KFParticleGpuFitState currentAtDca;
    KFParticleGpuMeasurement measurement;
    const bool built =
      KFParticleGpuMath::BuildLineDcaMeasurementSeed(first, second, currentAtDca, measurement);
    assert(built);
    assert(currentAtDca.X() == 0.f);
    assert(currentAtDca.Y() == 0.f);
    assert(currentAtDca.Px() == first.Px());
    assert(currentAtDca.Covariance(3, 3) == first.Covariance(3, 3));
    assert(measurement.Parameter(0) == 0.f);
    assert(measurement.Parameter(1) == 0.f);
    assert(measurement.Parameter(4) == second.Py());
    assert(measurement.Covariance(4, 4) == second.Covariance(4, 4));
    assert(KFParticleGpuMath::Abs(measurement.Correlation(0, 0)) > 1.e-6f);
    assert(measurement.Covariance(0, 0) >= 0.f);

    KFParticleGpuFitState farSecond = second;
    farSecond.Z() = 2001.f;
    farSecond.Pz() = -1.f;
    KFParticleGpuFitState rejectedCurrent;
    KFParticleGpuMeasurement rejectedMeasurement;
    assert(!KFParticleGpuMath::BuildLineDcaMeasurementSeed(
      first, farSecond, rejectedCurrent, rejectedMeasurement));
    Pass("measurement-seed", "straight-line DCA measurement arrays, covariance transport, and correlation block");
  }

  void TestFieldMeasurementSeedApprox()
  {
    KFParticleGpuFitState first;
    first.X() = 1.f;
    first.Y() = 2.f;
    first.Z() = 3.f;
    first.Px() = 4.f;
    first.Py() = 5.f;
    first.Pz() = 6.f;
    first.E() = 10.f;
    first.Q() = 1;
    first.NDF() = 0;
    first.NDF() = 0;
    first.Covariance(0, 0) = 0.4f;
    first.Covariance(1, 1) = 0.5f;
    first.Covariance(2, 2) = 0.6f;
    first.Covariance(3, 3) = 0.1f;
    first.Covariance(4, 4) = 0.2f;
    first.Covariance(5, 5) = 0.3f;

    KFParticleGpuFitState second;
    second.X() = -0.5f;
    second.Y() = 1.5f;
    second.Z() = 2.5f;
    second.Px() = -3.f;
    second.Py() = 2.f;
    second.Pz() = 5.f;
    second.E() = 8.f;
    second.Q() = -1;
    second.NDF() = 0;
    second.NDF() = 0;
    second.Covariance(0, 0) = 0.7f;
    second.Covariance(1, 1) = 0.8f;
    second.Covariance(2, 2) = 0.9f;
    second.Covariance(3, 3) = 0.15f;
    second.Covariance(4, 4) = 0.25f;
    second.Covariance(5, 5) = 0.35f;

    KFParticleGpuFitState lineCurrent;
    KFParticleGpuMeasurement lineMeasurement;
    assert(KFParticleGpuMath::BuildLineDcaMeasurementSeed(
      first, second, lineCurrent, lineMeasurement));

    KFParticleGpuFitState zeroCurrent;
    KFParticleGpuMeasurement zeroMeasurement;
    assert(KFParticleGpuMath::BuildConstantByDcaMeasurementSeedApprox(
      first, second, 0.f, zeroCurrent, zeroMeasurement));
    ExpectFitStateClose(zeroCurrent, lineCurrent, 1.e-5f);
    for (int i = 0; i < KFParticleGpuFitState::NumberOfParameters; ++i) {
      assert(AlmostEqual(zeroMeasurement.Parameter(i), lineMeasurement.Parameter(i), 1.e-5f));
    }

    KFParticleGpuFitState fieldCurrent;
    KFParticleGpuMeasurement fieldMeasurement;
    assert(KFParticleGpuMath::BuildConstantByDcaMeasurementSeedApprox(
      first, second, kDiagnosticFieldBy, fieldCurrent, fieldMeasurement));
    assert(!AlmostEqual(fieldCurrent.X(), lineCurrent.X(), 1.e-6f));
    assert(!AlmostEqual(fieldCurrent.Px(), lineCurrent.Px(), 1.e-6f));
    assert(!AlmostEqual(fieldMeasurement.Parameter(0), lineMeasurement.Parameter(0), 1.e-6f));
    for (int i = 0; i < KFParticleGpuFitState::NumberOfCovarianceElements; ++i) {
      assert(AlmostEqual(fieldCurrent.Covariance(i), lineCurrent.Covariance(i), 1.e-5f));
      assert(AlmostEqual(fieldMeasurement.Covariance(i), lineMeasurement.Covariance(i), 1.e-5f));
    }
    for (int row = 0; row < 3; ++row) {
      for (int column = 0; column < 3; ++column) {
        assert(AlmostEqual(fieldMeasurement.Correlation(row, column),
                           lineMeasurement.Correlation(row, column),
                           1.e-5f));
      }
    }

    KFParticleGpuFitState farSecond = second;
    farSecond.Z() = 3000.f;
    farSecond.Pz() = -1.f;
    KFParticleGpuFitState rejectedCurrent;
    KFParticleGpuMeasurement rejectedMeasurement;
    assert(!KFParticleGpuMath::BuildConstantByDcaMeasurementSeedApprox(
      first, farSecond, kDiagnosticFieldBy, rejectedCurrent, rejectedMeasurement));

    Pass("field-measurement-seed-approx",
         "field-aware measurement seed updates transported states while preserving line covariance approximation");
  }

  void TestFieldEnergyFit()
  {
    KFParticleGpuFitState first;
    first.Initialize();
    first.X() = 1.f;
    first.Y() = 2.f;
    first.Z() = 3.f;
    first.Px() = 4.f;
    first.Py() = 5.f;
    first.Pz() = 6.f;
    first.E() = 10.f;
    first.Q() = 1;
    first.NDF() = 0;
    first.Covariance(0, 0) = 0.4f;
    first.Covariance(1, 1) = 0.5f;
    first.Covariance(2, 2) = 0.6f;
    first.Covariance(3, 3) = 0.1f;
    first.Covariance(4, 4) = 0.2f;
    first.Covariance(5, 5) = 0.3f;
    first.Covariance(6, 6) = 0.4f;

    KFParticleGpuFitState second;
    second.Initialize();
    second.X() = -0.5f;
    second.Y() = 1.5f;
    second.Z() = 2.5f;
    second.Px() = -3.f;
    second.Py() = 2.f;
    second.Pz() = 5.f;
    second.E() = 8.f;
    second.Q() = -1;
    second.NDF() = 0;
    second.Covariance(0, 0) = 0.7f;
    second.Covariance(1, 1) = 0.8f;
    second.Covariance(2, 2) = 0.9f;
    second.Covariance(3, 3) = 0.15f;
    second.Covariance(4, 4) = 0.25f;
    second.Covariance(5, 5) = 0.35f;
    second.Covariance(6, 6) = 0.45f;

    KFParticleGpuFitState lineCurrent;
    KFParticleGpuMeasurement lineMeasurement;
    assert(KFParticleGpuMath::BuildLineDcaMeasurementSeed(
      first, second, lineCurrent, lineMeasurement));
    assert(KFParticleGpuMath::AddDaughterWithEnergyFit(
      lineCurrent, lineMeasurement, second.Q()));

    KFParticleGpuFitState fieldCurrent;
    KFParticleGpuMeasurement fieldMeasurement;
    assert(KFParticleGpuMath::BuildConstantByDcaMeasurementSeed(
      first, second, kDiagnosticFieldBy, fieldCurrent, fieldMeasurement));
    assert(KFParticleGpuMath::AddDaughterWithEnergyFit(
      fieldCurrent, fieldMeasurement, second.Q()));
    assert(fieldCurrent.NDF() == 2);
    assert(fieldCurrent.Q() == 0);
    assert(fieldCurrent.Chi2() >= 0.f);
    assert(!AlmostEqual(fieldCurrent.X(), lineCurrent.X(), 1.e-6f));
    assert(!AlmostEqual(fieldCurrent.Px(), lineCurrent.Px(), 1.e-6f));
    assert(fieldCurrent.Covariance(0, 0) >= 0.f);

    Pass("field-energy-fit",
         "field-aware measurement seed with numerical Jacobian can feed the Kalman energy-fit update");
  }

  void TestFieldTransportPrimitive()
  {
    KFParticleGpuFitState particle;
    particle.X() = 1.f;
    particle.Y() = 2.f;
    particle.Z() = 3.f;
    particle.Px() = 4.f;
    particle.Py() = 5.f;
    particle.Pz() = 6.f;
    particle.Q() = 1;

    KFParticleGpuFitState line;
    KFParticleGpuMath::TransportLine(particle, 0.25f, line);

    KFParticleGpuFitState zeroField;
    KFParticleGpuMath::TransportConstantBy(particle, 0.25f, 0.f, zeroField);
    assert(AlmostEqual(zeroField.X(), line.X()));
    assert(AlmostEqual(zeroField.Y(), line.Y()));
    assert(AlmostEqual(zeroField.Z(), line.Z()));
    assert(AlmostEqual(zeroField.Px(), line.Px()));
    assert(AlmostEqual(zeroField.Pz(), line.Pz()));

    KFParticleGpuFitState chargedField;
    KFParticleGpuMath::TransportConstantBy(particle, 0.25f, 20.f, chargedField);
    assert(AlmostEqual(chargedField.Y(), line.Y()));
    assert(!AlmostEqual(chargedField.X(), line.X()));
    assert(!AlmostEqual(chargedField.Px(), line.Px()));
    assert(AlmostEqual(KFParticleGpuMath::Momentum2(chargedField),
                       KFParticleGpuMath::Momentum2(particle),
                       1.e-5f));

    KFParticleGpuFitState neutral = particle;
    neutral.Q() = 0;
    KFParticleGpuFitState neutralField;
    KFParticleGpuMath::TransportConstantBy(neutral, 0.25f, 20.f, neutralField);
    assert(AlmostEqual(neutralField.X(), line.X()));
    assert(AlmostEqual(neutralField.Z(), line.Z()));

    Pass("field-transport-primitive", "constant-By single-track transport with exact zero-field and neutral fallbacks");
  }

  void TestFullFieldTransportPrimitive()
  {
    KFParticleGpuFitState particle;
    particle.X() = 1.f;
    particle.Y() = -0.5f;
    particle.Z() = 3.f;
    particle.Px() = 0.8f;
    particle.Py() = -0.35f;
    particle.Pz() = 1.1f;
    particle.E() = 1.5f;
    particle.S() = 0.75f;
    particle.SFromDecay() = 0.25f;
    particle.Q() = 1;
    for (int i = 0; i < KFParticleGpuFitState::NumberOfParameters; ++i) {
      particle.Covariance(i, i) = 0.01f * static_cast<float>(i + 1);
    }

    const float coefficients[KFParticleGpuFieldRegion::NumberOfCoefficients] = {
      0.4f, 0.02f, 0.001f, 20.f, 0.5f, 0.01f, -0.3f, 0.015f, 0.0005f, 3.f};
    const KFParticleGpuFieldRegion field(coefficients);
    const float dS = 0.15f;
    KFParticleGpuFitState transported;
    unsigned int status = KFParticleGpuMath::KFGpuFullFieldTransportInvalidInput;
    assert(KFParticleGpuMath::TransportFullField(particle, field, dS, transported, status));
    assert(status == KFParticleGpuMath::KFGpuFullFieldTransportSuccess);
    assert(KFParticleGpuMath::IsFiniteState(transported));

    const ReferenceTransportState reference = ReferenceFullFieldRk4(
      ReferenceTransportState{particle.X(), particle.Y(), particle.Z(), particle.Px(), particle.Py(), particle.Pz()},
      field,
      particle.Q(),
      dS);
    assert(AlmostEqual(transported.X(), reference.x, 2.e-4f));
    assert(AlmostEqual(transported.Y(), reference.y, 2.e-4f));
    assert(AlmostEqual(transported.Z(), reference.z, 2.e-4f));
    assert(AlmostEqual(transported.Px(), reference.px, 2.e-4f));
    assert(AlmostEqual(transported.Py(), reference.py, 2.e-4f));
    assert(AlmostEqual(transported.Pz(), reference.pz, 2.e-4f));
    assert(AlmostEqual(transported.S(), particle.S()));
    assert(AlmostEqual(transported.SFromDecay(), particle.SFromDecay() + dS));
    assert(transported.Covariance(0, 0) > 0.f);

    KFParticleGpuFieldRegion zeroField;
    KFParticleGpuFitState zeroTransport;
    assert(KFParticleGpuMath::TransportFullField(particle, zeroField, dS, zeroTransport, status));
    float zeroDsdr[6] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    KFParticleGpuFitState lineTransport;
    KFParticleGpuMath::TransportLineWithJacobian(particle, dS, zeroDsdr, lineTransport);
    ExpectFitStateClose(zeroTransport, lineTransport, 1.e-6f);

    KFParticleGpuFitState neutral = particle;
    neutral.Q() = 0;
    KFParticleGpuFitState neutralTransport;
    assert(KFParticleGpuMath::TransportFullField(neutral, field, dS, neutralTransport, status));
    assert(status == KFParticleGpuMath::KFGpuFullFieldTransportNeutral);
    KFParticleGpuFitState neutralLine;
    KFParticleGpuMath::TransportLineWithJacobian(neutral, dS, zeroDsdr, neutralLine);
    ExpectFitStateClose(neutralTransport, neutralLine, 1.e-6f);

    KFParticleGpuFitState limited;
    assert(KFParticleGpuMath::TransportFullField(particle, field, 1000.f, limited, status));
    assert(status == KFParticleGpuMath::KFGpuFullFieldTransportPathLimited);
    ExpectFitStateClose(limited, particle, 1.e-6f);

    float invalidCoefficients[KFParticleGpuFieldRegion::NumberOfCoefficients] = {};
    invalidCoefficients[0] = std::numeric_limits<float>::quiet_NaN();
    const KFParticleGpuFieldRegion invalidField(invalidCoefficients);
    KFParticleGpuFitState invalidOutput;
    assert(!KFParticleGpuMath::TransportFullField(particle, invalidField, dS, invalidOutput, status));
    assert(status == KFParticleGpuMath::KFGpuFullFieldTransportInvalidInput);
    ExpectFitStateClose(invalidOutput, particle, 1.e-6f);

    Pass("full-field-transport-primitive",
         "bounded ten-coefficient transport matches an independent RK4 reference and explicit guard paths");
  }

  void TestCpuCompatibleByDcaPrimitive()
  {
    assert(AlmostEqual(KFParticleGpuMath::CpuCompatibleAtan2(3.38f, 1.f),
                       1.2831439971923828f,
                       2.e-7f));
    float cpuSine = 0.f;
    float cpuCosine = 0.f;
    KFParticleGpuMath::CpuCompatibleSinCos(3.38f, cpuSine, cpuCosine);
    assert(AlmostEqual(cpuSine, -0.23615531623363495f, 2.e-7f));
    assert(AlmostEqual(cpuCosine, -0.9717153310775757f, 2.e-7f));

    KFParticleGpuFitState first;
    first.Initialize();
    first.X() = 0.f;
    first.Y() = 0.01f;
    first.Z() = 0.f;
    first.Px() = 0.32f;
    first.Py() = 0.08f;
    first.Pz() = 1.1f;
    first.Q() = 1;

    KFParticleGpuFitState second;
    second.Initialize();
    second.X() = 0.03f;
    second.Y() = -0.02f;
    second.Z() = 0.25f;
    second.Px() = -0.22f;
    second.Py() = 0.11f;
    second.Pz() = 0.9f;
    second.Q() = -1;

    float dS[2] = {0.f, 0.f};
    KFParticleGpuMath::GetDStoParticleByCpuCompatible(first, second, 18.f, dS);
    assert(AlmostEqual(dS[0], 0.16079706f, 1.e-6f));
    assert(AlmostEqual(dS[1], -0.075565495f, 1.e-6f));

    KFParticleGpuFitState firstAtDca;
    KFParticleGpuFitState secondAtDca;
    assert(KFParticleGpuMath::BuildConstantByDcaKinematicSeed(
      first, second, 18.f, firstAtDca, secondAtDca));
    KFParticleGpuFitState expectedFirst;
    KFParticleGpuFitState expectedSecond;
    KFParticleGpuMath::TransportConstantBy(first, dS[0], 18.f, expectedFirst);
    KFParticleGpuMath::TransportConstantBy(second, dS[1], 18.f, expectedSecond);
    ExpectFitStateClose(firstAtDca, expectedFirst, 1.e-6f);
    ExpectFitStateClose(secondAtDca, expectedSecond, 1.e-6f);

    Pass("cpu-compatible-by-dca",
         "constant-By DCA uses the CPU SIMD angle, trigonometric, helix-root, and line-correction math");
  }

  void TestFullFieldCoupledDcaPrimitive()
  {
    KFParticleGpuFitState first;
    first.Initialize();
    first.X() = 0.f;
    first.Y() = 0.01f;
    first.Z() = 0.f;
    first.Px() = 0.32f;
    first.Py() = 0.08f;
    first.Pz() = 1.1f;
    first.E() = 1.2f;
    first.Q() = 1;
    first.NDF() = 0;

    KFParticleGpuFitState second;
    second.Initialize();
    second.X() = 0.03f;
    second.Y() = -0.02f;
    second.Z() = 0.25f;
    second.Px() = -0.22f;
    second.Py() = 0.11f;
    second.Pz() = 0.9f;
    second.E() = 1.1f;
    second.Q() = -1;
    second.NDF() = 0;
    for (int parameter = 0; parameter < 6; ++parameter) {
      first.Covariance(parameter, parameter) = 0.01f * static_cast<float>(parameter + 1);
      second.Covariance(parameter, parameter) = 0.015f * static_cast<float>(parameter + 1);
      first.Covariance(6, parameter) = 0.001f * static_cast<float>(parameter + 1);
      second.Covariance(6, parameter) = -0.0015f * static_cast<float>(parameter + 1);
    }
    first.Covariance(6, 6) = 0.08f;
    second.Covariance(6, 6) = 0.09f;
    first.S() = 0.4f;
    second.S() = -0.3f;

    const float firstCoefficients[KFParticleGpuFieldRegion::NumberOfCoefficients] = {
      0.2f, 0.01f, 0.001f, 18.f, 0.25f, 0.01f, -0.1f, 0.02f, 0.001f, 0.f};
    const float secondCoefficients[KFParticleGpuFieldRegion::NumberOfCoefficients] = {
      0.18f, 0.012f, 0.0008f, 17.5f, 0.22f, 0.008f, -0.08f, 0.018f, 0.0007f, 0.25f};
    const KFParticleGpuFieldRegion firstField(firstCoefficients);
    const KFParticleGpuFieldRegion secondField(secondCoefficients);

    KFParticleGpuMath::KFParticleGpuFullFieldDcaResult result;
    unsigned int status = KFParticleGpuMath::KFGpuFullFieldDcaRejected;
    assert(KFParticleGpuMath::BuildFullFieldDcaCoupledResult(
      first, second, firstField, secondField, result, status));
    assert(status == KFParticleGpuMath::KFGpuFullFieldDcaSuccess);
    assert(KFParticleGpuMath::IsFiniteState(result.first));
    assert(KFParticleGpuMath::IsFiniteState(result.second));
    float correlationNorm = 0.f;
    for (int row = 0; row < 3; ++row) {
      for (int column = 0; column < 3; ++column) {
        assert(KFParticleGpuMath::IsFinite(result.Correlation(row, column)));
        correlationNorm += KFParticleGpuMath::Abs(result.Correlation(row, column));
      }
      assert(result.FirstCovariance(row, row) >= 0.f);
      assert(result.SecondCovariance(row, row) >= 0.f);
    }
    assert(correlationNorm > 0.f);
    assert(KFParticleGpuMath::Abs(result.FirstJacobian(0, 0)) > 0.f);
    assert(KFParticleGpuMath::Abs(result.SecondJacobian(0, 6)) > 0.f);
    assert(AlmostEqual(result.first.S(), first.S()));
    assert(AlmostEqual(result.second.S(), second.S()));

    // Independent central-difference oracle for the coupled DCA derivatives.
    float referenceFirstJacobian[6][12] = {};
    float referenceSecondJacobian[6][12] = {};
    for (int source = 0; source < 2; ++source) {
      for (int parameter = 0; parameter < 6; ++parameter) {
        KFParticleGpuFitState firstPlus = first;
        KFParticleGpuFitState firstMinus = first;
        KFParticleGpuFitState secondPlus = second;
        KFParticleGpuFitState secondMinus = second;
        const float value = source == 0 ? first.Parameter(parameter) : second.Parameter(parameter);
        const float step = KFParticleGpuMath::FullFieldDcaDerivativeStep(value);
        if (source == 0) {
          firstPlus.Parameter(parameter) += step;
          firstMinus.Parameter(parameter) -= step;
        }
        else {
          secondPlus.Parameter(parameter) += step;
          secondMinus.Parameter(parameter) -= step;
        }
        KFParticleGpuFitState firstAtPlus;
        KFParticleGpuFitState secondAtPlus;
        KFParticleGpuFitState firstAtMinus;
        KFParticleGpuFitState secondAtMinus;
        assert(KFParticleGpuMath::BuildFullFieldDcaKinematicSeed(
          firstPlus, secondPlus, firstField, secondField, firstAtPlus, secondAtPlus));
        assert(KFParticleGpuMath::BuildFullFieldDcaKinematicSeed(
          firstMinus, secondMinus, firstField, secondField, firstAtMinus, secondAtMinus));
        for (int row = 0; row < 6; ++row) {
          referenceFirstJacobian[row][source * 6 + parameter] =
            (firstAtPlus.Parameter(row) - firstAtMinus.Parameter(row)) / (2.f * step);
          referenceSecondJacobian[row][source * 6 + parameter] =
            (secondAtPlus.Parameter(row) - secondAtMinus.Parameter(row)) / (2.f * step);
          assert(AlmostEqual(result.FirstJacobian(row, source * 6 + parameter),
                             referenceFirstJacobian[row][source * 6 + parameter],
                             1.e-5f));
          assert(AlmostEqual(result.SecondJacobian(row, source * 6 + parameter),
                             referenceSecondJacobian[row][source * 6 + parameter],
                             1.e-5f));
        }
      }
    }
    float referenceCorrelation = 0.f;
    for (int source = 0; source < 2; ++source) {
      const KFParticleGpuFitState& input = source == 0 ? first : second;
      const int offset = source * 6;
      for (int left = 0; left < 6; ++left) {
        for (int right = 0; right < 6; ++right) {
          referenceCorrelation += referenceFirstJacobian[0][offset + left]
                                  * input.Covariance(left, right)
                                  * referenceSecondJacobian[0][offset + right];
        }
      }
    }
    assert(AlmostEqual(result.Correlation(0, 0), referenceCorrelation, 1.e-5f));
    for (int row = 0; row < 6; ++row) {
      float referenceFirstEnergyCovariance = 0.f;
      float referenceSecondEnergyCovariance = 0.f;
      for (int parameter = 0; parameter < 6; ++parameter) {
        referenceFirstEnergyCovariance += referenceFirstJacobian[row][parameter]
                                          * first.Covariance(parameter, 6);
        referenceSecondEnergyCovariance += referenceSecondJacobian[row][6 + parameter]
                                           * second.Covariance(parameter, 6);
      }
      assert(AlmostEqual(result.first.Covariance(6, row),
                         referenceFirstEnergyCovariance,
                         1.e-5f));
      assert(AlmostEqual(result.second.Covariance(6, row),
                         referenceSecondEnergyCovariance,
                         1.e-5f));
    }
    assert(AlmostEqual(result.first.Covariance(6, 6), first.Covariance(6, 6)));
    assert(AlmostEqual(result.second.Covariance(6, 6), second.Covariance(6, 6)));

    KFParticleGpuFieldRegion zeroField;
    KFParticleGpuMath::KFParticleGpuFullFieldDcaResult zeroResult;
    assert(KFParticleGpuMath::BuildFullFieldDcaCoupledResult(
      first, second, zeroField, zeroField, zeroResult, status));
    KFParticleGpuFitState lineCurrent;
    KFParticleGpuMeasurement lineMeasurement;
    assert(KFParticleGpuMath::BuildLineDcaMeasurementSeed(first, second, lineCurrent, lineMeasurement));
    assert(AlmostEqual(zeroResult.first.X(), lineCurrent.X(), 1.e-4f));
    assert(AlmostEqual(zeroResult.second.X(), lineMeasurement.Parameter(0), 1.e-4f));
    assert(AlmostEqual(zeroResult.Correlation(0, 0), lineMeasurement.Correlation(0, 0), 2.e-3f));

    KFParticleGpuFitState neutral = first;
    neutral.Q() = 0;
    assert(KFParticleGpuMath::BuildFullFieldDcaCoupledResult(
      neutral, second, firstField, secondField, result, status));
    assert(status & KFParticleGpuMath::KFGpuFullFieldDcaNeutral);

    float invalidCoefficients[KFParticleGpuFieldRegion::NumberOfCoefficients] = {};
    invalidCoefficients[3] = std::numeric_limits<float>::quiet_NaN();
    const KFParticleGpuFieldRegion invalidField(invalidCoefficients);
    assert(!KFParticleGpuMath::BuildFullFieldDcaCoupledResult(
      first, second, invalidField, secondField, result, status));
    assert(status == KFParticleGpuMath::KFGpuFullFieldDcaNonFinite);

    KFParticleGpuFitState distant = second;
    distant.Z() = 2000.f;
    assert(!KFParticleGpuMath::BuildFullFieldDcaCoupledResult(
      first, distant, firstField, secondField, result, status));
    assert(status & KFParticleGpuMath::KFGpuFullFieldDcaRejected);

    Pass("full-field-coupled-dca-primitive",
         "bounded two-track full-field derivatives produce coupled covariance and explicit neutral/rejection paths");
  }

  void TestFullFieldCoupledEnergyFit()
  {
    KFParticleGpuFitState first;
    first.Initialize();
    first.X() = 0.f;
    first.Y() = 0.01f;
    first.Z() = 0.f;
    first.Px() = 0.32f;
    first.Py() = 0.08f;
    first.Pz() = 1.1f;
    first.E() = 1.2f;
    first.Q() = 1;
    first.NDF() = 0;

    KFParticleGpuFitState second;
    second.Initialize();
    second.X() = 0.03f;
    second.Y() = -0.02f;
    second.Z() = 0.25f;
    second.Px() = -0.22f;
    second.Py() = 0.11f;
    second.Pz() = 0.9f;
    second.E() = 1.1f;
    second.Q() = -1;
    second.NDF() = 0;
    for (int parameter = 0; parameter < KFParticleGpuFitState::NumberOfParameters; ++parameter) {
      first.Covariance(parameter, parameter) = 0.01f * static_cast<float>(parameter + 1);
      second.Covariance(parameter, parameter) = 0.015f * static_cast<float>(parameter + 1);
    }

    const float firstCoefficients[KFParticleGpuFieldRegion::NumberOfCoefficients] = {
      0.2f, 0.01f, 0.001f, 18.f, 0.25f, 0.01f, -0.1f, 0.02f, 0.001f, 0.f};
    const float secondCoefficients[KFParticleGpuFieldRegion::NumberOfCoefficients] = {
      0.18f, 0.012f, 0.0008f, 17.5f, 0.22f, 0.008f, -0.08f, 0.018f, 0.0007f, 0.25f};
    const KFParticleGpuFieldRegion firstField(firstCoefficients);
    const KFParticleGpuFieldRegion secondField(secondCoefficients);

    KFParticleGpuFitState approximateCurrent;
    KFParticleGpuMeasurement approximateMeasurement;
    assert(KFParticleGpuMath::BuildFullFieldDcaMeasurementSeedApprox(
      first, second, firstField, secondField, approximateCurrent, approximateMeasurement));

    KFParticleGpuFitState coupledCurrent;
    KFParticleGpuMeasurement coupledMeasurement;
    assert(KFParticleGpuMath::BuildFullFieldDcaMeasurementSeedCoupled(
      first, second, firstField, secondField, coupledCurrent, coupledMeasurement));
    assert(!AlmostEqual(coupledMeasurement.Correlation(0, 0),
                        approximateMeasurement.Correlation(0, 0),
                        1.e-6f));
    for (int row = 0; row < 3; ++row) {
      for (int column = 0; column < 3; ++column) {
        assert(KFParticleGpuMath::IsFinite(coupledMeasurement.Correlation(row, column)));
      }
    }
    assert(coupledCurrent.Covariance(0, 0) >= 0.f);
    assert(coupledMeasurement.Covariance(0, 0) >= 0.f);

    KFParticleGpuFitState cpuCompatibleCurrent;
    KFParticleGpuMeasurement cpuCompatibleMeasurement;
    assert(KFParticleGpuMath::BuildFullFieldDcaMeasurementSeedCpuCompatible(
      first, second, firstField, secondField, cpuCompatibleCurrent, cpuCompatibleMeasurement));
    KFParticleGpuFitState expectedFirst;
    KFParticleGpuFitState expectedSecond;
    assert(KFParticleGpuMath::BuildFullFieldDcaIndependentTransportStates(
      first, second, firstField, secondField, expectedFirst, expectedSecond));
    ExpectFitStateClose(cpuCompatibleCurrent, expectedFirst, 1.e-6f);
    for (int parameter = 0; parameter < KFParticleGpuFitState::NumberOfParameters; ++parameter) {
      assert(AlmostEqual(cpuCompatibleMeasurement.Parameter(parameter),
                         expectedSecond.Parameter(parameter), 1.e-6f));
    }
    for (int covariance = 0;
         covariance < KFParticleGpuFitState::NumberOfCovarianceElements;
         ++covariance) {
      assert(AlmostEqual(cpuCompatibleMeasurement.Covariance(covariance),
                         expectedSecond.Covariance(covariance), 1.e-6f));
    }
    for (int row = 0; row < 3; ++row) {
      for (int column = 0; column < 3; ++column) {
        assert(cpuCompatibleMeasurement.Correlation(row, column) == 0.f);
      }
    }
    assert(!AlmostEqual(cpuCompatibleCurrent.Covariance(0, 0),
                        coupledCurrent.Covariance(0, 0),
                        1.e-7f));
    assert(KFParticleGpuMath::AddDaughterWithEnergyFit(
      cpuCompatibleCurrent, cpuCompatibleMeasurement, second.Q()));
    assert(KFParticleGpuMath::IsFiniteState(cpuCompatibleCurrent));
    Pass("full-field-cpu-compatible-energy-fit",
         "CPU V0 parity keeps coupled DCA positions but independently transported daughter covariance");

    KFParticleGpuFitState preliminaryFirst;
    KFParticleGpuFitState preliminarySecond;
    assert(KFParticleGpuMath::BuildFullFieldDcaIndependentTransportStates(
      first, second, firstField, secondField, preliminaryFirst, preliminarySecond));
    KFParticleGpuFitState manualSecondPassCurrent;
    KFParticleGpuMeasurement manualSecondPassMeasurement;
    assert(KFParticleGpuMath::BuildFullFieldDcaMeasurementSeedAnalytic(
      preliminaryFirst, preliminarySecond, firstField, secondField,
      manualSecondPassCurrent, manualSecondPassMeasurement));
    KFParticleGpuFitState fullProductCurrent;
    KFParticleGpuMeasurement fullProductMeasurement;
    assert(KFParticleGpuMath::BuildFullFieldDcaMeasurementSeedAnalytic<false>(
      preliminaryFirst, preliminarySecond, firstField, secondField,
      fullProductCurrent, fullProductMeasurement));
    for (int row = 0; row < 3; ++row) {
      for (int column = 0; column < 3; ++column) {
        assert(fullProductMeasurement.Correlation(row, column)
               == manualSecondPassMeasurement.Correlation(row, column));
      }
    }
    KFParticleGpuFitState twoStageCurrent;
    KFParticleGpuMeasurement twoStageMeasurement;
    assert(KFParticleGpuMath::BuildFullFieldDcaMeasurementSeedCpuConstructV0(
      first, second, firstField, secondField, twoStageCurrent, twoStageMeasurement));
    for (int parameter = 0; parameter < KFParticleGpuFitState::NumberOfParameters; ++parameter) {
      assert(AlmostEqual(twoStageCurrent.Parameter(parameter),
                         manualSecondPassCurrent.Parameter(parameter), 1.e-6f));
      assert(AlmostEqual(twoStageMeasurement.Parameter(parameter),
                         manualSecondPassMeasurement.Parameter(parameter), 1.e-6f));
    }
    for (int covariance = 0;
         covariance < KFParticleGpuFitState::NumberOfCovarianceElements;
         ++covariance) {
      assert(AlmostEqual(twoStageCurrent.Covariance(covariance),
                         manualSecondPassCurrent.Covariance(covariance), 1.e-6f));
      assert(AlmostEqual(twoStageMeasurement.Covariance(covariance),
                         manualSecondPassMeasurement.Covariance(covariance), 1.e-6f));
    }
    float secondPassCorrelationNorm = 0.f;
    for (int row = 0; row < 3; ++row) {
      for (int column = 0; column < 3; ++column) {
        assert(AlmostEqual(twoStageMeasurement.Correlation(row, column),
                           manualSecondPassMeasurement.Correlation(row, column), 1.e-6f));
        secondPassCorrelationNorm +=
          KFParticleGpuMath::Abs(twoStageMeasurement.Correlation(row, column));
      }
    }
    assert(secondPassCorrelationNorm > 0.f);
    assert(KFParticleGpuMath::AddDaughterWithEnergyFit(
      twoStageCurrent, twoStageMeasurement, second.Q()));
    assert(KFParticleGpuMath::IsFiniteState(twoStageCurrent));
    Pass("full-field-cpu-construct-v0-two-stage",
         "CPU ConstructV0 parity performs independent pre-transport followed by coupled GetMeasurement DCA");

    assert(KFParticleGpuMath::AddDaughterWithEnergyFit(coupledCurrent, coupledMeasurement, second.Q()));
    assert(coupledCurrent.NDF() == 2);
    assert(coupledCurrent.Q() == 0);
    assert(coupledCurrent.Chi2() >= 0.f);
    assert(KFParticleGpuMath::IsFiniteState(coupledCurrent));

    Pass("full-field-coupled-energy-fit",
         "full-field energy-fit uses coupled DCA covariance instead of the line-DCA correlation approximation");
  }

  void TestRealScaleCpuConstructV0Regression()
  {
    const float firstParameters[6] = {
      7.08508921f, -4.49616003f, -13.5050001f, 0.0439769514f, -0.0652863979f, 0.400178373f};
    const float firstCovariance[21] = {
      1.2805292e-06f, -8.82166387e-06f, 0.000134082511f, 0.f, 0.f, 0.f,
      -7.95641668e-08f, 5.14834369e-07f, 0.f, 1.99038163e-06f,
      3.31165296e-07f, -4.77195363e-06f, 0.f, -8.63523212e-07f, 2.42128908e-06f,
      -1.47383176e-07f, 6.66178437e-07f, 0.f, 5.13457508e-06f, -5.6415447e-06f,
      3.54706499e-05f};
    const float secondParameters[6] = {
      9.92090988f, 1.76413846f, -13.2010002f, 0.44889009f, 0.0761831626f, 1.25327694f};
    const float secondCovariance[21] = {
      1.11535496e-06f, -8.00729504e-06f, 6.94773189e-05f, 0.f, 0.f, 0.f,
      -4.72003904e-07f, 3.06515744e-06f, 0.f, 3.9576189e-05f,
      4.76141878e-07f, -4.27469467e-06f, 0.f, 5.92749439e-06f, 2.32458592e-06f,
      -1.02152626e-06f, 6.62802904e-06f, 0.f, 0.000103554361f, 1.60161526e-05f,
      0.000277544052f};
    KFParticleGpuTrackState firstTrack;
    KFParticleGpuTrackState secondTrack;
    firstTrack.Initialize(firstParameters, firstCovariance);
    secondTrack.Initialize(secondParameters, secondCovariance);
    KFParticleGpuFitState first;
    KFParticleGpuFitState second;
    first.Initialize(firstTrack, -1, 0.139570385f);
    second.Initialize(secondTrack, 1, 0.139570385f);
    first.NDF() = -1;

    const float firstFieldCoefficients[10] = {
      -0.0660794005f, -0.00331308902f, 2.95575155e-05f, -10.7440863f,
      -0.0322167352f, 0.0010213994f, 0.127144858f, -0.00752550736f,
      -0.0001830908f, -13.5050001f};
    const float secondFieldCoefficients[10] = {
      0.0361264572f, 0.00241193734f, 5.82812645e-05f, -10.6532497f,
      -0.0234668963f, 0.00127589644f, -0.049883008f, 0.00207339227f,
      0.000108895299f, -13.2010002f};
    const KFParticleGpuFieldRegion firstField(firstFieldCoefficients);
    const KFParticleGpuFieldRegion secondField(secondFieldCoefficients);

    KFParticleGpuFitState mother;
    KFParticleGpuMeasurement measurement;
    assert(KFParticleGpuMath::BuildFullFieldDcaMeasurementSeedCpuConstructV0(
      first, second, firstField, secondField, mother, measurement));
    assert(KFParticleGpuMath::AddDaughterWithEnergyFit(mother, measurement, second.Q()));
    const float cpuParameters[7] = {
      1.01353443f, 0.097291939f, -40.9884109f, 0.492407382f,
      0.0101595512f, 1.66291451f, 1.77252567f};
    const float cpuCovariance[21] = {
      0.0148513131f, 0.00166124001f, 0.00106151076f, 0.0485299751f,
      0.00592603162f, 0.166866854f, -0.000142359699f, -8.58122075e-06f,
      -4.36072878e-05f, 4.06726831e-05f, -6.37166522e-05f, -2.80646891e-05f,
      -0.0001440419f, 5.94478615e-06f, 3.52609504e-06f, -0.000329983595f,
      -2.12061386e-05f, -0.000231254613f, 0.00010339688f, 1.26528848e-05f,
      0.000280292006f};
    for (int parameter = 0; parameter < 3; ++parameter) {
      assert(AlmostEqual(mother.Parameter(parameter), cpuParameters[parameter], 1.e-3f));
    }
    for (int parameter = 3; parameter < 7; ++parameter) {
      assert(AlmostEqual(mother.Parameter(parameter), cpuParameters[parameter], 1.e-4f));
    }
    for (int covariance = 0; covariance < 21; ++covariance) {
      assert(AlmostEqual(mother.Covariance(covariance), cpuCovariance[covariance], 3.e-3f));
    }
    assert(AlmostEqual(mother.Chi2(), 0.018545378f, 1.e-3f));
    Pass("real-scale-cpu-construct-v0-regression",
         "a traced long-path detector pair reproduces CPU SIMD parameters, covariance, and chi2");
  }

  void TestCpuDerivativeDcaMiddlePointRegression()
  {
    const float firstParameters[6] = {
      -17.1380177f, -1.89276993f, 18.5489998f, -0.827033043f, -0.0706946999f, 2.58645773f};
    const float firstCovariance[21] = {
      2.56391957e-07f, -1.64389371e-06f, 2.49271907e-05f, 0.f, 0.f, 0.f,
      3.3885803e-07f, -4.28756863e-07f, 0.f, 0.00012837573f, 2.18080444e-07f,
      -2.81243069e-06f, 0.f, 1.18039206e-05f, 2.83803615e-06f, -1.22988604e-06f,
      1.88474507e-06f, 0.f, -0.000408377964f, -3.79987396e-05f, 0.00130853883f};
    const float secondParameters[6] = {
      -5.30288982f, 18.6413879f, 30.4950008f, 0.0865562782f, 0.158387274f, 0.510374367f};
    const float secondCovariance[21] = {
      3.09851998e-07f, -2.3027751e-06f, 2.96841608e-05f, 0.f, 0.f, 0.f,
      -5.16627452e-08f, 4.37198338e-07f, 0.f, 5.11777398e-06f, 4.57005633e-08f,
      -8.04988986e-07f, 0.f, 6.45002729e-06f, 1.08809536e-05f, -1.80463417e-07f,
      1.62960976e-06f, 0.f, 2.10140806e-05f, 3.19475694e-05f, 0.000104721446f};
    KFParticleGpuTrackState firstTrack;
    KFParticleGpuTrackState secondTrack;
    firstTrack.Initialize(firstParameters, firstCovariance);
    secondTrack.Initialize(secondParameters, secondCovariance);
    KFParticleGpuFitState first;
    KFParticleGpuFitState second;
    first.Initialize(firstTrack, -1, 0.139570385f);
    second.Initialize(secondTrack, 1, 0.139570385f);
    first.NDF() = -1;

    const float firstFieldCoefficients[10] = {
      0.0655412152f, 0.00218684319f, -7.64091965e-07f, -10.2738333f, 0.053929206f,
      0.000775120454f, -0.0756340399f, -0.00533406343f, -2.39997462e-06f, 18.5489998f};
    const float secondFieldCoefficients[10] = {
      -0.202597678f, 0.00315024564f, 0.000278466498f, -10.662899f, 0.0494437255f,
      0.000581318978f, 1.15643466f, 0.0659745708f, 3.37450765e-05f, 30.4950008f};
    const KFParticleGpuFieldRegion firstField(firstFieldCoefficients);
    const KFParticleGpuFieldRegion secondField(secondFieldCoefficients);

    KFParticleGpuFitState firstAtDca;
    KFParticleGpuFitState secondAtDca;
    assert(KFParticleGpuMath::BuildFullFieldDcaKinematicSeed(
      first, second, firstField, secondField, firstAtDca, secondAtDca));
    const float cpuFirstAtDca[3] = {-2.84036231f, -0.542365134f, -31.4906445f};
    const float cpuSecondAtDca[3] = {-3.00465107f, -0.574299991f, -31.6438408f};
    for (int parameter = 0; parameter < 3; ++parameter) {
      assert(AlmostEqual(firstAtDca.Parameter(parameter), cpuFirstAtDca[parameter], 2.e-3f));
      assert(AlmostEqual(secondAtDca.Parameter(parameter), cpuSecondAtDca[parameter], 2.e-3f));
    }

    KFParticleGpuFitState mother;
    KFParticleGpuMeasurement measurement;
    assert(KFParticleGpuMath::BuildFullFieldDcaMeasurementSeedCpuConstructV0(
      first, second, firstField, secondField, mother, measurement));
    assert(KFParticleGpuMath::AddDaughterWithEnergyFit(mother, measurement, second.Q()));
    const float cpuMother[7] = {
      -2.84004831f, -0.542201757f, -31.4969997f, -0.790131986f,
      0.0830615535f, 3.12727094f, 3.27151442f};
    for (int parameter = 0; parameter < 7; ++parameter) {
      assert(AlmostEqual(mother.Parameter(parameter), cpuMother[parameter], 3.e-3f));
    }
    assert(AlmostEqual(mother.Chi2(), 0.393042386f, 3.e-3f));
    Pass("cpu-derivative-dca-middle-point-regression",
         "a real detector pair exercises the covariance-selected middle DCA root used by CPU ConstructV0");
  }

  void TestFieldTransportProbe(KFParticleGpuRuntime& runtime)
  {
    xpu::buffer<float> floatChecks(30, xpu::buf_io);
    xpu::buffer<int> integerChecks(12, xpu::buf_io);

    runtime.GetQueue().launch<KFParticleGpuFieldTransportProbe>(
      xpu::n_threads(1), floatChecks.get(), integerChecks.get());
    runtime.GetQueue().wait();
    runtime.GetQueue().copy(floatChecks, xpu::d2h);
    runtime.GetQueue().copy(integerChecks, xpu::d2h);
    runtime.GetQueue().wait();

    const float* floats = HostPointer(floatChecks);
    const int* integers = HostPointer(integerChecks);
    assert(integers[0] == 1);
    assert(AlmostEqual(floats[0], 2.f));
    assert(AlmostEqual(floats[1], 4.5f));
    assert(AlmostEqual(floats[2], floats[0]));
    assert(AlmostEqual(floats[3], floats[1]));
    assert(!AlmostEqual(floats[4], floats[0]));
    assert(!AlmostEqual(floats[6], 4.f));
    assert(AlmostEqual(floats[8], floats[0]));
    assert(AlmostEqual(floats[9], floats[1]));
    assert(integers[1] == 1);
    assert(integers[2] == 1);
    assert(integers[3] == 1);
    assert(integers[4] == 1);
    assert(integers[5] == 1);
    assert(integers[6] == 2);
    assert(integers[7] == 0);
    assert(AlmostEqual(floats[10], floats[11], 1.e-5f));
    assert(!AlmostEqual(floats[12], floats[10], 1.e-6f));
    assert(!AlmostEqual(floats[14], floats[13], 1.e-6f));
    assert(!AlmostEqual(floats[16], floats[15], 1.e-6f));
    assert(floats[18] >= 0.f);
    assert(!AlmostEqual(floats[19], floats[14], 1.e-6f));
    assert(integers[8] == 1);
    assert(integers[9] == static_cast<int>(KFParticleGpuMath::KFGpuFullFieldTransportSuccess));
    assert(std::isfinite(floats[20]));
    assert(std::isfinite(floats[21]));
    assert(std::isfinite(floats[22]));
    assert(std::isfinite(floats[23]));
    assert(std::isfinite(floats[24]));
    assert(std::isfinite(floats[25]));
    assert(floats[26] > 0.f);
    assert(integers[10] == 1);
    assert(integers[11] == static_cast<int>(KFParticleGpuMath::KFGpuFullFieldDcaSuccess));
    assert(std::isfinite(floats[27]));
    assert(floats[28] > 0.f);
    assert(KFParticleGpuMath::Abs(floats[29]) > 0.f);
    Pass("field-transport-probe",
         "device-side constant-By, full-field transport, and coupled DCA covariance checks");
  }

  void TestV0LineTopologyProbe(KFParticleGpuRuntime& runtime)
  {
    xpu::buffer<float> floatChecks(8, xpu::buf_io);
    xpu::buffer<unsigned int> statusChecks(2, xpu::buf_io);
    runtime.GetQueue().launch<KFParticleGpuV0LineTopologyProbe>(
      xpu::n_threads(1), floatChecks.get(), statusChecks.get());
    runtime.GetQueue().wait();
    runtime.GetQueue().copy(floatChecks, xpu::d2h);
    runtime.GetQueue().copy(statusChecks, xpu::d2h);
    runtime.GetQueue().wait();

    const float* floats = HostPointer(floatChecks);
    const unsigned int* status = HostPointer(statusChecks);
    assert(status[0] == 1u);
    assert(status[1] == KFGpuV0LineTopologyValid);
    assert(AlmostEqual(floats[0], 3.f));
    assert(AlmostEqual(floats[1], 5.f));
    assert(AlmostEqual(floats[2], std::sqrt(29.8f) / 5.f));
    assert(AlmostEqual(floats[3], 5.f / (std::sqrt(29.8f) / 5.f)));
    assert(AlmostEqual(floats[4], 2.69427f, 1.e-4f));
    assert(AlmostEqual(floats[5], 0.991419f, 1.e-4f));
    assert(AlmostEqual(floats[6], 0.6f));
    assert(AlmostEqual(floats[7], (20.2f / 0.96f) / 3.f));
    Pass("v0-line-topology-probe",
         "device-side correlated line-to-PV topology contract matches the scalar fixture");
  }

  void TestFieldDcaKinematicSeed()
  {
    KFParticleGpuFitState first;
    first.X() = 1.f;
    first.Y() = 2.f;
    first.Z() = 3.f;
    first.Px() = 4.f;
    first.Py() = 5.f;
    first.Pz() = 6.f;
    first.E() = 10.f;
    first.Q() = 1;

    KFParticleGpuFitState second;
    second.X() = -0.5f;
    second.Y() = 1.5f;
    second.Z() = 2.5f;
    second.Px() = -3.f;
    second.Py() = 2.f;
    second.Pz() = 5.f;
    second.E() = 8.f;
    second.Q() = -1;

    KFParticleGpuFitState lineMother;
    KFParticleGpuMath::BuildLineDcaKinematicMother(first, second, lineMother);

    KFParticleGpuFitState zeroFieldMother;
    assert(KFParticleGpuMath::BuildConstantByDcaKinematicMother(
      first, second, 0.f, zeroFieldMother));
    ExpectFitStateClose(zeroFieldMother, lineMother, 1.e-5f);

    KFParticleGpuFitState neutralFirst = first;
    KFParticleGpuFitState neutralSecond = second;
    neutralFirst.Q() = 0;
    neutralSecond.Q() = 0;
    KFParticleGpuFitState neutralLineMother;
    KFParticleGpuMath::BuildLineDcaKinematicMother(neutralFirst, neutralSecond, neutralLineMother);
    KFParticleGpuFitState neutralFieldMother;
    assert(KFParticleGpuMath::BuildConstantByDcaKinematicMother(
      neutralFirst, neutralSecond, 200.f, neutralFieldMother));
    ExpectFitStateClose(neutralFieldMother, neutralLineMother, 1.e-5f);

    KFParticleGpuFitState fieldMother;
    assert(KFParticleGpuMath::BuildConstantByDcaKinematicMother(
      first, second, 200.f, fieldMother));
    assert(!AlmostEqual(fieldMother.X(), lineMother.X(), 1.e-6f));
    assert(!AlmostEqual(fieldMother.Px(), lineMother.Px(), 1.e-6f));
    assert(fieldMother.Q() == lineMother.Q());
    assert(AlmostEqual(fieldMother.E(), lineMother.E()));

    KFParticleGpuFitState farSecond = second;
    farSecond.Z() = 3000.f;
    farSecond.Pz() = -1.f;
    KFParticleGpuFitState rejectedMother;
    assert(!KFParticleGpuMath::BuildConstantByDcaKinematicMother(
      first, farSecond, 200.f, rejectedMother));

    Pass("field-dca-kinematic-seed", "two-daughter constant-By DCA seed with zero-field equivalence and bounded rejection");
  }

  void TestFieldDcaReferenceComparison()
  {
    KFParticleGpuFitState first;
    first.X() = 1.f;
    first.Y() = 2.f;
    first.Z() = 3.f;
    first.Px() = 4.f;
    first.Py() = 5.f;
    first.Pz() = 6.f;
    first.E() = 10.f;
    first.Q() = 1;

    KFParticleGpuFitState second;
    second.X() = -0.5f;
    second.Y() = 1.5f;
    second.Z() = 2.5f;
    second.Px() = -3.f;
    second.Py() = 2.f;
    second.Pz() = 5.f;
    second.E() = 8.f;
    second.Q() = -1;

    ReferenceState zeroReference;
    assert(ReferenceBuildConstantByDcaKinematicMother(first, second, 0.0, zeroReference));
    KFParticleGpuFitState zeroMother;
    assert(KFParticleGpuMath::BuildConstantByDcaKinematicMother(
      first, second, 0.f, zeroMother));
    ExpectFitStateMatchesReference(zeroMother, zeroReference, 1.e-5f);

    ReferenceState fieldReference;
    assert(ReferenceBuildConstantByDcaKinematicMother(
      first, second, kDiagnosticFieldBy, fieldReference));
    KFParticleGpuFitState fieldMother;
    assert(KFParticleGpuMath::BuildConstantByDcaKinematicMother(
      first, second, kDiagnosticFieldBy, fieldMother));
    ExpectFitStateMatchesReference(fieldMother, fieldReference, 5.e-5f);

    KFParticleGpuFitState farSecond = second;
    farSecond.Z() = 3000.f;
    farSecond.Pz() = -1.f;
    ReferenceState rejectedReference;
    assert(!ReferenceBuildConstantByDcaKinematicMother(
      first, farSecond, kDiagnosticFieldBy, rejectedReference));
    KFParticleGpuFitState rejectedMother;
    assert(!KFParticleGpuMath::BuildConstantByDcaKinematicMother(
      first, farSecond, kDiagnosticFieldBy, rejectedMother));

    Pass("field-dca-reference-comparison",
         "constant-By two-daughter DCA seed matches an independent scalar reference");
  }

  void TestKalmanUpdateProbe(KFParticleGpuRuntime& runtime)
  {
    xpu::buffer<float> floatChecks(8, xpu::buf_io);
    xpu::buffer<int> integerChecks(3, xpu::buf_io);

    runtime.GetQueue().launch<KFParticleGpuKalmanUpdateProbe>(
      xpu::n_threads(1), floatChecks.get(), integerChecks.get());
    runtime.GetQueue().wait();
    runtime.GetQueue().copy(floatChecks, xpu::d2h);
    runtime.GetQueue().copy(integerChecks, xpu::d2h);
    runtime.GetQueue().wait();

    const float* floats = HostPointer(floatChecks);
    const int* integers = HostPointer(integerChecks);
    assert(integers[0] == 1);
    assert(integers[1] == 0);
    assert(integers[2] == 2);
    assert(AlmostEqual(floats[0], 0.8f));
    assert(AlmostEqual(floats[1], 5.f / 3.f));
    assert(AlmostEqual(floats[2], 18.f / 7.f));
    assert(AlmostEqual(floats[3], 1.5f));
    assert(AlmostEqual(floats[4], 5.f));
    assert(AlmostEqual(floats[5], 0.8f));
    assert(AlmostEqual(floats[6], 0.11f));
    assert(AlmostEqual(floats[7], 2.1523809f, 1.e-5f));
    Pass("kalman-energy-fit-probe", "device-side one-daughter energy-fit Kalman update on diagonal covariance");
  }

