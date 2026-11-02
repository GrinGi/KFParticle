  void TestTransactionTableUpdate(KFParticleGpuRuntime& runtime,
                                  KFParticleGpuBufferManager& buffers)
  {
    auto requested = buffers.Capacities();
    requested.transactions = std::max(requested.transactions, 1u);
    requested.domains = std::max(requested.domains, 1u);
    requested.workRanges = std::max(requested.workRanges, 1u);
    requested.domainStatuses = std::max(requested.domainStatuses, 1u);
    requested.domainChannelStatuses = std::max(requested.domainChannelStatuses, 1u);
    requested.twoDaughterPairTiles = std::max(requested.twoDaughterPairTiles, 1u);
    buffers.EnsureCapacity(requested);
    buffers.SetTransactionSizes(1u, 1u, 1u, 1u, 1u);
    buffers.SetTwoDaughterPairTileSize(1u);
    const auto host = buffers.HostTransaction();
    const auto device = buffers.DeviceTransaction();
    host.Transaction()[0] = {};
    host.Domains()[0] = {};
    host.WorkRanges()[0] = {};
    host.WorkStatuses()[0] = {};
    host.DomainStatuses()[0] = {};
    host.DomainChannelStatuses()[0] = {};
    buffers.HostTwoDaughterPairTiles()[0] = {};
    buffers.UploadTransaction();

    KFParticleGpuWorkStatus workStatus;
    workStatus.visitedCombinations = 17u;
    workStatus.flags = KFParticleGpuWorkSourceResolved;
    KFParticleGpuDomainStatus domainStatus;
    domainStatus.storedTasks = 13u;
    domainStatus.overflowFlags = CandidateCapacityExceeded;
    KFParticleGpuDomainChannelStatus channelStatus;
    channelStatus.visited = 23u;
    channelStatus.stored = 11u;
    auto& queue = runtime.GetQueue();
    queue.memcpy(device.WorkStatuses(), &workStatus, sizeof(workStatus));
    queue.memcpy(device.DomainStatuses(), &domainStatus, sizeof(domainStatus));
    queue.memcpy(device.DomainChannelStatuses(), &channelStatus, sizeof(channelStatus));
    queue.wait();
    // CPU XPU aliases these buffers; HIP/CUDA mirrors can remain stale.
    if (host.WorkStatuses() != device.WorkStatuses()) { host.WorkStatuses()[0] = {}; }
    if (host.DomainStatuses() != device.DomainStatuses()) { host.DomainStatuses()[0] = {}; }
    if (host.DomainChannelStatuses() != device.DomainChannelStatuses()) {
      host.DomainChannelStatuses()[0] = {};
    }
    host.Transaction()[0].decayPlanRevision = 73u;
    host.Domains()[0].sourceId = 91u;
    host.WorkRanges()[0].generation = 2u;
    buffers.HostTwoDaughterPairTiles()[0].firstPair = 64u;
    buffers.UploadTransaction(false);

    KFParticleGpuTransactionDesc transaction;
    KFParticleGpuDomainDesc domain;
    KFParticleGpuWorkRange work;
    KFParticleGpuPairTile tile;
    queue.memcpy(&transaction, device.Transaction(), sizeof(transaction));
    queue.memcpy(&domain, device.Domains(), sizeof(domain));
    queue.memcpy(&work, device.WorkRanges(), sizeof(work));
    queue.memcpy(&tile, buffers.DeviceTwoDaughterPairTiles(), sizeof(tile));
    buffers.DownloadTransactionStatus(true, true, false);
    assert(transaction.decayPlanRevision == 73u);
    assert(domain.sourceId == 91u);
    assert(work.generation == 2u);
    assert(tile.firstPair == 64u);
    assert(host.WorkStatuses()[0].visitedCombinations == workStatus.visitedCombinations);
    assert(host.WorkStatuses()[0].flags == workStatus.flags);
    assert(host.DomainStatuses()[0].storedTasks == domainStatus.storedTasks);
    assert(host.DomainStatuses()[0].overflowFlags == domainStatus.overflowFlags);
    assert(host.DomainChannelStatuses()[0].visited == channelStatus.visited);
    assert(host.DomainChannelStatuses()[0].stored == channelStatus.stored);

    host.WorkStatuses()[0] = {};
    host.DomainStatuses()[0] = {};
    host.DomainChannelStatuses()[0] = {};
    buffers.UploadTransaction();
    buffers.DownloadTransactionStatus(true, true, false);
    assert(host.WorkStatuses()[0].visitedCombinations == 0u);
    assert(host.DomainStatuses()[0].overflowFlags == 0u);
    assert(host.DomainChannelStatuses()[0].visited == 0u);
    buffers.SetTransactionSizes(0u, 0u, 0u, 0u, 0u);
    buffers.SetTwoDaughterPairTileSize(0u);
    Pass("transaction-table-update",
         "metadata updates preserve device statuses; initial submission resets them");
  }

  void TestTransactionValidationComplexity()
  {
    const auto validate = [](unsigned int count, bool permuted,
                             KFParticleGpuTransactionValidationStats& stats) {
      std::vector<KFParticleGpuTransactionDesc> transactions(1u);
      std::vector<KFParticleGpuDomainDesc> domains(count);
      std::vector<KFParticleGpuWorkRange> workRanges(count);
      std::vector<KFParticleGpuWorkStatus> workStatuses(count);
      std::vector<KFParticleGpuDomainStatus> domainStatuses(count);
      transactions[0].decayPlanRevision = 73u;
      transactions[0].domains = {0u, count};
      transactions[0].workRanges = {0u, count};
      for (unsigned int index = 0u; index < count; ++index) {
        const unsigned int rank = permuted ? (index * 37u) % count : index;
        const unsigned int base = rank * 4u;
        auto& domain = domains[index];
        domain.inputDescriptorIndex = index;
        domain.statusIndex = index;
        domain.sourceTracks = {base, 2u};
        domain.primaryVertices = {base, 1u};
        domain.workRanges = {index, 1u};
        domain.taskReservation = {base, 2u};
        domain.candidateReservation = {base, 2u};
        domain.daughterReservation = {base, 2u};
        domain.selectedReservation = {base, 2u};
        auto& work = workRanges[index];
        work.domainIndex = index;
        work.family = KFParticleGpuTwoDaughterWork;
        work.sourceTracks = domain.sourceTracks;
        work.secondarySourceTracks = domain.sourceTracks;
        work.primaryVertices = domain.primaryVertices;
        work.outputReservation = domain.taskReservation;
      }
      KFParticleGpuTransactionView view(
        transactions.data(), domains.data(), workRanges.data(),
        workStatuses.data(), domainStatuses.data(), count, count, count);
      constexpr unsigned int capacity = 1000000u;
      const auto optimized = KFParticleGpuValidateTransaction(
        view, 73u, capacity, capacity, count, capacity, capacity,
        capacity, capacity, &stats);
      const auto reference = KFParticleGpuValidateTransactionReference(
        view, 73u, capacity, capacity, count, capacity, capacity,
        capacity, capacity);
      assert(optimized == KFParticleGpuTransactionValid);
      assert(optimized == reference);
    };

    KFParticleGpuTransactionValidationStats ordered;
    validate(2048u, false, ordered);
    assert(ordered.sortedFallbackFamilies == 0u);
    assert(ordered.domainPairComparisons == 0u);
    assert(ordered.adjacentRangeComparisons == 6ull * 2047ull);

    KFParticleGpuTransactionValidationStats permuted;
    validate(257u, true, permuted);
    assert(permuted.sortedFallbackFamilies == 6u);
    assert(permuted.domainPairComparisons == 0u);
    assert(permuted.adjacentRangeComparisons == 6ull * 256ull);
    Pass("transaction-validation-complexity",
         "ordered domains use linear scans and permuted domains use the safe sorted fallback");
  }

  void TestParityAndPromotionContract()
  {
    const unsigned int topologies[] = {
      KFGpuGraphTopologyTrackTrack,
      KFGpuGraphTopologyTrackComposite,
      KFGpuGraphTopologyCompositeTrack,
      KFGpuGraphTopologyCompositeComposite,
      KFGpuGraphTopologyNeutralDaughter,
      KFGpuGraphTopologyUnaryComposite,
    };
    const unsigned int outputClasses[] = {
      KFGpuGraphOutputPrimaryAndSecondary,
      KFGpuGraphOutputSecondary,
      KFGpuGraphOutputSecondary,
      KFGpuGraphOutputFinal,
      KFGpuGraphOutputTemporary,
      KFGpuGraphOutputFinal,
    };
    KFParticleGpuParityTolerance tolerance;
    for (unsigned int topologyIndex = 0u;
         topologyIndex < sizeof(topologies) / sizeof(topologies[0]);
         ++topologyIndex) {
      KFParticleGpuParitySnapshot cpu;
      cpu.key.eventId = 71u;
      cpu.key.channelId = 100u + topologyIndex;
      cpu.key.pdg = 3000 + static_cast<int>(topologyIndex);
      cpu.key.lineageSize = 4u;
      cpu.key.lineage[0] = 19;
      cpu.key.lineage[1] = 3;
      cpu.key.lineage[2] = 11;
      cpu.key.lineage[3] = 7;
      KFParticleGpuParity::Canonicalize(cpu.key);
      cpu.parameters[3] = 0.5f;
      cpu.covariance[9] = 0.01f;
      cpu.chi2 = 1.f;
      cpu.ndf = 2;
      cpu.charge = 1;
      cpu.mass = 1.1f;
      cpu.massError = 0.02f;
      cpu.massValid = 1u;
      cpu.topology = topologies[topologyIndex];
      cpu.outputClass = outputClasses[topologyIndex];
      cpu.operationStatus = KFGpuCandidateOperationAccepted;
      cpu.available = KFGpuParityKeyAvailable | KFGpuParityParametersAvailable
                      | KFGpuParityCovarianceAvailable
                      | KFGpuParityFitQualityAvailable | KFGpuParityMassAvailable
                      | KFGpuParityOperationAvailable;

      KFParticleGpuParitySnapshot gpu = cpu;
      gpu.parameters[3] += 0.5f * tolerance.parameterAbsolute;
      assert(KFParticleGpuParity::Compare(cpu, gpu, tolerance).Equivalent());
      gpu.parameters[3] += 2.f * tolerance.parameterAbsolute;
      const auto mismatch = KFParticleGpuParity::Compare(cpu, gpu, tolerance);
      assert((mismatch.issues & KFGpuParityParameterMismatch) != 0u);
      assert(mismatch.firstParameter == 3u);
      assert(mismatch.maxParameterResidual > tolerance.parameterAbsolute);

      gpu = cpu;
      tolerance.parameterRelative = 1.e-3f;
      cpu.parameters[3] = 10.f;
      gpu.parameters[3] = 10.005f;
      assert(KFParticleGpuParity::Compare(cpu, gpu, tolerance).Equivalent());
      gpu.parameters[3] = 10.02f;
      assert((KFParticleGpuParity::Compare(cpu, gpu, tolerance).issues
              & KFGpuParityParameterMismatch) != 0u);
      tolerance.parameterRelative = 0.f;

      gpu = cpu;
      gpu.massValid = 0u;
      gpu.massError = 1.e8f;
      cpu.massError = 0.5f * tolerance.massErrorAbsolute;
      assert(KFParticleGpuParity::Compare(cpu, gpu, tolerance).Equivalent());
      cpu.massError = 2.f * tolerance.massErrorAbsolute;
      assert((KFParticleGpuParity::Compare(cpu, gpu, tolerance).issues
              & KFGpuParityMassMismatch) != 0u);
    }

    const auto ready = KFParticleGpuParity::PromotionVerdict(
      KFGpuPromotionReady, 6u, 0u);
    assert(ready.CanPromote());
    const unsigned int injectedBlockers[] = {
      KFGpuPromotionUnsupportedTopology,
      KFGpuPromotionOverflow,
      KFGpuPromotionInvalidInput,
      KFGpuPromotionInvalidField,
      KFGpuPromotionIncompleteLineage,
      KFGpuPromotionRuntimeUnavailable,
      KFGpuPromotionExecutionFailure,
    };
    for (const unsigned int blocker : injectedBlockers) {
      const auto blocked =
        KFParticleGpuParity::PromotionVerdict(blocker, 6u, 0u);
      assert(!blocked.CanPromote());
      assert((blocked.blockers & blocker) != 0u);
    }
    const auto mismatched =
      KFParticleGpuParity::PromotionVerdict(KFGpuPromotionReady, 6u, 1u);
    assert(!mismatched.CanPromote());
    assert((mismatched.blockers & KFGpuPromotionPhysicsMismatch) != 0u);
    Pass("parity-promotion-contract",
         "all graph topologies expose canonical lineage, bounded fit comparison, and explicit promotion blockers");
  }

  constexpr float kCpuReferencePionMass = 0.13957039f;
  constexpr float kCpuReferenceProtonMass = 0.9382720813f;
  constexpr float kCpuReferenceK0ShortMass = 0.497614f;
  constexpr float kCpuReferenceLambdaMass = 1.115683f;

#ifdef PANDA_STT
  constexpr float kCpuReferenceK0ShortMassSigma = 12.0e-3f;
  constexpr float kCpuReferenceLambdaMassSigma = 2.7e-3f;
  constexpr float kCpuReferenceSecondaryTopoChi2 = -3.f;
#elif defined ALICE_ITS
  constexpr float kCpuReferenceK0ShortMassSigma = 17.7e-3f;
  constexpr float kCpuReferenceLambdaMassSigma = 5.9e-3f;
  constexpr float kCpuReferenceSecondaryTopoChi2 = 5.f;
#elif defined STAR_HFT
  constexpr float kCpuReferenceK0ShortMassSigma = 17.7e-3f;
  constexpr float kCpuReferenceLambdaMassSigma = 5.9e-3f;
  constexpr float kCpuReferenceSecondaryTopoChi2 = 5.f;
#elif defined CBM
  constexpr float kCpuReferenceK0ShortMassSigma = 3.7e-3f;
  constexpr float kCpuReferenceLambdaMassSigma = 1.5e-3f;
  constexpr float kCpuReferenceSecondaryTopoChi2 = 5.f;
#else
  constexpr float kCpuReferenceK0ShortMassSigma = 4.9e-3f;
  constexpr float kCpuReferenceLambdaMassSigma = 2.1e-3f;
  constexpr float kCpuReferenceSecondaryTopoChi2 = 5.f;
#endif

  template<typename T>
  T* HostPointer(xpu::buffer<T>& buffer)
  {
    return xpu::buffer_prop(buffer).template h_ptr<T>();
  }

  void Pass(const char* name, const char* details)
  {
    static int checkNumber = 0;
    ++checkNumber;
    std::cerr << "PASS [" << std::setw(2) << std::setfill('0') << checkNumber
              << "] " << name << " - " << details << '\n';
  }

  bool AlmostEqual(float lhs, float rhs, float tolerance = 1.e-6f)
  {
    return std::fabs(lhs - rhs) <= tolerance;
  }

  bool AlmostEqualScaled(float lhs,
                         float rhs,
                         float absoluteTolerance,
                         float relativeTolerance)
  {
    const float scale = std::max(std::fabs(lhs), std::fabs(rhs));
    return std::fabs(lhs - rhs) <= absoluteTolerance + relativeTolerance * scale;
  }

  struct ReferenceTransportState
  {
    float x;
    float y;
    float z;
    float px;
    float py;
    float pz;
  };

  KFParticleGpuFieldValue ReferenceField(const KFParticleGpuFieldRegion& field, float z)
  {
    const float* coefficients = field.Coefficients();
    const float dz = z - coefficients[9];
    const float dz2 = dz * dz;
    return KFParticleGpuFieldValue(coefficients[0] + coefficients[1] * dz + coefficients[2] * dz2,
                                   coefficients[3] + coefficients[4] * dz + coefficients[5] * dz2,
                                   coefficients[6] + coefficients[7] * dz + coefficients[8] * dz2);
  }

  ReferenceTransportState ReferenceFullFieldRk4(ReferenceTransportState state,
                                                const KFParticleGpuFieldRegion& field,
                                                int charge,
                                                float dS)
  {
    // Independent numerical oracle for the analytic GPU TransportCBM port.
    const int steps = 128;
    const float step = dS / static_cast<float>(steps);
    const float chargeLight = static_cast<float>(charge) * 0.000299792458f;
    const auto derivative = [&field, chargeLight](const ReferenceTransportState& value) {
      const KFParticleGpuFieldValue magneticField = ReferenceField(field, value.z);
      return ReferenceTransportState{value.px,
                                     value.py,
                                     value.pz,
                                     chargeLight * (value.py * magneticField.z - value.pz * magneticField.y),
                                     chargeLight * (value.pz * magneticField.x - value.px * magneticField.z),
                                     chargeLight * (value.px * magneticField.y - value.py * magneticField.x)};
    };
    const auto advance = [](const ReferenceTransportState& value,
                            const ReferenceTransportState& derivativeValue,
                            float scale) {
      return ReferenceTransportState{value.x + scale * derivativeValue.x,
                                     value.y + scale * derivativeValue.y,
                                     value.z + scale * derivativeValue.z,
                                     value.px + scale * derivativeValue.px,
                                     value.py + scale * derivativeValue.py,
                                     value.pz + scale * derivativeValue.pz};
    };

    for (int i = 0; i < steps; ++i) {
      const ReferenceTransportState k1 = derivative(state);
      const ReferenceTransportState k2 = derivative(advance(state, k1, 0.5f * step));
      const ReferenceTransportState k3 = derivative(advance(state, k2, 0.5f * step));
      const ReferenceTransportState k4 = derivative(advance(state, k3, step));
      state = ReferenceTransportState{
        state.x + step * (k1.x + 2.f * (k2.x + k3.x) + k4.x) / 6.f,
        state.y + step * (k1.y + 2.f * (k2.y + k3.y) + k4.y) / 6.f,
        state.z + step * (k1.z + 2.f * (k2.z + k3.z) + k4.z) / 6.f,
        state.px + step * (k1.px + 2.f * (k2.px + k3.px) + k4.px) / 6.f,
        state.py + step * (k1.py + 2.f * (k2.py + k3.py) + k4.py) / 6.f,
        state.pz + step * (k1.pz + 2.f * (k2.pz + k3.pz) + k4.pz) / 6.f};
    }
    return state;
  }

  bool SelectionResultClose(const KFParticleGpuV0SelectionResult& actual,
                            const KFParticleGpuV0SelectionResult& expected,
                            float absoluteTolerance,
                            float relativeTolerance)
  {
    const KFParticleGpuV0SelectionObservables& lhs = actual.observables;
    const KFParticleGpuV0SelectionObservables& rhs = expected.observables;
    return actual.candidateIndex == expected.candidateIndex
           && actual.channelId == expected.channelId
           && actual.eventIndex == expected.eventIndex
           && actual.bestPrimaryVertexIndex == expected.bestPrimaryVertexIndex
           && actual.selectionClass == expected.selectionClass
           && actual.rejectionReasons == expected.rejectionReasons
           && actual.topologyStatus == expected.topologyStatus
           && AlmostEqualScaled(lhs.mass, rhs.mass, absoluteTolerance, relativeTolerance)
           && AlmostEqualScaled(lhs.massError, rhs.massError, absoluteTolerance, relativeTolerance)
           && AlmostEqualScaled(lhs.geometricChi2PerNdf,
                                rhs.geometricChi2PerNdf,
                                absoluteTolerance,
                                relativeTolerance)
           && AlmostEqualScaled(lhs.nearestPrimaryVertexDistance,
                                rhs.nearestPrimaryVertexDistance,
                                absoluteTolerance,
                                relativeTolerance)
           && AlmostEqualScaled(lhs.nearestPrimaryVertexDistanceError,
                                rhs.nearestPrimaryVertexDistanceError,
                                absoluteTolerance,
                                relativeTolerance)
           && AlmostEqualScaled(lhs.nearestPrimaryVertexLdL,
                                rhs.nearestPrimaryVertexLdL,
                                absoluteTolerance,
                                relativeTolerance)
           && AlmostEqualScaled(lhs.bestPrimaryVertexTopoChi2PerNdf,
                                rhs.bestPrimaryVertexTopoChi2PerNdf,
                                absoluteTolerance,
                                relativeTolerance)
           && AlmostEqualScaled(lhs.bestPrimaryVertexDecayLength,
                                rhs.bestPrimaryVertexDecayLength,
                                absoluteTolerance,
                                relativeTolerance)
           && AlmostEqualScaled(lhs.bestPrimaryVertexDecayLengthError,
                                rhs.bestPrimaryVertexDecayLengthError,
                                absoluteTolerance,
                                relativeTolerance)
           && AlmostEqualScaled(lhs.bestPrimaryVertexLdL,
                                rhs.bestPrimaryVertexLdL,
                                absoluteTolerance,
                                relativeTolerance);
  }

  void DumpSelectionResultComparison(const char* label,
                                     const KFParticleGpuV0SelectionResult& actual,
                                     const KFParticleGpuV0SelectionResult& expected,
                                     float absoluteTolerance,
                                     float relativeTolerance)
  {
    const std::streamsize oldPrecision = std::cerr.precision();
    std::cerr << std::setprecision(9)
              << "TRACE selection label=" << label
              << " candidate=" << actual.candidateIndex << '/' << expected.candidateIndex
              << " channel=" << actual.channelId << '/' << expected.channelId
              << " event=" << actual.eventIndex << '/' << expected.eventIndex
              << " best-pv=" << actual.bestPrimaryVertexIndex << '/'
              << expected.bestPrimaryVertexIndex
              << " class=" << actual.selectionClass << '/' << expected.selectionClass
              << " rejection=" << actual.rejectionReasons << '/' << expected.rejectionReasons
              << " topology=" << actual.topologyStatus << '/' << expected.topologyStatus << '\n';
    const auto dump = [absoluteTolerance, relativeTolerance](const char* name,
                                                              float gpu,
                                                              float host) {
      const float scale = std::max(std::fabs(gpu), std::fabs(host));
      const float effectiveTolerance = absoluteTolerance + relativeTolerance * scale;
      std::cerr << "TRACE selection-observable name=" << name
                << " gpu=" << gpu << " host=" << host
                << " delta=" << (gpu - host)
                << " relative-delta=" << (scale > 0.f ? std::fabs(gpu - host) / scale : 0.f)
                << " effective-tolerance=" << effectiveTolerance
                << " close=" << (std::fabs(gpu - host) <= effectiveTolerance ? 1 : 0) << '\n';
    };
    const KFParticleGpuV0SelectionObservables& lhs = actual.observables;
    const KFParticleGpuV0SelectionObservables& rhs = expected.observables;
    dump("mass", lhs.mass, rhs.mass);
    dump("mass-error", lhs.massError, rhs.massError);
    dump("geometric-chi2-ndf", lhs.geometricChi2PerNdf, rhs.geometricChi2PerNdf);
    dump("nearest-pv-distance", lhs.nearestPrimaryVertexDistance,
         rhs.nearestPrimaryVertexDistance);
    dump("nearest-pv-distance-error", lhs.nearestPrimaryVertexDistanceError,
         rhs.nearestPrimaryVertexDistanceError);
    dump("nearest-pv-ldl", lhs.nearestPrimaryVertexLdL, rhs.nearestPrimaryVertexLdL);
    dump("best-pv-topo-chi2-ndf", lhs.bestPrimaryVertexTopoChi2PerNdf,
         rhs.bestPrimaryVertexTopoChi2PerNdf);
    dump("best-pv-decay-length", lhs.bestPrimaryVertexDecayLength,
         rhs.bestPrimaryVertexDecayLength);
    dump("best-pv-decay-length-error", lhs.bestPrimaryVertexDecayLengthError,
         rhs.bestPrimaryVertexDecayLengthError);
    dump("best-pv-ldl", lhs.bestPrimaryVertexLdL, rhs.bestPrimaryVertexLdL);
    std::cerr.precision(oldPrecision);
  }

  void ExpectSelectionResultClose(const KFParticleGpuV0SelectionResult& actual,
                                  const KFParticleGpuV0SelectionResult& expected,
                                  float absoluteTolerance = 1.e-5f,
                                  float relativeTolerance = 2.e-5f)
  {
    if (!SelectionResultClose(
          actual, expected, absoluteTolerance, relativeTolerance)) {
      DumpSelectionResultComparison(
        "mismatch", actual, expected, absoluteTolerance, relativeTolerance);
      assert(false);
    }
  }

  bool VerboseDiagnosticTraceEnabled()
  {
    const char* enabled = std::getenv("KFPARTICLE_GPU_TEST_VERBOSE_TRACE");
    return enabled && enabled[0] != '\0' && enabled[0] != '0';
  }

  constexpr float kDiagnosticFieldBy = 200.f;

  void ExpectFitStateClose(const KFParticleGpuFitState& actual,
                           const KFParticleGpuFitState& expected,
                           float tolerance = 1.e-5f)
  {
    for (int i = 0; i < KFParticleGpuFitState::NumberOfParameters; ++i) {
      assert(AlmostEqual(actual.Parameter(i), expected.Parameter(i), tolerance));
    }
    assert(AlmostEqual(actual.Chi2(), expected.Chi2(), tolerance));
    assert(actual.NDF() == expected.NDF());
    assert(actual.Q() == expected.Q());
    assert(AlmostEqual(actual.SumDaughterMass(), expected.SumDaughterMass(), tolerance));
    assert(AlmostEqual(actual.MassHypo(), expected.MassHypo(), tolerance));
  }

  void ExpectFieldAwareSeedClose(const KFParticleGpuFitState& actual,
                                 const KFParticleGpuFitState& expected)
  {
    // HIP and host libm can differ slightly in the diagnostic constant-By path.
    // Keep this comparison strict enough to catch wrong data flow, but do not
    // require bitwise-like agreement from iterative trigonometric transport.
    ExpectFitStateClose(actual, expected, 5.e-4f);
  }

  struct FieldAwareEnergyFitTolerance
  {
    float stateAbsolute = 0.f;
    float stateRelative = 0.f;
    float chi2 = 0.f;
    float mass = 0.f;
  };

  FieldAwareEnergyFitTolerance DefaultV0EnergyFitTolerance(int motherPdg)
  {
    // These tolerances cover HIP/CPU float and libm differences in the
    // coupled full-field DCA derivatives plus energy-fit path.
    if (motherPdg == 310) {
      return {7.e-4f, 2.e-5f, 2.e-3f, 2.e-4f};
    }
    return {1.e-3f, 2.e-5f, 3.e-3f, 3.e-4f};
  }

  void ExpectFullFieldEnergyFitClose(const KFParticleGpuFitState& actual,
                                     const KFParticleGpuFitState& expected,
                                     int motherPdg,
                                     const char* label = "v0")
  {
    const FieldAwareEnergyFitTolerance tolerance =
      DefaultV0EnergyFitTolerance(motherPdg);
    for (int i = 0; i < KFParticleGpuFitState::NumberOfParameters; ++i) {
      if (!AlmostEqualScaled(actual.Parameter(i),
                             expected.Parameter(i),
                             tolerance.stateAbsolute,
                             tolerance.stateRelative)) {
        const float stateScale =
          std::max(std::fabs(actual.Parameter(i)), std::fabs(expected.Parameter(i)));
        std::cerr << "DETAIL full-field-" << label << "-fit state mother=" << motherPdg
                  << " first-parameter=" << i
                  << " state-absolute-tolerance=" << tolerance.stateAbsolute
                  << " state-relative-tolerance=" << tolerance.stateRelative
                  << " state-effective-tolerance="
                  << (tolerance.stateAbsolute + tolerance.stateRelative * stateScale) << '\n';
        for (int parameter = 0; parameter < KFParticleGpuFitState::NumberOfParameters;
             ++parameter) {
          std::cerr << "DETAIL full-field-" << label << "-fit parameter=" << parameter
                    << " gpu=" << actual.Parameter(parameter)
                    << " host=" << expected.Parameter(parameter)
                    << " delta=" << (actual.Parameter(parameter) - expected.Parameter(parameter))
                    << '\n';
        }
        std::cerr << "DETAIL full-field-" << label << "-fit chi2 gpu=" << actual.Chi2()
                  << " host=" << expected.Chi2()
                  << " delta=" << (actual.Chi2() - expected.Chi2())
                  << " ndf=" << actual.NDF() << '/' << expected.NDF()
                  << " q=" << actual.Q() << '/' << expected.Q() << '\n';
        for (int parameter = 0; parameter < KFParticleGpuFitState::NumberOfParameters;
             ++parameter) {
          std::cerr << "DETAIL full-field-" << label << "-fit covariance-diagonal=" << parameter
                    << " gpu=" << actual.Covariance(parameter, parameter)
                    << " host=" << expected.Covariance(parameter, parameter)
                    << " delta=" << (actual.Covariance(parameter, parameter)
                                   - expected.Covariance(parameter, parameter))
                    << '\n';
        }
        assert(false);
      }
    }
    if (!AlmostEqual(actual.Chi2(), expected.Chi2(), tolerance.chi2)) {
      std::cerr << "DETAIL full-field-" << label << "-fit chi2"
                << " mother=" << motherPdg
                << " gpu=" << actual.Chi2()
                << " host=" << expected.Chi2()
                << " delta=" << (actual.Chi2() - expected.Chi2())
                << " tolerance=" << tolerance.chi2 << '\n';
      assert(false);
    }
    assert(actual.NDF() == expected.NDF());
    assert(actual.Q() == expected.Q());
    if (!AlmostEqual(actual.SumDaughterMass(), expected.SumDaughterMass(), tolerance.mass)
        || !AlmostEqual(actual.MassHypo(), expected.MassHypo(), tolerance.mass)) {
      std::cerr << "DETAIL full-field-" << label << "-fit metadata"
                << " mother=" << motherPdg
                << " gpu-sum-mass=" << actual.SumDaughterMass()
                << " host-sum-mass=" << expected.SumDaughterMass()
                << " gpu-mass-hypo=" << actual.MassHypo()
                << " host-mass-hypo=" << expected.MassHypo()
                << " tolerance=" << tolerance.mass << '\n';
      assert(false);
    }

    float actualMass = 0.f;
    float actualMassError = 0.f;
    float expectedMass = 0.f;
    float expectedMassError = 0.f;
    const bool actualMassValid =
      KFParticleGpuMath::GetMass(actual, actualMass, actualMassError);
    const bool expectedMassValid =
      KFParticleGpuMath::GetMass(expected, expectedMass, expectedMassError);
    assert(actualMassValid == expectedMassValid);
    if (actualMassValid
        && !AlmostEqual(actualMass, expectedMass, tolerance.mass)) {
      std::cerr << "DETAIL full-field-" << label << "-fit mass"
                << " mother=" << motherPdg
                << " gpu=" << actualMass
                << " host=" << expectedMass
                << " delta=" << (actualMass - expectedMass)
                << " tolerance=" << tolerance.mass << '\n';
      assert(false);
    }
  }

  void ExpectDefaultV0EnergyFitClose(const KFParticleGpuFitState& actual,
                                     const KFParticleGpuFitState& expected,
                                     int motherPdg)
  {
    ExpectFullFieldEnergyFitClose(actual, expected, motherPdg);
  }

  void StoreDiagnosticField(KFParticleGpuInputTrackSoAView& tracks, unsigned int track)
  {
    KFParticleGpuFieldRegion diagnosticField;
    diagnosticField.Coefficient(0) = 0.4f;
    diagnosticField.Coefficient(1) = 0.02f;
    diagnosticField.Coefficient(2) = 0.001f;
    diagnosticField.Coefficient(3) = kDiagnosticFieldBy;
    diagnosticField.Coefficient(4) = 0.5f;
    diagnosticField.Coefficient(5) = 0.01f;
    diagnosticField.Coefficient(6) = -0.3f;
    diagnosticField.Coefficient(7) = 0.015f;
    diagnosticField.Coefficient(8) = 0.0005f;
    diagnosticField.Coefficient(9) = tracks.Numerical().Parameter(2, track);
    StoreFieldRegion(diagnosticField, tracks, track);
  }

  struct ReferenceState
  {
    double x = 0.0;
    double y = 0.0;
    double z = 0.0;
    double px = 0.0;
    double py = 0.0;
    double pz = 0.0;
    double energy = 0.0;
    int charge = 0;
  };

  ReferenceState MakeReferenceState(const KFParticleGpuFitState& source)
  {
    ReferenceState state;
    state.x = source.X();
    state.y = source.Y();
    state.z = source.Z();
    state.px = source.Px();
    state.py = source.Py();
    state.pz = source.Pz();
    state.energy = source.E();
    state.charge = source.Q();
    return state;
  }

  void ReferenceGetDStoParticleLine(const ReferenceState& first,
                                    const ReferenceState& second,
                                    double dS[2])
  {
    const double dx = first.x - second.x;
    const double dy = first.y - second.y;
    const double dz = first.z - second.z;
    const double p1p1 = first.px * first.px + first.py * first.py + first.pz * first.pz;
    const double p2p2 = second.px * second.px + second.py * second.py + second.pz * second.pz;
    const double p1p2 = first.px * second.px + first.py * second.py + first.pz * second.pz;
    const double r1p1 = dx * first.px + dy * first.py + dz * first.pz;
    const double r1p2 = dx * second.px + dy * second.py + dz * second.pz;
    double determinant = p1p2 * p1p2 - p1p1 * p2p2;

    if (determinant > -1.e-8 && determinant < 1.e-8) {
      determinant = determinant < 0.0 ? -1.e-8 : 1.e-8;
    }

    dS[0] = (r1p1 * p2p2 - r1p2 * p1p2) / determinant;
    dS[1] = (r1p1 * p1p2 - r1p2 * p1p1) / determinant;
  }

  ReferenceState ReferenceTransportConstantBy(const ReferenceState& particle,
                                              double dS,
                                              double by)
  {
    const double kCLight = 0.000299792458;
    const double bq = by * static_cast<double>(particle.charge) * kCLight;
    ReferenceState transported = particle;

    if (std::fabs(bq) < 1.e-8) {
      transported.x = particle.x + particle.px * dS;
      transported.y = particle.y + particle.py * dS;
      transported.z = particle.z + particle.pz * dS;
      return transported;
    }

    const double bs = bq * dS;
    const double sinBs = std::sin(bs);
    const double cosBs = std::cos(bs);
    double sB = 0.0;
    double cB = 0.0;
    if (std::fabs(bs) > 1.e-8) {
      sB = sinBs / bq;
      cB = (1.0 - cosBs) / bq;
    }
    else {
      sB = dS;
      cB = 0.5 * dS * bs;
    }

    transported.x = particle.x + sB * particle.px - cB * particle.pz;
    transported.y = particle.y + dS * particle.py;
    transported.z = particle.z + cB * particle.px + sB * particle.pz;
    transported.px = cosBs * particle.px - sinBs * particle.pz;
    transported.py = particle.py;
    transported.pz = sinBs * particle.px + cosBs * particle.pz;
    return transported;
  }

  bool ReferenceBuildConstantByDcaKinematicMother(const ReferenceState& referenceFirst,
                                                  const ReferenceState& referenceSecond,
                                                  double by,
                                                  ReferenceState& mother)
  {
    const bool firstIsStraight =
      std::fabs(by * static_cast<double>(referenceFirst.charge)) < 1.e-8;
    const bool secondIsStraight =
      std::fabs(by * static_cast<double>(referenceSecond.charge)) < 1.e-8;

    double dS[2] = {0.0, 0.0};
    ReferenceGetDStoParticleLine(referenceFirst, referenceSecond, dS);
    if (std::fabs(dS[0] * referenceFirst.pz) > 1000.0
        || std::fabs(dS[1] * referenceSecond.pz) > 1000.0) {
      return false;
    }

    ReferenceState firstAtDca;
    ReferenceState secondAtDca;
    if (firstIsStraight && secondIsStraight) {
      firstAtDca = ReferenceTransportConstantBy(referenceFirst, dS[0], 0.0);
      secondAtDca = ReferenceTransportConstantBy(referenceSecond, dS[1], 0.0);
    }
    else {
      for (int iteration = 0; iteration < 2; ++iteration) {
        firstAtDca = ReferenceTransportConstantBy(referenceFirst, dS[0], by);
        secondAtDca = ReferenceTransportConstantBy(referenceSecond, dS[1], by);
        double correction[2] = {0.0, 0.0};
        ReferenceGetDStoParticleLine(firstAtDca, secondAtDca, correction);
        dS[0] += correction[0];
        dS[1] += correction[1];
        if (std::fabs(dS[0] * referenceFirst.pz) > 1000.0
            || std::fabs(dS[1] * referenceSecond.pz) > 1000.0) {
          return false;
        }
      }
      firstAtDca = ReferenceTransportConstantBy(referenceFirst, dS[0], by);
      secondAtDca = ReferenceTransportConstantBy(referenceSecond, dS[1], by);
    }

    mother.x = 0.5 * (firstAtDca.x + secondAtDca.x);
    mother.y = 0.5 * (firstAtDca.y + secondAtDca.y);
    mother.z = 0.5 * (firstAtDca.z + secondAtDca.z);
    mother.px = firstAtDca.px + secondAtDca.px;
    mother.py = firstAtDca.py + secondAtDca.py;
    mother.pz = firstAtDca.pz + secondAtDca.pz;
    mother.energy = firstAtDca.energy + secondAtDca.energy;
    mother.charge = firstAtDca.charge + secondAtDca.charge;
    return true;
  }

  bool ReferenceBuildConstantByDcaKinematicMother(const KFParticleGpuFitState& first,
                                                  const KFParticleGpuFitState& second,
                                                  double by,
                                                  ReferenceState& mother)
  {
    return ReferenceBuildConstantByDcaKinematicMother(
      MakeReferenceState(first), MakeReferenceState(second), by, mother);
  }

  void ExpectFitStateMatchesReference(const KFParticleGpuFitState& actual,
                                      const ReferenceState& expected,
                                      float tolerance)
  {
    assert(AlmostEqual(actual.X(), static_cast<float>(expected.x), tolerance));
    assert(AlmostEqual(actual.Y(), static_cast<float>(expected.y), tolerance));
    assert(AlmostEqual(actual.Z(), static_cast<float>(expected.z), tolerance));
    assert(AlmostEqual(actual.Px(), static_cast<float>(expected.px), tolerance));
    assert(AlmostEqual(actual.Py(), static_cast<float>(expected.py), tolerance));
    assert(AlmostEqual(actual.Pz(), static_cast<float>(expected.pz), tolerance));
    assert(AlmostEqual(actual.E(), static_cast<float>(expected.energy), tolerance));
    assert(actual.Q() == expected.charge);
  }

  void FillInput(KFParticleGpuBufferManager& buffers,
                 const KFParticleGpuBufferCapacities& capacities)
  {
    buffers.SetInputSizes(2, 1, 1);

    KFParticleGpuInputTrackSoAView tracks = buffers.HostInputTracks();
    for (unsigned int i = 0; i < capacities.tracks; ++i) {
      KFParticleGpuTrackState track;
      track.X() = 1.f + i;
      track.Y() = 10.f + i;
      track.Z() = 20.f + i;
      track.Px() = 2.f + i;
      track.Py() = 3.f + i;
      track.Pz() = 4.f + i;
      track.Covariance(0) = 100.f + i;
      StoreTrackState(track, tracks, i);
      tracks.SourceId(i) = 17 + static_cast<int>(2 * i);
      tracks.Pdg(i) = (i % 2 == 0) ? 211 : -211;
      tracks.Charge(i) = (i % 2 == 0) ? 1 : -1;
      tracks.PrimaryVertexIndex(i) = static_cast<int>(i) - 1;
      tracks.NumberOfPixelHits(i) = 4 + static_cast<int>(i);
      tracks.ChiToPrimaryVertex(i) = 12.f + i;

      KFParticleGpuFieldRegion field;
      field.Coefficient(0) = 3.f + i;
      field.Coefficient(1) = 0.25f;
      field.Coefficient(2) = 0.125f;
      field.Coefficient(3) = 4.f + i;
      field.Coefficient(4) = -0.5f;
      field.Coefficient(5) = 0.25f;
      field.Coefficient(6) = 5.f + i;
      field.Coefficient(7) = 0.75f;
      field.Coefficient(8) = -0.125f;
      field.Coefficient(9) = track.Z();
      StoreFieldRegion(field, tracks, i);
    }

    KFParticleGpuVertexState vertex;
    vertex.X() = 4.f;
    vertex.Covariance(0) = 40.f;
    vertex.Chi2() = 2.5f;
    vertex.NDF() = 5;
    vertex.NContributors() = 6;
    KFParticleGpuVertexSoAView vertices = buffers.HostPrimaryVertices();
    StoreVertexState(vertex, vertices, 0);

    KFParticleGpuEventDesc* events = buffers.HostEvents();
    events[0].eventId = 23;
    events[0].TrackSet(SecondaryPositiveFirst).tracks = KFParticleGpuRange(0, 2);
    events[0].TrackSet(SecondaryPositiveFirst).Species(Pion) = KFParticleGpuRange(0, 2);
    events[0].primaryVertices = KFParticleGpuRange(0, 1);
  }

  void TestHostViews(KFParticleGpuBufferManager& buffers)
  {
    assert(buffers.TrackSize() == 2);
    assert(buffers.VertexSize() == 1);
    assert(buffers.EventSize() == 1);

    KFParticleGpuInputTrackSoAView tracks = buffers.HostInputTracks();
    assert(tracks.Size() == 2);
    assert(tracks.Stride() == buffers.Capacities().tracks);
    assert(tracks.Numerical().Parameter(0, 1) == 2.f);
    assert(tracks.Numerical().Parameter(3, 1) == 3.f);
    assert(tracks.Numerical().Covariance(0, 1) == 101.f);
    assert(tracks.FieldCoefficient(0, 1) == 4.f);
    assert(HasFieldRegions(MakeConstView(tracks)));
    KFParticleGpuFieldRegion loadedField;
    assert(LoadFieldRegionOrZero(MakeConstView(tracks), 1, loadedField));
    assert(loadedField.Coefficient(3) == 5.f);
    const KFParticleGpuFieldValue fieldAtTrack =
      EvaluateTrackField(MakeConstView(tracks), 1, tracks.Numerical().Parameter(2, 1));
    assert(fieldAtTrack.x == 4.f);
    assert(fieldAtTrack.y == 5.f);
    assert(fieldAtTrack.z == 6.f);
    assert(!LoadFieldRegionOrZero(MakeConstView(tracks), tracks.Size(), loadedField));
    assert(loadedField.Get(0.f).x == 0.f);
    assert(tracks.ChiToPrimaryVertex(1) == 13.f);
    assert(tracks.SourceId(1) == 19);
    assert(tracks.Pdg(1) == -211);
    assert(tracks.Charge(1) == -1);
    assert(tracks.PrimaryVertexIndex(1) == 0);
    assert(tracks.NumberOfPixelHits(1) == 5);

    KFParticleGpuVertexSoAView vertices = buffers.HostPrimaryVertices();
    assert(vertices.Size() == 1);
    assert(vertices.Stride() == buffers.Capacities().vertices);
    assert(vertices.Parameter(0, 0) == 4.f);
    assert(vertices.Covariance(0, 0) == 40.f);
    assert(vertices.Chi2(0) == 2.5f);
    assert(vertices.NDF(0) == 5);
    assert(vertices.NContributors(0) == 6);

    KFParticleGpuEventDesc* events = buffers.HostEvents();
    assert(events[0].eventId == 23);
    assert(events[0].TrackSet(SecondaryPositiveFirst).tracks.End() == 2);
    assert(events[0].TrackSet(SecondaryPositiveFirst).Species(Pion).End() == 2);
    assert(events[0].primaryVertices.End() == 1);
    Pass("host-views", "component-major SoA stride and packed track/vertex/event accessors");
  }

  void TestDeviceInputProbe(KFParticleGpuRuntime& runtime, KFParticleGpuBufferManager& buffers)
  {
    buffers.UploadInput();

    xpu::buffer<float> floatChecks(11, xpu::buf_io);
    xpu::buffer<int> integerChecks(8, xpu::buf_io);
    xpu::buffer<unsigned int> unsignedChecks(9, xpu::buf_io);

    runtime.GetQueue().launch<KFParticleGpuInputLayoutProbe>(
      xpu::n_threads(1),
      MakeConstView(buffers.DeviceInputTracks()),
      MakeConstView(buffers.DevicePrimaryVertices()),
      buffers.DeviceEvents(),
      floatChecks.get(),
      integerChecks.get(),
      unsignedChecks.get());
    runtime.GetQueue().wait();
    runtime.GetQueue().copy(floatChecks, xpu::d2h);
    runtime.GetQueue().copy(integerChecks, xpu::d2h);
    runtime.GetQueue().copy(unsignedChecks, xpu::d2h);
    runtime.GetQueue().wait();

    const float* floats = HostPointer(floatChecks);
    const int* integers = HostPointer(integerChecks);
    const unsigned int* unsigneds = HostPointer(unsignedChecks);
    assert(floats[0] == 2.f);
    assert(floats[1] == 3.f);
    assert(floats[2] == 101.f);
    assert(floats[3] == 4.f);
    assert(floats[4] == 13.f);
    assert(floats[5] == 4.f);
    assert(floats[6] == 40.f);
    assert(floats[7] == 2.5f);
    assert(floats[8] == 4.f);
    assert(floats[9] == 5.f);
    assert(floats[10] == 6.f);
    assert(integers[0] == 19);
    assert(integers[1] == -211);
    assert(integers[2] == -1);
    assert(integers[3] == 0);
    assert(integers[4] == 5);
    assert(integers[5] == 5);
    assert(integers[6] == 6);
    assert(integers[7] == 23);
    assert(unsigneds[0] == 2);
    assert(unsigneds[1] == buffers.Capacities().tracks);
    assert(unsigneds[2] == 1);
    assert(unsigneds[3] == buffers.Capacities().vertices);
    assert(unsigneds[4] == 0);
    assert(unsigneds[5] == 2);
    assert(unsigneds[6] == 0);
    assert(unsigneds[7] == 1);
    assert(unsigneds[8] == 1);
    Pass("device-input-probe", "host-to-device upload plus device-side SoA/event and field-region reads");
  }

  void TestCandidateReset(KFParticleGpuBufferManager& buffers)
  {
    KFParticleGpuCandidatePoolView candidates = buffers.HostCandidates();
    candidates.SizeData()[0] = 99;
    candidates.Daughters().SizeData()[0] = 98;
    candidates.OverflowFlagsData()[0] = 97;

    buffers.ResetCandidates();
    buffers.DownloadCandidates();

    const KFParticleGpuConstCandidatePoolView resetCandidates =
      MakeConstView(buffers.HostCandidates());
    assert(resetCandidates.Size() == 0);
    assert(resetCandidates.Daughters().Size() == 0);
    assert(resetCandidates.OverflowFlags() == 0);
    Pass("candidate-reset", "candidate, daughter, and overflow counters reset on device and read back");
  }

  std::vector<unsigned int> ExplicitV0TrackRoutingOracle(
    const KFParticleGpuDecayPlan& plan,
    unsigned int eventIndex,
    unsigned int selectedEventIndex,
    unsigned int bachelorEventIndex,
    const KFParticleGpuRange& selectedRange,
    const KFParticleGpuRange& bachelorRange,
    unsigned int bachelorTrackIndex,
    KFParticleGpuTrackSet bachelorTrackSet,
    KFParticleGpuTrackSpecies bachelorSpecies,
    int selectedV0Pdg,
    int bachelorPdg)
  {
    std::vector<unsigned int> result;
    if (selectedRange.size == 0u
        || eventIndex != selectedEventIndex
        || eventIndex != bachelorEventIndex
        || !bachelorRange.Contains(bachelorTrackIndex)) {
      return result;
    }
    for (std::size_t index = 0u; index < plan.NumberOfV0TrackCascadeChannels(); ++index) {
      const KFParticleGpuV0TrackCascadeChannel& channel =
        plan.V0TrackCascadeChannel(index);
      if (channel.v0Pdg == selectedV0Pdg
          && channel.bachelorPdg == bachelorPdg
          && channel.bachelorTrackSet == bachelorTrackSet
          && channel.bachelorSpecies == bachelorSpecies) {
        result.push_back(channel.channelId);
      }
    }
    return result;
  }

  std::vector<unsigned int> MaskedV0TrackRoutingResult(
    const KFParticleGpuV0TrackRoutingPlan& routing,
    int selectedV0Pdg,
    int bachelorPdg)
  {
    const unsigned int selectedV0Role =
      static_cast<unsigned int>(KFParticleGpuClassifySelectedV0Role(selectedV0Pdg));
    const unsigned int bachelorRole =
      static_cast<unsigned int>(KFParticleGpuClassifyBachelorRole(bachelorPdg));
    const KFParticleGpuChannelMask channels =
      routing.CompatibleChannels(selectedV0Role, bachelorRole);
    std::vector<unsigned int> result;
    for (unsigned int bit = channels.NextSetBit(0u);
         bit != KFParticleGpuChannelMask::InvalidBit;
         bit = channels.NextSetBit(bit + 1u)) {
      assert(bit < routing.Descriptors().size());
      result.push_back(routing.Descriptors()[bit].channelId);
    }
    return result;
  }

  void TestV0TrackRoutingPlan(KFParticleGpuBufferManager& buffers)
  {
    KFParticleGpuChannelMask mask;
    assert(mask.Empty());
    assert(mask.Set(0u));
    assert(mask.Set(31u));
    assert(mask.Set(32u));
    assert(mask.Set(63u));
    assert(mask.Set(64u));
    assert(mask.Set(127u));
    assert(mask.Set(128u));
    assert(mask.Set(255u));
    assert(mask.Set(KFParticleGpuChannelMask::BitCapacity - 1u));
    assert(!mask.Set(KFParticleGpuChannelMask::BitCapacity));
    assert(mask.Count() == 9u);
    assert(mask.NextSetBit(0u) == 0u);
    assert(mask.NextSetBit(1u) == 31u);
    assert(mask.NextSetBit(32u) == 32u);
    assert(mask.NextSetBit(33u) == 63u);
    assert(mask.NextSetBit(64u) == 64u);
    assert(mask.NextSetBit(65u) == 127u);
    assert(mask.NextSetBit(128u) == 128u);
    assert(mask.NextSetBit(129u) == 255u);
    assert(mask.NextSetBit(256u) == KFParticleGpuChannelMask::BitCapacity - 1u);
    assert(mask.NextSetBit(KFParticleGpuChannelMask::BitCapacity)
           == KFParticleGpuChannelMask::InvalidBit);
    KFParticleGpuChannelMask upperWord;
    upperWord.Set(128u);
    upperWord.Set(255u);
    assert(mask.Intersected(upperWord) == upperWord);

    bool uncompiledUploadRejected = false;
    try {
      KFParticleGpuV0TrackRoutingPlan uncompiled;
      buffers.UploadV0TrackRoutingPlan(uncompiled);
    }
    catch (const std::invalid_argument&) {
      uncompiledUploadRejected = true;
    }
    assert(uncompiledUploadRejected);

    KFParticleGpuDecayPlan plan;
    AddDefaultV0TrackCascadeChannels(plan);
    KFParticleGpuV0TrackRoutingPlan routing;
    routing.Compile(plan);
    assert(routing.SourceRevision() == plan.Revision());
    assert(routing.Descriptors().size() == 4u);
    assert(routing.CompatibilityEntries().size() == 4u);
    assert(routing.Groups().size() == 2u);
    assert(routing.EnabledChannels().Count() == 4u);
    assert(routing.Groups()[0].bachelorTrackSet == SecondaryNegativeFirst);
    assert(routing.Groups()[0].AcceptsSelectedV0Role(KFGpuSelectedV0RoleLambda));
    assert(routing.Groups()[0].AcceptsBachelorRole(KFGpuBachelorRolePiMinus));
    assert(routing.Groups()[0].AcceptsBachelorRole(KFGpuBachelorRoleKMinus));
    assert(routing.Groups()[0].AcceptsBachelorSpecies(Pion));
    assert(routing.Groups()[0].AcceptsBachelorSpecies(Kaon));
    assert(routing.Groups()[0].channels.Test(0u));
    assert(routing.Groups()[0].channels.Test(2u));
    assert(routing.Groups()[1].bachelorTrackSet == SecondaryPositiveFirst);
    assert(routing.Groups()[1].channels.Test(1u));
    assert(routing.Groups()[1].channels.Test(3u));

    const KFParticleGpuRange selectedRange(4u, 2u);
    const KFParticleGpuRange negativeBachelors(10u, 3u);
    const std::vector<unsigned int> explicitXi = ExplicitV0TrackRoutingOracle(
      plan, 7u, 7u, 7u, selectedRange, negativeBachelors, 11u,
      SecondaryNegativeFirst, Pion, 3122, -211);
    const std::vector<unsigned int> maskedXi =
      MaskedV0TrackRoutingResult(routing, 3122, -211);
    assert(explicitXi == maskedXi);
    assert(explicitXi.size() == 1u
           && explicitXi[0] == KFGpuChannelXiMinusToLambdaPiMinus);
    assert(ExplicitV0TrackRoutingOracle(
             plan, 7u, 7u, 8u, selectedRange, negativeBachelors, 11u,
             SecondaryNegativeFirst, Pion, 3122, -211).empty());
    assert(ExplicitV0TrackRoutingOracle(
             plan, 7u, 7u, 7u, KFParticleGpuRange(4u, 0u), negativeBachelors, 11u,
             SecondaryNegativeFirst, Pion, 3122, -211).empty());
    assert(ExplicitV0TrackRoutingOracle(
             plan, 7u, 7u, 7u, selectedRange, negativeBachelors, 14u,
             SecondaryNegativeFirst, Pion, 3122, -211).empty());

    KFParticleGpuV0TrackCascadeChannel alternateXi =
      plan.V0TrackCascadeChannel(0u);
    alternateXi.channelId = 1011u;
    plan.AddV0TrackCascadeChannel(alternateXi);
    routing.Compile(plan);
    const std::vector<unsigned int> explicitMulti = ExplicitV0TrackRoutingOracle(
      plan, 7u, 7u, 7u, selectedRange, negativeBachelors, 11u,
      SecondaryNegativeFirst, Pion, 3122, -211);
    const std::vector<unsigned int> maskedMulti =
      MaskedV0TrackRoutingResult(routing, 3122, -211);
    assert(explicitMulti == maskedMulti);
    assert(maskedMulti.size() == 2u);

    KFParticleGpuDecayPlan widePlan;
    for (unsigned int index = 0u; index < 96u; ++index) {
      KFParticleGpuV0TrackCascadeChannel channel = alternateXi;
      channel.channelId = 2000u + index;
      widePlan.AddV0TrackCascadeChannel(channel);
    }
    KFParticleGpuV0TrackRoutingPlan wideRouting;
    wideRouting.Compile(widePlan);
    const KFParticleGpuChannelMask wideMask =
      wideRouting.CompatibleChannels(KFGpuSelectedV0RoleLambda, KFGpuBachelorRolePiMinus);
    assert(wideMask.Count() == 96u);
    assert(wideMask.Test(31u));
    assert(wideMask.Test(32u));
    assert(wideMask.Test(33u));
    assert(wideMask.Test(63u));
    assert(wideMask.Test(64u));
    assert(wideMask.Test(95u));

    bool duplicateRejected = false;
    try {
      KFParticleGpuDecayPlan invalid;
      invalid.AddV0TrackCascadeChannel(alternateXi);
      invalid.AddV0TrackCascadeChannel(alternateXi);
      KFParticleGpuV0TrackRoutingPlan invalidRouting;
      invalidRouting.Compile(invalid);
    }
    catch (const std::invalid_argument&) {
      duplicateRejected = true;
    }
    assert(duplicateRejected);

    KFParticleGpuDecayPlan genericParentPlan;
    KFParticleGpuV0TrackCascadeChannel genericParent = alternateXi;
    genericParent.channelId = 3001u;
    genericParent.parentChannelId = 701u;
    genericParent.v0Pdg = 310;
    genericParent.bachelorPdg = 2212;
    genericParent.bachelorSpecies = Proton;
    genericParent.bachelorTrackSet = SecondaryPositiveFirst;
    genericParentPlan.AddV0TrackCascadeChannel(genericParent);
    KFParticleGpuV0TrackRoutingPlan genericParentRouting;
    genericParentRouting.Compile(genericParentPlan);
    assert(genericParentRouting.CompatibleChannelsByPdg(701u, 310, 2212).Test(0u));
    assert(genericParentRouting.CompatibleChannelsByPdg(702u, 310, 2212).Empty());

    bool inconsistentSpeciesRejected = false;
    try {
      KFParticleGpuDecayPlan invalid;
      KFParticleGpuV0TrackCascadeChannel channel = alternateXi;
      channel.channelId = 3002u;
      channel.bachelorSpecies = Kaon;
      invalid.AddV0TrackCascadeChannel(channel);
      KFParticleGpuV0TrackRoutingPlan invalidRouting;
      invalidRouting.Compile(invalid);
    }
    catch (const std::invalid_argument&) {
      inconsistentSpeciesRejected = true;
    }
    assert(inconsistentSpeciesRejected);

    bool inconsistentSignRejected = false;
    try {
      KFParticleGpuDecayPlan invalid;
      KFParticleGpuV0TrackCascadeChannel channel = alternateXi;
      channel.channelId = 3003u;
      channel.bachelorTrackSet = SecondaryPositiveFirst;
      invalid.AddV0TrackCascadeChannel(channel);
      KFParticleGpuV0TrackRoutingPlan invalidRouting;
      invalidRouting.Compile(invalid);
    }
    catch (const std::invalid_argument&) {
      inconsistentSignRejected = true;
    }
    assert(inconsistentSignRejected);

    plan.Clear();
    AddDefaultV0TrackCascadeChannels(plan);
    routing.Compile(plan);
    assert(buffers.UploadV0TrackRoutingPlan(routing));
    assert(!buffers.UploadV0TrackRoutingPlan(routing));
    assert(buffers.V0TrackRoutingRevision() == plan.Revision());
    const KFParticleGpuV0TrackRoutingView hostRouting = buffers.HostV0TrackRouting();
    assert(hostRouting.DescriptorCount() == 4u);
    assert(hostRouting.CompatibilityCount() == 4u);
    assert(hostRouting.GroupCount() == 2u);
    assert(hostRouting.EnabledChannelsData()[0].Count() == 4u);
    assert(hostRouting.ChannelAcceptedCounters()[0] == 0u);
    assert(hostRouting.CompatibleChannels(
             KFGpuSelectedV0RoleLambda, KFGpuBachelorRoleKMinus).Test(2u));
    const KFParticleGpuDeviceStorage& storage = buffers.Storage();
    assert(storage.fV0TrackRoutingDescriptors.get() != nullptr);
    assert(storage.fV0TrackRoutingCompatibility.get() != nullptr);
    assert(storage.fV0TrackRoutingGroups.get() != nullptr);
    assert(storage.fV0TrackRoutingEnabledChannels.get() != nullptr);
    assert(storage.fV0TrackRoutingChannelAcceptedCounters.get() != nullptr);

    Pass("v0-track-routing-plan",
         "composite-track routing keeps legacy role hints while exact parent-channel and PDG masks own correctness");
  }

  struct TwoDaughterRoutingTaskKey
  {
    unsigned int eventIndex = 0u;
    unsigned int channelId = 0u;
    unsigned int firstSourceId = 0u;
    unsigned int secondSourceId = 0u;

    bool operator==(const TwoDaughterRoutingTaskKey& other) const
    {
      return eventIndex == other.eventIndex
             && channelId == other.channelId
             && firstSourceId == other.firstSourceId
             && secondSourceId == other.secondSourceId;
    }
  };

  std::vector<TwoDaughterRoutingTaskKey> ExplicitTwoDaughterRoutingOracle(
    const KFParticleGpuDecayPlan& plan,
    unsigned int eventIndex,
    unsigned int firstEventIndex,
    unsigned int secondEventIndex,
    const KFParticleGpuRange& firstRange,
    const KFParticleGpuRange& secondRange,
    unsigned int firstTrackIndex,
    unsigned int secondTrackIndex,
    unsigned int firstSourceId,
    unsigned int secondSourceId,
    KFParticleGpuTrackSet firstTrackSet,
    KFParticleGpuTrackSet secondTrackSet,
    KFParticleGpuTrackSpecies firstSpecies,
    KFParticleGpuTrackSpecies secondSpecies,
    int firstPdg,
    int secondPdg)
  {
    std::vector<TwoDaughterRoutingTaskKey> result;
    if (eventIndex != firstEventIndex
        || eventIndex != secondEventIndex
        || !firstRange.Contains(firstTrackIndex)
        || !secondRange.Contains(secondTrackIndex)
        || firstSourceId == secondSourceId) {
      return result;
    }
    const auto sourceSpecies = [](int pdg) {
      switch (pdg < 0 ? -pdg : pdg) {
        case 11: return Electron;
        case 13:
        case 19: return Muon;
        case 211: return Pion;
        case 321: return Kaon;
        case 2212: return Proton;
        case 1000010020: return Deuteron;
        case 1000010030: return Triton;
        case 1000020030: return Helium3;
        case 1000020040: return Helium4;
        case 1000020060: return Helium6;
        case 1000030060: return Lithium6;
        case 1000030070: return Lithium7;
        case 1000040070: return Beryllium7;
        default: return NumberOfTrackSpecies;
      }
    };
    for (std::size_t index = 0u; index < plan.NumberOfTwoDaughterChannels(); ++index) {
      const KFParticleGpuTwoDaughterChannel& channel =
        plan.TwoDaughterChannel(index);
      const int firstSourcePdg = channel.firstSourcePdg != 0
                                   ? channel.firstSourcePdg
                                   : channel.firstDaughterPdg;
      const int secondSourcePdg = channel.secondSourcePdg != 0
                                    ? channel.secondSourcePdg
                                    : channel.secondDaughterPdg;
      const bool firstSourceMatches =
        firstPdg == firstSourcePdg
        || (channel.firstAlternateSourcePdg != 0
            && firstPdg == channel.firstAlternateSourcePdg);
      const bool secondSourceMatches =
        secondPdg == secondSourcePdg
        || (channel.secondAlternateSourcePdg != 0
            && secondPdg == channel.secondAlternateSourcePdg);
      if (channel.firstTrackSet == firstTrackSet
          && channel.secondTrackSet == secondTrackSet
          && sourceSpecies(firstPdg) == firstSpecies
          && sourceSpecies(secondPdg) == secondSpecies
          && firstSourceMatches
          && secondSourceMatches) {
        result.push_back(
          {eventIndex, channel.channelId, firstSourceId, secondSourceId});
      }
    }
    return result;
  }

  std::vector<TwoDaughterRoutingTaskKey> MaskedTwoDaughterRoutingResult(
    const KFParticleGpuTwoDaughterRoutingPlan& routing,
    unsigned int eventIndex,
    unsigned int firstSourceId,
    unsigned int secondSourceId,
    int firstPdg,
    int secondPdg)
  {
    const unsigned int firstRole =
      static_cast<unsigned int>(KFParticleGpuClassifyTrackRole(firstPdg));
    const unsigned int secondRole =
      static_cast<unsigned int>(KFParticleGpuClassifyTrackRole(secondPdg));
    const KFParticleGpuChannelMask channels =
      routing.CompatibleChannels(firstRole, secondRole);
    std::vector<TwoDaughterRoutingTaskKey> result;
    for (unsigned int bit = channels.NextSetBit(0u);
         bit != KFParticleGpuChannelMask::InvalidBit;
         bit = channels.NextSetBit(bit + 1u)) {
      assert(bit < routing.Descriptors().size());
      result.push_back(
        {eventIndex, routing.Descriptors()[bit].channelId,
         firstSourceId, secondSourceId});
    }
    return result;
  }

  bool RejectsTwoDaughterRoutingPlan(const KFParticleGpuDecayPlan& plan)
  {
    try {
      KFParticleGpuTwoDaughterRoutingPlan routing;
      routing.Compile(plan);
    }
    catch (const std::invalid_argument&) {
      return true;
    }
    catch (const std::length_error&) {
      return true;
    }
    return false;
  }

  void TestTwoDaughterRoutingPlan(KFParticleGpuBufferManager& buffers)
  {
    bool uncompiledUploadRejected = false;
    try {
      KFParticleGpuTwoDaughterRoutingPlan uncompiled;
      buffers.UploadTwoDaughterRoutingPlan(uncompiled);
    }
    catch (const std::invalid_argument&) {
      uncompiledUploadRejected = true;
    }
    assert(uncompiledUploadRejected);

    KFParticleGpuDecayPlan plan;
    AddDefaultV0TwoDaughterChannels(plan);
    KFParticleGpuTwoDaughterRoutingPlan routing;
    routing.Compile(plan);
    assert(routing.SourceRevision() == plan.Revision());
    assert(routing.Descriptors().size() == 3u);
    assert(routing.CompatibilityEntries().size() == 3u);
    assert(routing.Groups().size() == 2u);
    assert(routing.EnabledChannels().Count() == 3u);

    const KFParticleGpuTwoDaughterRoutingDescriptor& k0 = routing.Descriptors()[0];
    assert(k0.channelId == KFGpuChannelK0ShortToPiPlusPiMinus);
    assert(k0.firstRole == KFGpuTrackRolePiPlus);
    assert(k0.secondRole == KFGpuTrackRolePiMinus);
    assert(k0.firstTrackSet == SecondaryPositiveFirst);
    assert(k0.secondTrackSet == SecondaryNegativeFirst);
    assert(k0.firstSpecies == Pion && k0.secondSpecies == Pion);
    assert(k0.firstDaughterPdg == 211 && k0.secondDaughterPdg == -211);
    assert(k0.firstCharge == 1 && k0.secondCharge == -1);
    assert(k0.transportMode == KFGpuTransportFullField);
    assert(AlmostEqual(k0.firstMass, plan.TwoDaughterChannel(0u).firstMass));
    assert(AlmostEqual(k0.motherMass, plan.TwoDaughterChannel(0u).motherMass));
    assert(k0.selection.topologyMode
           == plan.TwoDaughterChannel(0u).selection.topologyMode);
    assert(AlmostEqual(
      k0.selection.maxPrimaryVertexDistance,
      plan.TwoDaughterChannel(0u).selection.maxPrimaryVertexDistance));

    const KFParticleGpuTwoDaughterExecutionGroup& positiveNegative =
      routing.Groups()[0];
    assert(positiveNegative.firstTrackSet == SecondaryPositiveFirst);
    assert(positiveNegative.secondTrackSet == SecondaryNegativeFirst);
    assert(positiveNegative.AcceptsFirstRole(KFGpuTrackRolePiPlus));
    assert(positiveNegative.AcceptsFirstRole(KFGpuTrackRoleProton));
    assert(positiveNegative.AcceptsSecondRole(KFGpuTrackRolePiMinus));
    assert(positiveNegative.AcceptsFirstSpecies(Pion));
    assert(positiveNegative.AcceptsFirstSpecies(Proton));
    assert(positiveNegative.channels.Test(0u));
    assert(positiveNegative.channels.Test(1u));
    assert(routing.Groups()[1].firstTrackSet == SecondaryNegativeFirst);
    assert(routing.Groups()[1].secondTrackSet == SecondaryPositiveFirst);
    assert(routing.Groups()[1].channels.Test(2u));

    const KFParticleGpuRange firstRange(4u, 2u);
    const KFParticleGpuRange secondRange(9u, 3u);
    const std::vector<TwoDaughterRoutingTaskKey> explicitK0 =
      ExplicitTwoDaughterRoutingOracle(
        plan, 7u, 7u, 7u, firstRange, secondRange, 4u, 10u, 100u, 101u,
        SecondaryPositiveFirst, SecondaryNegativeFirst, Pion, Pion, 211, -211);
    const std::vector<TwoDaughterRoutingTaskKey> maskedK0 =
      MaskedTwoDaughterRoutingResult(routing, 7u, 100u, 101u, 211, -211);
    assert(explicitK0 == maskedK0);
    assert(explicitK0.size() == 2u);
    assert(explicitK0[0].channelId == KFGpuChannelK0ShortToPiPlusPiMinus);
    assert(explicitK0[1].channelId == KFGpuChannelLambdaToProtonPiMinus);
    assert(ExplicitTwoDaughterRoutingOracle(
             plan, 7u, 7u, 8u, firstRange, secondRange, 4u, 10u, 100u, 101u,
             SecondaryPositiveFirst, SecondaryNegativeFirst,
             Pion, Pion, 211, -211).empty());
    assert(ExplicitTwoDaughterRoutingOracle(
             plan, 7u, 7u, 7u, firstRange, secondRange, 4u, 10u, 100u, 101u,
             SecondaryPositiveFirst, SecondaryNegativeFirst,
             Kaon, Pion, 211, -211).empty());
    assert(ExplicitTwoDaughterRoutingOracle(
             plan, 7u, 7u, 7u, firstRange, secondRange, 4u, 10u, 100u, 100u,
             SecondaryPositiveFirst, SecondaryNegativeFirst,
             Pion, Pion, 211, -211).empty());
    assert(ExplicitTwoDaughterRoutingOracle(
             plan, 7u, 7u, 7u, KFParticleGpuRange(4u, 0u), secondRange,
             4u, 10u, 100u, 101u, SecondaryPositiveFirst,
             SecondaryNegativeFirst, Pion, Pion, 211, -211).empty());

    KFParticleGpuTwoDaughterChannel alternateK0 = plan.TwoDaughterChannel(0u);
    alternateK0.channelId = 1011u;
    plan.AddTwoDaughterChannel(alternateK0);
    routing.Compile(plan);
    const std::vector<TwoDaughterRoutingTaskKey> explicitMulti =
      ExplicitTwoDaughterRoutingOracle(
        plan, 7u, 7u, 7u, firstRange, secondRange, 4u, 10u, 100u, 101u,
        SecondaryPositiveFirst, SecondaryNegativeFirst, Pion, Pion, 211, -211);
    const std::vector<TwoDaughterRoutingTaskKey> maskedMulti =
      MaskedTwoDaughterRoutingResult(routing, 7u, 100u, 101u, 211, -211);
    assert(explicitMulti == maskedMulti);
    assert(maskedMulti.size() == 3u);

    KFParticleGpuDecayPlan widePlan;
    for (unsigned int index = 0u; index < 96u; ++index) {
      KFParticleGpuTwoDaughterChannel channel = alternateK0;
      channel.channelId = 2000u + index;
      widePlan.AddTwoDaughterChannel(channel);
    }
    KFParticleGpuTwoDaughterRoutingPlan wideRouting;
    wideRouting.Compile(widePlan);
    const KFParticleGpuChannelMask wideMask =
      wideRouting.CompatibleChannels(KFGpuTrackRolePiPlus, KFGpuTrackRolePiMinus);
    assert(wideMask.Count() == 96u);
    assert(wideMask.Test(31u));
    assert(wideMask.Test(32u));
    assert(wideMask.Test(33u));
    assert(wideMask.Test(63u));
    assert(wideMask.Test(64u));
    assert(wideMask.Test(95u));

    const int heavyNucleiPdgs[4] = {
      1000020060, 1000030060, 1000030070, 1000040070};
    const KFParticleGpuTrackSpecies heavyNucleiSpecies[4] = {
      Helium6, Lithium6, Lithium7, Beryllium7};
    KFParticleGpuDecayPlan heavyNucleiPlan;
    for (unsigned int index = 0u; index < 4u; ++index) {
      KFParticleGpuTwoDaughterChannel channel = alternateK0;
      channel.channelId = 4000u + index;
      channel.motherPdg = 5000 + static_cast<int>(index);
      channel.firstDaughterPdg = heavyNucleiPdgs[index];
      channel.firstSourcePdg = heavyNucleiPdgs[index];
      channel.firstSpecies = heavyNucleiSpecies[index];
      heavyNucleiPlan.AddTwoDaughterChannel(channel);
    }
    KFParticleGpuTwoDaughterRoutingPlan heavyNucleiRouting;
    heavyNucleiRouting.Compile(heavyNucleiPlan);
    assert(heavyNucleiRouting.Descriptors().size() == 4u);
    for (unsigned int index = 0u; index < 4u; ++index) {
      const KFParticleGpuChannelMask compatible =
        heavyNucleiRouting.CompatibleChannelsByPdg(heavyNucleiPdgs[index], -211);
      assert(compatible.Count() == 1u);
      assert(compatible.Test(index));
      assert(heavyNucleiRouting.Descriptors()[index].firstSpecies
             == heavyNucleiSpecies[index]);
    }

    KFParticleGpuDecayPlan invalid;
    KFParticleGpuTwoDaughterChannel invalidChannel = alternateK0;
    invalidChannel.channelId = 0u;
    invalid.AddTwoDaughterChannel(invalidChannel);
    assert(RejectsTwoDaughterRoutingPlan(invalid));

    invalid.Clear();
    invalidChannel = alternateK0;
    invalid.AddTwoDaughterChannel(invalidChannel);
    invalid.AddTwoDaughterChannel(invalidChannel);
    assert(RejectsTwoDaughterRoutingPlan(invalid));

    invalid.Clear();
    invalidChannel = alternateK0;
    invalidChannel.channelId = 3001u;
    invalidChannel.firstDaughterPdg = 11;
    invalid.AddTwoDaughterChannel(invalidChannel);
    assert(RejectsTwoDaughterRoutingPlan(invalid));

    invalid.Clear();
    invalidChannel = alternateK0;
    invalidChannel.channelId = 3002u;
    invalidChannel.firstSpecies = Kaon;
    invalid.AddTwoDaughterChannel(invalidChannel);
    assert(RejectsTwoDaughterRoutingPlan(invalid));

    invalid.Clear();
    invalidChannel = alternateK0;
    invalidChannel.channelId = 3003u;
    invalidChannel.firstCharge = -1;
    invalid.AddTwoDaughterChannel(invalidChannel);
    assert(RejectsTwoDaughterRoutingPlan(invalid));

    invalid.Clear();
    invalidChannel = alternateK0;
    invalidChannel.channelId = 3004u;
    invalidChannel.firstTrackSet = SecondaryNegativeFirst;
    invalid.AddTwoDaughterChannel(invalidChannel);
    assert(RejectsTwoDaughterRoutingPlan(invalid));

    invalid.Clear();
    invalidChannel = alternateK0;
    invalidChannel.channelId = 3005u;
    invalidChannel.motherMass = std::numeric_limits<float>::quiet_NaN();
    invalid.AddTwoDaughterChannel(invalidChannel);
    assert(RejectsTwoDaughterRoutingPlan(invalid));

    invalid.Clear();
    invalidChannel = alternateK0;
    invalidChannel.channelId = 3006u;
    invalidChannel.selection.topologyMode = 17;
    invalid.AddTwoDaughterChannel(invalidChannel);
    assert(RejectsTwoDaughterRoutingPlan(invalid));

    invalid.Clear();
    invalidChannel = alternateK0;
    invalidChannel.channelId = 3007u;
    invalidChannel.selection.maxPrimaryVertexDistance =
      std::numeric_limits<float>::infinity();
    invalid.AddTwoDaughterChannel(invalidChannel);
    assert(RejectsTwoDaughterRoutingPlan(invalid));

    invalid.Clear();
    for (unsigned int index = 0u;
         index <= KFParticleGpuChannelMask::BitCapacity;
         ++index) {
      invalidChannel = alternateK0;
      invalidChannel.channelId = 4000u + index;
      invalid.AddTwoDaughterChannel(invalidChannel);
    }
    assert(RejectsTwoDaughterRoutingPlan(invalid));

    plan.Clear();
    AddDefaultV0TwoDaughterChannels(plan);
    routing.Compile(plan);
    assert(buffers.UploadTwoDaughterRoutingPlan(routing));
    assert(!buffers.UploadTwoDaughterRoutingPlan(routing));
    assert(buffers.TwoDaughterRoutingRevision() == plan.Revision());

    buffers.EnsureTwoDaughterRoutedTaskCapacity(7u);
    buffers.ResetTwoDaughterRoutingStatus();
    const KFParticleGpuTwoDaughterRoutingStatus status =
      buffers.DownloadTwoDaughterRoutingStatus();
    assert(status.visitedPairs == 0u);
    assert(status.activeChannelBits == 0u);
    assert(status.acceptedTasks == 0u);
    assert(status.storedTasks == 0u);
    assert(status.blockReservations == 0u);
    assert(status.overflowFlags == 0u);
    assert(buffers.HostTwoDaughterRoutedTasks() != nullptr);

    const KFParticleGpuTwoDaughterRoutingView hostRouting =
      buffers.HostTwoDaughterRouting();
    assert(hostRouting.DescriptorCount() == 3u);
    assert(hostRouting.CompatibilityCount() == 3u);
    assert(hostRouting.GroupCount() == 2u);
    assert(hostRouting.EnabledChannelsData()[0].Count() == 3u);
    assert(hostRouting.ChannelAcceptedCounters()[0] == 0u);
    assert(hostRouting.CompatibleChannels(
             KFGpuTrackRoleProton, KFGpuTrackRolePiMinus).Test(1u));

    KFParticleGpuChannelMask lambdaOnly;
    lambdaOnly.Set(1u);
    buffers.SetTwoDaughterRoutingEnabledChannels(lambdaOnly);
    assert(buffers.HostTwoDaughterRouting().CompatibleChannels(
             KFGpuTrackRolePiPlus, KFGpuTrackRolePiMinus).Test(1u));
    assert(buffers.HostTwoDaughterRouting().CompatibleChannels(
             KFGpuTrackRoleProton, KFGpuTrackRolePiMinus).Test(1u));
    buffers.SetTwoDaughterRoutingEnabledChannels(routing.EnabledChannels());

    KFParticleGpuCandidateDescriptorIndexView descriptorIndices =
      buffers.HostCandidateDescriptorIndices();
    assert(descriptorIndices.Capacity() == buffers.Capacities().candidates);
    descriptorIndices.Index(0u) = 2u;
    buffers.UploadCandidates();

    const KFParticleGpuDeviceStorage& storage = buffers.Storage();
    assert(storage.fTwoDaughterRoutedTasks.get() != nullptr);
    assert(storage.fTwoDaughterRoutingDescriptors.get() != nullptr);
    assert(storage.fTwoDaughterRoutingCompatibility.get() != nullptr);
    assert(storage.fTwoDaughterRoutingGroups.get() != nullptr);
    assert(storage.fTwoDaughterRoutingEnabledChannels.get() != nullptr);
    assert(storage.fTwoDaughterRoutingChannelAcceptedCounters.get() != nullptr);
    assert(storage.fCandidateRoutingDescriptorIndices.get() != nullptr);

    Pass("two-daughter-routing-plan",
         "two-track routing compiles exact charged-PDG masks, execution groups, revisioned tables, and descriptor tags");
  }

  bool RejectsDecayGraphManifest(
    const KFParticleGpuDecayGraphManifest& manifest,
    const KFParticleGpuDecayGraphCompileLimits& limits =
      KFParticleGpuDecayGraphCompileLimits())
  {
    try {
      KFParticleGpuDecayGraphPlan graph;
      graph.Compile(manifest, limits);
    }
    catch (const std::invalid_argument&) {
      return true;
    }
    catch (const std::length_error&) {
      return true;
    }
    return false;
  }

  bool RejectsCpuChannelCatalogue(
    const std::vector<KFParticleGpuCpuChannelContract>& catalogue)
  {
    try {
      ValidateKFParticleGpuCpuChannelCatalogue(catalogue);
    }
    catch (const std::invalid_argument&) {
      return true;
    }
    return false;
  }

  void TestCpuFinderChannelCatalogue()
  {
    const std::vector<KFParticleGpuCpuChannelContract>& catalogue =
      KFParticleGpuCpuChannelCatalogue();
    assert(catalogue.size() == 260u);
    assert(catalogue.size() <= KFParticleGpuChannelMask::BitCapacity);
    assert(KFParticleGpuCpuChannelCatalogueRevision() == 7326650681912440158ull);
    ValidateKFParticleGpuCpuChannelCatalogue(catalogue);

    std::array<unsigned int, KFGpuCpuFamilyCount> total = {};
    std::array<unsigned int, KFGpuCpuFamilyCount> active = {};
    std::array<unsigned int, KFGpuCpuFamilyCount> disabled = {};
    for (const KFParticleGpuCpuChannelContract& entry : catalogue) {
      ++total[entry.family];
      if (entry.activation == KFGpuCpuChannelConfigurationDisabled) {
        ++disabled[entry.family];
      }
      else {
        ++active[entry.family];
      }
    }
    for (unsigned int family = 0u; family < KFGpuCpuFamilyCount; ++family) {
      assert(total[family] != 0u);
    }
    assert(active[KFGpuCpuFamilyTwoDaughter] > 3u);
    assert(active[KFGpuCpuFamilyTrackComposite] > 4u);
    assert(active[KFGpuCpuFamilyNeutralMissingMass] != 0u);
    assert(active[KFGpuCpuFamilyNeutralMissingMass] == 22u);
    assert(disabled[KFGpuCpuFamilySameSignPrimaryResonance] != 0u);
    assert(active[KFGpuCpuFamilySameSignPrimaryResonance] == 0u);
    for (const KFParticleGpuCpuChannelContract& entry : catalogue) {
      if (entry.family != KFGpuCpuFamilyNeutralMissingMass) continue;
      assert(entry.activation == KFGpuCpuChannelActive);
      assert(entry.firstSourceKind == KFGpuGraphSourceTrackRange);
      assert(entry.secondSourceKind == KFGpuGraphSourceTrackRange);
      assert((entry.requiredInputs & KFGpuCpuInputNeutralCandidates) == 0u);
      assert(entry.firstDaughterPdg != 0);
      assert(entry.secondDaughterPdg != 0);
      assert(entry.supportStatus == KFGpuGraphSupported);
      assert(entry.unsupportedReason == KFGpuGraphUnsupportedNone);
    }

    KFParticleGpuDecayPlan plan;
    AddDefaultV0TwoDaughterChannels(plan);
    AddDefaultV0TrackCascadeChannels(plan);
    const std::vector<KFParticleGpuCpuChannelCoverage> coverage =
      MakeCpuFinderChannelCoverage(plan);
    assert(coverage.size() == catalogue.size());
    unsigned int mappedDefaultChannels = 0u;
    unsigned int explicitRequirements = 0u;
    for (const KFParticleGpuCpuChannelCoverage& item : coverage) {
      if (item.supportStatus == KFGpuGraphSupported
          && item.unsupportedReason == KFGpuGraphUnsupportedNone) {
        ++mappedDefaultChannels;
      }
      else if (item.activation != KFGpuCpuChannelConfigurationDisabled) {
        assert(item.unsupportedReason != KFGpuGraphUnsupportedNone);
        ++explicitRequirements;
      }
    }
    assert(mappedDefaultChannels == 7u);
    assert(explicitRequirements != 0u);

    KFParticleGpuDecayPlan chargedPlan;
    AddCpuFinderTwoDaughterChannels(chargedPlan);
    assert(chargedPlan.NumberOfTwoDaughterChannels() == 50u);
    for (std::size_t index = 0u;
         index < chargedPlan.NumberOfTwoDaughterChannels(); ++index) {
      assert(chargedPlan.TwoDaughterChannel(index).transportMode
             == KFGpuTransportFullField);
    }
    KFParticleGpuTwoDaughterRoutingPlan chargedRouting;
    chargedRouting.Compile(chargedPlan);
    assert(chargedRouting.Descriptors().size() == 50u);
    assert(chargedRouting.EnabledChannels().Count() == 50u);
    KFParticleGpuDecayGraphPlan chargedGraph;
    chargedGraph.Compile(MakeDefaultCpuFinderDecayGraphManifest(chargedPlan));
    assert(chargedGraph.Nodes().size() == 50u);
    assert(chargedGraph.Groups().size() == 1u);
    assert(chargedGraph.Groups()[0].generation == 1u);
    assert(chargedGraph.Groups()[0].topology == KFGpuGraphTopologyTrackTrack);
    assert(chargedGraph.Groups()[0].nodeCount == 50u);
    assert(chargedGraph.FamilyCoverage()[KFGpuCpuFamilyTwoDaughter].supportStatus
           == KFGpuGraphSupported);
    unsigned int supportedTwoDaughter = 0u;
    for (const KFParticleGpuCpuChannelCoverage& item :
         MakeCpuFinderChannelCoverage(chargedPlan)) {
      if (item.family == KFGpuCpuFamilyTwoDaughter
          && item.activation == KFGpuCpuChannelActive) {
        assert(item.supportStatus == KFGpuGraphSupported);
        assert(item.unsupportedReason == KFGpuGraphUnsupportedNone);
        ++supportedTwoDaughter;
      }
    }
    assert(supportedTwoDaughter == 50u);
    const KFParticleGpuTwoDaughterChannel& charmPiPi =
      chargedPlan.TwoDaughterChannel(49u);
    assert(charmPiPi.motherPdg == 420);
    assert(charmPiPi.minFirstPixelHits == 3);
    assert(charmPiPi.minFirstChiToPrimaryVertex == 8.f);
    assert(charmPiPi.minFirstPt == 0.2f);
    bool foundPrimaryHypernucleus = false;
    for (std::size_t index = 0u;
         index < chargedPlan.NumberOfTwoDaughterChannels(); ++index) {
      const KFParticleGpuTwoDaughterChannel& channel =
        chargedPlan.TwoDaughterChannel(index);
      if (channel.motherPdg == 100015) {
        assert(channel.firstDaughterPdg == 1000040070);
        assert(channel.firstSpecies == Beryllium7);
        assert(channel.primaryVertexIndex == KFGpuPrimaryVertexFromDaughters);
        foundPrimaryHypernucleus = true;
      }
    }
    assert(foundPrimaryHypernucleus);

    KFParticleGpuDecayPlan narrowDiagnosticPlan;
    AddCpuFinderTwoDaughterChannels(narrowDiagnosticPlan);
    AddCpuFinderCompositeTrackChannels(narrowDiagnosticPlan);
    assert(narrowDiagnosticPlan.NumberOfV0TrackCascadeChannels() == 120u);
    KFParticleGpuDecayGraphPlan narrowDiagnosticGraph;
    narrowDiagnosticGraph.Compile(
      MakeDefaultCpuFinderDecayGraphManifest(narrowDiagnosticPlan));
    assert(narrowDiagnosticGraph.Nodes().size() == 170u);
    assert(narrowDiagnosticGraph.FamilyCoverage()[KFGpuCpuFamilyTrackComposite].supportStatus
           == KFGpuGraphPartiallySupported);
    assert(narrowDiagnosticGraph.FamilyCoverage()[KFGpuCpuFamilyPrimaryProjection].supportStatus
           == KFGpuGraphUnsupported);

    AddCpuFinderPrimaryProjectionChannels(narrowDiagnosticPlan);
    assert(narrowDiagnosticPlan.NumberOfGraphOperationChannels() == 8u);
    KFParticleGpuDecayGraphPlan narrowProjectedGraph;
    narrowProjectedGraph.Compile(
      MakeDefaultCpuFinderDecayGraphManifest(narrowDiagnosticPlan));
    assert(narrowProjectedGraph.Nodes().size() == 178u);
    assert(narrowProjectedGraph.FamilyCoverage()[KFGpuCpuFamilyPrimaryProjection].supportStatus
           == KFGpuGraphPartiallySupported);

    AddCpuFinderCompositeCompositeChannels(chargedPlan);
    assert(chargedPlan.NumberOfGraphOperationChannels() == 24u);
    AddCpuFinderCompositeTrackChannels(chargedPlan);
    assert(chargedPlan.NumberOfV0TrackCascadeChannels() == 124u);
    KFParticleGpuV0TrackRoutingPlan compositeRouting;
    compositeRouting.Compile(chargedPlan);
    assert(compositeRouting.Descriptors().size() == 124u);
    assert(compositeRouting.EnabledChannels().Count() == 124u);
    bool foundCharmContinuation = false;
    bool foundFourthGeneration = false;
    bool foundSecondaryDistanceCut = false;
    bool foundLongLivedWithoutDistanceCut = false;
    for (const KFParticleGpuV0TrackRoutingDescriptor& descriptor :
         compositeRouting.Descriptors()) {
      assert(descriptor.transportMode == KFGpuTransportFullField);
      if (descriptor.motherPdg == 411) {
        assert(descriptor.v0Pdg == 421);
        assert(descriptor.bachelorPdg == 211);
        assert(descriptor.minBachelorPixelHits == 3);
        assert(descriptor.minBachelorPt == 0.2f);
        foundCharmContinuation = true;
      }
      if (descriptor.motherPdg == 3029) {
        assert(descriptor.generation == 3u);
        assert(descriptor.v0Pdg == 3028);
      }
      if (descriptor.family == KFGpuCpuFamilyTrackComposite
          && descriptor.primaryVertexIndex == -1) {
        assert(descriptor.maxV0TrackDistance == 1.f);
        foundSecondaryDistanceCut = true;
      }
      if (descriptor.family == KFGpuCpuFamilyLongLivedComposite) {
        assert(descriptor.maxV0TrackDistance < 0.f);
        foundLongLivedWithoutDistanceCut = true;
      }
      foundFourthGeneration |= descriptor.generation == 4u;
    }
    assert(foundCharmContinuation);
    assert(foundFourthGeneration);
    assert(foundSecondaryDistanceCut);
    assert(foundLongLivedWithoutDistanceCut);
    unsigned int supportedCompositeTrack = 0u;
    unsigned int deferredPi0CompositeTrack = 0u;
    for (const KFParticleGpuCpuChannelCoverage& item :
         MakeCpuFinderChannelCoverage(chargedPlan)) {
      if ((item.family == KFGpuCpuFamilyTrackComposite
          || item.family == KFGpuCpuFamilyLongLivedComposite)
          && item.activation == KFGpuCpuChannelActive) {
        if (item.supportStatus == KFGpuGraphSupported) {
          assert(item.unsupportedReason == KFGpuGraphUnsupportedNone);
          ++supportedCompositeTrack;
        }
        else {
          assert(item.family == KFGpuCpuFamilyTrackComposite);
          assert(item.unsupportedReason == KFGpuGraphUnsupportedValidationPending);
          ++deferredPi0CompositeTrack;
        }
      }
    }
    assert(supportedCompositeTrack == 124u);
    assert(deferredPi0CompositeTrack == 0u);

    KFParticleGpuDecayGraphPlan compositeGraph;
    compositeGraph.Compile(MakeDefaultCpuFinderDecayGraphManifest(chargedPlan));
    assert(compositeGraph.Nodes().size() == 198u);
    assert(compositeGraph.FamilyCoverage()[KFGpuCpuFamilyTrackComposite].supportStatus
           == KFGpuGraphSupported);
    assert(compositeGraph.FamilyCoverage()[KFGpuCpuFamilyLongLivedComposite].supportStatus
           == KFGpuGraphSupported);
    assert(compositeGraph.FamilyCoverage()[KFGpuCpuFamilyCompositeComposite].supportStatus
           == KFGpuGraphSupported);
    unsigned int supportedCompositeComposite = 0u;
    for (const KFParticleGpuCpuChannelCoverage& item :
         MakeCpuFinderChannelCoverage(chargedPlan)) {
      if (item.family == KFGpuCpuFamilyCompositeComposite
          && item.activation == KFGpuCpuChannelActive) {
        assert(item.supportStatus == KFGpuGraphSupported);
        assert(item.unsupportedReason == KFGpuGraphUnsupportedNone);
        ++supportedCompositeComposite;
      }
    }
    assert(supportedCompositeComposite == 24u);
    const KFParticleGpuGraphOperationChannel& pi0 =
      chargedPlan.GraphOperationChannel(18u);
    assert(pi0.node.channelId == 5019u);
    assert(pi0.node.firstSource.sourceId == pi0.node.secondSource.sourceId);
    assert((pi0.descriptor.flags & KFGpuGraphRequireOrderedCandidatePair) != 0u);

    AddCpuFinderPrimaryProjectionChannels(chargedPlan);
    assert(chargedPlan.NumberOfGraphOperationChannels() == 36u);
    KFParticleGpuDecayGraphPlan projectedGraph;
    projectedGraph.Compile(MakeDefaultCpuFinderDecayGraphManifest(chargedPlan));
    assert(projectedGraph.Nodes().size() == 210u);
    assert(projectedGraph.FamilyCoverage()[KFGpuCpuFamilyPrimaryProjection].supportStatus
           == KFGpuGraphSupported);
    unsigned int supportedProjection = 0u;
    for (const KFParticleGpuCpuChannelCoverage& item :
         MakeCpuFinderChannelCoverage(chargedPlan)) {
      if (item.family == KFGpuCpuFamilyPrimaryProjection
          && item.activation == KFGpuCpuChannelActive) {
        assert(item.supportStatus == KFGpuGraphSupported);
        ++supportedProjection;
      }
    }
    assert(supportedProjection == 12u);
    const KFParticleGpuGraphOperationChannel& k0Projection =
      chargedPlan.GraphOperationChannel(24u);
    assert(k0Projection.node.channelId == 8001u);
    assert(k0Projection.node.firstSource.sourceId == 1u);
    assert(k0Projection.descriptor.operationMask == KFGpuGraphExtrapolate);
    assert(k0Projection.descriptor.primaryVertexIndex
           == KFGpuGraphPrimaryVertexFromCandidate);

    AddCpuFinderNeutralMissingMassChannels(chargedPlan);
    assert(chargedPlan.NumberOfGraphOperationChannels() == 58u);
    KFParticleGpuDecayGraphPlan neutralGraph;
    neutralGraph.Compile(MakeDefaultCpuFinderDecayGraphManifest(chargedPlan));
    assert(neutralGraph.Nodes().size() == 232u);
    assert(neutralGraph.FamilyCoverage()[KFGpuCpuFamilyNeutralMissingMass].supportStatus
           == KFGpuGraphSupported);
    unsigned int neutralChannels = 0u;
    for (std::size_t index = 36u;
         index < chargedPlan.NumberOfGraphOperationChannels(); ++index) {
      const KFParticleGpuGraphOperationChannel& channel =
        chargedPlan.GraphOperationChannel(index);
      assert(channel.node.firstSource.kind == KFGpuGraphSourceTrackRange);
      assert(channel.node.secondSource.kind == KFGpuGraphSourceTrackRange);
      assert(channel.descriptor.missingMassMode
             == KFGpuGraphMissingMassFilteredReconstruction);
      assert(channel.descriptor.firstMass >= 0.f);
      assert(channel.descriptor.secondMass >= 0.f);
      assert(channel.descriptor.neutralMass >= 0.f);
      assert(KFParticleGpuGraphOperations::ValidateDescriptor(channel.descriptor));
      ++neutralChannels;
    }
    assert(neutralChannels == 22u);

    KFParticleGpuDecayPlan ordinaryDiagnosticPlan;
    AddCpuFinderTwoDaughterChannels(ordinaryDiagnosticPlan);
    AddCpuFinderCompositeTrackChannels(ordinaryDiagnosticPlan);
    AddCpuFinderNeutralMissingMassChannels(ordinaryDiagnosticPlan);
    AddCpuFinderKaonMatchingChannels(ordinaryDiagnosticPlan);
    AddCpuFinderFinalSelectionChannels(ordinaryDiagnosticPlan);
    assert(ordinaryDiagnosticPlan.NumberOfGraphOperationChannels() == 42u);
    KFParticleGpuDecayGraphPlan ordinaryDiagnosticGraph;
    ordinaryDiagnosticGraph.Compile(
      MakeDefaultCpuFinderDecayGraphManifest(ordinaryDiagnosticPlan));
    assert(ordinaryDiagnosticGraph.Nodes().size() == 212u);
    assert(ordinaryDiagnosticGraph.FamilyCoverage()[KFGpuCpuFamilyKaonMatching]
             .supportStatus == KFGpuGraphSupported);
    unsigned int kaonMatchingChannels = 0u;
    for (std::size_t index = 0u;
         index < ordinaryDiagnosticPlan.NumberOfGraphOperationChannels();
         ++index) {
      const KFParticleGpuGraphOperationChannel& channel =
        ordinaryDiagnosticPlan.GraphOperationChannel(index);
      if (channel.node.channelId != 7001u
          && channel.node.channelId != 7002u) {
        continue;
      }
      ++kaonMatchingChannels;
      assert(channel.node.firstSource.kind
             == KFGpuGraphSourceCandidateGeneration);
      assert(channel.node.secondSource.kind == KFGpuGraphSourceTrackRange);
      assert(channel.node.generation == 3u);
      assert(channel.descriptor.topology
             == KFGpuGraphTopologyCompositeTrack);
      assert((channel.descriptor.operationMask & KFGpuGraphMatch) != 0u);
      assert((channel.descriptor.flags & KFGpuGraphStoreFirstOnMatch) != 0u);
      assert((channel.descriptor.flags
              & KFGpuGraphMatchTrackPrimaryVertex) != 0u);
      assert(channel.descriptor.maxMatchDistance == 20.f);
      assert(channel.descriptor.maxMatchMomentumSigma == 5.f);
      assert(channel.descriptor.maxMatchTopologyChi2PerNdf == 3.f);
      assert(channel.descriptor.matchMassWindowSigma == 3.7e-3f);
      assert(channel.descriptor.matchMassWindowCut == 3.f);
      assert(channel.descriptor.massConstraint == 0.497614f);
      assert(channel.descriptor.massConstraintSigma == 0.f);
      assert(channel.descriptor.maxGeometricChi2PerNdf == 3.f);
    }
    assert(kaonMatchingChannels == 2u);
    assert(ordinaryDiagnosticGraph.FamilyCoverage()[KFGpuCpuFamilyFinalSelection]
             .supportStatus == KFGpuGraphSupported);
    unsigned int finalSelectionChannels = 0u;
    for (std::size_t index = 0u;
         index < ordinaryDiagnosticPlan.NumberOfGraphOperationChannels();
         ++index) {
      const KFParticleGpuGraphOperationChannel& channel =
        ordinaryDiagnosticPlan.GraphOperationChannel(index);
      if (channel.node.channelId < 9001u
          || channel.node.channelId > 9018u) {
        continue;
      }
      ++finalSelectionChannels;
      assert(channel.node.topology == KFGpuGraphTopologyUnaryComposite);
      assert(channel.node.firstSource.kind
             == KFGpuGraphSourceCandidateGeneration);
      assert((channel.descriptor.operationMask & KFGpuGraphSelect) != 0u);
      assert((channel.descriptor.operationMask
              & KFGpuGraphMassConstraint) != 0u);
      assert((channel.descriptor.flags
              & KFGpuGraphSelectEventPrimaryVertices) != 0u);
      assert(channel.descriptor.maxTopologyChi2PerNdf == 3.f);
      assert(channel.descriptor.maxSelectionVertexDistance == 200.f);
      assert(channel.descriptor.minSelectionDecayLengthOverError == 10.f);
      assert(channel.descriptor.matchMassWindowSigma == 0.0145f);
      assert(channel.descriptor.matchMassWindowCut == 3.f);
    }
    assert(finalSelectionChannels == 18u);

    KFParticleGpuEventDesc capacityEvent;
    capacityEvent.TrackSet(PrimaryPositiveLast).tracks =
      KFParticleGpuRange(0u, 2u);
    capacityEvent.TrackSet(PrimaryNegativeLast).tracks =
      KFParticleGpuRange(2u, 3u);
    capacityEvent.TrackSet(SecondaryPositiveFirst).Species(Muon) =
      KFParticleGpuRange(5u, 5u);
    capacityEvent.TrackSet(SecondaryNegativeFirst).Species(Muon) =
      KFParticleGpuRange(10u, 7u);
    // Two particle/antiparticle muon pairs are active (pi and K mothers):
    // 2 * (2*5 + 3*7) = 62. Every other raw species range is empty.
    assert(KFParticleGpuCpuFinderRawTaskCapacity(
             capacityEvent, ordinaryDiagnosticPlan) == 62u);

    KFParticleGpuDecayPlan cascadeCapacityPlan;
    AddDefaultV0TwoDaughterChannels(cascadeCapacityPlan);
    AddDefaultV0TrackCascadeChannels(cascadeCapacityPlan);
    KFParticleGpuEventDesc cascadeCapacityEvent;
    cascadeCapacityEvent.TrackSet(SecondaryPositiveFirst).tracks = {0u, 2u};
    cascadeCapacityEvent.TrackSet(SecondaryPositiveFirst).Species(Pion) = {0u, 1u};
    cascadeCapacityEvent.TrackSet(SecondaryPositiveFirst).Species(Proton) = {1u, 1u};
    cascadeCapacityEvent.TrackSet(SecondaryNegativeFirst).tracks = {2u, 4u};
    cascadeCapacityEvent.TrackSet(SecondaryNegativeFirst).Species(Pion) = {2u, 3u};
    cascadeCapacityEvent.TrackSet(SecondaryNegativeFirst).Species(Kaon) = {5u, 1u};
    const KFParticleGpuEventCapacityPlan cascadeCapacity =
      KFParticleGpuCpuFinderEventCapacityPlan(cascadeCapacityEvent,
                                              cascadeCapacityPlan);
    assert(cascadeCapacity.twoDaughterCandidates == 9u);
    assert(cascadeCapacity.cascadeCandidates == 24u);
    assert(cascadeCapacity.graphCandidates == 0u);
    assert(cascadeCapacity.taskCapacity == 24u);
    assert(cascadeCapacity.candidates == 33u);
    assert(cascadeCapacity.daughterIds == 90u);
    assert(cascadeCapacity.selectedCandidates == 33u);

    std::vector<KFParticleGpuCpuChannelContract> invalid = catalogue;
    invalid.push_back(catalogue.front());
    assert(RejectsCpuChannelCatalogue(invalid));
    invalid = catalogue;
    invalid.front().conjugateChannelId = 999999u;
    assert(RejectsCpuChannelCatalogue(invalid));
    invalid = catalogue;
    invalid.front().selectionProfile = 0u;
    assert(RejectsCpuChannelCatalogue(invalid));
    invalid = catalogue;
    invalid.back().parentChannelId = 999999u;
    assert(RejectsCpuChannelCatalogue(invalid));
    invalid = catalogue;
    invalid.back().secondParentChannelId = 999999u;
    assert(RejectsCpuChannelCatalogue(invalid));

    Pass("cpu-finder-channel-catalogue",
         "all CPU families have stable IDs, conjugates, dependencies, inputs, cuts, and explicit GPU requirements");
  }

  void TestCompleteCpuFinderManifest(KFParticleGpuBufferManager& buffers)
  {
    KFParticleGpuDecayPlan plan;
    AddCompleteCpuFinderChannels(plan);
    ValidateCompleteCpuFinderPlan(plan);

    const KFParticleGpuCpuFinderCoverageSummary summary =
      MakeCompleteCpuFinderCoverageSummary(plan);
    assert(summary.Complete());
    assert(summary.catalogueChannels == 260u);
    assert(summary.requiredChannels == 252u);
    assert(summary.implementedChannels == 252u);
    assert(summary.disabledChannels == 8u);
    assert(summary.unsupportedChannels == 0u);
    assert(plan.NumberOfTwoDaughterChannels() == 50u);
    assert(plan.NumberOfV0TrackCascadeChannels() == 124u);
    assert(plan.NumberOfGraphOperationChannels() == 78u);

    const KFParticleGpuDecayGraphManifest manifest =
      MakeDefaultCpuFinderDecayGraphManifest(plan);
    assert(manifest.Nodes().size() == summary.requiredChannels);
    assert(manifest.FamilyCoverage().size() == KFGpuCpuFamilyCount);
    for (unsigned int descriptorIndex = 0u;
         descriptorIndex < plan.NumberOfGraphOperationChannels();
         ++descriptorIndex) {
      const unsigned int channelId =
        plan.GraphOperationChannel(descriptorIndex).node.channelId;
      const auto found = std::find_if(
        manifest.Nodes().begin(), manifest.Nodes().end(),
        [channelId](const KFParticleGpuGraphNode& node) {
          return node.channelId == channelId;
        });
      assert(found != manifest.Nodes().end());
      assert(found->operationDescriptorIndex == descriptorIndex);
    }
    for (const KFParticleGpuGraphFamilyCoverage& family :
         manifest.FamilyCoverage()) {
      assert(family.family < KFGpuCpuFamilyCount);
      assert(family.supportStatus == KFGpuGraphSupported);
      assert(family.unsupportedReason == KFGpuGraphUnsupportedNone);
      assert(family.implementedChannelCount
             == summary.implementedByFamily[family.family]);
      assert(summary.requiredByFamily[family.family]
               == summary.implementedByFamily[family.family]
             || summary.disabledByFamily[family.family] != 0u);
    }

    const std::vector<KFParticleGpuCpuChannelContract>& catalogue =
      KFParticleGpuCpuChannelCatalogue();
    for (const KFParticleGpuCpuChannelContract& channel : catalogue) {
      unsigned int matches = 0u;
      const KFParticleGpuGraphNode* matchedNode = nullptr;
      for (const KFParticleGpuGraphNode& node : manifest.Nodes()) {
        if (node.channelId == channel.channelId) {
          ++matches;
          matchedNode = &node;
        }
      }
      if (channel.activation == KFGpuCpuChannelConfigurationDisabled) {
        assert(matches == 0u);
        continue;
      }
      assert(channel.supportStatus == KFGpuGraphSupported);
      assert(channel.unsupportedReason == KFGpuGraphUnsupportedNone);
      assert(matches == 1u);
      assert(matchedNode != nullptr);
      assert(matchedNode->motherPdg == channel.motherPdg);
      assert(matchedNode->topology == channel.topology);
      assert(matchedNode->generation == channel.generation);
      if (matchedNode->operationMask != channel.operationMask) {
        std::cerr << "COMPLETE_MANIFEST_OPERATION_MISMATCH channel="
                  << channel.channelId << " family=" << channel.family
                  << " catalogue=" << channel.operationMask
                  << " graph=" << matchedNode->operationMask << '\n';
      }
      assert(matchedNode->operationMask == channel.operationMask);
      if (matchedNode->outputClass != channel.outputClass) {
        std::cerr << "COMPLETE_MANIFEST_OUTPUT_MISMATCH channel="
                  << channel.channelId << " family=" << channel.family
                  << " catalogue=" << channel.outputClass
                  << " graph=" << matchedNode->outputClass << '\n';
      }
      assert(matchedNode->outputClass == channel.outputClass);
      assert(matchedNode->supportStatus == KFGpuGraphSupported);
      assert(matchedNode->unsupportedReason == KFGpuGraphUnsupportedNone);
    }

    KFParticleGpuDecayGraphPlan graph;
    graph.Compile(manifest);
    assert(graph.Nodes().size() == 252u);
    assert(graph.SourceRevision() != 0u);
    bool payloadCovered[KFGpuGraphPayloadCount] = {};
    for (const KFParticleGpuGraphNode& node : graph.Nodes()) {
      assert(node.payloadKind > KFGpuGraphPayloadNone);
      assert(node.payloadKind < KFGpuGraphPayloadCount);
      payloadCovered[node.payloadKind] = true;

      KFParticleGpuParitySnapshot cpu;
      cpu.key.eventId = 21u;
      cpu.key.channelId = node.channelId;
      cpu.key.pdg = node.motherPdg;
      cpu.key.lineageSize = 2u;
      cpu.key.lineage[0] = 102;
      cpu.key.lineage[1] = 101;
      KFParticleGpuParity::Canonicalize(cpu.key);
      cpu.parameters[3] = 0.4f;
      cpu.covariance[9] = 0.01f;
      cpu.chi2 = 1.f;
      cpu.ndf = 1;
      cpu.mass = 1.f;
      cpu.massError = 0.01f;
      cpu.massValid = 1u;
      cpu.topology = node.topology;
      cpu.outputClass = node.outputClass;
      cpu.operationStatus = KFGpuCandidateOperationAccepted;
      cpu.available = KFGpuParityKeyAvailable
        | KFGpuParityParametersAvailable
        | KFGpuParityCovarianceAvailable
        | KFGpuParityFitQualityAvailable
        | KFGpuParityMassAvailable
        | KFGpuParityOperationAvailable;
      const KFParticleGpuParitySnapshot gpu = cpu;
      assert(KFParticleGpuParity::Compare(
               cpu, gpu, KFParticleGpuParityTolerance()).Equivalent());
    }
    for (unsigned int payload = KFGpuGraphPayloadTwoDaughter;
         payload < KFGpuGraphPayloadCount; ++payload) {
      assert(payloadCovered[payload]);
    }
    for (const KFParticleGpuGraphFamilyCoverage& family :
         graph.FamilyCoverage()) {
      assert(family.supportStatus == KFGpuGraphSupported);
      assert(family.unsupportedReason == KFGpuGraphUnsupportedNone);
    }

    KFParticleGpuTwoDaughterRoutingPlan twoDaughterRouting;
    twoDaughterRouting.Compile(plan);
    assert(twoDaughterRouting.Descriptors().size() == 50u);
    assert(buffers.UploadTwoDaughterRoutingPlan(twoDaughterRouting, true));
    const KFParticleGpuTwoDaughterRoutingView hostTwoDaughter =
      buffers.HostTwoDaughterRouting();
    assert(hostTwoDaughter.DescriptorCount() == 50u);
    for (unsigned int index = 0u;
         index < hostTwoDaughter.DescriptorCount(); ++index) {
      assert(hostTwoDaughter.Descriptors()[index].channelId
             == plan.TwoDaughterChannel(index).channelId);
      assert(hostTwoDaughter.ChannelVisitedCounters()[index] == 0u);
      assert(hostTwoDaughter.ChannelAcceptedCounters()[index] == 0u);
      assert(hostTwoDaughter.ChannelStoredCounters()[index] == 0u);
      assert(hostTwoDaughter.ChannelConstructedCounters()[index] == 0u);
    }

    KFParticleGpuV0TrackRoutingPlan compositeRouting;
    compositeRouting.Compile(plan);
    assert(compositeRouting.Descriptors().size() == 124u);
    assert(buffers.UploadV0TrackRoutingPlan(compositeRouting, true));
    const KFParticleGpuV0TrackRoutingView hostComposite =
      buffers.HostV0TrackRouting();
    assert(hostComposite.DescriptorCount() == 124u);
    for (unsigned int index = 0u;
         index < hostComposite.DescriptorCount(); ++index) {
      assert(hostComposite.Descriptors()[index].channelId
             == plan.V0TrackCascadeChannel(index).channelId);
      assert(hostComposite.ChannelVisitedCounters()[index] == 0u);
      assert(hostComposite.ChannelAcceptedCounters()[index] == 0u);
      assert(hostComposite.ChannelStoredCounters()[index] == 0u);
      assert(hostComposite.ChannelConstructedCounters()[index] == 0u);
    }

    assert(buffers.UploadGraphOperationPlan(plan, true));
    assert(buffers.GraphOperationDescriptorSize() == 78u);
    const KFParticleGpuGraphOperationStorageView operationStorage =
      buffers.DeviceGraphOperationStorage();
    assert(operationStorage.ChannelVisited() != nullptr);
    assert(operationStorage.ChannelAccepted() != nullptr);
    assert(operationStorage.ChannelStored() != nullptr);
    assert(operationStorage.ChannelConstructed() != nullptr);
    assert(operationStorage.ChannelRejected() != nullptr);
    const KFParticleGpuGraphOperationDescriptor* operationDescriptors =
      buffers.HostGraphOperationDescriptors();
    assert(operationDescriptors != nullptr);
    for (unsigned int index = 0u;
         index < buffers.GraphOperationDescriptorSize(); ++index) {
      assert(operationDescriptors[index].channelId
             == plan.GraphOperationChannel(index).node.channelId);
    }

    assert(buffers.UploadDecayGraphPlan(graph, true));
    const KFParticleGpuDecayGraphView hostGraph = buffers.HostDecayGraph();
    assert(hostGraph.NodeCount() == 252u);
    assert(hostGraph.FamilyCount() == KFGpuCpuFamilyCount);
    for (const KFParticleGpuGraphNode& node : graph.Nodes()) {
      assert(hostGraph.FindNode(node.channelId) != nullptr);
    }

    KFParticleGpuDecayPlan incomplete;
    AddCpuFinderTwoDaughterChannels(incomplete);
    bool rejectedIncomplete = false;
    try {
      ValidateCompleteCpuFinderPlan(incomplete);
    }
    catch (const std::invalid_argument&) {
      rejectedIncomplete = true;
    }
    assert(rejectedIncomplete);

    KFParticleGpuDecayPlan defaultPlan;
    AddDefaultV0TwoDaughterChannels(defaultPlan);
    AddDefaultV0TrackCascadeChannels(defaultPlan);
    KFParticleGpuTwoDaughterRoutingPlan defaultTwoDaughterRouting;
    defaultTwoDaughterRouting.Compile(defaultPlan);
    assert(buffers.UploadTwoDaughterRoutingPlan(
      defaultTwoDaughterRouting, true));
    KFParticleGpuV0TrackRoutingPlan defaultCompositeRouting;
    defaultCompositeRouting.Compile(defaultPlan);
    assert(buffers.UploadV0TrackRoutingPlan(defaultCompositeRouting, true));

    Pass("complete-cpu-finder-manifest",
         "all 252 declared active channels and eight disabled contracts have exact graph and nine-family coverage");
  }

  void TestDecayGraphContract(KFParticleGpuBufferManager& buffers)
  {
    KFParticleGpuDecayPlan decayPlan;
    AddDefaultV0TwoDaughterChannels(decayPlan);
    AddDefaultV0TrackCascadeChannels(decayPlan);
    const KFParticleGpuDecayGraphManifest manifest =
      MakeDefaultCpuFinderDecayGraphManifest(decayPlan);

    KFParticleGpuDecayGraphPlan graph;
    graph.Compile(manifest);
    assert(graph.SourceRevision() != 0u);
    assert(graph.Nodes().size() == 7u);
    assert(graph.Groups().size() == 2u);
    assert(graph.FamilyCoverage().size() == KFGpuCpuFamilyCount);
    assert(graph.Groups()[0].generation == 1u);
    assert(graph.Groups()[0].topology == KFGpuGraphTopologyTrackTrack);
    assert(graph.Groups()[0].nodeOffset == 0u);
    assert(graph.Groups()[0].nodeCount == 3u);
    assert(graph.Groups()[1].generation == 2u);
    assert(graph.Groups()[1].topology == KFGpuGraphTopologyCompositeTrack);
    assert(graph.Groups()[1].nodeOffset == 3u);
    assert(graph.Groups()[1].nodeCount == 4u);

    KFParticleGpuDecayGraphPlan identicalGraph;
    identicalGraph.Compile(manifest);
    assert(identicalGraph.SourceRevision() == graph.SourceRevision());

    KFParticleGpuDecayGraphManifest changedManifest;
    for (const KFParticleGpuGraphNode& original : manifest.Nodes()) {
      KFParticleGpuGraphNode changed = original;
      if (changed.channelId == KFGpuChannelK0ShortToPiPlusPiMinus) {
        ++changed.selectionProfile;
      }
      changedManifest.AddNode(changed);
    }
    for (const KFParticleGpuGraphFamilyCoverage& family : manifest.FamilyCoverage()) {
      changedManifest.AddFamilyCoverage(family);
    }
    KFParticleGpuDecayGraphPlan changedGraph;
    changedGraph.Compile(changedManifest);
    assert(changedGraph.SourceRevision() != graph.SourceRevision());

    KFParticleGpuDecayPlan emptyDecayPlan;
    KFParticleGpuDecayGraphPlan emptyGraph;
    emptyGraph.Compile(MakeDefaultCpuFinderDecayGraphManifest(emptyDecayPlan));
    assert(emptyGraph.Nodes().empty());
    assert(emptyGraph.Groups().empty());
    assert(emptyGraph.FamilyCoverage().size() == KFGpuCpuFamilyCount);

    const KFParticleGpuGraphNode* lambda = nullptr;
    const KFParticleGpuGraphNode* xi = nullptr;
    for (const KFParticleGpuGraphNode& node : graph.Nodes()) {
      if (node.channelId == KFGpuChannelLambdaToProtonPiMinus) {
        lambda = &node;
      }
      if (node.channelId == KFGpuChannelXiMinusToLambdaPiMinus) {
        xi = &node;
      }
    }
    assert(lambda);
    assert(lambda->motherPdg == 3122);
    assert(lambda->payloadKind == KFGpuGraphPayloadTwoDaughter);
    assert(lambda->payloadIndex < 3u);
    assert(lambda->generation == 1u);
    assert(lambda->firstSource.kind == KFGpuGraphSourceTrackRange);
    assert(lambda->secondSource.kind == KFGpuGraphSourceTrackRange);
    assert(lambda->supportStatus == KFGpuGraphSupported);
    assert(xi);
    assert(xi->generation == 2u);
    assert(xi->payloadKind == KFGpuGraphPayloadCompositeTrack);
    assert(xi->payloadIndex < 4u);
    assert(xi->firstSource.kind == KFGpuGraphSourceCandidateGeneration);
    assert(xi->firstSource.generation == 1u);
    assert(xi->firstSource.sourceId == KFGpuChannelLambdaToProtonPiMinus);

    assert(graph.FamilyCoverage()[KFGpuCpuFamilyTwoDaughter].supportStatus
           == KFGpuGraphPartiallySupported);
    assert(graph.FamilyCoverage()[KFGpuCpuFamilyNeutralMissingMass].supportStatus
           == KFGpuGraphUnsupported);
    assert(graph.FamilyCoverage()[KFGpuCpuFamilyNeutralMissingMass].unsupportedReason
           == KFGpuGraphUnsupportedNotImplemented);

    KFParticleGpuDecayGraphManifest duplicateNode = manifest;
    duplicateNode.AddNode(manifest.Nodes()[0]);
    assert(RejectsDecayGraphManifest(duplicateNode));

    KFParticleGpuDecayGraphManifest missingFamily;
    for (const KFParticleGpuGraphNode& node : manifest.Nodes()) {
      missingFamily.AddNode(node);
    }
    for (const KFParticleGpuGraphFamilyCoverage& family : manifest.FamilyCoverage()) {
      if (family.family != KFGpuCpuFamilyFinalSelection) {
        missingFamily.AddFamilyCoverage(family);
      }
    }
    assert(RejectsDecayGraphManifest(missingFamily));

    KFParticleGpuDecayGraphManifest duplicateFamily = manifest;
    duplicateFamily.AddFamilyCoverage(manifest.FamilyCoverage()[0]);
    assert(RejectsDecayGraphManifest(duplicateFamily));

    KFParticleGpuDecayGraphManifest invalidCoverage;
    for (const KFParticleGpuGraphNode& node : manifest.Nodes()) {
      invalidCoverage.AddNode(node);
    }
    for (const KFParticleGpuGraphFamilyCoverage& original :
         manifest.FamilyCoverage()) {
      KFParticleGpuGraphFamilyCoverage family = original;
      if (family.family == KFGpuCpuFamilyNeutralMissingMass) {
        family.unsupportedReason = KFGpuGraphUnsupportedNone;
      }
      invalidCoverage.AddFamilyCoverage(family);
    }
    assert(RejectsDecayGraphManifest(invalidCoverage));

    KFParticleGpuDecayGraphManifest forwardDependency = manifest;
    KFParticleGpuGraphNode forward = manifest.Nodes()[3];
    forward.channelId = 1001u;
    forward.firstSource.generation = forward.generation;
    forwardDependency.AddNode(forward);
    assert(RejectsDecayGraphManifest(forwardDependency));

    KFParticleGpuDecayGraphManifest unknownParent = manifest;
    KFParticleGpuGraphNode orphan = manifest.Nodes()[3];
    orphan.channelId = 1002u;
    orphan.firstSource.sourceId = 999999u;
    unknownParent.AddNode(orphan);
    assert(RejectsDecayGraphManifest(unknownParent));

    KFParticleGpuDecayGraphManifest invalidOperations = manifest;
    KFParticleGpuGraphNode invalid = manifest.Nodes()[0];
    invalid.channelId = 1003u;
    invalid.operationMask = KFGpuGraphSelect;
    invalidOperations.AddNode(invalid);
    assert(RejectsDecayGraphManifest(invalidOperations));

    KFParticleGpuDecayGraphManifest unknownOperation = manifest;
    KFParticleGpuGraphNode unknown = manifest.Nodes()[0];
    unknown.channelId = 1004u;
    unknown.operationMask |= 1u << 31u;
    unknownOperation.AddNode(unknown);
    assert(RejectsDecayGraphManifest(unknownOperation));

    KFParticleGpuDecayGraphManifest duplicatePayload;
    for (const KFParticleGpuGraphNode& original : manifest.Nodes()) {
      KFParticleGpuGraphNode node = original;
      if (node.channelId == KFGpuChannelLambdaToProtonPiMinus) {
        node.payloadIndex = 0u;
      }
      duplicatePayload.AddNode(node);
    }
    for (const KFParticleGpuGraphFamilyCoverage& family : manifest.FamilyCoverage()) {
      duplicatePayload.AddFamilyCoverage(family);
    }
    assert(RejectsDecayGraphManifest(duplicatePayload));

    KFParticleGpuDecayGraphCompileLimits nodeLimit;
    nodeLimit.nodes = 6u;
    assert(RejectsDecayGraphManifest(manifest, nodeLimit));
    KFParticleGpuDecayGraphCompileLimits groupLimit;
    groupLimit.groups = 1u;
    assert(RejectsDecayGraphManifest(manifest, groupLimit));
    KFParticleGpuDecayGraphCompileLimits familyLimit;
    familyLimit.families = KFGpuCpuFamilyCount - 1u;
    assert(RejectsDecayGraphManifest(manifest, familyLimit));

    {
      KFParticleGpuDecayGraphPlan uncompiled;
      bool rejected = false;
      try {
        buffers.UploadDecayGraphPlan(uncompiled);
      }
      catch (const std::invalid_argument&) {
        rejected = true;
      }
      assert(rejected);
    }

    assert(buffers.UploadDecayGraphPlan(graph));
    assert(!buffers.UploadDecayGraphPlan(graph));
    assert(buffers.DecayGraphRevision() == graph.SourceRevision());
    assert(buffers.Capacities().decayGraphNodes >= graph.Nodes().size());
    assert(buffers.Capacities().decayGraphGroups >= graph.Groups().size());
    assert(buffers.Capacities().decayGraphFamilies >= graph.FamilyCoverage().size());

    const KFParticleGpuDecayGraphView hostGraph = buffers.HostDecayGraph();
    assert(hostGraph.NodeCount() == 7u);
    assert(hostGraph.GroupCount() == 2u);
    assert(hostGraph.FamilyCount() == KFGpuCpuFamilyCount);
    assert(hostGraph.Revision() == graph.SourceRevision());
    assert(hostGraph.FindNode(KFGpuChannelLambdaToProtonPiMinus));
    assert(hostGraph.FindNode(KFGpuChannelLambdaToProtonPiMinus)->motherPdg == 3122);
    assert(hostGraph.FindFamily(KFGpuCpuFamilyNeutralMissingMass));
    assert(hostGraph.FindFamily(KFGpuCpuFamilyNeutralMissingMass)->unsupportedReason
           == KFGpuGraphUnsupportedNotImplemented);

    const KFParticleGpuDeviceStorage& storage = buffers.Storage();
    assert(storage.fDecayGraphNodes.get() != nullptr);
    assert(storage.fDecayGraphGroups.get() != nullptr);
    assert(storage.fDecayGraphFamilyCoverage.get() != nullptr);
    Pass("decay-graph-contract",
         "CPU-family inventory compiles into validated generation nodes and persistent XPU tables");
  }

  void TestKernelStateContract(KFParticleGpuBufferManager& buffers)
  {
    const KFParticleGpuKernelState state(
      MakeConstView(buffers.DeviceInputTracks()), buffers.DeviceCandidates());

    assert(state.InputTracks().Size() == buffers.TrackSize());
    assert(state.InputTracks().Stride() == buffers.Capacities().tracks);
    assert(state.Candidates().Capacity() == buffers.Capacities().candidates);
    assert(state.Candidates().SizeData() == buffers.DeviceCandidates().SizeData());
    assert(state.V0TrackRouting().DescriptorCount() == 0u);
    assert(state.TwoDaughterRouting().DescriptorCount() == 0u);
    assert(state.CandidateDescriptorIndices().Capacity() == 0u);
    assert(state.DecayGraph().NodeCount() == 0u);

    Pass("device-state-contract",
         "flat non-owning kernel ABI retains device views without buffer ownership");
  }

  void TestVisibleDeviceStorage(KFParticleGpuBufferManager& buffers)
  {
    KFParticleGpuBufferCapacities requested = buffers.Capacities();
    requested.twoDaughterTasks = 17u;
    requested.graphOperationDescriptors =
      std::max(requested.graphOperationDescriptors, 3u);
    requested.graphOperationTasks = std::max(requested.graphOperationTasks, 8u);
    requested.transactions = std::max(requested.transactions, 1u);
    requested.domains = std::max(requested.domains, 2u);
    requested.workRanges = std::max(requested.workRanges, 2u);
    requested.domainStatuses = std::max(requested.domainStatuses, 2u);
    buffers.EnsureCapacity(requested);
    buffers.SetTransactionSizes(1u, 2u, 2u, 2u);

    const KFParticleGpuDeviceStorage& storage = buffers.Storage();
    assert(storage.fTrackParameters.get() != nullptr);
    assert(storage.fEvents.get() != nullptr);
    assert(storage.fTransactions.get() != nullptr);
    assert(storage.fDomains.get() != nullptr);
    assert(storage.fWorkRanges.get() != nullptr);
    assert(storage.fDomainStatuses.get() != nullptr);
    assert(storage.fCandidateParameters.get() != nullptr);
    assert(storage.fV0SelectionResults.get() != nullptr);
    assert(storage.fSelectedCandidateIndices.get() != nullptr);
    assert(storage.fSelectedCandidateChannelIds.get() != nullptr);
    assert(storage.fTwoDaughterTasks.get() != nullptr);
    assert(storage.fTwoDaughterTaskCount.get() != nullptr);
    assert(storage.fTwoDaughterTotalPairCount.get() != nullptr);
    assert(storage.fTwoDaughterTaskOverflowFlags.get() != nullptr);
    assert(storage.fTwoDaughterRoutedTasks.get() != nullptr);
    assert(storage.fTwoDaughterRoutingDescriptors.get() != nullptr);
    assert(storage.fTwoDaughterRoutingCompatibility.get() != nullptr);
    assert(storage.fTwoDaughterRoutingGroups.get() != nullptr);
    assert(storage.fTwoDaughterRoutingEnabledChannels.get() != nullptr);
    assert(storage.fCandidateRoutingDescriptorIndices.get() != nullptr);
    assert(storage.fV0TrackRoutingDescriptors.get() != nullptr);
    assert(storage.fV0TrackRoutingCompatibility.get() != nullptr);
    assert(storage.fV0TrackRoutingGroups.get() != nullptr);
    assert(storage.fV0TrackRoutingEnabledChannels.get() != nullptr);
    assert(storage.fV0TrackRoutingChannelAcceptedCounters.get() != nullptr);
    assert(storage.fDecayGraphNodes.get() != nullptr);
    assert(storage.fDecayGraphGroups.get() != nullptr);
    assert(storage.fDecayGraphFamilyCoverage.get() != nullptr);
    assert(storage.fGraphOperationDescriptors.get() != nullptr);
    assert(storage.fGraphOperationTasks.get() != nullptr);
    assert(storage.fGraphOperationResults.get() != nullptr);
    assert(storage.fGraphOperationVisitedCombinations.get() != nullptr);
    assert(storage.fGraphOperationAcceptedTasks.get() != nullptr);
    assert(storage.fGraphOperationStoredTasks.get() != nullptr);
    assert(storage.fGraphOperationConstructedCandidates.get() != nullptr);
    assert(storage.fGraphOperationRejectedTasks.get() != nullptr);
    assert(storage.fGraphOperationOverflowFlags.get() != nullptr);
    assert(storage.fGraphOperationChannelVisitedCounters.get() != nullptr);
    assert(storage.fGraphOperationChannelAcceptedCounters.get() != nullptr);
    assert(storage.fGraphOperationChannelStoredCounters.get() != nullptr);
    assert(storage.fGraphOperationChannelConstructedCounters.get() != nullptr);
    assert(storage.fGraphOperationChannelRejectedCounters.get() != nullptr);
    assert(buffers.Capacities().twoDaughterTasks == requested.twoDaughterTasks);
    assert(buffers.Capacities().graphOperationDescriptors
           == requested.graphOperationDescriptors);
    assert(buffers.Capacities().graphOperationTasks
           == requested.graphOperationTasks);
    assert(buffers.DeviceTransaction().DomainCapacity() == 2u);
    assert(buffers.DeviceTransaction().WorkRangeCapacity() == 2u);
    assert(buffers.DeviceTransaction().StatusCapacity() == 2u);
    assert(buffers.DeviceTransaction().TimeDomainCapacity() == 0u);

    Pass("device-storage-owner",
         "input, transaction-domain, routing, graph-operation work, raw-output, and selected-output XPU buffers persist together");
  }

  void TestPublishedKernelState(KFParticleGpuRuntime& runtime,
                                KFParticleGpuBufferManager& buffers)
  {
    if (buffers.Capacities().twoDaughterRoutedTasks == 0u) {
      buffers.EnsureTwoDaughterRoutedTaskCapacity(1u);
    }
    if (buffers.Capacities().v0TrackRoutedTasks == 0u) {
      buffers.EnsureV0TrackRoutedTaskCapacity(1u);
    }
    const KFParticleGpuDeviceStorage& storage = buffers.Storage();
    const KFParticleGpuKernels state(MakeConstView(buffers.DeviceInputTracks()),
                                     MakeConstView(buffers.DevicePrimaryVertices()),
                                     buffers.DeviceEvents(),
                                     storage.fTwoDaughterTasks.get(),
                                     buffers.Capacities().twoDaughterTasks,
                                     buffers.DeviceCandidates(),
                                     buffers.DeviceSelectedCandidates(),
                                     buffers.DeviceV0TrackRouting(),
                                     buffers.DeviceTwoDaughterRouting(),
                                     buffers.DeviceCandidateDescriptorIndices(),
                                     buffers.DeviceDecayGraph(),
                                     buffers.DeviceGraphOperationStorage(),
                                     buffers.DeviceTwoDaughterGenerationStorage(),
                                     buffers.DeviceV0TrackGenerationStorage());
    xpu::set<TheKFParticleFinder>(state);

    constexpr unsigned int CheckCount = 63u;
    xpu::buffer<unsigned int> checks(CheckCount, xpu::buf_io);
    unsigned int* hostChecks = HostPointer(checks);
    for (unsigned int i = 0; i < CheckCount; ++i) {
      hostChecks[i] = 0u;
    }
    runtime.GetQueue().copy(checks, xpu::h2d);
    runtime.GetQueue().launch<KFParticleGpuKernelStateProbe>(xpu::n_threads(1), checks.get());
    runtime.GetQueue().wait();
    runtime.GetQueue().copy(checks, xpu::d2h);
    runtime.GetQueue().wait();

    assert(hostChecks[0] == buffers.TrackSize());
    assert(hostChecks[1] == buffers.VertexSize());
    assert(hostChecks[2] == buffers.HostEvents()[0].eventId);
    assert(hostChecks[3] == buffers.Capacities().twoDaughterTasks);
    assert(hostChecks[4] == buffers.Capacities().candidates);
    assert(hostChecks[5] == buffers.Capacities().selectedCandidates);
    assert(hostChecks[6] == 4u);
    assert(hostChecks[7] == 2u);
    assert(hostChecks[8] == 1u);
    assert(hostChecks[9] == 4u);
    assert(hostChecks[10] == 3u);
    assert(hostChecks[11] == 2u);
    assert(hostChecks[12] == 2u);
    assert(hostChecks[13] == 3u);
    assert(hostChecks[14] == buffers.Capacities().candidates);
    assert(hostChecks[15] == 2u);
    assert(hostChecks[16] == 7u);
    assert(hostChecks[17] == 2u);
    assert(hostChecks[18] == KFGpuCpuFamilyCount);
    assert(hostChecks[19] == 1u);
    assert(hostChecks[20] == KFGpuGraphUnsupported);
    assert(hostChecks[21] == static_cast<unsigned int>(buffers.DecayGraphRevision()));
    assert(hostChecks[22] == KFGpuGraphPayloadTwoDaughter);
    assert(hostChecks[23] < 3u);
    assert(hostChecks[24] == buffers.GraphOperationDescriptorSize());
    assert(hostChecks[25] == buffers.Capacities().graphOperationDescriptors);
    assert(hostChecks[26] == buffers.Capacities().graphOperationTasks);
    assert(hostChecks[27] == static_cast<unsigned int>(
             buffers.GraphOperationRevision()));
    for (unsigned int index = 28u; index < 42u; ++index) {
      assert(hostChecks[index] == 1u);
    }
    assert(hostChecks[42] == buffers.Capacities().twoDaughterRoutedTasks);
    for (unsigned int index = 43u; index < 50u; ++index) {
      assert(hostChecks[index] == 1u);
    }
    assert(hostChecks[50] == buffers.DeviceTwoDaughterRouting().DescriptorCount());
    for (unsigned int index = 51u; index < 55u; ++index) {
      assert(hostChecks[index] == 1u);
    }
    assert(hostChecks[55] == buffers.Capacities().v0TrackRoutedTasks);
    for (unsigned int index = 56u; index < CheckCount; ++index) {
      assert(hostChecks[index] == 1u);
    }
    Pass("published-device-state",
         "XPU constant memory exposes input, work, output, routing, and decay-graph views");

    const unsigned int grownTaskCapacity =
      buffers.Capacities().graphOperationTasks + 5u;
    const unsigned int grownTwoDaughterCapacity =
      buffers.Capacities().twoDaughterRoutedTasks + 5u;
    const unsigned int grownV0TrackCapacity =
      buffers.Capacities().v0TrackRoutedTasks + 5u;
    buffers.EnsureGraphOperationTaskCapacity(grownTaskCapacity);
    buffers.EnsureTwoDaughterRoutedTaskCapacity(grownTwoDaughterCapacity);
    buffers.EnsureV0TrackRoutedTaskCapacity(grownV0TrackCapacity);
    const KFParticleGpuKernels grownState(
      MakeConstView(buffers.DeviceInputTracks()),
      MakeConstView(buffers.DevicePrimaryVertices()),
      buffers.DeviceEvents(),
      storage.fTwoDaughterTasks.get(),
      buffers.Capacities().twoDaughterTasks,
      buffers.DeviceCandidates(),
      buffers.DeviceSelectedCandidates(),
      buffers.DeviceV0TrackRouting(),
      buffers.DeviceTwoDaughterRouting(),
      buffers.DeviceCandidateDescriptorIndices(),
      buffers.DeviceDecayGraph(),
      buffers.DeviceGraphOperationStorage(),
      buffers.DeviceTwoDaughterGenerationStorage(),
      buffers.DeviceV0TrackGenerationStorage());
    xpu::set<TheKFParticleFinder>(grownState);
    for (unsigned int i = 0; i < CheckCount; ++i) {
      hostChecks[i] = 0u;
    }
    runtime.GetQueue().copy(checks, xpu::h2d);
    runtime.GetQueue().launch<KFParticleGpuKernelStateProbe>(
      xpu::n_threads(1), checks.get());
    runtime.GetQueue().wait();
    runtime.GetQueue().copy(checks, xpu::d2h);
    runtime.GetQueue().wait();
    assert(hostChecks[26] == grownTaskCapacity);
    assert(hostChecks[42] == grownTwoDaughterCapacity);
    assert(hostChecks[55] == grownV0TrackCapacity);
    for (unsigned int index = 28u; index < 42u; ++index) {
      assert(hostChecks[index] == 1u);
    }
    for (unsigned int index = 43u; index < 50u; ++index) {
      assert(hostChecks[index] == 1u);
    }
    for (unsigned int index = 51u; index < 55u; ++index) {
      assert(hostChecks[index] == 1u);
    }
    for (unsigned int index = 56u; index < CheckCount; ++index) {
      assert(hostChecks[index] == 1u);
    }
    Pass("published-generation-state",
         "graph and routed-generation workspaces are republished after capacity replacement");
  }

  void TestTimeDomainSeam(KFParticleGpuBufferManager& buffers)
  {
    const KFParticleGpuTransactionView eventTransaction = buffers.DeviceTransaction();
    assert(eventTransaction.TimeDomains() == nullptr);
    assert(eventTransaction.TimeDomainCapacity() == 0u);

    KFParticleGpuTimeDomainDesc timeDomain;
    timeDomain.flags = KFParticleGpuTimeDomainHasTrackParameters
                       | KFParticleGpuTimeDomainHasCovariance
                       | KFParticleGpuTimeDomainHasTime
                       | KFParticleGpuTimeDomainHasSourceIdentity
                       | KFParticleGpuTimeDomainHasFirstState
                       | KFParticleGpuTimeDomainHasLastState
                       | KFParticleGpuTimeDomainHasField
                       | KFParticleGpuTimeDomainHasPid
                       | KFParticleGpuTimeDomainHasPrimaryVertices
                       | KFParticleGpuTimeDomainHasLeftHalo
                       | KFParticleGpuTimeDomainHasRightHalo
                       | KFParticleGpuTimeDomainDeviceResident;
    timeDomain.domainIndex = 3u;
    timeDomain.windowIndex = 5u;
    timeDomain.chunkIndex = 1u;
    timeDomain.chunkCount = 4u;
    timeDomain.producerDeviceIndex = 1u;
    timeDomain.producerQueueIndex = 2u;
    timeDomain.timesliceId = 17001u;
    timeDomain.storageLifetimeToken = 17002u;
    timeDomain.dependencyToken = 17003u;
    timeDomain.coreTracks = {20u, 10u};
    timeDomain.leftHaloTracks = {16u, 4u};
    timeDomain.rightHaloTracks = {30u, 3u};
    timeDomain.coreStartTime = 100.f;
    timeDomain.coreEndTime = 200.f;
    timeDomain.haloStartTime = 90.f;
    timeDomain.haloEndTime = 210.f;

    assert(timeDomain.schemaVersion == KFParticleGpuTimeDomainSchemaVersion);
    assert(timeDomain.coreTracks.offset == 20u && timeDomain.coreTracks.size == 10u);
    assert(timeDomain.leftHaloTracks.offset + timeDomain.leftHaloTracks.size
           == timeDomain.coreTracks.offset);
    assert(timeDomain.rightHaloTracks.offset
           == timeDomain.coreTracks.offset + timeDomain.coreTracks.size);
    assert(timeDomain.haloStartTime <= timeDomain.coreStartTime);
    assert(timeDomain.haloEndTime >= timeDomain.coreEndTime);

    Pass("time-domain-seam",
         "event batches allocate no time metadata; the frozen CA seam names core/halo ownership, producer lifetime, device, queue, and dependency identity");
  }

  void TestDecayPlanDataTypes()
  {
    static_assert(std::is_trivially_copyable<KFParticleGpuTwoDaughterChannel>::value,
                  "Two-daughter channel descriptors must remain device-copyable values");
    static_assert(std::is_trivially_copyable<KFParticleGpuCandidateRange>::value,
                  "Candidate ranges must remain flat host/device bookkeeping values");
    static_assert(std::is_trivially_copyable<KFParticleGpuTwoDaughterChannelResult>::value,
                  "Channel results must remain flat host/device bookkeeping values");
    static_assert(std::is_trivially_copyable<KFParticleGpuGraphOperationChannel>::value,
                  "Graph-operation channel descriptors must remain flat device values");

    KFParticleGpuTwoDaughterChannel channel;
    channel.channelId = 7;
    channel.firstTrackSet = PrimaryPositiveFirst;
    channel.secondTrackSet = SecondaryNegativeFirst;
    channel.firstSpecies = Proton;
    channel.secondSpecies = Pion;
    channel.flags = KFGpuTwoDaughterUseEnergyFit;
    channel.motherPdg = 3122;
    channel.firstDaughterPdg = 2212;
    channel.secondDaughterPdg = -211;
    channel.primaryVertexIndex = 3;
    channel.firstMass = 0.938272f;
    channel.secondMass = 0.13957f;
    channel.motherMass = 1.115683f;
    channel.motherMassSigma = 0.0021f;
    channel.secondaryMassSigmaCut = 3.f;
    channel.maxSecondaryTopoChi2PerNdf = 5.f;
    channel.minSecondaryLdL = 10.f;
    channel.transportMode = KFGpuTransportFieldAware;
    channel.firstCharge = 1;
    channel.secondCharge = -1;
    channel.minFirstPixelHits = 4;
    channel.minSecondPixelHits = 5;
    channel.maxFirstChiToPrimaryVertex = 2.5f;
    channel.maxSecondChiToPrimaryVertex = 3.5f;

    const KFParticleGpuTwoDaughterTaskSource source =
      MakeTwoDaughterTaskSource(channel, 11u);
    assert(source.channelId == channel.channelId);
    assert(source.eventIndex == 11u);
    assert(source.firstTrackSet == PrimaryPositiveFirst);
    assert(source.secondTrackSet == SecondaryNegativeFirst);
    assert(source.firstSpecies == Proton);
    assert(source.secondSpecies == Pion);
    assert(source.flags == KFGpuTwoDaughterUseEnergyFit);
    assert(source.motherPdg == 3122);
    assert(source.firstDaughterPdg == 2212);
    assert(source.secondDaughterPdg == -211);
    assert(source.primaryVertexIndex == 3);
    assert(AlmostEqual(source.firstMass, 0.938272f));
    assert(AlmostEqual(source.secondMass, 0.13957f));
    assert(AlmostEqual(source.motherMass, 1.115683f));
    assert(AlmostEqual(source.motherMassSigma, 0.0021f));
    assert(AlmostEqual(source.secondaryMassSigmaCut, 3.f));
    assert(AlmostEqual(source.maxSecondaryTopoChi2PerNdf, 5.f));
    assert(AlmostEqual(source.minSecondaryLdL, 10.f));
    assert(source.transportMode == KFGpuTransportFieldAware);
    assert(source.firstCharge == 1);
    assert(source.secondCharge == -1);
    assert(source.minFirstPixelHits == 4);
    assert(source.minSecondPixelHits == 5);
    assert(AlmostEqual(source.maxFirstChiToPrimaryVertex, 2.5f));
    assert(AlmostEqual(source.maxSecondChiToPrimaryVertex, 3.5f));

    KFParticleGpuCandidateRange range;
    assert(range.Empty());
    range.offset = 4;
    range.size = 2;
    range.daughterOffset = 8;
    range.daughterSize = 4;
    range.overflowFlags = CandidateCapacityExceeded;
    assert(!range.Empty());
    assert(range.End() == 6);
    assert(range.ContainsCandidate(4));
    assert(range.ContainsCandidate(5));
    assert(!range.ContainsCandidate(6));
    assert(range.LocalCandidateIndex(5) == 1);
    assert(range.DaughterEnd() == 12);
    assert(range.ContainsDaughter(8));
    assert(range.ContainsDaughter(11));
    assert(!range.ContainsDaughter(12));
    assert(range.LocalDaughterIndex(10) == 2);
    assert(range.HasOverflow(CandidateCapacityExceeded));
    assert(!range.HasOverflow(DaughterCapacityExceeded));
    assert(range.HasAnyOverflow());

    KFParticleGpuTwoDaughterChannelResult result;
    result.channelId = channel.channelId;
    result.motherPdg = channel.motherPdg;
    result.eventIndex = source.eventIndex;
    result.totalPairs = 10;
    result.acceptedTasks = 6;
    result.storedTasks = 4;
    result.candidates = range;
    assert(result.Truncated());
    assert(!result.Empty());
    assert(result.HasOverflow(CandidateCapacityExceeded));
    assert(result.HasAnyOverflow());
    assert(result.ContainsCandidate(4));
    result.storedTasks = result.acceptedTasks;
    assert(!result.Truncated());
    assert(result.candidates.overflowFlags == CandidateCapacityExceeded);
    assert(IsSupportedTwoDaughterTransportMode(KFGpuTransportStraightLine));
    assert(IsSupportedTwoDaughterTransportMode(KFGpuTransportFieldAware));
    assert(IsSupportedTwoDaughterTransportMode(KFGpuTransportFullField));
    assert(IsSupportedTwoDaughterEnergyFitTransportMode(KFGpuTransportStraightLine));
    assert(IsSupportedTwoDaughterEnergyFitTransportMode(KFGpuTransportFieldAware));
    assert(IsSupportedTwoDaughterEnergyFitTransportMode(KFGpuTransportFullField));

    Pass("decay-plan-data-types",
         "standalone two-daughter channel descriptors and result ranges are flat values");
  }

  void TestTwoDaughterPairTilePlan()
  {
    constexpr std::array<unsigned int, 6u> pairs{{0u, 1u, 63u, 64u, 65u, 99u}};
    constexpr std::array<unsigned int, 6u> expected{{0u, 1u, 1u, 1u, 2u, 2u}};
    for (unsigned int fixture = 0u; fixture < pairs.size(); ++fixture) {
      assert(KFParticleGpuPairTileCount(pairs[fixture]) == expected[fixture]);
      std::array<KFParticleGpuPairTile, 2u> tiles{};
      unsigned int cursor = 0u;
      assert(KFParticleGpuAppendPairTiles(
        13u, pairs[fixture], tiles.data(), expected[fixture], cursor));
      assert(cursor == expected[fixture]);
      for (unsigned int tile = 0u; tile < cursor; ++tile) {
        assert(tiles[tile].workIndex == 13u);
        assert(tiles[tile].firstPair == tile * KFParticleGpuPairTileSize);
      }
    }
    Pass("two-daughter-pair-tile-plan",
         "host planning covers exact 0/1/63/64/65/99-pair boundaries without rectangular work padding");
  }

