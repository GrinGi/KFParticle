/*
 * This file is part of KFParticle package
 * Copyright (C) 2007-2026
 *
 * KFParticle is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 */

#include "KFParticleGpuBufferManager.h"
#include "KFParticleGpuCandidateTransfer.h"
#include "KFParticleGpuChannelRouting.h"
#include "KFParticleGpuCpuChannelCatalogue.h"
#include "KFParticleGpuDecayGraphPlan.h"
#include "KFParticleGpuDecayPlan.h"
#include "KFParticleGpuDeviceStorage.h"
#include "KFParticleGpuGraphOperations.h"
#include "KFParticleGpuInputDataTransfer.h"
#include "KFParticleGpuKernelState.h"
#include "KFParticleGpuKernels.h"
#include "KFParticleGpuMath.h"
#include "KFParticleGpuMaterializer.h"
#include "KFParticleGpuParity.h"
#include "KFParticleGpuRuntime.h"
#include "KFParticleGpuRoutingPlan.h"
#include "KFParticleGpuSelection.h"
#include "KFParticleGpuSteering.h"
#include "KFParticleGpuTwoDaughter.h"
#include "KFParticleGpuTwoDaughterRouting.h"
#include "KFParticleGpuV0Track.h"
#include "xpu/KFParticleGpuXpuBaseline.h"

#include <xpu/host.h>

#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <vector>

#ifndef KFPARTICLE_GPU_TEST_DEVICE
#define KFPARTICLE_GPU_TEST_DEVICE "cpu"
#endif

static_assert(std::is_trivially_copyable<KFParticleGpuKernels>::value,
              "KFParticle GPU constant-memory state must remain trivially copyable");
static_assert(std::is_trivially_default_constructible<KFParticleGpuKernels>::value,
              "KFParticle GPU constant-memory state must not require dynamic initialization");
static_assert(std::is_trivially_copyable<KFParticleGpuKernelState>::value,
              "KFParticle GPU kernel state must remain a flat device ABI");
static_assert(std::is_trivially_default_constructible<KFParticleGpuKernelState>::value,
              "KFParticle GPU kernel state must not require dynamic initialization");
static_assert(std::is_trivially_copyable<KFParticleGpuMeasurement>::value,
              "KFParticle GPU measurement must remain a flat device value type");
static_assert(std::is_trivially_copyable<KFParticleGpuV0SelectionObservables>::value,
              "KFParticle GPU selection observables must remain flat device values");
static_assert(std::is_trivially_copyable<KFParticleGpuV0SelectionResult>::value,
              "KFParticle GPU selection results must remain flat device values");
static_assert(std::is_trivially_copyable<KFParticleGpuV0LineTopologyResult>::value,
              "KFParticle GPU line-topology results must remain flat device values");
static_assert(std::is_trivially_copyable<KFParticleGpuV0TrackChannel>::value,
              "KFParticle GPU V0-track channels must remain flat device values");
static_assert(std::is_trivially_copyable<KFParticleGpuV0TrackTask>::value,
              "KFParticle GPU V0-track tasks must remain flat device values");
static_assert(std::is_trivially_copyable<KFParticleGpuV0TrackInputView>::value,
              "KFParticle GPU V0-track input must remain a non-owning device view");
static_assert(std::is_trivially_copyable<KFParticleGpuV0TrackLineage>::value,
              "KFParticle GPU V0-track lineage must remain flat device values");
static_assert(std::is_trivially_copyable<KFParticleGpuChannelMask>::value,
              "KFParticle GPU channel masks must remain flat device values");
static_assert(std::is_trivially_copyable<KFParticleGpuV0TrackRoutingDescriptor>::value,
              "KFParticle GPU routing descriptors must remain flat device values");
static_assert(std::is_trivially_copyable<KFParticleGpuV0TrackExecutionGroup>::value,
              "KFParticle GPU execution groups must remain flat device values");
static_assert(std::is_trivially_copyable<KFParticleGpuV0TrackRoutedTask>::value,
              "KFParticle GPU routed tasks must remain flat device values");
static_assert(std::is_trivially_copyable<KFParticleGpuV0TrackRoutingStatus>::value,
              "KFParticle GPU routing status must remain a flat device value");
static_assert(std::is_trivially_copyable<KFParticleGpuTwoDaughterRoutingDescriptor>::value,
              "KFParticle GPU two-daughter routing descriptors must remain flat values");
static_assert(std::is_trivially_copyable<KFParticleGpuTwoDaughterRoutedTask>::value,
              "KFParticle GPU two-daughter routed tasks must remain flat values");
static_assert(std::is_trivially_copyable<KFParticleGpuGraphNode>::value,
              "KFParticle GPU graph nodes must remain flat values");
static_assert(std::is_trivially_copyable<KFParticleGpuGraphExecutionGroup>::value,
              "KFParticle GPU graph groups must remain flat values");
static_assert(std::is_trivially_copyable<KFParticleGpuGraphFamilyCoverage>::value,
              "KFParticle GPU graph coverage must remain flat values");
static_assert(std::is_trivially_copyable<KFParticleGpuDecayGraphView>::value,
              "KFParticle GPU graph view must remain a flat non-owning ABI");
static_assert(std::is_trivially_copyable<KFParticleGpuGraphOperationDescriptor>::value,
              "KFParticle GPU graph operation descriptors must remain flat values");
static_assert(std::is_trivially_copyable<KFParticleGpuGraphOperationTask>::value,
              "KFParticle GPU graph operation tasks must remain flat values");
static_assert(std::is_trivially_copyable<KFParticleGpuGraphOperationResult>::value,
              "KFParticle GPU graph operation results must remain flat values");

namespace
{
  void Pass(const char* name, const char* details);

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

  unsigned int FieldAwareProfileIterations()
  {
    const char* enabled = std::getenv("KFPARTICLE_GPU_TEST_PROFILE_FIELD_AWARE");
    if (!enabled || enabled[0] == '\0' || enabled[0] == '0') {
      return 0u;
    }

    const char* requested = std::getenv("KFPARTICLE_GPU_TEST_PROFILE_ITERATIONS");
    if (!requested || requested[0] == '\0') {
      return 500u;
    }
    const unsigned long parsed = std::strtoul(requested, nullptr, 10);
    return parsed > 0u && parsed <= 1000000u ? static_cast<unsigned int>(parsed) : 500u;
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
    buffers.EnsureCapacity(requested);

    const KFParticleGpuDeviceStorage& storage = buffers.Storage();
    assert(storage.fTrackParameters.get() != nullptr);
    assert(storage.fEvents.get() != nullptr);
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

    Pass("device-storage-owner",
         "input, routing, graph-operation work, raw-output, and selected-output XPU buffers persist together");
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
         "graph and routed-generation workspaces are republished after device-buffer growth");
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

  void TestGeneratedTwoDaughterPipeline(KFParticleGpuRuntime& runtime,
                                        KFParticleGpuBufferManager& buffers)
  {
    buffers.SetInputSizes(4, 1, 1);
    KFParticleGpuInputTrackSoAView tracks = buffers.HostInputTracks();
    StoreSyntheticTrack(tracks, 0, -0.2f, 0.1f, 0.0f, 0.8f, 0.1f, 1.0f, 211, 1, 701);
    StoreSyntheticTrack(tracks, 1, 0.4f, -0.3f, 0.2f, 0.5f, 0.6f, 0.9f, 211, 1, 702);
    StoreSyntheticTrack(tracks, 2, 0.1f, 0.5f, -0.4f, -0.3f, 0.7f, 1.1f, -211, -1, 801);
    StoreSyntheticTrack(tracks, 3, -0.6f, -0.2f, 0.3f, -0.4f, 0.2f, 0.8f, -211, -1, 802);
    tracks.NumberOfPixelHits(1) = 3;
    tracks.ChiToPrimaryVertex(3) = 9.f;

    KFParticleGpuEventDesc* events = buffers.HostEvents();
    events[0] = KFParticleGpuEventDesc();
    events[0].eventId = 43;
    events[0].TrackSet(SecondaryPositiveFirst).tracks = KFParticleGpuRange(0, 2);
    events[0].TrackSet(SecondaryPositiveFirst).Species(Pion) = KFParticleGpuRange(0, 2);
    events[0].TrackSet(SecondaryNegativeFirst).tracks = KFParticleGpuRange(2, 2);
    events[0].TrackSet(SecondaryNegativeFirst).Species(Pion) = KFParticleGpuRange(2, 2);
    buffers.UploadInput();
    buffers.ResetCandidates();

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

    xpu::buffer<KFParticleGpuTwoDaughterTask> taskBuffer(3, xpu::buf_io);
    xpu::buffer<unsigned int> writtenTasks(1, xpu::buf_io);
    xpu::buffer<unsigned int> totalPairs(1, xpu::buf_io);

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

    runtime.GetQueue().launch<KFParticleGpuTwoDaughterTaskKernel>(
      xpu::n_threads(3),
      MakeConstView(buffers.DeviceInputTracks()),
      taskBuffer.get(),
      3u,
      buffers.DeviceCandidates());
    runtime.GetQueue().wait();
    buffers.DownloadCandidates();

    KFParticleGpuTwoDaughterTask referenceTask;
    KFParticleGpuRange firstRange(0, 2);
    KFParticleGpuRange secondRange(2, 2);
    FillTwoDaughterTask(firstRange, secondRange, 0, source, referenceTask);
    KFParticleGpuFitState reference;
    assert(BuildTwoDaughterCandidate(MakeConstView(buffers.HostInputTracks()), referenceTask, reference));

    const KFParticleGpuConstCandidatePoolView candidates = MakeConstView(buffers.HostCandidates());
    assert(candidates.Size() == 3);
    assert(candidates.Daughters().Size() == 4);
    assert(candidates.OverflowFlags() == DaughterCapacityExceeded);

    KFParticleGpuFitState actual;
    LoadCandidateFit(candidates, 0, actual);
    ExpectFitStateClose(actual, reference, 2.e-5f);
    assert(candidates.Metadata().Pdg(0) == 310);
    assert(candidates.Metadata().DaughterCount(0) == 2);
    assert(candidates.Metadata().Flags(0)
           == static_cast<unsigned int>(
             KFGpuCandidateValid | KFGpuCandidateLineDca | KFGpuCandidateEnergyFit));
    assert(candidates.Daughters().SourceId(0) == 701);
    assert(candidates.Daughters().SourceId(1) == 801);

    assert(candidates.Metadata().DaughterCount(1) == 0);
    assert(candidates.Metadata().Flags(1) & KFGpuCandidateBuildFailed);
    assert(candidates.Metadata().DaughterCount(2) == 0);
    assert(candidates.Metadata().Flags(2) & KFGpuCandidateBuildFailed);

    Pass("two-daughter-generated-pipeline", "GPU-generated pair tasks feed the two-daughter construction kernel");
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
    assert(candidates.Metadata().Flags(1) & KFGpuCandidateBuildFailed);
    assert(candidates.Metadata().Flags(2) & KFGpuCandidateBuildFailed);

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
    assert(candidates.Metadata().Pdg(0) == 997);
    assert(candidates.Metadata().Flags(0)
           == static_cast<unsigned int>(KFGpuCandidateValid | KFGpuCandidateLineDca));
    assert(HasCandidateDaughters(candidates, 901, 1001));
    assert(HasCandidateDaughters(candidates, 902, 1001));

    KFParticleGpuFitState fieldMother;
    LoadCandidateFit(candidates, 0, fieldMother);
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
    unsigned int fieldCandidate = candidates.Size();
    unsigned int lineCandidate = candidates.Size();
    for (unsigned int candidate = 0u; candidate < candidates.Size(); ++candidate) {
      if (candidates.Metadata().ChannelId(candidate) == 78u) {
        fieldCandidate = candidate;
      }
      if (candidates.Metadata().ChannelId(candidate) == 79u) {
        lineCandidate = candidate;
      }
    }
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

  void DumpFullFieldTraceMeasurement(const char* stage,
                                     const KFParticleGpuMeasurement& gpu,
                                     const KFParticleGpuMeasurement& host)
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
  }

  void DumpDcaArithmeticTrace(const KFParticleGpuMath::KFParticleGpuDcaArithmeticTrace& gpu,
                              const KFParticleGpuMath::KFParticleGpuDcaArithmeticTrace& host)
  {
    const std::streamsize oldPrecision = std::cerr.precision();
    std::cerr << std::setprecision(9);
    const auto dump = [](const char* name, float gpuValue, float hostValue) {
      std::cerr << "TRACE dca-arithmetic value=" << name
                << " gpu=" << gpuValue << " host=" << hostValue
                << " delta=" << (gpuValue - hostValue) << '\n';
    };
    for (int particle = 0; particle < 2; ++particle) {
      for (int parameter = 0; parameter < 6; ++parameter) {
        std::cerr << "TRACE dca-arithmetic input particle=" << particle
                  << " parameter=" << parameter
                  << " gpu=" << gpu.parameters[particle][parameter]
                  << " host=" << host.parameters[particle][parameter]
                  << " delta="
                  << (gpu.parameters[particle][parameter]
                      - host.parameters[particle][parameter]) << '\n';
      }
      dump(particle == 0 ? "bq-first" : "bq-second", gpu.bq[particle], host.bq[particle]);
      dump(particle == 0 ? "pt2-first" : "pt2-second", gpu.pt2[particle], host.pt2[particle]);
      dump(particle == 0 ? "drp-first" : "drp-second", gpu.drp[particle], host.drp[particle]);
      dump(particle == 0 ? "dxyp-first" : "dxyp-second", gpu.dxyp[particle], host.dxyp[particle]);
    }
    dump("dx0", gpu.delta0[0], host.delta0[0]);
    dump("dy0", gpu.delta0[1], host.delta0[1]);
    dump("dr02", gpu.dr02, host.dr02);
    dump("p1p2", gpu.p1p2, host.p1p2);
    dump("dp1p2", gpu.dp1p2, host.dp1p2);
    for (int index = 0; index < 4; ++index) {
      std::cerr << "TRACE dca-arithmetic k index=" << index
                << " gpu=" << gpu.k[index] << " host=" << host.k[index]
                << " delta=" << (gpu.k[index] - host.k[index]) << '\n';
    }
    dump("kp", gpu.kp, host.kp);
    dump("kd", gpu.kd, host.kd);
    dump("c1", gpu.c[0], host.c[0]);
    dump("c2", gpu.c[1], host.c[1]);
    dump("discriminant", gpu.discriminant, host.discriminant);
    dump("root", gpu.root, host.root);
    for (int particle = 0; particle < 2; ++particle) {
      for (int root = 0; root < 2; ++root) {
        std::cerr << "TRACE dca-arithmetic atan particle=" << particle
                  << " root=" << root
                  << " numerator-gpu=" << gpu.atanNumerator[particle][root]
                  << " numerator-host=" << host.atanNumerator[particle][root]
                  << " numerator-delta="
                  << (gpu.atanNumerator[particle][root]
                      - host.atanNumerator[particle][root])
                  << " denominator-gpu=" << gpu.atanDenominator[particle][root]
                  << " denominator-host=" << host.atanDenominator[particle][root]
                  << " denominator-delta="
                  << (gpu.atanDenominator[particle][root]
                      - host.atanDenominator[particle][root])
                  << " dS-gpu=" << gpu.roots[particle][root]
                  << " dS-host=" << host.roots[particle][root]
                  << " dS-delta="
                  << (gpu.roots[particle][root] - host.roots[particle][root]) << '\n';
        const KFParticleGpuMath::KFParticleGpuAtan2ArithmeticTrace& gpuAtan =
          gpu.atan[particle][root];
        const KFParticleGpuMath::KFParticleGpuAtan2ArithmeticTrace& hostAtan =
          host.atan[particle][root];
        std::cerr << "TRACE dca-arithmetic atan-detail particle=" << particle
                  << " root=" << root
                  << " ratio-gpu=" << gpuAtan.initialRatio
                  << " ratio-host=" << hostAtan.initialRatio
                  << " ratio-delta=" << (gpuAtan.initialRatio - hostAtan.initialRatio)
                  << " transformed-gpu=" << gpuAtan.transformedRatio
                  << " transformed-host=" << hostAtan.transformedRatio
                  << " transformed-delta="
                  << (gpuAtan.transformedRatio - hostAtan.transformedRatio)
                  << " result-gpu=" << gpuAtan.result
                  << " result-host=" << hostAtan.result
                  << " result-delta=" << (gpuAtan.result - hostAtan.result)
                  << " flags=" << gpuAtan.branchFlags << '/' << hostAtan.branchFlags << '\n';
        for (int stage = 0; stage < 4; ++stage) {
          std::cerr << "TRACE dca-arithmetic atan-polynomial particle=" << particle
                    << " root=" << root << " stage=" << stage
                    << " gpu=" << gpuAtan.polynomialStage[stage]
                    << " host=" << hostAtan.polynomialStage[stage]
                    << " delta="
                    << (gpuAtan.polynomialStage[stage]
                        - hostAtan.polynomialStage[stage]) << '\n';
        }
      }
      dump(particle == 0 ? "selected-dS-first" : "selected-dS-second",
           gpu.selectedDs[particle],
           host.selectedDs[particle]);
      dump(particle == 0 ? "bs-first" : "bs-second", gpu.bs[particle], host.bs[particle]);
      dump(particle == 0 ? "sin-first" : "sin-second", gpu.sine[particle], host.sine[particle]);
      dump(particle == 0 ? "cos-first" : "cos-second", gpu.cosine[particle], host.cosine[particle]);
      const KFParticleGpuMath::KFParticleGpuSinCosArithmeticTrace& gpuSinCos =
        gpu.sincos[particle];
      const KFParticleGpuMath::KFParticleGpuSinCosArithmeticTrace& hostSinCos =
        host.sincos[particle];
      std::cerr << "TRACE dca-arithmetic sincos particle=" << particle
                << " scaled-gpu=" << gpuSinCos.scaled
                << " scaled-host=" << hostSinCos.scaled
                << " scaled-delta=" << (gpuSinCos.scaled - hostSinCos.scaled)
                << " rounded=" << gpuSinCos.rounded << '/' << hostSinCos.rounded
                << " quadrant-count=" << gpuSinCos.quadrantCount << '/'
                << hostSinCos.quadrantCount
                << " quadrant=" << gpuSinCos.quadrant << '/' << hostSinCos.quadrant
                << " reduced-gpu=" << gpuSinCos.reduced
                << " reduced-host=" << hostSinCos.reduced
                << " reduced-delta=" << (gpuSinCos.reduced - hostSinCos.reduced)
                << " sine-series-delta="
                << (gpuSinCos.sineSeries - hostSinCos.sineSeries)
                << " cosine-series-delta="
                << (gpuSinCos.cosineSeries - hostSinCos.cosineSeries) << '\n';
      for (int component = 0; component < 3; ++component) {
        std::cerr << "TRACE dca-arithmetic transported particle=" << particle
                  << " component=" << component
                  << " position-gpu=" << gpu.position[particle][component]
                  << " position-host=" << host.position[particle][component]
                  << " position-delta="
                  << (gpu.position[particle][component] - host.position[particle][component])
                  << " momentum-gpu=" << gpu.momentum[particle][component]
                  << " momentum-host=" << host.momentum[particle][component]
                  << " momentum-delta="
                  << (gpu.momentum[particle][component] - host.momentum[particle][component])
                  << '\n';
      }
      dump(particle == 0 ? "momentum2-first" : "momentum2-second",
           gpu.momentum2[particle],
           host.momentum2[particle]);
      dump(particle == 0 ? "separation-dot-first" : "separation-dot-second",
           gpu.separationDotMomentum[particle],
           host.separationDotMomentum[particle]);
      dump(particle == 0 ? "correction-numerator-first" : "correction-numerator-second",
           gpu.correctionNumerator[particle],
           host.correctionNumerator[particle]);
      dump(particle == 0 ? "correction-first" : "correction-second",
           gpu.correction[particle],
           host.correction[particle]);
      dump(particle == 0 ? "final-dS-first" : "final-dS-second",
           gpu.finalDs[particle],
           host.finalDs[particle]);
    }
    dump("root-distance2-first", gpu.rootDistance2[0], host.rootDistance2[0]);
    dump("root-distance2-second", gpu.rootDistance2[1], host.rootDistance2[1]);
    dump("momentum-dot", gpu.momentumDot, host.momentumDot);
    for (int component = 0; component < 3; ++component) {
      std::cerr << "TRACE dca-arithmetic separation component=" << component
                << " gpu=" << gpu.separation[component]
                << " host=" << host.separation[component]
                << " delta=" << (gpu.separation[component] - host.separation[component])
                << '\n';
    }
    dump("determinant-raw", gpu.determinantRaw, host.determinantRaw);
    dump("determinant-used", gpu.determinantUsed, host.determinantUsed);
    std::cerr.precision(oldPrecision);
  }

  void DumpFullFieldTrace(const KFParticleGpuFullFieldTwoDaughterTrace& gpu,
                          const KFParticleGpuFullFieldTwoDaughterTrace& host)
  {
    std::cerr << "TRACE full-field stage-status gpu=" << gpu.stage << " host=" << host.stage
              << " middle=" << gpu.useMiddlePoint << '/' << host.useMiddlePoint
              << " by=" << gpu.by << '/' << host.by
              << " by-delta=" << (gpu.by - host.by) << '\n';
    DumpDcaArithmeticTrace(gpu.dcaArithmetic, host.dcaArithmetic);
    for (int root = 0; root < 2; ++root) {
      std::cerr << "TRACE full-field roots index=" << root
                << " first-gpu=" << gpu.firstRoots[root]
                << " first-host=" << host.firstRoots[root]
                << " first-delta=" << (gpu.firstRoots[root] - host.firstRoots[root])
                << " second-gpu=" << gpu.secondRoots[root]
                << " second-host=" << host.secondRoots[root]
                << " second-delta=" << (gpu.secondRoots[root] - host.secondRoots[root]) << '\n';
    }
    for (int daughter = 0; daughter < 2; ++daughter) {
      std::cerr << "TRACE full-field dS daughter=" << daughter
                << " gpu=" << gpu.dS[daughter] << " host=" << host.dS[daughter]
                << " delta=" << (gpu.dS[daughter] - host.dS[daughter]) << '\n';
    }
    for (int derivative = 0; derivative < 4; ++derivative) {
      for (int parameter = 0; parameter < 6; ++parameter) {
        std::cerr << "TRACE full-field dsdr block=" << derivative
                  << " parameter=" << parameter
                  << " gpu=" << gpu.dsdr[derivative][parameter]
                  << " host=" << host.dsdr[derivative][parameter]
                  << " delta=" << (gpu.dsdr[derivative][parameter]
                                   - host.dsdr[derivative][parameter]) << '\n';
      }
    }
    DumpFullFieldTraceState(
      "preliminary-first", gpu.preliminaryFirst, host.preliminaryFirst);
    DumpFullFieldTraceState(
      "preliminary-second", gpu.preliminarySecond, host.preliminarySecond);
    DumpFullFieldTraceState(
      "second-pass-current", gpu.secondPassCurrent, host.secondPassCurrent);
    DumpFullFieldTraceMeasurement(
      "second-pass-measurement", gpu.secondPassMeasurement, host.secondPassMeasurement);
    for (int row = 0; row < 3; ++row) {
      for (int column = 0; column < 3; ++column) {
        std::cerr << "TRACE full-field correlation row=" << row << " column=" << column
                  << " gpu=" << gpu.secondPassMeasurement.Correlation(row, column)
                  << " host=" << host.secondPassMeasurement.Correlation(row, column)
                  << " delta=" << (gpu.secondPassMeasurement.Correlation(row, column)
                                   - host.secondPassMeasurement.Correlation(row, column)) << '\n';
      }
    }
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

  void TestDecayPlanDefaultK0FieldAwareFixture(KFParticleGpuRuntime& runtime,
                                               KFParticleGpuBufferManager& buffers)
  {
    KFParticleGpuSteering& steering = runtime.GetSteering();
    KFParticleGpuDecayPlan& plan = steering.GetDecayPlan();
    plan.Clear();
    KFParticleGpuTwoDaughterChannel channel =
      MakeK0ShortToPiPlusPiMinusChannel();
    assert(channel.transportMode == KFGpuTransportFullField);
    channel.maxDaughterDistance = -1.f;
    plan.AddTwoDaughterChannel(channel);

    KFParticleGpuBufferCapacities requested = buffers.Capacities();
    requested.candidates = requested.candidates > 4u ? requested.candidates : 4u;
    requested.daughterIds = requested.daughterIds > 8u ? requested.daughterIds : 8u;
    buffers.EnsureCapacity(requested);
    FillGeneratedPipelineFixture(buffers);

    KFParticleGpuInputTrackSoAView tracks = buffers.HostInputTracks();
    StoreDiagnosticField(tracks, 0);

    const KFParticleGpuTwoDaughterTaskSource source =
      MakeTwoDaughterTaskSource(channel, 0u);
    const KFParticleGpuRange firstRange(0, 2);
    const KFParticleGpuRange secondRange(2, 2);
    KFParticleGpuTwoDaughterTask fieldTask;
    FillTwoDaughterTask(firstRange, secondRange, 0u, source, fieldTask);

    KFParticleGpuFitState referenceMother;
    assert(BuildTwoDaughterCandidate(
      MakeConstView(buffers.HostInputTracks()), fieldTask, referenceMother));

    KFParticleGpuTwoDaughterTask straightTask = fieldTask;
    straightTask.transportMode = KFGpuTransportStraightLine;
    KFParticleGpuFitState straightMother;
    assert(BuildTwoDaughterCandidate(
      MakeConstView(buffers.HostInputTracks()), straightTask, straightMother));

    // First isolate the exact task from compact task generation. A failure here
    // is a full-field device-math mismatch; a later executor-only failure is a
    // generated-task or compact-pool data-flow defect.
    buffers.UploadInput();
    xpu::buffer<KFParticleGpuTwoDaughterTask> directTaskBuffer(1u, xpu::buf_io);
    HostPointer(directTaskBuffer)[0] = fieldTask;
    runtime.GetQueue().copy(directTaskBuffer, xpu::h2d);
    runtime.GetQueue().wait();
    xpu::buffer<KFParticleGpuFullFieldTwoDaughterTrace> traceBuffer(1u, xpu::buf_io);
    runtime.GetQueue().launch<KFParticleGpuFullFieldTwoDaughterProbe>(
      xpu::n_threads(1),
      MakeConstView(buffers.DeviceInputTracks()),
      directTaskBuffer.get(),
      traceBuffer.get());
    runtime.GetQueue().copy(traceBuffer, xpu::d2h);
    runtime.GetQueue().wait();
    KFParticleGpuFullFieldTwoDaughterTrace hostTrace;
    assert(BuildHostFullFieldTrace(
      MakeConstView(buffers.HostInputTracks()), fieldTask, hostTrace));
    buffers.ResetCandidates();
    runtime.GetQueue().launch<KFParticleGpuTwoDaughterTaskKernel>(
      xpu::n_threads(1),
      MakeConstView(buffers.DeviceInputTracks()),
      directTaskBuffer.get(),
      1u,
      buffers.DeviceCandidates());
    runtime.GetQueue().wait();
    buffers.DownloadCandidates();
    const KFParticleGpuConstCandidatePoolView directCandidates =
      MakeConstView(buffers.HostCandidates());
    assert(directCandidates.Size() == 1u);
    KFParticleGpuFitState directMother;
    LoadCandidateFit(directCandidates, 0u, directMother);
    if (VerboseDiagnosticTraceEnabled()
        || NeedsFullFieldTrace(directMother, referenceMother, 310)) {
      DumpFullFieldTrace(HostPointer(traceBuffer)[0], hostTrace);
    }
    ExpectDefaultV0EnergyFitClose(directMother, referenceMother, 310);
    Pass("decay-plan-default-k0-direct-reference",
         "direct full-field K0S task matches the host before compact task generation");

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
    const unsigned int candidateIndex = FindCandidateByDaughters(candidates, 901, 1001);
    assert(candidateIndex < candidates.Size());
    assert(candidates.Metadata().Pdg(candidateIndex) == 310);
    assert(candidates.Metadata().Flags(candidateIndex) & KFGpuCandidateValid);
    assert(candidates.Metadata().Flags(candidateIndex) & KFGpuCandidateLineDca);
    assert(candidates.Metadata().Flags(candidateIndex) & KFGpuCandidateEnergyFit);

    KFParticleGpuFitState fieldMother;
    LoadCandidateFit(candidates, candidateIndex, fieldMother);
    ExpectDefaultV0EnergyFitClose(fieldMother, referenceMother, 310);
    assert(!AlmostEqual(fieldMother.X(), straightMother.X(), 1.e-6f));
    assert(!AlmostEqual(fieldMother.Px(), straightMother.Px(), 1.e-6f));

    plan.Clear();
    Pass("decay-plan-default-k0-field-fixture",
         "default K0S channel uses full-field transport on a nonzero-field fixture");
  }

  void TestDecayPlanDefaultLambdaFieldAwareFixture(KFParticleGpuRuntime& runtime,
                                                   KFParticleGpuBufferManager& buffers)
  {
    KFParticleGpuSteering& steering = runtime.GetSteering();
    KFParticleGpuDecayPlan& plan = steering.GetDecayPlan();
    plan.Clear();
    KFParticleGpuTwoDaughterChannel channel = MakeLambdaToProtonPiMinusChannel();
    assert(channel.transportMode == KFGpuTransportFullField);
    channel.maxDaughterDistance = -1.f;
    plan.AddTwoDaughterChannel(channel);

    FillDefaultV0Fixture(buffers);
    KFParticleGpuInputTrackSoAView tracks = buffers.HostInputTracks();
    // The proton is the first daughter for Lambda and therefore supplies the
    // deterministic nonzero-field transport used by this fixture.
    StoreDiagnosticField(tracks, 1);

    const KFParticleGpuTwoDaughterTaskSource source =
      MakeTwoDaughterTaskSource(channel, 0u);
    KFParticleGpuTwoDaughterTask fieldTask;
    FillTwoDaughterTask(KFParticleGpuRange(1, 1),
                         KFParticleGpuRange(2, 1),
                         0u,
                         source,
                         fieldTask);

    KFParticleGpuFitState referenceMother;
    assert(BuildTwoDaughterCandidate(
      MakeConstView(buffers.HostInputTracks()), fieldTask, referenceMother));

    KFParticleGpuTwoDaughterTask straightTask = fieldTask;
    straightTask.transportMode = KFGpuTransportStraightLine;
    KFParticleGpuFitState straightMother;
    assert(BuildTwoDaughterCandidate(
      MakeConstView(buffers.HostInputTracks()), straightTask, straightMother));

    KFParticleGpuFullFieldTwoDaughterTrace hostTrace;
    assert(BuildHostFullFieldTrace(
      MakeConstView(buffers.HostInputTracks()), fieldTask, hostTrace));
    KFParticleGpuFullFieldTwoDaughterTrace deviceTrace;
    const KFParticleGpuFitState directMother =
      RunDirectFullFieldTask(runtime, buffers, fieldTask, deviceTrace);
    if (VerboseDiagnosticTraceEnabled()
        || NeedsFullFieldTrace(directMother, referenceMother, 3122)) {
      DumpFullFieldTask("lambda-direct", fieldTask);
      DumpFullFieldTrace(deviceTrace, hostTrace);
    }
    ExpectDefaultV0EnergyFitClose(directMother, referenceMother, 3122);
    Pass("decay-plan-default-lambda-direct-reference",
         "direct full-field Lambda task matches the host before compact task generation");

    const std::vector<KFParticleGpuTwoDaughterChannelResult>& results =
      steering.RunDecayPlan(0u, 2u);
    assert(results.size() == 1u);
    assert(results[0].channelId == KFGpuChannelLambdaToProtonPiMinus);
    assert(results[0].motherPdg == 3122);
    assert(results[0].acceptedTasks == 2u);
    assert(results[0].candidates.size == 2u);

    const KFParticleGpuConstCandidatePoolView candidates =
      MakeConstView(buffers.HostCandidates());
    assert(candidates.Size() == 2u);
    const unsigned int candidateIndex =
      FindCandidateByDaughters(candidates, 3201, 3301);
    assert(candidateIndex < candidates.Size());
    assert(candidates.Metadata().Pdg(candidateIndex) == 3122);
    assert(candidates.Metadata().Flags(candidateIndex) & KFGpuCandidateValid);
    assert(HasCandidateDaughters(candidates, 3201, 3301));

    KFParticleGpuFitState fieldMother;
    LoadCandidateFit(candidates, candidateIndex, fieldMother);
    if (VerboseDiagnosticTraceEnabled()
        || NeedsFullFieldTrace(fieldMother, referenceMother, 3122)) {
      DumpFullFieldTask("lambda-decay-plan", fieldTask);
      DumpFullFieldTrace(deviceTrace, hostTrace);
      DumpFullFieldTraceState("decay-plan-vs-direct", fieldMother, directMother);
    }
    ExpectDefaultV0EnergyFitClose(fieldMother, referenceMother, 3122);
    assert(!AlmostEqual(fieldMother.X(), straightMother.X(), 1.e-6f));
    assert(!AlmostEqual(fieldMother.Px(), straightMother.Px(), 1.e-6f));

    plan.Clear();
    Pass("decay-plan-default-lambda-field-fixture",
         "default Lambda channel uses full-field proton-pion transport on a nonzero-field fixture");
  }

  void TestDecayPlanDefaultAntiLambdaFieldAwareFixture(KFParticleGpuRuntime& runtime,
                                                       KFParticleGpuBufferManager& buffers)
  {
    KFParticleGpuSteering& steering = runtime.GetSteering();
    KFParticleGpuDecayPlan& plan = steering.GetDecayPlan();
    plan.Clear();
    KFParticleGpuTwoDaughterChannel channel =
      MakeAntiLambdaToAntiProtonPiPlusChannel();
    assert(channel.transportMode == KFGpuTransportFullField);
    channel.maxDaughterDistance = -1.f;
    plan.AddTwoDaughterChannel(channel);

    FillDefaultV0Fixture(buffers);
    KFParticleGpuInputTrackSoAView tracks = buffers.HostInputTracks();
    // anti-Lambda reverses the track-set order: the anti-proton remains the
    // first daughter, now taken from the negative secondary track set.
    StoreDiagnosticField(tracks, 3);

    const KFParticleGpuTwoDaughterTaskSource source =
      MakeTwoDaughterTaskSource(channel, 0u);
    KFParticleGpuTwoDaughterTask fieldTask;
    FillTwoDaughterTask(KFParticleGpuRange(3, 1),
                         KFParticleGpuRange(0, 1),
                         0u,
                         source,
                         fieldTask);

    KFParticleGpuFitState referenceMother;
    assert(BuildTwoDaughterCandidate(
      MakeConstView(buffers.HostInputTracks()), fieldTask, referenceMother));

    KFParticleGpuTwoDaughterTask straightTask = fieldTask;
    straightTask.transportMode = KFGpuTransportStraightLine;
    KFParticleGpuFitState straightMother;
    assert(BuildTwoDaughterCandidate(
      MakeConstView(buffers.HostInputTracks()), straightTask, straightMother));

    KFParticleGpuFullFieldTwoDaughterTrace hostTrace;
    assert(BuildHostFullFieldTrace(
      MakeConstView(buffers.HostInputTracks()), fieldTask, hostTrace));
    KFParticleGpuFullFieldTwoDaughterTrace deviceTrace;
    const KFParticleGpuFitState directMother =
      RunDirectFullFieldTask(runtime, buffers, fieldTask, deviceTrace);
    if (VerboseDiagnosticTraceEnabled()
        || NeedsFullFieldTrace(directMother, referenceMother, -3122)) {
      DumpFullFieldTask("anti-lambda-direct", fieldTask);
      DumpFullFieldTrace(deviceTrace, hostTrace);
    }
    ExpectDefaultV0EnergyFitClose(directMother, referenceMother, -3122);
    Pass("decay-plan-default-anti-lambda-direct-reference",
         "direct full-field anti-Lambda task matches the host before compact task generation");

    const std::vector<KFParticleGpuTwoDaughterChannelResult>& results =
      steering.RunDecayPlan(0u, 1u);
    assert(results.size() == 1u);
    assert(results[0].channelId == KFGpuChannelAntiLambdaToAntiProtonPiPlus);
    assert(results[0].motherPdg == -3122);
    assert(results[0].acceptedTasks == 1u);
    assert(results[0].candidates.size == 1u);

    const KFParticleGpuConstCandidatePoolView candidates =
      MakeConstView(buffers.HostCandidates());
    assert(candidates.Size() == 1u);
    assert(candidates.Metadata().Pdg(0) == -3122);
    assert(candidates.Metadata().Flags(0) & KFGpuCandidateValid);
    assert(HasCandidateDaughters(candidates, 3401, 3101));

    KFParticleGpuFitState fieldMother;
    LoadCandidateFit(candidates, 0u, fieldMother);
    if (VerboseDiagnosticTraceEnabled()
        || NeedsFullFieldTrace(fieldMother, referenceMother, -3122)) {
      DumpFullFieldTask("anti-lambda-decay-plan", fieldTask);
      DumpFullFieldTrace(deviceTrace, hostTrace);
      DumpFullFieldTraceState("decay-plan-vs-direct", fieldMother, directMother);
    }
    ExpectDefaultV0EnergyFitClose(fieldMother, referenceMother, -3122);
    assert(!AlmostEqual(fieldMother.X(), straightMother.X(), 1.e-6f));
    assert(!AlmostEqual(fieldMother.Px(), straightMother.Px(), 1.e-6f));

    plan.Clear();
    Pass("decay-plan-default-anti-lambda-field-fixture",
         "default anti-Lambda channel uses full-field anti-proton-pion transport on a nonzero-field fixture");
  }

  void TestDefaultV0FullFieldDeviceReference(KFParticleGpuRuntime& runtime,
                                              KFParticleGpuBufferManager& buffers)
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
      tasks[i].flags = KFGpuTwoDaughterUseLineDca;

      assert(BuildTwoDaughterKinematicCandidate(inputTracks, tasks[i], references[i]));
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
      ExpectFieldAwareSeedClose(actual, references[i]);
      assert(candidates.Metadata().Pdg(i) == tasks[i].motherPdg);
      const unsigned int flags = candidates.Metadata().Flags(i);
      const unsigned int requiredFlags =
        static_cast<unsigned int>(KFGpuCandidateValid | KFGpuCandidateLineDca);
      assert((flags & requiredFlags) == requiredFlags);
      assert((flags & static_cast<unsigned int>(KFGpuCandidateEnergyFit)) == 0u);
      // The synthetic fixture is not tuned to pass each physical V0 cut.
      // Selection rejection is valid here; transport and stored fit remain
      // observable for the host/device full-field comparison.
      assert((flags & ~static_cast<unsigned int>(
                        requiredFlags | KFGpuCandidateSelectionRejected)) == 0u);
      assert(candidates.Daughters().SourceId(2u * i) == inputTracks.SourceId(tasks[i].firstTrack));
      assert(candidates.Daughters().SourceId(2u * i + 1u)
             == inputTracks.SourceId(tasks[i].secondTrack));
    }

    Pass("default-v0-full-field-device-reference",
         "GPU full-field K0S, Lambda, and anti-Lambda kinematic seeds match host construction");
  }

  void TestDefaultV0FieldAwareEnergyFit(KFParticleGpuRuntime& runtime,
                                        KFParticleGpuBufferManager& buffers)
  {
    FillDefaultV0Fixture(buffers);
    KFParticleGpuInputTrackSoAView tracks = buffers.HostInputTracks();
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
      FillTwoDaughterTask(
        event.TrackSet(channel.firstTrackSet).Species(
          channel.firstSpecies == NumberOfTrackSpecies ? Proton : channel.firstSpecies),
        event.TrackSet(channel.secondTrackSet).Species(channel.secondSpecies),
        0u,
        source,
        tasks[i]);
      assert((tasks[i].flags & KFGpuTwoDaughterUseLineDca) != 0u);
      assert((tasks[i].flags & KFGpuTwoDaughterUseEnergyFit) != 0u);
      assert(BuildTwoDaughterCandidate(inputTracks, tasks[i], references[i]));
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
      ExpectDefaultV0EnergyFitClose(actual, references[i], tasks[i].motherPdg);
      assert(candidates.Metadata().Pdg(i) == tasks[i].motherPdg);
      assert(candidates.Metadata().Flags(i)
             == (TwoDaughterCandidateFlags(tasks[i])
                 | (PassTwoDaughterPostBuildSelection(tasks[i], references[i])
                      ? 0u
                      : static_cast<unsigned int>(KFGpuCandidateSelectionRejected))));
      assert(candidates.Daughters().SourceId(2u * i) == inputTracks.SourceId(tasks[i].firstTrack));
      assert(candidates.Daughters().SourceId(2u * i + 1u)
             == inputTracks.SourceId(tasks[i].secondTrack));
    }

    Pass("default-v0-field-energy-fit",
         "GPU field-aware K0S, Lambda, and anti-Lambda energy fits match channel-specific state, chi2, and mass tolerances");
  }

  void ProfileDefaultV0FieldAwareEnergyFit(KFParticleGpuRuntime& runtime,
                                           KFParticleGpuBufferManager& buffers)
  {
    const unsigned int iterations = FieldAwareProfileIterations();
    if (iterations == 0u) {
      return;
    }

    FillDefaultV0Fixture(buffers);
    KFParticleGpuInputTrackSoAView tracks = buffers.HostInputTracks();
    StoreDiagnosticField(tracks, 0);
    StoreDiagnosticField(tracks, 1);
    StoreDiagnosticField(tracks, 3);

    const KFParticleGpuTwoDaughterChannel channels[] = {
      MakeK0ShortToPiPlusPiMinusChannel(),
      MakeLambdaToProtonPiMinusChannel(),
      MakeAntiLambdaToAntiProtonPiPlusChannel()
    };
    constexpr unsigned int numberOfChannels = sizeof(channels) / sizeof(channels[0]);
    KFParticleGpuTwoDaughterTask fieldTasks[numberOfChannels];
    KFParticleGpuTwoDaughterTask lineTasks[numberOfChannels];
    const KFParticleGpuEventDesc& event = buffers.HostEvents()[0];
    for (unsigned int i = 0; i < numberOfChannels; ++i) {
      const KFParticleGpuTwoDaughterChannel& channel = channels[i];
      FillTwoDaughterTask(
        event.TrackSet(channel.firstTrackSet).Species(
          channel.firstSpecies == NumberOfTrackSpecies ? Proton : channel.firstSpecies),
        event.TrackSet(channel.secondTrackSet).Species(channel.secondSpecies),
        0u,
        MakeTwoDaughterTaskSource(channel, 0u),
        fieldTasks[i]);
      lineTasks[i] = fieldTasks[i];
      lineTasks[i].transportMode = KFGpuTransportStraightLine;
    }

    xpu::buffer<KFParticleGpuTwoDaughterTask> taskBuffer(numberOfChannels, xpu::buf_io);
    KFParticleGpuTwoDaughterTask* hostTasks = HostPointer(taskBuffer);
    buffers.UploadInput();

    const auto runProfile = [&](const KFParticleGpuTwoDaughterTask* tasks) {
      for (unsigned int i = 0; i < numberOfChannels; ++i) {
        hostTasks[i] = tasks[i];
      }
      runtime.GetQueue().copy(taskBuffer, xpu::h2d);
      runtime.GetQueue().wait();

      // Warm-up makes image loading and first-use allocations irrelevant to
      // the reported steady-state queue submission and reconstruction cost.
      for (unsigned int warmup = 0; warmup < 8u; ++warmup) {
        buffers.ResetCandidates();
        runtime.GetQueue().launch<KFParticleGpuTwoDaughterTaskKernel>(
          xpu::n_threads(numberOfChannels),
          MakeConstView(buffers.DeviceInputTracks()),
          taskBuffer.get(),
          numberOfChannels,
          buffers.DeviceCandidates());
        runtime.GetQueue().wait();
      }

      const std::chrono::steady_clock::time_point started = std::chrono::steady_clock::now();
      for (unsigned int iteration = 0; iteration < iterations; ++iteration) {
        buffers.ResetCandidates();
        runtime.GetQueue().launch<KFParticleGpuTwoDaughterTaskKernel>(
          xpu::n_threads(numberOfChannels),
          MakeConstView(buffers.DeviceInputTracks()),
          taskBuffer.get(),
          numberOfChannels,
          buffers.DeviceCandidates());
        runtime.GetQueue().wait();
      }
      const std::chrono::steady_clock::time_point finished = std::chrono::steady_clock::now();
      return std::chrono::duration<double, std::micro>(finished - started).count()
             / static_cast<double>(iterations * numberOfChannels);
    };

    const double fieldMicrosecondsPerCandidate = runProfile(fieldTasks);
    const double lineMicrosecondsPerCandidate = runProfile(lineTasks);
    assert(fieldMicrosecondsPerCandidate > 0.0);
    assert(lineMicrosecondsPerCandidate > 0.0);
    std::cerr << "PROFILE field-aware-default-v0-energy-fit"
              << " iterations=" << iterations
              << " candidates-per-iteration=" << numberOfChannels
              << " field-us-per-candidate=" << fieldMicrosecondsPerCandidate
              << " line-us-per-candidate=" << lineMicrosecondsPerCandidate
              << " field-to-line=" << fieldMicrosecondsPerCandidate / lineMicrosecondsPerCandidate
              << '\n';
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
    ExpectDefaultV0ChannelRegression(candidates,
                                     results[0],
                                     plan.TwoDaughterChannel(0),
                                     KFGpuChannelK0ShortToPiPlusPiMinus,
                                     310,
                                     0u,
                                     0u);
    ExpectDefaultV0ChannelRegression(candidates,
                                     results[1],
                                     plan.TwoDaughterChannel(1),
                                     KFGpuChannelLambdaToProtonPiMinus,
                                     3122,
                                     1u,
                                     2u,
                                     2u);
    ExpectDefaultV0ChannelRegression(candidates,
                                     results[2],
                                     plan.TwoDaughterChannel(2),
                                     KFGpuChannelAntiLambdaToAntiProtonPiPlus,
                                     -3122,
                                     3u,
                                     6u);
    assert(HasCandidateDaughters(candidates, 3101, 3301));
    assert(HasCandidateDaughters(candidates, 3201, 3301));
    assert(HasCandidateDaughters(candidates, 3401, 3101));
    const KFParticleGpuConstSelectedCandidateIndexView selected =
      MakeConstView(buffers.HostSelectedCandidates());
    const KFParticleGpuSelectedCandidateRange selectedRange =
      steering.LastDecayPlanSelectedCandidates();
    assert(selectedRange.size == selected.Size());
    assert(selectedRange.overflowFlags == selected.OverflowFlags());

    plan.Clear();
    Pass("decay-plan-default-v0-executor",
         "default V0 regression validates K0S, Lambda, and anti-Lambda ranges, flags, and daughters");
  }

  void TestDecayPlanGenerationWidePipeline(KFParticleGpuRuntime& runtime,
                                           KFParticleGpuBufferManager& buffers)
  {
    KFParticleGpuSteering& steering = runtime.GetSteering();
    KFParticleGpuDecayPlan& plan = steering.GetDecayPlan();
    plan.Clear();
    AddDefaultV0ChannelsWithoutCpuPairGate(plan);
    FillDefaultV0Fixture(buffers);

    const std::vector<KFParticleGpuTwoDaughterChannelResult>& results =
      steering.RunDecayPlan(0u, 4u);
    const std::vector<KFParticleGpuDecayPlanEventResult>& events =
      steering.LastDecayPlanEventResults();
    const KFParticleGpuTwoDaughterRoutingMonitorData& monitoring =
      steering.LastTwoDaughterRoutingMonitorData();
    const KFParticleGpuConstCandidatePoolView candidates =
      MakeConstView(buffers.HostCandidates());
    const KFParticleGpuConstSelectedCandidateIndexView selected =
      MakeConstView(buffers.HostSelectedCandidates());

    assert(results.size() == 3u);
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

    plan.Clear();
    Pass("decay-plan-generation-wide-pipeline",
         "Stage 16.3 routes, constructs, selects, and summarizes all default V0 descriptors in one queued generation");
  }

  void TestDecayPlanDefaultV0SelectionRegression(KFParticleGpuRuntime& runtime,
                                                 KFParticleGpuBufferManager& buffers)
  {
    KFParticleGpuSteering& steering = runtime.GetSteering();
    KFParticleGpuDecayPlan& plan = steering.GetDecayPlan();
    plan.Clear();
    AddDefaultV0ChannelsWithoutCpuPairGate(plan);
    FillDefaultV0Fixture(buffers);

    const std::vector<KFParticleGpuTwoDaughterChannelResult>& results =
      steering.RunDecayPlan(0u, 4u);
    const KFParticleGpuConstCandidatePoolView candidates = MakeConstView(buffers.HostCandidates());
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

    const KFParticleGpuConstSelectedCandidateIndexView selected =
      MakeConstView(buffers.HostSelectedCandidates());
    const KFParticleGpuSelectedCandidateRange selectedRange =
      steering.LastDecayPlanSelectedCandidates();
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

    std::vector<KFParticleGpuTwoDaughterChannel> boundaryChannels;
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
      bool isBoundaryReference = false;
      for (const KFParticleGpuTwoDaughterChannelResult& result : boundaryResults) {
        if (result.channelId == channelId
            && candidateIndex
                 == FindChannelCandidate(candidates,
                                         result.generationCandidates,
                                         result.channelId)) {
          isBoundaryReference = true;
          break;
        }
      }
      if (isBoundaryReference) {
        assert(KFParticleGpuSelection::HasRejection(
          deviceSelections.Result(candidateIndex), KFGpuV0SelectionRejectDistance));
      }
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
    const std::vector<KFParticleGpuTwoDaughterChannelResult>& batch =
      steering.RunDecayPlanBatch(0u, 2u, 4u);
    const std::vector<KFParticleGpuDecayPlanEventResult>& events =
      steering.LastDecayPlanEventResults();

    assert(batch.size() == 4u);
    assert(events.size() == 2u);
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
    assert(performance.kernelLaunches > performance.descriptorGroups);
    assert(performance.queueWaits >= 7u);
    assert(performance.hostToDeviceBytes > 0u);
    assert(performance.deviceToHostBytes > 0u);
    assert(performance.allocatedBytesHighWater >= performance.hostToDeviceBytes);
    assert(performance.candidatePoolOccupancy > 0.);
    steering.SetPerformanceMonitoringEnabled(false);
    assert(!steering.LastPerformanceSnapshot().enabled);

    plan.Clear();
    Pass("decay-plan-batch-executor",
         "one upload executes two isolated event/channel plans with stable event ranges and serial-equivalent lineage");
    Pass("performance-monitoring-contract",
         "opt-in work, launch, wait, transfer, growth, and high-water counters reuse the completed transaction without changing default execution");
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
    assert(overflow[0].constructedCandidates == 4u);
    assert(overflow[0].constructedDaughters == 8u);
    assert(overflow[1].acceptedTasks == 4u);
    assert(overflow[1].HasOverflow(CandidateCapacityExceeded));
    assert(overflow[1].HasOverflow(DaughterCapacityExceeded));
    const KFParticleGpuConstCandidatePoolView overflowCandidates =
      MakeConstView(buffers.HostCandidates());
    assert(overflowCandidates.Size() == overflowCandidates.Capacity());
    assert(overflowCandidates.OverflowFlags()
           == static_cast<unsigned int>(CandidateCapacityExceeded | DaughterCapacityExceeded));

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
    requested.daughterIds = requested.daughterIds > 8u ? requested.daughterIds : 8u;
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
    buffers.SetInputSizes(1u, 1u, 1u);

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
    buffers.HostEvents()[0] = KFParticleGpuEventDesc();
    buffers.HostEvents()[0].eventId = 1730u;
    buffers.HostEvents()[0].primaryVertices = KFParticleGpuRange(0u, 1u);

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

    buffers.ResetCandidates();
    Pass("composite-neutral-final-graph-operations",
         "composite-composite, filtered missing mass, CPU-compatible kaon matching, PV projection, mass/topology selection, output classes, lineage/overflow rejection, and graph payloads agree on host and device");
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
    requested.candidates = requested.candidates > 64u ? requested.candidates : 64u;
    requested.daughterIds = requested.daughterIds > 160u ? requested.daughterIds : 160u;
    requested.selectedCandidates = requested.selectedCandidates > 12u
                                     ? requested.selectedCandidates : 12u;
    requested.twoDaughterTasks = requested.twoDaughterTasks > 8u
                                   ? requested.twoDaughterTasks : 8u;
    requested.v0TrackRoutedTasks = requested.v0TrackRoutedTasks > 32u
                                     ? requested.v0TrackRoutedTasks : 32u;
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

    const std::vector<KFParticleGpuTwoDaughterChannelResult>& v0Results =
      steering.RunDecayPlanBatch(0u, 2u, 8u);
    const std::vector<KFParticleGpuV0TrackChannelResult>& cascadeResults =
      steering.LastV0TrackCascadeResults();
    const std::vector<KFParticleGpuDecayPlanEventResult>& eventResults =
      steering.LastDecayPlanEventResults();
    assert(v0Results.size() == 6u);
    assert(cascadeResults.size() == 8u);
    assert(eventResults.size() == 2u);

    const unsigned int expectedPerChannel[4] = {10u, 0u, 5u, 0u};
    const KFParticleGpuConstCandidatePoolView candidates =
      MakeConstView(buffers.HostCandidates());
    for (unsigned int eventIndex = 0u; eventIndex < 2u; ++eventIndex) {
      const KFParticleGpuDecayPlanEventResult& event = eventResults[eventIndex];
      assert(event.eventIndex == eventIndex);
      assert(event.channelOffset == 3u * eventIndex);
      assert(event.channelCount == 3u);
      assert(event.candidates.size == 9u);
      assert(event.cascadeChannelOffset == 4u * eventIndex);
      assert(event.cascadeChannelCount == 4u);
      assert(event.cascadeCandidates.size == 15u);
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
    }
    assert(eventResults[0].cascadeCandidates.End()
           == eventResults[1].cascadeCandidates.offset);
    assert(candidates.Size() == 48u);
    assert(candidates.Daughters().Size() == 126u);

    const KFParticleGpuV0TrackRoutingMonitorData& monitoring =
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

    plan.Clear();
    Pass("decay-plan-v0-track-cascade-batch",
         "two-event production block-scan routing preserves event isolation, queued snapshots, channel counters, generation ranges, and aggregate monitoring");
  }

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
    void SetFieldCoeff(float, int) {}
    void AddDaughterId(int value) { daughters.push_back(value); }
  };

  void TestBoundedOutputMaterialization(KFParticleGpuBufferManager& buffers)
  {
    KFParticleGpuBufferCapacities requested = buffers.Capacities();
    requested.tracks = requested.tracks < 3u ? 3u : requested.tracks;
    requested.candidates = requested.candidates < 4u ? 4u : requested.candidates;
    requested.daughterIds = requested.daughterIds < 12u ? 12u : requested.daughterIds;
    buffers.EnsureCapacity(requested);
    buffers.SetInputSizes(3u, 0u, 1u);

    KFParticleGpuInputTrackSoAView tracks = buffers.HostInputTracks();
    StoreSyntheticTrack(tracks, 0u, 0.f, 0.f, 0.f, 0.3f, 0.1f, 0.5f, 211, 1, 101);
    StoreSyntheticTrack(tracks, 1u, 0.f, 0.f, 0.f, 0.2f, 0.2f, 0.4f, -211, -1, 102);
    StoreSyntheticTrack(tracks, 2u, 0.f, 0.f, 0.f, 0.1f, 0.3f, 0.6f, -211, -1, 103);

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
      candidates.Metadata(), 1u, 1u, KFGpuDirectDaughterInputTrack, 2u);

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
    const KFParticleGpuMaterializationRequest request{
      MakeConstView(tracks), MakeConstView(candidates), &range, 1u, 0u};
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

    candidates.Metadata().DirectFirstIndex(3u) = 3u;
    std::vector<MaterializedParticleProbe> unchanged(1u);
    unchanged[0].id = 91;
    KFParticleGpuMaterializationStatus status = KFGpuMaterializationSucceeded;
    assert(!KFParticleGpuMaterializer::CommitToParticles(
      request, unchanged, nullptr, &status));
    assert(status == KFGpuMaterializationCyclicAncestry);
    assert(unchanged.size() == 1u && unchanged[0].id == 91);
    candidates.Metadata().DirectFirstIndex(3u) = 2u;

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
    candidates.Daughters().SourceId(10u) = 103;
    const float savedParameter = candidates.Fit().Parameter(0u, 3u);
    candidates.Fit().Parameter(0u, 3u) =
      std::numeric_limits<float>::quiet_NaN();
    assert(KFParticleGpuMaterializer::Build(request).status
           == KFGpuMaterializationInvalidNumericalState);
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
      assert(root.gpuCandidateIndex == 3u);
      assert(root.channelId == node.channelId);
      assert(root.pdg == node.motherPdg);
      assert(root.topology == node.topology);
      assert(root.outputClass == node.outputClass);
    }
    Pass("bounded-output-materialization",
         "all 252 active channel identities plus V0/cascade/composite/unary ancestry, bounded failures, full fit state, and atomic commit");
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
    assert(buffers.Capacities().tracks == 12);
    assert(buffers.Capacities().nonhomogeneousField);
    Pass("capacity-policy", "buffers reuse smaller requests and grow monotonically without dropping field storage");
  }
}

int main()
{
  KFParticleGpuRuntime& runtime = KFParticleGpuRuntime::Instance();
  KFParticleGpuRuntimeSettings settings;
  settings.device = KFPARTICLE_GPU_TEST_DEVICE;
  settings.initializeXpuIfNeeded = true;
  settings.verbose = false;

  runtime.Initialize(settings);
  assert(runtime.IsInitialized());
  assert(runtime.InitializedXpu());
  xpu::queue* queue = &runtime.GetQueue();
  assert(queue == &runtime.GetQueue());
  Pass("runtime", "XPU runtime initialized once and exposes a stable process queue");

  xpu::preload<KFParticleGpuXpuBaselineImage>();

  xpu::buffer<unsigned int> baselineMarker(1, xpu::buf_device);
  unsigned int baselineMarkerHost = 0;
  runtime.GetQueue().memcpy(baselineMarker.get(), &baselineMarkerHost, sizeof(baselineMarkerHost));

  runtime.GetQueue().launch<KFParticleGpuXpuBaselineMarker>(
    xpu::n_threads(1), baselineMarker.get());
  runtime.GetQueue().wait();
  runtime.GetQueue().memcpy(&baselineMarkerHost, baselineMarker.get(), sizeof(baselineMarkerHost));
  assert(baselineMarkerHost == 0x58505542u);
  Pass("baseline-image", "independent test image kernel launches and writes a device marker");

  xpu::buffer<unsigned int> launchMarker(1, xpu::buf_device);
  unsigned int launchMarkerHost = 0;
  runtime.GetQueue().memcpy(launchMarker.get(), &launchMarkerHost, sizeof(launchMarkerHost));

  runtime.GetQueue().launch<KFParticleGpuLaunchSmoke>(
    xpu::n_threads(1), launchMarker.get());
  runtime.GetQueue().wait();
  runtime.GetQueue().memcpy(&launchMarkerHost, launchMarker.get(), sizeof(launchMarkerHost));
  assert(launchMarkerHost == 0x4b465047u);
  Pass("kfparticle-image", "KFParticle GPU image kernel launches and writes a device marker");

  KFParticleGpuBufferManager& buffers = runtime.GetSteering().GetBuffers();
  KFParticleGpuBufferCapacities capacities;
  capacities.tracks = 8;
  capacities.vertices = 3;
  capacities.events = 2;
  capacities.candidates = 6;
  capacities.daughterIds = 4;
  capacities.selectedCandidates = 1;
  capacities.nonhomogeneousField = true;
  buffers.EnsureCapacity(capacities);
  FillInput(buffers, capacities);
  Pass("input-packing", "host buffers allocated and filled with packed tracks, vertices, events, and field data");
  TestHostViews(buffers);
  TestDeviceInputProbe(runtime, buffers);
  TestCandidateReset(buffers);
  TestV0TrackRoutingPlan(buffers);
  TestTwoDaughterRoutingPlan(buffers);
  TestCpuFinderChannelCatalogue();
  TestCompleteCpuFinderManifest(buffers);
  TestDecayGraphContract(buffers);
  TestKernelStateContract(buffers);
  TestVisibleDeviceStorage(buffers);
  TestPublishedKernelState(runtime, buffers);
  TestParityAndPromotionContract();
  TestDecayPlanSelectedOutputTruncation(runtime, buffers);
  TestDecayPlanDataTypes();
  TestV0SelectionResultDataTypes(runtime);
  TestPrimaryVertexTopologyObservables();
  TestV0LineTopologyContract();
  TestV0SelectionDecision();
  TestV0LineTopologySelectionDecision();
  TestSelectedCandidateOutput(runtime, buffers);
  TestDecayPlanChannelList();
  TestDefaultV0DecayPlanBuilders();
  TestKinematicMathSeed();
  TestMeasurementSeed();
  TestFieldMeasurementSeedApprox();
  TestFieldEnergyFit();
  TestFieldTransportPrimitive();
  TestCpuCompatibleByDcaPrimitive();
  TestFullFieldTransportPrimitive();
  TestFullFieldCoupledDcaPrimitive();
  TestFullFieldCoupledEnergyFit();
  TestRealScaleCpuConstructV0Regression();
  TestCpuDerivativeDcaMiddlePointRegression();
  TestFieldDcaKinematicSeed();
  TestFieldDcaReferenceComparison();
  TestFieldTransportProbe(runtime);
  TestV0LineTopologyProbe(runtime);
  TestKalmanUpdateProbe(runtime);
  TestTwoDaughterTaskKernel(runtime, buffers);
  TestTwoDaughterTaskGeneration(runtime, buffers);
  TestCompactTwoDaughterTaskGeneration(runtime, buffers);
  TestCompactTwoDaughterCandidateConstruction(runtime, buffers);
  TestCompactTwoDaughterCandidateOverflow(runtime, buffers);
  TestGeneratedTwoDaughterPipeline(runtime, buffers);
  TestSteeringTwoDaughterStage(runtime, buffers);
  TestSteeringCompactTwoDaughterStage(runtime, buffers);
  TestSteeringCompactTwoDaughterEdgeCases(runtime, buffers);
  TestDecayPlanOneChannelExecutor(runtime, buffers);
  TestDecayPlanFieldAwareDiagnosticExecutor(runtime, buffers);
  TestDecayPlanMixedTransportExecutor(runtime, buffers);
  TestDecayPlanDefaultK0Executor(runtime, buffers);
  TestCpuFinderTwoDaughterPairGate(runtime, buffers);
  TestDecayPlanDefaultK0FieldAwareFixture(runtime, buffers);
  TestDecayPlanDefaultLambdaFieldAwareFixture(runtime, buffers);
  TestDecayPlanDefaultAntiLambdaFieldAwareFixture(runtime, buffers);
  TestDefaultV0FullFieldDeviceReference(runtime, buffers);
  TestDefaultV0FieldAwareEnergyFit(runtime, buffers);
  ProfileDefaultV0FieldAwareEnergyFit(runtime, buffers);
  TestDecayPlanDefaultV0Executor(runtime, buffers);
  TestDecayPlanGenerationWidePipeline(runtime, buffers);
  TestDecayPlanDefaultV0SelectionRegression(runtime, buffers);
  TestDecayPlanSelectionMultiPvBoundary(runtime, buffers);
  TestDecayPlanMultiChannelExecutor(runtime, buffers);
  TestDecayPlanExecutorEdgeCases(runtime, buffers);
  TestDecayPlanBatchExecutor(runtime, buffers);
  TestDecayPlanBatchSelectedOutput(runtime, buffers);
  TestTwoDaughterSyntheticGrid(runtime, buffers);
  TestSelectionHelpers(buffers);
  TestV0TrackTaskContract(runtime, buffers);
  TestV0TrackChannelBuilders();
  TestV0TrackCompactCandidateConstruction(runtime, buffers);
  TestFusedTwoDaughterRoutingAndConstruction(runtime, buffers);
  TestFusedV0TrackRoutingAndConstruction(runtime, buffers);
  TestGenericChargedDecayGraph(runtime, buffers);
  TestCompositeNeutralAndFinalGraphOperations(runtime, buffers);
  TestCpuFinderFinalSelectionGraph(runtime, buffers);
  TestFilteredMissingMassRawTrackGraph(runtime, buffers);
  TestDecayPlanV0TrackCascade(runtime, buffers);
  TestDecayPlanV0TrackCascadeBatch(runtime, buffers);
  TestBoundedOutputMaterialization(buffers);
  FillInput(buffers, capacities);
  TestRoundTrip(runtime.GetSteering());
  TestGuardsAndEmptyRun(runtime.GetSteering());
  TestCapacityPolicy(buffers);

  runtime.Finalize();
  assert(!runtime.IsInitialized());

  settings.initializeXpuIfNeeded = false;
  runtime.Initialize(settings);
  assert(runtime.IsInitialized());
  assert(!runtime.InitializedXpu());

  KFParticleGpuBufferManager& emptyBuffers = runtime.GetSteering().GetBuffers();
  assert(emptyBuffers.HostInputTracks().Numerical().ParametersData() == nullptr);
  assert(emptyBuffers.HostEvents() == nullptr);
  emptyBuffers.UploadInput();
  bool resetRejected = false;
  try {
    emptyBuffers.ResetCandidates();
  }
  catch (const std::logic_error&) {
    resetRejected = true;
  }
  assert(resetRejected);

  runtime.Finalize();
  Pass("reattach-finalize", "runtime finalizes cleanly and can reattach without owning XPU initialization");
  return 0;
}
