/*
 * This file is part of KFParticle package
 * Copyright (C) 2007-2026
 *
 * KFParticle is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 */

#ifndef KFPARTICLEGPUMATH_H
#define KFPARTICLEGPUMATH_H

#include "KFParticleGpuField.h"
#include "KFParticleGpuFitState.h"
#include "KFParticleGpuMeasurement.h"
#include "KFParticleGpuVertexState.h"

namespace KFParticleGpuMath
{
  struct KFParticleGpuAtan2ArithmeticTrace
  {
    float y;
    float x;
    float absY;
    float absX;
    float initialRatio;
    float transformedRatio;
    float ratio2;
    float polynomialStage[4];
    float result;
    unsigned int branchFlags;
  };

  struct KFParticleGpuSinCosArithmeticTrace
  {
    float value;
    float scaled;
    float rounded;
    float reduced;
    float reduced2;
    float sineSeries;
    float cosineSeries;
    float sine;
    float cosine;
    int quadrantCount;
    int quadrant;
  };

  KFPARTICLE_GPU_HOST_DEVICE inline float CpuCompatibleAtan2(
    float y,
    float x,
    KFParticleGpuAtan2ArithmeticTrace* trace = nullptr)
  {
#if defined(__clang__)
#pragma clang fp contract(off)
#endif
    const float pi = 3.1415926535897932f;
    const bool xZero = x == 0.f;
    const bool yZero = y == 0.f;
    const bool xNegative = x < 0.f;
    const bool yNegative = y < 0.f;
    const float absX = x < 0.f ? -x : x;
    const float absY = y < 0.f ? -y : y;

    float a = absY / absX;
    const bool aboveThreePiOverEight = a > 2.414213562373095f;
    const bool abovePiOverEight = a > 0.4142135623730950f && !aboveThreePiOverEight;
    float result = aboveThreePiOverEight ? pi / 2.f : 0.f;
    result = abovePiOverEight ? pi / 4.f : result;
    a = aboveThreePiOverEight ? -1.f / a : a;
    a = abovePiOverEight ? (absY - absX) / (absY + absX) : a;
    const float a2 = a * a;
    result += (((8.05374449538e-2f * a2 - 1.38776856032e-1f) * a2
                + 1.99777106478e-1f) * a2
               - 3.33329491539e-1f) * a2 * a + a;
    result = xNegative != yNegative ? -result : result;
    result = xNegative && !yNegative ? result + pi : result;
    result = xNegative && yNegative ? result - pi : result;
    result = xZero && yZero ? 0.f : result;
    result = xZero && yNegative ? -pi / 2.f : result;
    if (trace) {
      trace->y = y;
      trace->x = x;
      trace->absY = absY;
      trace->absX = absX;
      trace->initialRatio = absY / absX;
      trace->transformedRatio = a;
      trace->ratio2 = a2;
      trace->polynomialStage[0] = 8.05374449538e-2f * a2 - 1.38776856032e-1f;
      trace->polynomialStage[1] = trace->polynomialStage[0] * a2 + 1.99777106478e-1f;
      trace->polynomialStage[2] = trace->polynomialStage[1] * a2 - 3.33329491539e-1f;
      trace->polynomialStage[3] = trace->polynomialStage[2] * a2 * a + a;
      trace->result = result;
      trace->branchFlags = (xZero ? 1u : 0u) | (yZero ? 2u : 0u)
                           | (xNegative ? 4u : 0u) | (yNegative ? 8u : 0u)
                           | (aboveThreePiOverEight ? 16u : 0u)
                           | (abovePiOverEight ? 32u : 0u);
    }
    return result;
  }

  KFPARTICLE_GPU_HOST_DEVICE inline void CpuCompatibleSinCos(
    float value,
    float& sine,
    float& cosine,
    KFParticleGpuSinCosArithmeticTrace* trace = nullptr)
  {
#if defined(__clang__)
#pragma clang fp contract(off)
#endif
    const float halfPiInverse = 6.36619772e-1f;
    const int quadrantCount = static_cast<int>(KFParticleGpuRint(value * halfPiInverse));
    const int quadrant = quadrantCount & 3;
    const float quadrantFloat = static_cast<float>(quadrantCount);
    const float reduced = value - 1.5707969666f * quadrantFloat
                          + 6.3975784e-7f * quadrantFloat;
    const float reduced2 = reduced * reduced;
    float sineSeries = -1.984126984e-4f;
    sineSeries = sineSeries * reduced2 + 8.333333333e-3f;
    sineSeries = sineSeries * reduced2 - 1.666666667e-1f;
    sineSeries = sineSeries * (reduced2 * reduced) + reduced;
    float cosineSeries = 2.48015873e-5f;
    cosineSeries = cosineSeries * reduced2 - 1.388888889e-3f;
    cosineSeries = cosineSeries * reduced2 + 4.166666667e-2f;
    cosineSeries = cosineSeries * (reduced2 * reduced2) - 0.5f * reduced2 + 1.f;

    const bool evenQuadrant = quadrant == 0 || quadrant == 2;
    sine = evenQuadrant ? sineSeries : cosineSeries;
    cosine = evenQuadrant ? cosineSeries : sineSeries;
    if ((quadrant & 2) != 0) { sine = -sine; }
    if (((quadrant + 1) & 2) != 0) { cosine = -cosine; }
    if (trace) {
      trace->value = value;
      trace->scaled = value * halfPiInverse;
      trace->rounded = KFParticleGpuRint(value * halfPiInverse);
      trace->reduced = reduced;
      trace->reduced2 = reduced2;
      trace->sineSeries = sineSeries;
      trace->cosineSeries = cosineSeries;
      trace->sine = sine;
      trace->cosine = cosine;
      trace->quadrantCount = quadrantCount;
      trace->quadrant = quadrant;
    }
  }

  enum FullFieldTransportStatus : unsigned int
  {
    KFGpuFullFieldTransportSuccess = 0u,
    KFGpuFullFieldTransportNeutral = 1u << 0u,
    KFGpuFullFieldTransportPathLimited = 1u << 1u,
    KFGpuFullFieldTransportInvalidInput = 1u << 2u,
    KFGpuFullFieldTransportNonFiniteOutput = 1u << 3u
  };

  enum FullFieldDcaStatus : unsigned int
  {
    KFGpuFullFieldDcaSuccess = 0u,
    KFGpuFullFieldDcaNeutral = 1u << 0u,
    KFGpuFullFieldDcaRejected = 1u << 1u,
    KFGpuFullFieldDcaNonFinite = 1u << 2u
  };

  // Test-only arithmetic snapshot for the CPU-compatible two-track DCA.
  // Keeping every conditioning-sensitive scalar in one POD lets the HIP
  // lifecycle test locate the first backend divergence without device printf.
  struct KFParticleGpuDcaArithmeticTrace
  {
    float parameters[2][6];
    float bq[2];
    float pt2[2];
    float delta0[2];
    float dr02;
    float drp[2];
    float dxyp[2];
    float p1p2;
    float dp1p2;
    float k[4];
    float kp;
    float kd;
    float c[2];
    float discriminant;
    float root;
    float atanNumerator[2][2];
    float atanDenominator[2][2];
    KFParticleGpuAtan2ArithmeticTrace atan[2][2];
    float roots[2][2];
    float rootDistance2[2];
    float selectedDs[2];
    float bs[2];
    float sine[2];
    float cosine[2];
    KFParticleGpuSinCosArithmeticTrace sincos[2];
    float position[2][3];
    float momentum[2][3];
    float momentum2[2];
    float momentumDot;
    float separation[3];
    float separationDotMomentum[2];
    float determinantRaw;
    float determinantUsed;
    float correctionNumerator[2];
    float correction[2];
    float finalDs[2];
  };

  /**
   * Flat coupled DCA result for the two charged-daughter full-field route.
   *
   * The 6x12 Jacobians cover (x,y,z,px,py,pz) at the common DCA with
   * respect to the two independent six-parameter input states. The covariance
   * blocks assume no input cross-correlation, as do the scalar KFParticle
   * two-track construction inputs.
   */
  struct KFParticleGpuFullFieldDcaResult
  {
    KFParticleGpuFitState first;
    KFParticleGpuFitState second;
    float firstCovariance[21];
    float secondCovariance[21];
    float correlation[9];
    float firstJacobian[72];
    float secondJacobian[72];

    KFPARTICLE_GPU_HOST_DEVICE void Initialize()
    {
      first.Initialize();
      second.Initialize();
      for (int i = 0; i < 21; ++i) {
        firstCovariance[i] = 0.f;
        secondCovariance[i] = 0.f;
      }
      for (int i = 0; i < 9; ++i) {
        correlation[i] = 0.f;
      }
      for (int i = 0; i < 72; ++i) {
        firstJacobian[i] = 0.f;
        secondJacobian[i] = 0.f;
      }
    }

    KFPARTICLE_GPU_HOST_DEVICE float& FirstJacobian(int row, int column)
    {
      return firstJacobian[row * 12 + column];
    }
    KFPARTICLE_GPU_HOST_DEVICE const float& FirstJacobian(int row, int column) const
    {
      return firstJacobian[row * 12 + column];
    }
    KFPARTICLE_GPU_HOST_DEVICE float& SecondJacobian(int row, int column)
    {
      return secondJacobian[row * 12 + column];
    }
    KFPARTICLE_GPU_HOST_DEVICE const float& SecondJacobian(int row, int column) const
    {
      return secondJacobian[row * 12 + column];
    }
    KFPARTICLE_GPU_HOST_DEVICE float& FirstCovariance(int row, int column)
    {
      return firstCovariance[KFParticleGpuFitState::CovarianceIndex(row, column)];
    }
    KFPARTICLE_GPU_HOST_DEVICE const float& FirstCovariance(int row, int column) const
    {
      return firstCovariance[KFParticleGpuFitState::CovarianceIndex(row, column)];
    }
    KFPARTICLE_GPU_HOST_DEVICE float& SecondCovariance(int row, int column)
    {
      return secondCovariance[KFParticleGpuFitState::CovarianceIndex(row, column)];
    }
    KFPARTICLE_GPU_HOST_DEVICE const float& SecondCovariance(int row, int column) const
    {
      return secondCovariance[KFParticleGpuFitState::CovarianceIndex(row, column)];
    }
    KFPARTICLE_GPU_HOST_DEVICE float& Correlation(int row, int column)
    {
      return correlation[row * 3 + column];
    }
    KFPARTICLE_GPU_HOST_DEVICE const float& Correlation(int row, int column) const
    {
      return correlation[row * 3 + column];
    }
  };

  KFPARTICLE_GPU_HOST_DEVICE inline float Abs(float value)
  {
    return value >= 0.f ? value : -value;
  }

  KFPARTICLE_GPU_HOST_DEVICE inline bool IsFinite(float value)
  {
    // This form is available on every XPU backend and rejects NaN and infinities
    // without depending on backend-specific stdlib overloads.
    return value == value && value > -3.4e38f && value < 3.4e38f;
  }

  KFPARTICLE_GPU_HOST_DEVICE inline bool IsFiniteState(const KFParticleGpuFitState& state)
  {
    for (int i = 0; i < KFParticleGpuFitState::NumberOfParameters; ++i) {
      if (!IsFinite(state.Parameter(i))) {
        return false;
      }
    }
    for (int i = 0; i < KFParticleGpuFitState::NumberOfCovarianceElements; ++i) {
      if (!IsFinite(state.Covariance(i))) {
        return false;
      }
    }
    return IsFinite(state.Chi2()) && IsFinite(state.SFromDecay())
           && IsFinite(state.SumDaughterMass()) && IsFinite(state.MassHypo());
  }

  KFPARTICLE_GPU_HOST_DEVICE inline void InvertCholetsky3(float a[6])
  {
    float d[3];
    float u[3][3];

    for (int i = 0; i < 3; ++i) {
      d[i] = 0.f;
      for (int j = 0; j < 3; ++j) {
        u[i][j] = 0.f;
      }
    }

    for (int i = 0; i < 3; ++i) {
      float uud = 0.f;
      for (int j = 0; j < i; ++j) {
        uud += u[j][i] * u[j][i] * d[j];
      }

      uud = a[i * (i + 3) / 2] - uud;
      if (Abs(uud) < 1.e-12f) {
        uud = 1.e-12f;
      }

      d[i] = uud / Abs(uud);
      u[i][i] = KFParticleGpuSqrt(Abs(uud));

      for (int j = i + 1; j < 3; ++j) {
        uud = 0.f;
        for (int k = 0; k < i; ++k) {
          uud += u[k][i] * u[k][j] * d[k];
        }
        uud = a[j * (j + 1) / 2 + i] - uud;
        u[i][j] = d[i] / u[i][i] * uud;
      }
    }

    float u1[3];
    for (int i = 0; i < 3; ++i) {
      u1[i] = u[i][i];
      u[i][i] = 1.f / u[i][i];
    }

    for (int i = 0; i < 2; ++i) {
      u[i][i + 1] = -u[i][i + 1] * u[i][i] * u[i + 1][i + 1];
    }

    u[0][2] = u[0][1] * u1[1] * u[1][2] - u[0][2] * u[0][0] * u[2][2];

    for (int i = 0; i < 3; ++i) {
      a[i + 3] = u[i][2] * u[2][2] * d[2];
    }

    for (int i = 0; i < 2; ++i) {
      a[i + 1] = u[i][1] * u[1][1] * d[1] + u[i][2] * u[1][2] * d[2];
    }

    a[0] = u[0][0] * u[0][0] * d[0]
           + u[0][1] * u[0][1] * d[1]
           + u[0][2] * u[0][2] * d[2];
  }

  template<int N>
  KFPARTICLE_GPU_HOST_DEVICE inline void MultQSQt(const float Q[], const float S[], float SOut[])
  {
#if defined(__clang__)
#pragma clang fp contract(off)
#endif
    float mA[N * N];

    for (int i = 0, ij = 0; i < N; ++i) {
      for (int j = 0; j < N; ++j, ++ij) {
        mA[ij] = 0.f;
        for (int k = 0; k < N; ++k) {
          mA[ij] += S[KFParticleGpuFitState::CovarianceIndex(i, k)] * Q[j * N + k];
        }
      }
    }

    for (int i = 0; i < N; ++i) {
      for (int j = 0; j <= i; ++j) {
        const int ij = KFParticleGpuFitState::CovarianceIndex(i, j);
        SOut[ij] = 0.f;
        for (int k = 0; k < N; ++k) {
          SOut[ij] += Q[i * N + k] * mA[k * N + j];
        }
      }
    }
  }

  KFPARTICLE_GPU_HOST_DEVICE inline void BuildKinematicMother(const KFParticleGpuFitState& first,
                                                              const KFParticleGpuFitState& second,
                                                              KFParticleGpuFitState& mother);

  KFPARTICLE_GPU_HOST_DEVICE inline float Momentum2(const KFParticleGpuFitState& particle)
  {
    return particle.Px() * particle.Px()
           + particle.Py() * particle.Py()
           + particle.Pz() * particle.Pz();
  }

  KFPARTICLE_GPU_HOST_DEVICE inline float Mass2(const KFParticleGpuFitState& particle)
  {
    return particle.E() * particle.E() - Momentum2(particle);
  }

  KFPARTICLE_GPU_HOST_DEVICE inline float Mass(const KFParticleGpuFitState& particle)
  {
    const float mass2 = Mass2(particle);
    return mass2 > 0.f ? KFParticleGpuSqrt(mass2) : -KFParticleGpuSqrt(-mass2);
  }

  KFPARTICLE_GPU_HOST_DEVICE inline bool GetMass(const KFParticleGpuFitState& particle,
                                                 float& mass,
                                                 float& error)
  {
    const float big = 1.e8f;
    const float localSmall = 1.e-8f;
    const float px = particle.Px();
    const float py = particle.Py();
    const float pz = particle.Pz();
    const float energy = particle.E();

    const float mass2 = Mass2(particle);
    mass = mass2 >= 0.f ? KFParticleGpuSqrt(mass2) : -KFParticleGpuSqrt(-mass2);

    const float variance =
      px * px * particle.Covariance(9)
      + py * py * particle.Covariance(14)
      + pz * pz * particle.Covariance(20)
      + energy * energy * particle.Covariance(27)
      + 2.f * (px * py * particle.Covariance(13)
                + pz * (px * particle.Covariance(18) + py * particle.Covariance(19))
                - energy * (px * particle.Covariance(24)
                            + py * particle.Covariance(25)
                            + pz * particle.Covariance(26)));

    const bool valid = mass2 >= 0.f && variance >= 0.f && mass > localSmall;
    error = valid ? KFParticleGpuSqrt(variance) / mass : big;
    return valid;
  }

  KFPARTICLE_GPU_HOST_DEVICE inline float TransverseMomentum2(const KFParticleGpuFitState& particle)
  {
    return particle.Px() * particle.Px() + particle.Py() * particle.Py();
  }

  KFPARTICLE_GPU_HOST_DEVICE inline float GetDStoPointLine(const KFParticleGpuFitState& particle,
                                                           const float point[3],
                                                           float dsdr[6])
  {
    float momentum2 = Momentum2(particle);
    if (momentum2 < 1.e-4f) {
      momentum2 = 1.f;
    }

    const float dx = point[0] - particle.X();
    const float dy = point[1] - particle.Y();
    const float dz = point[2] - particle.Z();
    const float projection = particle.Px() * dx + particle.Py() * dy + particle.Pz() * dz;

    dsdr[0] = -particle.Px() / momentum2;
    dsdr[1] = -particle.Py() / momentum2;
    dsdr[2] = -particle.Pz() / momentum2;
    dsdr[3] = (dx * momentum2 - 2.f * particle.Px() * projection) / (momentum2 * momentum2);
    dsdr[4] = (dy * momentum2 - 2.f * particle.Py() * projection) / (momentum2 * momentum2);
    dsdr[5] = (dz * momentum2 - 2.f * particle.Pz() * projection) / (momentum2 * momentum2);
    return projection / momentum2;
  }

  KFPARTICLE_GPU_HOST_DEVICE inline void TransportLine(const KFParticleGpuFitState& particle,
                                                       float dS,
                                                       KFParticleGpuFitState& transported)
  {
    transported = particle;
    transported.X() = particle.X() + particle.Px() * dS;
    transported.Y() = particle.Y() + particle.Py() * dS;
    transported.Z() = particle.Z() + particle.Pz() * dS;
    // Scalar KFParticle transport keeps parameter 7 unchanged. The travelled
    // path is accumulated separately in SFromDecay.
    transported.S() = particle.S();
    transported.SFromDecay() = particle.SFromDecay() + dS;
  }

  KFPARTICLE_GPU_HOST_DEVICE inline void TransportConstantBy(
    const KFParticleGpuFitState& particle,
    float dS,
    float by,
    KFParticleGpuFitState& transported)
  {
    const float kCLight = 0.000299792458f;
    const float bq = by * static_cast<float>(particle.Q()) * kCLight;

    if (Abs(bq) < 1.e-8f) {
      TransportLine(particle, dS, transported);
      return;
    }

    const float bs = bq * dS;
    const float sinBs = KFParticleGpuSin(bs);
    const float cosBs = KFParticleGpuCos(bs);
    float sB = 0.f;
    float cB = 0.f;

    if (Abs(bs) > 1.e-8f) {
      sB = sinBs / bq;
      cB = (1.f - cosBs) / bq;
    }
    else {
      // Small-angle form keeps charged transport continuous with the line case.
      sB = dS;
      cB = 0.5f * dS * bs;
    }

    transported = particle;
    transported.X() = particle.X() + sB * particle.Px() - cB * particle.Pz();
    transported.Y() = particle.Y() + dS * particle.Py();
    transported.Z() = particle.Z() + cB * particle.Px() + sB * particle.Pz();
    transported.Px() = cosBs * particle.Px() - sinBs * particle.Pz();
    transported.Py() = particle.Py();
    transported.Pz() = sinBs * particle.Px() + cosBs * particle.Pz();
    transported.S() = particle.S();
    transported.SFromDecay() = particle.SFromDecay() + dS;
  }

  KFPARTICLE_GPU_HOST_DEVICE inline void TransportLineWithJacobian(
    const KFParticleGpuFitState& particle,
    float dS,
    const float dsdr[6],
    KFParticleGpuFitState& transported,
    const float dsdrOther[6],
    float* selfDerivative,
    float* otherDerivative);

  /**
   * CBM-style fixed-order transport through the packed parabolic field region.
   *
   * The implementation mirrors the CPU TransportCBM three-point field
   * approximation. It stays a value-only device primitive: no allocation,
   * exceptions, or mutable global state are involved.
   */
  KFPARTICLE_GPU_HOST_DEVICE inline bool TransportFullField(
    const KFParticleGpuFitState& particle,
    const KFParticleGpuFieldRegion& field,
    float dS,
    KFParticleGpuFitState& transported,
    unsigned int& status,
    const float* dsdr = nullptr,
    const float* dsdrOther = nullptr,
    float* selfDerivative = nullptr,
    float* otherDerivative = nullptr)
  {
#if defined(__clang__)
#pragma clang fp contract(off)
#endif
    status = KFGpuFullFieldTransportSuccess;
    transported = particle;

    if (!IsFiniteState(particle) || !IsFinite(dS)) {
      status = KFGpuFullFieldTransportInvalidInput;
      return false;
    }
    for (int i = 0; i < KFParticleGpuFieldRegion::NumberOfCoefficients; ++i) {
      if (!IsFinite(field.Coefficient(i))) {
        status = KFGpuFullFieldTransportInvalidInput;
        return false;
      }
    }

    float localDsdr[6] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    if (dsdr) {
      for (int i = 0; i < 6; ++i) {
        if (!IsFinite(dsdr[i])) {
          status = KFGpuFullFieldTransportInvalidInput;
          return false;
        }
        localDsdr[i] = dsdr[i];
      }
    }

    if (particle.Q() == 0) {
      TransportLineWithJacobian(
        particle,
        dS,
        localDsdr,
        transported,
        dsdrOther,
        selfDerivative,
        otherDerivative);
      status = KFGpuFullFieldTransportNeutral;
      return true;
    }

    if (Abs(dS * particle.Pz()) > 1000.f) {
      status = KFGpuFullFieldTransportPathLimited;
      return true;
    }

    const float chargeLight = static_cast<float>(particle.Q()) * 0.000299792458f;
    const float px = particle.Px();
    const float py = particle.Py();
    const float pz = particle.Pz();
    const float z0 = particle.Z();

    float z1 = z0 + 0.5f * pz * dS;
    float z2 = z0 + pz * dS;

    const KFParticleGpuFieldValue field0 = field.Get(z0);
    const KFParticleGpuFieldValue field1Line = field.Get(z1);
    const KFParticleGpuFieldValue field2Line = field.Get(z2);
    const float ssy1 = (7.f * field0.y + 6.f * field1Line.y - field2Line.y)
                       * chargeLight * dS * dS / 96.f;
    const float ssy2 = (field0.y + 2.f * field1Line.y) * chargeLight * dS * dS / 6.f;
    z1 += ssy1 * px;
    z2 += ssy2 * px;

    const KFParticleGpuFieldValue fields[3] = {field0, field.Get(z1), field.Get(z2)};
    const float sx = chargeLight * (fields[0].x + 4.f * fields[1].x + fields[2].x) * dS / 6.f;
    const float sy = chargeLight * (fields[0].y + 4.f * fields[1].y + fields[2].y) * dS / 6.f;
    const float sz = chargeLight * (fields[0].z + 4.f * fields[1].z + fields[2].z) * dS / 6.f;
    const float ssx = chargeLight * (fields[0].x + 2.f * fields[1].x) * dS * dS / 6.f;
    const float ssy = chargeLight * (fields[0].y + 2.f * fields[1].y) * dS * dS / 6.f;
    const float ssz = chargeLight * (fields[0].z + 2.f * fields[1].z) * dS * dS / 6.f;

    float syz = 0.f;
    float ssyz = 0.f;
    const float c2[3][3] = {{5.f, -4.f, -1.f}, {44.f, 80.f, -4.f}, {11.f, 44.f, 5.f}};
    const float cc2[3][3] = {{38.f, 8.f, -4.f}, {148.f, 208.f, -20.f}, {3.f, 36.f, 3.f}};
    for (int n = 0; n < 3; ++n) {
      for (int m = 0; m < 3; ++m) {
        syz += c2[n][m] * fields[n].y * fields[m].z;
        ssyz += cc2[n][m] * fields[n].y * fields[m].z;
      }
    }
    syz *= chargeLight * chargeLight * dS * dS / 360.f;
    ssyz *= chargeLight * chargeLight * dS * dS * dS / 2520.f;

    float syy = chargeLight * (fields[0].y + 4.f * fields[1].y + fields[2].y) * dS;
    const float syyy = syy * syy * syy / 1296.f;
    syy = syy * syy / 72.f;
    const float ssyy =
      (fields[0].y * (38.f * fields[0].y + 156.f * fields[1].y - fields[2].y)
       + fields[1].y * (208.f * fields[1].y + 16.f * fields[2].y)
       + 3.f * fields[2].y * fields[2].y)
      * dS * dS * dS * chargeLight * chargeLight / 2520.f;
    const float ssyyy =
      (fields[0].y * (fields[0].y * (85.f * fields[0].y + 526.f * fields[1].y
                                      - 7.f * fields[2].y)
                       + fields[1].y * (1376.f * fields[1].y + 84.f * fields[2].y)
                       + 19.f * fields[2].y * fields[2].y)
       + fields[1].y * (fields[1].y * (1376.f * fields[1].y + 256.f * fields[2].y)
                         + 62.f * fields[2].y * fields[2].y)
       + 3.f * fields[2].y * fields[2].y * fields[2].y)
      * dS * dS * dS * dS * chargeLight * chargeLight * chargeLight / 90720.f;

    float jacobian[8][8] = {};
    jacobian[0][0] = jacobian[1][1] = jacobian[2][2] = 1.f;
    jacobian[6][6] = jacobian[7][7] = 1.f;
    jacobian[0][3] = dS - ssyy;
    jacobian[0][4] = ssx;
    jacobian[0][5] = ssyyy - ssy;
    jacobian[1][3] = -ssz;
    jacobian[1][4] = dS;
    jacobian[1][5] = ssx + ssyz;
    jacobian[2][3] = ssy - ssyyy;
    jacobian[2][4] = -ssx;
    jacobian[2][5] = dS - ssyy;
    jacobian[3][3] = 1.f - syy;
    jacobian[3][4] = sx;
    jacobian[3][5] = syyy - sy;
    jacobian[4][3] = -sz;
    jacobian[4][4] = 1.f;
    jacobian[4][5] = sx + syz;
    jacobian[5][3] = sy - syyy;
    jacobian[5][4] = -sx;
    jacobian[5][5] = 1.f - syy;

    // Parameters depend on dS, but not on the derivatives of dS. CPU
    // TransportCBM evaluates them before augmenting the covariance Jacobian.
    transported.X() = particle.X() + jacobian[0][3] * px + jacobian[0][4] * py + jacobian[0][5] * pz;
    transported.Y() = particle.Y() + jacobian[1][3] * px + jacobian[1][4] * py + jacobian[1][5] * pz;
    transported.Z() = z0 + jacobian[2][3] * px + jacobian[2][4] * py + jacobian[2][5] * pz;
    transported.Px() = jacobian[3][3] * px + jacobian[3][4] * py + jacobian[3][5] * pz;
    transported.Py() = jacobian[4][3] * px + jacobian[4][4] * py + jacobian[4][5] * pz;
    transported.Pz() = jacobian[5][3] * px + jacobian[5][4] * py + jacobian[5][5] * pz;
    transported.S() = particle.S();
    transported.SFromDecay() = particle.SFromDecay() + dS;

    float pathDerivative[6] = {};
    if (Abs(dS) > 0.f) {
      float derivative[6][6] = {};
      derivative[0][3] = 1.f - 3.f * ssyy / dS;
      derivative[0][4] = 2.f * ssx / dS;
      derivative[0][5] = (4.f * ssyyy - 2.f * ssy) / dS;
      derivative[1][3] = -2.f * ssz / dS;
      derivative[1][4] = 1.f;
      derivative[1][5] = (2.f * ssx + 3.f * ssyz) / dS;
      derivative[2][3] = (2.f * ssy - 4.f * ssyyy) / dS;
      derivative[2][4] = -2.f * ssx / dS;
      derivative[2][5] = 1.f - 3.f * ssyy / dS;
      derivative[3][3] = -2.f * syy / dS;
      derivative[3][4] = sx / dS;
      derivative[3][5] = (3.f * syyy - sy) / dS;
      derivative[4][3] = -sz / dS;
      derivative[4][5] = (sx + 2.f * syz) / dS;
      derivative[5][3] = (sy - 3.f * syyy) / dS;
      derivative[5][4] = -sx / dS;
      derivative[5][5] = -2.f * syy / dS;
      for (int row = 0; row < 6; ++row) {
        const float transportDerivative = derivative[row][3] * px
                                          + derivative[row][4] * py
                                          + derivative[row][5] * pz;
        pathDerivative[row] = transportDerivative;
        for (int column = 0; column < 6; ++column) {
          jacobian[row][column] += transportDerivative * localDsdr[column];
        }
      }
    }

    if (selfDerivative) {
      for (int row = 0; row < 6; ++row) {
        for (int column = 0; column < 6; ++column) {
          selfDerivative[row * 6 + column] = jacobian[row][column];
        }
      }
    }
    if (otherDerivative && dsdrOther) {
      for (int row = 0; row < 6; ++row) {
        for (int column = 0; column < 6; ++column) {
          otherDerivative[row * 6 + column] = pathDerivative[row] * dsdrOther[column];
        }
      }
    }

    float covariance[KFParticleGpuFitState::NumberOfCovarianceElements];
    MultQSQt<8>(jacobian[0], particle.Covariances(), covariance);
    for (int i = 0; i < KFParticleGpuFitState::NumberOfCovarianceElements; ++i) {
      transported.Covariance(i) = covariance[i];
    }
    if (!IsFiniteState(transported)) {
      transported = particle;
      status = KFGpuFullFieldTransportNonFiniteOutput;
      return false;
    }
    return true;
  }

  KFPARTICLE_GPU_HOST_DEVICE inline void TransportLineWithJacobian(
    const KFParticleGpuFitState& particle,
    float dS,
    const float dsdr[6],
    KFParticleGpuFitState& transported,
    const float dsdrOther[6] = nullptr,
    float* selfDerivative = nullptr,
    float* otherDerivative = nullptr)
  {
    float jacobian[8][8];
    for (int i = 0; i < 8; ++i) {
      for (int j = 0; j < 8; ++j) {
        jacobian[i][j] = 0.f;
      }
    }

    for (int i = 0; i < 8; ++i) {
      jacobian[i][i] = 1.f;
    }
    jacobian[0][3] = dS;
    jacobian[1][4] = dS;
    jacobian[2][5] = dS;

    const float momentum[3] = {particle.Px(), particle.Py(), particle.Pz()};
    for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 6; ++j) {
        jacobian[i][j] += momentum[i] * dsdr[j];
      }
    }

    transported = particle;
    transported.X() = particle.X() + dS * particle.Px();
    transported.Y() = particle.Y() + dS * particle.Py();
    transported.Z() = particle.Z() + dS * particle.Pz();
    transported.S() = particle.S();
    transported.SFromDecay() = particle.SFromDecay() + dS;

    float transportedCovariance[36];
    MultQSQt<8>(jacobian[0], particle.Covariances(), transportedCovariance);
    for (int i = 0; i < KFParticleGpuFitState::NumberOfCovarianceElements; ++i) {
      transported.Covariance(i) = transportedCovariance[i];
    }

    if (selfDerivative) {
      for (int i = 0; i < 6; ++i) {
        for (int j = 0; j < 6; ++j) {
          selfDerivative[i * 6 + j] = jacobian[i][j];
        }
      }
    }

    if (otherDerivative && dsdrOther) {
      for (int i = 0; i < 6; ++i) {
        for (int j = 0; j < 6; ++j) {
          otherDerivative[i * 6 + j] = i < 3 ? momentum[i] * dsdrOther[j] : 0.f;
        }
      }
    }
  }

  KFPARTICLE_GPU_HOST_DEVICE inline void GetDStoParticleLine(const KFParticleGpuFitState& first,
                                                             const KFParticleGpuFitState& second,
                                                             float dS[2])
  {
    const float dx = first.X() - second.X();
    const float dy = first.Y() - second.Y();
    const float dz = first.Z() - second.Z();
    const float p1p1 = Momentum2(first);
    const float p2p2 = Momentum2(second);
    const float p1p2 = first.Px() * second.Px() + first.Py() * second.Py() + first.Pz() * second.Pz();
    const float r1p1 = dx * first.Px() + dy * first.Py() + dz * first.Pz();
    const float r1p2 = dx * second.Px() + dy * second.Py() + dz * second.Pz();
    float determinant = p1p2 * p1p2 - p1p1 * p2p2;

    if (determinant > -1.e-8f && determinant < 1.e-8f) {
      determinant = determinant < 0.f ? -1.e-8f : 1.e-8f;
    }

    dS[0] = (r1p1 * p2p2 - r1p2 * p1p2) / determinant;
    dS[1] = (r1p1 * p1p2 - r1p2 * p1p1) / determinant;
  }

  KFPARTICLE_GPU_HOST_DEVICE inline void GetDStoParticleLine(const KFParticleGpuFitState& first,
                                                             const KFParticleGpuFitState& second,
                                                             float dS[2],
                                                             float dsdr[4][6])
  {
    const float p12 = Momentum2(first);
    const float p22 = Momentum2(second);
    const float p1p2 = first.Px() * second.Px() + first.Py() * second.Py() + first.Pz() * second.Pz();

    const float drp1 =
      first.Px() * (second.X() - first.X())
      + first.Py() * (second.Y() - first.Y())
      + first.Pz() * (second.Z() - first.Z());
    const float drp2 =
      second.Px() * (second.X() - first.X())
      + second.Py() * (second.Y() - first.Y())
      + second.Pz() * (second.Z() - first.Z());

    float detp = p1p2 * p1p2 - p12 * p22;
    if (Abs(detp) < 1.e-4f) {
      detp = 1.f;
    }

    dS[0] = (drp2 * p1p2 - drp1 * p22) / detp;
    dS[1] = (drp2 * p12 - drp1 * p1p2) / detp;

    const float dx = second.X() - first.X();
    const float dy = second.Y() - first.Y();
    const float dz = second.Z() - first.Z();
    const float drp1Dr1[6] = {-first.Px(), -first.Py(), -first.Pz(), dx, dy, dz};
    const float drp1Dr2[6] = {first.Px(), first.Py(), first.Pz(), 0.f, 0.f, 0.f};
    const float drp2Dr1[6] = {-second.Px(), -second.Py(), -second.Pz(), 0.f, 0.f, 0.f};
    const float drp2Dr2[6] = {second.Px(), second.Py(), second.Pz(), dx, dy, dz};
    const float dp1p2Dr1[6] = {0.f, 0.f, 0.f, second.Px(), second.Py(), second.Pz()};
    const float dp1p2Dr2[6] = {0.f, 0.f, 0.f, first.Px(), first.Py(), first.Pz()};
    const float dp12Dr1[6] = {0.f, 0.f, 0.f, 2.f * first.Px(), 2.f * first.Py(), 2.f * first.Pz()};
    const float dp12Dr2[6] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    const float dp22Dr1[6] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    const float dp22Dr2[6] = {0.f, 0.f, 0.f, 2.f * second.Px(), 2.f * second.Py(), 2.f * second.Pz()};
    const float ddetpDr1[6] = {
      0.f, 0.f, 0.f,
      -2.f * p22 * first.Px() + 2.f * p1p2 * second.Px(),
      -2.f * p22 * first.Py() + 2.f * p1p2 * second.Py(),
      -2.f * p22 * first.Pz() + 2.f * p1p2 * second.Pz()};
    const float ddetpDr2[6] = {
      0.f, 0.f, 0.f,
      2.f * p1p2 * first.Px() - 2.f * p12 * second.Px(),
      2.f * p1p2 * first.Py() - 2.f * p12 * second.Py(),
      2.f * p1p2 * first.Pz() - 2.f * p12 * second.Pz()};

    const float a1 = drp2 * p1p2 - drp1 * p22;
    const float a2 = drp2 * p12 - drp1 * p1p2;
    for (int i = 0; i < 6; ++i) {
      const float da1Dr1 = drp2Dr1[i] * p1p2 + drp2 * dp1p2Dr1[i]
                           - drp1Dr1[i] * p22 - drp1 * dp22Dr1[i];
      const float da1Dr2 = drp2Dr2[i] * p1p2 + drp2 * dp1p2Dr2[i]
                           - drp1Dr2[i] * p22 - drp1 * dp22Dr2[i];
      const float da2Dr1 = drp2Dr1[i] * p12 + drp2 * dp12Dr1[i]
                           - drp1Dr1[i] * p1p2 - drp1 * dp1p2Dr1[i];
      const float da2Dr2 = drp2Dr2[i] * p12 + drp2 * dp12Dr2[i]
                           - drp1Dr2[i] * p1p2 - drp1 * dp1p2Dr2[i];

      dsdr[0][i] = da1Dr1 / detp - a1 / (detp * detp) * ddetpDr1[i];
      dsdr[1][i] = da1Dr2 / detp - a1 / (detp * detp) * ddetpDr2[i];
      dsdr[2][i] = da2Dr1 / detp - a2 / (detp * detp) * ddetpDr1[i];
      dsdr[3][i] = da2Dr2 / detp - a2 / (detp * detp) * ddetpDr2[i];
    }
  }

  KFPARTICLE_GPU_HOST_DEVICE inline void BuildLineDcaKinematicMother(
    const KFParticleGpuFitState& first,
    const KFParticleGpuFitState& second,
    KFParticleGpuFitState& mother)
  {
    float dS[2] = {0.f, 0.f};
    GetDStoParticleLine(first, second, dS);

    KFParticleGpuFitState firstAtDca;
    KFParticleGpuFitState secondAtDca;
    TransportLine(first, dS[0], firstAtDca);
    TransportLine(second, dS[1], secondAtDca);
    BuildKinematicMother(firstAtDca, secondAtDca, mother);
  }

  KFPARTICLE_GPU_HOST_DEVICE inline void GetDStoParticleBzCpuCompatibleDerivatives(const KFParticleGpuFitState& particle1,
                                      float Bz,
                                      const KFParticleGpuFitState& particle2,
                                      float dS[2],
                                      float dsdr[4][6],
                                      const float* param1,
                                      const float* param2, bool useMiddlePoint,
                                      KFParticleGpuDcaArithmeticTrace* arithmeticTrace = nullptr)
  {
#if defined(__clang__)
#pragma clang fp contract(off)
#endif
    if (!param1) {
      param1 = particle1.Parameters();
      param2 = particle2.Parameters();
    }

    const float kOvSqr6 = float(1) / KFParticleGpuSqrt(float(6));
    const float kCLight = float(0.000299792458);

    const float bq1 = Bz * particle1.Q() * kCLight;
    const float bq2 = Bz * particle2.Q() * kCLight;

    const bool isStraight1 = Abs(bq1) < float(1.e-8);
    const bool isStraight2 = Abs(bq2) < float(1.e-8);

    if (isStraight1 && isStraight2) {
      GetDStoParticleLine(particle1, particle2, dS, dsdr);
      return;
    }

    const float px1 = param1[3];
    const float py1 = param1[4];
    const float pz1 = param1[5];

    const float px2 = param2[3];
    const float py2 = param2[4];
    const float pz2 = param2[5];

    const float pt12 = px1 * px1 + py1 * py1;
    const float pt22 = px2 * px2 + py2 * py2;

    const float x01 = param1[0];
    const float y01 = param1[1];
    const float z01 = param1[2];

    const float x02 = param2[0];
    const float y02 = param2[1];
    const float z02 = param2[2];

    float dS1[2] = {float(0), float(0)};
    float dS2[2] = {float(0), float(0)};

    const float dx0 = x01 - x02;
    const float dy0 = y01 - y02;
    const float dr02 = dx0 * dx0 + dy0 * dy0;
    const float drp1 = dx0 * px1 + dy0 * py1;
    const float dxyp1 = dx0 * py1 - dy0 * px1;
    const float drp2 = dx0 * px2 + dy0 * py2;
    const float dxyp2 = dx0 * py2 - dy0 * px2;
    const float p1p2 = px1 * px2 + py1 * py2;
    const float dp1p2 = px1 * py2 - px2 * py1;

    const float k11 = bq2 * drp1 - dp1p2;
    const float k21 = bq1 * (bq2 * dxyp1 - p1p2) + bq2 * pt12;
    const float k12 = bq1 * drp2 - dp1p2;
    const float k22 = bq2 * (bq1 * dxyp2 + p1p2) - bq1 * pt22;

    const float kp = dxyp1 * bq2 - dxyp2 * bq1 - p1p2;
    const float kd = dr02 / float(2) * bq1 * bq2 + kp;
    const float c1 = -(bq1 * kd + pt12 * bq2);
    const float c2 = bq2 * kd + pt22 * bq1;

    const float discriminant = pt12 * pt22 - kd * kd;
    float d1 = discriminant;
    if (d1 < float(0)) {
      d1 = float(0);
    }
    d1 = KFParticleGpuSqrt(d1);

    float d2 = pt12 * pt22 - kd * kd;
    if (d2 < float(0)) {
      d2 = float(0);
    }
    d2 = KFParticleGpuSqrt(d2);

    if (arithmeticTrace) {
      for (int parameter = 0; parameter < 6; ++parameter) {
        arithmeticTrace->parameters[0][parameter] = param1[parameter];
        arithmeticTrace->parameters[1][parameter] = param2[parameter];
      }
      arithmeticTrace->bq[0] = bq1;
      arithmeticTrace->bq[1] = bq2;
      arithmeticTrace->pt2[0] = pt12;
      arithmeticTrace->pt2[1] = pt22;
      arithmeticTrace->delta0[0] = dx0;
      arithmeticTrace->delta0[1] = dy0;
      arithmeticTrace->dr02 = dr02;
      arithmeticTrace->drp[0] = drp1;
      arithmeticTrace->drp[1] = drp2;
      arithmeticTrace->dxyp[0] = dxyp1;
      arithmeticTrace->dxyp[1] = dxyp2;
      arithmeticTrace->p1p2 = p1p2;
      arithmeticTrace->dp1p2 = dp1p2;
      arithmeticTrace->k[0] = k11;
      arithmeticTrace->k[1] = k21;
      arithmeticTrace->k[2] = k12;
      arithmeticTrace->k[3] = k22;
      arithmeticTrace->kp = kp;
      arithmeticTrace->kd = kd;
      arithmeticTrace->c[0] = c1;
      arithmeticTrace->c[1] = c2;
      arithmeticTrace->discriminant = discriminant;
      arithmeticTrace->root = d1;
    }

    float dS1dR1[2][6] = {};
    float dS2dR2[2][6] = {};
    float dS1dR2[2][6] = {};
    float dS2dR1[2][6] = {};

    float dk11dr1[6] = {bq2 * px1, bq2 * py1, float(0), bq2 * dx0 - py2, bq2 * dy0 + px2, float(0)};
    float dk11dr2[6] = {-bq2 * px1, -bq2 * py1, float(0), py1, -px1, float(0)};
    float dk12dr1[6] = {bq1 * px2, bq1 * py2, float(0), -py2, px2, float(0)};
    float dk12dr2[6] = {-bq1 * px2, -bq1 * py2, float(0), bq1 * dx0 + py1, bq1 * dy0 - px1, float(0)};
    float dk21dr1[6] = {bq1 * bq2 * py1, -bq1 * bq2 * px1, float(0),
                    float(2) * bq2 * px1 + bq1 * (-(bq2 * dy0) - px2),
                    float(2) * bq2 * py1 + bq1 * (bq2 * dx0 - py2), float(0)};
    float dk21dr2[6] = {-(bq1 * bq2 * py1), bq1 * bq2 * px1, float(0), -(bq1 * px1), -(bq1 * py1), float(0)};
    float dk22dr1[6] = {bq1 * bq2 * py2, -(bq1 * bq2 * px2), float(0), bq2 * px2, bq2 * py2, float(0)};
    float dk22dr2[6] = {-(bq1 * bq2 * py2), bq1 * bq2 * px2, float(0),
                    bq2 * (-(bq1 * dy0) + px1) - float(2) * bq1 * px2,
                    bq2 * (bq1 * dx0 + py1) - float(2) * bq1 * py2, float(0)};

    float dkddr1[6] = {bq1 * bq2 * dx0 + bq2 * py1 - bq1 * py2,
                   bq1 * bq2 * dy0 - bq2 * px1 + bq1 * px2,
                   float(0),
                   -bq2 * dy0 - px2,
                   bq2 * dx0 - py2,
                   float(0)};
    float dkddr2[6] = {-bq1 * bq2 * dx0 - bq2 * py1 + bq1 * py2,
                   -bq1 * bq2 * dy0 + bq2 * px1 - bq1 * px2,
                   float(0),
                   bq1 * dy0 - px1,
                   -bq1 * dx0 - py1,
                   float(0)};

    float dc1dr1[6] = {-(bq1 * (bq1 * bq2 * dx0 + bq2 * py1 - bq1 * py2)),
                   -(bq1 * (bq1 * bq2 * dy0 - bq2 * px1 + bq1 * px2)),
                   float(0),
                   -float(2) * bq2 * px1 - bq1 * (-(bq2 * dy0) - px2),
                   -float(2) * bq2 * py1 - bq1 * (bq2 * dx0 - py2),
                   float(0)};
    float dc1dr2[6] = {-(bq1 * (-(bq1 * bq2 * dx0) - bq2 * py1 + bq1 * py2)),
                   -(bq1 * (-(bq1 * bq2 * dy0) + bq2 * px1 - bq1 * px2)),
                   float(0),
                   -(bq1 * (bq1 * dy0 - px1)),
                   -(bq1 * (-(bq1 * dx0) - py1)),
                   float(0)};

    float dc2dr1[6] = {bq2 * (bq1 * bq2 * dx0 + bq2 * py1 - bq1 * py2),
                   bq2 * (bq1 * bq2 * dy0 - bq2 * px1 + bq1 * px2),
                   float(0),
                   bq2 * (-(bq2 * dy0) - px2),
                   bq2 * (bq2 * dx0 - py2),
                   float(0)};
    float dc2dr2[6] = {bq2 * (-(bq1 * bq2 * dx0) - bq2 * py1 + bq1 * py2),
                   bq2 * (-(bq1 * bq2 * dy0) + bq2 * px1 - bq1 * px2),
                   float(0),
                   bq2 * (bq1 * dy0 - px1) + float(2) * bq1 * px2,
                   bq2 * (-(bq1 * dx0) - py1) + float(2) * bq1 * py2,
                   float(0)};

    float dd1dr1[6] = {float(0), float(0), float(0), float(0), float(0), float(0)};
    float dd1dr2[6] = {float(0), float(0), float(0), float(0), float(0), float(0)};

    if (d1 > float(0)) {
      for (int i = 0; i < 6; ++i) {
        dd1dr1[i] = -kd / d1 * dkddr1[i];
        dd1dr2[i] = -kd / d1 * dkddr2[i];
      }
      dd1dr1[3] += px1 / d1 * pt22;
      dd1dr1[4] += py1 / d1 * pt22;
      dd1dr2[3] += px2 / d1 * pt12;
      dd1dr2[4] += py2 / d1 * pt12;
    }

    if (!isStraight1) {
      dS1[0] = CpuCompatibleAtan2(
                  bq1 * (k11 * c1 + k21 * d1),
                  bq1 * k11 * d1 * bq1 - k21 * c1,
                  arithmeticTrace ? &arithmeticTrace->atan[0][0] : nullptr)
                / bq1;
      dS1[1] = CpuCompatibleAtan2(
                  bq1 * (k11 * c1 - k21 * d1),
                  -bq1 * k11 * d1 * bq1 - k21 * c1,
                  arithmeticTrace ? &arithmeticTrace->atan[0][1] : nullptr)
                / bq1;

      float a = bq1 * (k11 * c1 + k21 * d1);
      float b = bq1 * k11 * d1 * bq1 - k21 * c1;

      for (int iP = 0; iP < 6; ++iP) {
        if ((b * b + a * a) > float(0)) {
          const float dadr1 = bq1 * (dk11dr1[iP] * c1 + k11 * dc1dr1[iP] + dk21dr1[iP] * d1 + k21 * dd1dr1[iP]);
          const float dadr2 = bq1 * (dk11dr2[iP] * c1 + k11 * dc1dr2[iP] + dk21dr2[iP] * d1 + k21 * dd1dr2[iP]);
          const float dbdr1 = bq1 * bq1 * (dk11dr1[iP] * d1 + k11 * dd1dr1[iP]) - (dk21dr1[iP] * c1 + k21 * dc1dr1[iP]);
          const float dbdr2 = bq1 * bq1 * (dk11dr2[iP] * d1 + k11 * dd1dr2[iP]) - (dk21dr2[iP] * c1 + k21 * dc1dr2[iP]);

          dS1dR1[0][iP] = float(1) / bq1 * float(1) / (b * b + a * a) * (dadr1 * b - dbdr1 * a);
          dS1dR2[0][iP] = float(1) / bq1 * float(1) / (b * b + a * a) * (dadr2 * b - dbdr2 * a);
        }
        else {
          dS1dR1[0][iP] = float(0);
          dS1dR2[0][iP] = float(0);
        }
      }

      a = bq1 * (k11 * c1 - k21 * d1);
      b = -bq1 * k11 * d1 * bq1 - k21 * c1;

      for (int iP = 0; iP < 6; ++iP) {
        if ((b * b + a * a) > float(0)) {
          const float dadr1 = bq1 * (dk11dr1[iP] * c1 + k11 * dc1dr1[iP] - (dk21dr1[iP] * d1 + k21 * dd1dr1[iP]));
          const float dadr2 = bq1 * (dk11dr2[iP] * c1 + k11 * dc1dr2[iP] - (dk21dr2[iP] * d1 + k21 * dd1dr2[iP]));
          const float dbdr1 = -bq1 * bq1 * (dk11dr1[iP] * d1 + k11 * dd1dr1[iP]) - (dk21dr1[iP] * c1 + k21 * dc1dr1[iP]);
          const float dbdr2 = -bq1 * bq1 * (dk11dr2[iP] * d1 + k11 * dd1dr2[iP]) - (dk21dr2[iP] * c1 + k21 * dc1dr2[iP]);

          dS1dR1[1][iP] = float(1) / bq1 * float(1) / (b * b + a * a) * (dadr1 * b - dbdr1 * a);
          dS1dR2[1][iP] = float(1) / bq1 * float(1) / (b * b + a * a) * (dadr2 * b - dbdr2 * a);
        }
        else {
          dS1dR1[1][iP] = float(0);
          dS1dR2[1][iP] = float(0);
        }
      }
    }

    if (!isStraight2) {
      dS2[0] = CpuCompatibleAtan2(
                  bq2 * k12 * c2 + k22 * d2 * bq2,
                  bq2 * k12 * d2 * bq2 - k22 * c2,
                  arithmeticTrace ? &arithmeticTrace->atan[1][0] : nullptr)
                / bq2;
      dS2[1] = CpuCompatibleAtan2(
                  bq2 * k12 * c2 - k22 * d2 * bq2,
                  -bq2 * k12 * d2 * bq2 - k22 * c2,
                  arithmeticTrace ? &arithmeticTrace->atan[1][1] : nullptr)
                / bq2;

      float a = bq2 * (k12 * c2 + k22 * d2);
      float b = bq2 * k12 * d2 * bq2 - k22 * c2;

      for (int iP = 0; iP < 6; ++iP) {
        if ((b * b + a * a) > float(0)) {
          const float dadr1 = bq2 * (dk12dr1[iP] * c2 + k12 * dc2dr1[iP] + dk22dr1[iP] * d1 + k22 * dd1dr1[iP]);
          const float dadr2 = bq2 * (dk12dr2[iP] * c2 + k12 * dc2dr2[iP] + dk22dr2[iP] * d1 + k22 * dd1dr2[iP]);
          const float dbdr1 = bq2 * bq2 * (dk12dr1[iP] * d1 + k12 * dd1dr1[iP]) - (dk22dr1[iP] * c2 + k22 * dc2dr1[iP]);
          const float dbdr2 = bq2 * bq2 * (dk12dr2[iP] * d1 + k12 * dd1dr2[iP]) - (dk22dr2[iP] * c2 + k22 * dc2dr2[iP]);

          dS2dR1[0][iP] = float(1) / bq2 * float(1) / (b * b + a * a) * (dadr1 * b - dbdr1 * a);
          dS2dR2[0][iP] = float(1) / bq2 * float(1) / (b * b + a * a) * (dadr2 * b - dbdr2 * a);
        }
        else {
          dS2dR1[0][iP] = float(0);
          dS2dR2[0][iP] = float(0);
        }
      }

      a = bq2 * (k12 * c2 - k22 * d2);
      b = -bq2 * k12 * d2 * bq2 - k22 * c2;

      for (int iP = 0; iP < 6; ++iP) {
        if ((b * b + a * a) > float(0)) {
          const float dadr1 = bq2 * (dk12dr1[iP] * c2 + k12 * dc2dr1[iP] - (dk22dr1[iP] * d1 + k22 * dd1dr1[iP]));
          const float dadr2 = bq2 * (dk12dr2[iP] * c2 + k12 * dc2dr2[iP] - (dk22dr2[iP] * d1 + k22 * dd1dr2[iP]));
          const float dbdr1 = -bq2 * bq2 * (dk12dr1[iP] * d1 + k12 * dd1dr1[iP]) - (dk22dr1[iP] * c2 + k22 * dc2dr1[iP]);
          const float dbdr2 = -bq2 * bq2 * (dk12dr2[iP] * d1 + k12 * dd1dr2[iP]) - (dk22dr2[iP] * c2 + k22 * dc2dr2[iP]);

          dS2dR1[1][iP] = float(1) / bq2 * float(1) / (b * b + a * a) * (dadr1 * b - dbdr1 * a);
          dS2dR2[1][iP] = float(1) / bq2 * float(1) / (b * b + a * a) * (dadr2 * b - dbdr2 * a);
        }
        else {
          dS2dR1[1][iP] = float(0);
          dS2dR2[1][iP] = float(0);
        }
      }
    }

    if (arithmeticTrace) {
      arithmeticTrace->atanNumerator[0][0] = bq1 * (k11 * c1 + k21 * d1);
      arithmeticTrace->atanNumerator[0][1] = bq1 * (k11 * c1 - k21 * d1);
      arithmeticTrace->atanDenominator[0][0] = bq1 * k11 * d1 * bq1 - k21 * c1;
      arithmeticTrace->atanDenominator[0][1] = -bq1 * k11 * d1 * bq1 - k21 * c1;
      arithmeticTrace->atanNumerator[1][0] = bq2 * (k12 * c2 + k22 * d2);
      arithmeticTrace->atanNumerator[1][1] = bq2 * (k12 * c2 - k22 * d2);
      arithmeticTrace->atanDenominator[1][0] = bq2 * k12 * d2 * bq2 - k22 * c2;
      arithmeticTrace->atanDenominator[1][1] = -bq2 * k12 * d2 * bq2 - k22 * c2;
      arithmeticTrace->roots[0][0] = dS1[0];
      arithmeticTrace->roots[0][1] = dS1[1];
      arithmeticTrace->roots[1][0] = dS2[0];
      arithmeticTrace->roots[1][1] = dS2[1];
    }

    if (isStraight1 && (pt12 > float(0))) {
      dS1[0] = (k11 * c1 + k21 * d1) / (-k21 * c1);
      dS1[1] = (k11 * c1 - k21 * d1) / (-k21 * c1);

      float a = k11 * c1 + k21 * d1;
      float b = -k21 * c1;

      for (int iP = 0; iP < 6; ++iP) {
        if (b * b > float(0)) {
          const float dadr1 = dk11dr1[iP] * c1 + k11 * dc1dr1[iP] + dk21dr1[iP] * d1 + k21 * dd1dr1[iP];
          const float dadr2 = dk11dr2[iP] * c1 + k11 * dc1dr2[iP] + dk21dr2[iP] * d1 + k21 * dd1dr2[iP];
          const float dbdr1 = -(dk21dr1[iP] * c1 + k21 * dc1dr1[iP]);
          const float dbdr2 = -(dk21dr2[iP] * c1 + k21 * dc1dr2[iP]);

          dS1dR1[0][iP] = dadr1 / b - dbdr1 * a / (b * b);
          dS1dR2[0][iP] = dadr2 / b - dbdr2 * a / (b * b);
        }
        else {
          dS1dR1[0][iP] = float(0);
          dS1dR2[0][iP] = float(0);
        }
      }

      a = k11 * c1 - k21 * d1;
      for (int iP = 0; iP < 6; ++iP) {
        if (b * b > float(0)) {
          const float dadr1 = dk11dr1[iP] * c1 + k11 * dc1dr1[iP] - dk21dr1[iP] * d1 - k21 * dd1dr1[iP];
          const float dadr2 = dk11dr2[iP] * c1 + k11 * dc1dr2[iP] - dk21dr2[iP] * d1 - k21 * dd1dr2[iP];
          const float dbdr1 = -(dk21dr1[iP] * c1 + k21 * dc1dr1[iP]);
          const float dbdr2 = -(dk21dr2[iP] * c1 + k21 * dc1dr2[iP]);

          dS1dR1[1][iP] = dadr1 / b - dbdr1 * a / (b * b);
          dS1dR2[1][iP] = dadr2 / b - dbdr2 * a / (b * b);
        }
        else {
          dS1dR1[1][iP] = float(0);
          dS1dR2[1][iP] = float(0);
        }
      }
    }

    if (isStraight2 && (pt22 > float(0))) {
      dS2[0] = (k12 * c2 + k22 * d2) / (-k22 * c2);
      dS2[1] = (k12 * c2 - k22 * d2) / (-k22 * c2);

      float a = k12 * c2 + k22 * d1;
      float b = -k22 * c2;

      for (int iP = 0; iP < 6; ++iP) {
        if (b * b > float(0)) {
          const float dadr1 = dk12dr1[iP] * c2 + k12 * dc2dr1[iP] + dk22dr1[iP] * d1 + k22 * dd1dr1[iP];
          const float dadr2 = dk12dr2[iP] * c2 + k12 * dc2dr2[iP] + dk22dr2[iP] * d1 + k22 * dd1dr2[iP];
          const float dbdr1 = -(dk22dr1[iP] * c2 + k22 * dc2dr1[iP]);
          const float dbdr2 = -(dk22dr2[iP] * c2 + k22 * dc2dr2[iP]);

          dS2dR1[0][iP] = dadr1 / b - dbdr1 * a / (b * b);
          dS2dR2[0][iP] = dadr2 / b - dbdr2 * a / (b * b);
        }
        else {
          dS2dR1[0][iP] = float(0);
          dS2dR2[0][iP] = float(0);
        }
      }

      //TODO: why d1? should it be d2? a = k12 * c2 - k22 * d2?
      // Oh, I know why... d1 and d2 is actually the same. The real question is why do we have both?
      a = k12 * c2 - k22 * d1;
      for (int iP = 0; iP < 6; ++iP) {
        if (b * b > float(0)) {
          const float dadr1 = dk12dr1[iP] * c2 + k12 * dc2dr1[iP] - dk22dr1[iP] * d1 - k22 * dd1dr1[iP];
          const float dadr2 = dk12dr2[iP] * c2 + k12 * dc2dr2[iP] - dk22dr2[iP] * d1 - k22 * dd1dr2[iP];
          const float dbdr1 = -(dk22dr1[iP] * c2 + k22 * dc2dr1[iP]);
          const float dbdr2 = -(dk22dr2[iP] * c2 + k22 * dc2dr2[iP]);

          dS2dR1[1][iP] = dadr1 / b - dbdr1 * a / (b * b);
          dS2dR2[1][iP] = dadr2 / b - dbdr2 * a / (b * b);
        }
        else {
          dS2dR1[1][iP] = float(0);
          dS2dR2[1][iP] = float(0);
        }
      }
    }

    float dr2[2];
    for (int iP = 0; iP < 2; ++iP) {
      const float bs1 = bq1 * dS1[iP];
      const float bs2 = bq2 * dS2[iP];

      float sss = 0.f;
      float ccc = 0.f;
      CpuCompatibleSinCos(bs1, sss, ccc);

      const bool bs1Big = Abs(bs1) > float(1.e-8);
      const bool bs2Big = Abs(bs2) > float(1.e-8);

      float sB = float(0);
      float cB = float(0);
      if (bs1Big) {
        sB = sss / bq1;
        cB = (float(1) - ccc) / bq1;
      }
      else {
        sB = ((float(1) - bs1 * kOvSqr6) * (float(1) + bs1 * kOvSqr6) * dS1[iP]);
        cB = float(0.5) * sB * bs1;
      }

      const float x1 = param1[0] + sB * px1 + cB * py1;
      const float y1 = param1[1] - cB * px1 + sB * py1;
      const float z1 = param1[2] + dS1[iP] * param1[5];

      CpuCompatibleSinCos(bs2, sss, ccc);

      if (bs2Big) {
        sB = sss / bq2;
        cB = (float(1) - ccc) / bq2;
      }
      else {
        sB = ((float(1) - bs2 * kOvSqr6) * (float(1) + bs2 * kOvSqr6) * dS2[iP]);
        cB = float(0.5) * sB * bs2;
      }

      const float x2 = param2[0] + sB * px2 + cB * py2;
      const float y2 = param2[1] - cB * px2 + sB * py2;
      const float z2 = param2[2] + dS2[iP] * param2[5];

      const float dx = x1 - x2;
      const float dy = y1 - y2;
      const float dz = z1 - z2;

      dr2[iP] = dx * dx + dy * dy + dz * dz;
    }

    const bool isFirstRoot = dr2[0] < dr2[1];
    if (useMiddlePoint) {
      dS[0] = 0.5f * (dS1[0] + dS1[1]);
      dS[1] = 0.5f * (dS2[0] + dS2[1]);

      for (int iP = 0; iP < 6; ++iP) {
        dsdr[0][iP] = 0.5f * (dS1dR1[0][iP] + dS1dR1[1][iP]);
        dsdr[1][iP] = 0.5f * (dS1dR2[0][iP] + dS1dR2[1][iP]);
        dsdr[2][iP] = 0.5f * (dS2dR1[0][iP] + dS2dR1[1][iP]);
        dsdr[3][iP] = 0.5f * (dS2dR2[0][iP] + dS2dR2[1][iP]);
      }
    }
    else if (isFirstRoot) {
      dS[0] = dS1[0];
      dS[1] = dS2[0];

      for (int iP = 0; iP < 6; ++iP) {
        dsdr[0][iP] = dS1dR1[0][iP];
        dsdr[1][iP] = dS1dR2[0][iP];
        dsdr[2][iP] = dS2dR1[0][iP];
        dsdr[3][iP] = dS2dR2[0][iP];
      }
    }
    else {
      dS[0] = dS1[1];
      dS[1] = dS2[1];

      for (int iP = 0; iP < 6; ++iP) {
        dsdr[0][iP] = dS1dR1[1][iP];
        dsdr[1][iP] = dS1dR2[1][iP];
        dsdr[2][iP] = dS2dR1[1][iP];
        dsdr[3][iP] = dS2dR2[1][iP];
      }
    }

    if (arithmeticTrace) {
      arithmeticTrace->rootDistance2[0] = dr2[0];
      arithmeticTrace->rootDistance2[1] = dr2[1];
      arithmeticTrace->selectedDs[0] = dS[0];
      arithmeticTrace->selectedDs[1] = dS[1];
    }

    {
      const float bs1 = bq1 * dS[0];
      const float bs2 = bq2 * dS[1];

      float sss = 0.f;
      float ccc = 0.f;
      CpuCompatibleSinCos(
        bs1, sss, ccc, arithmeticTrace ? &arithmeticTrace->sincos[0] : nullptr);

      const bool bs1Big = Abs(bs1) > float(1.e-8);
      const bool bs2Big = Abs(bs2) > float(1.e-8);

      float sB = float(0);
      float cB = float(0);
      if (bs1Big) {
        sB = sss / bq1;
        cB = (float(1) - ccc) / bq1;
      }
      else {
        sB = ((float(1) - bs1 * kOvSqr6) * (float(1) + bs1 * kOvSqr6) * dS[0]);
        cB = float(0.5) * sB * bs1;
      }

      const float x1 = x01 + sB * px1 + cB * py1;
      const float y1 = y01 - cB * px1 + sB * py1;
      const float z1 = z01 + dS[0] * pz1;
      const float ppx1 = ccc * px1 + sss * py1;
      const float ppy1 = -sss * px1 + ccc * py1;
      const float ppz1 = pz1;

      float sss1 = 0.f;
      float ccc1 = 0.f;
      CpuCompatibleSinCos(
        bs2, sss1, ccc1, arithmeticTrace ? &arithmeticTrace->sincos[1] : nullptr);

      float sB1 = float(0);
      float cB1 = float(0);
      if (bs2Big) {
        sB1 = sss1 / bq2;
        cB1 = (float(1) - ccc1) / bq2;
      }
      else {
        sB1 = ((float(1) - bs2 * kOvSqr6) * (float(1) + bs2 * kOvSqr6) * dS[1]);
        cB1 = float(0.5) * sB1 * bs2;
      }

      const float x2 = x02 + sB1 * px2 + cB1 * py2;
      const float y2 = y02 - cB1 * px2 + sB1 * py2;
      const float z2 = z02 + dS[1] * pz2;
      const float ppx2 = ccc1 * px2 + sss1 * py2;
      const float ppy2 = -sss1 * px2 + ccc1 * py2;
      const float ppz2 = pz2;

      const float p12 = ppx1 * ppx1 + ppy1 * ppy1 + ppz1 * ppz1;
      const float p22 = ppx2 * ppx2 + ppy2 * ppy2 + ppz2 * ppz2;
      const float lp1p2 = ppx1 * ppx2 + ppy1 * ppy2 + ppz1 * ppz2;

      const float dx = x2 - x1;
      const float dy = y2 - y1;
      const float dz = z2 - z1;

      const float ldrp1 = ppx1 * dx + ppy1 * dy + ppz1 * dz;
      const float ldrp2 = ppx2 * dx + ppy2 * dy + ppz2 * dz;

      float detp = lp1p2 * lp1p2 - p12 * p22;
      if (Abs(detp) < float(1.e-4)) {
        detp = float(1);
      }

      const float a1 = ldrp2 * lp1p2 - ldrp1 * p22;
      const float a2 = ldrp2 * p12 - ldrp1 * lp1p2;

      if (arithmeticTrace) {
        arithmeticTrace->bs[0] = bs1;
        arithmeticTrace->bs[1] = bs2;
        arithmeticTrace->sine[0] = sss;
        arithmeticTrace->sine[1] = sss1;
        arithmeticTrace->cosine[0] = ccc;
        arithmeticTrace->cosine[1] = ccc1;
        arithmeticTrace->position[0][0] = x1;
        arithmeticTrace->position[0][1] = y1;
        arithmeticTrace->position[0][2] = z1;
        arithmeticTrace->position[1][0] = x2;
        arithmeticTrace->position[1][1] = y2;
        arithmeticTrace->position[1][2] = z2;
        arithmeticTrace->momentum[0][0] = ppx1;
        arithmeticTrace->momentum[0][1] = ppy1;
        arithmeticTrace->momentum[0][2] = ppz1;
        arithmeticTrace->momentum[1][0] = ppx2;
        arithmeticTrace->momentum[1][1] = ppy2;
        arithmeticTrace->momentum[1][2] = ppz2;
        arithmeticTrace->momentum2[0] = p12;
        arithmeticTrace->momentum2[1] = p22;
        arithmeticTrace->momentumDot = lp1p2;
        arithmeticTrace->separation[0] = dx;
        arithmeticTrace->separation[1] = dy;
        arithmeticTrace->separation[2] = dz;
        arithmeticTrace->separationDotMomentum[0] = ldrp1;
        arithmeticTrace->separationDotMomentum[1] = ldrp2;
        arithmeticTrace->determinantRaw = lp1p2 * lp1p2 - p12 * p22;
        arithmeticTrace->determinantUsed = detp;
        arithmeticTrace->correctionNumerator[0] = a1;
        arithmeticTrace->correctionNumerator[1] = a2;
        arithmeticTrace->correction[0] = a1 / detp;
        arithmeticTrace->correction[1] = a2 / detp;
      }
      const float lp1p2_ds0 = bq1 * (ppx2 * ppy1 - ppy2 * ppx1);
      const float lp1p2_ds1 = bq2 * (ppx1 * ppy2 - ppy1 * ppx2);
      const float ldrp1_ds0 = -p12 + bq1 * (ppy1 * dx - ppx1 * dy);
      const float ldrp1_ds1 = lp1p2;
      const float ldrp2_ds0 = -lp1p2;
      const float ldrp2_ds1 = p22 + bq2 * (ppy2 * dx - ppx2 * dy);
      const float detp_ds0 = float(2) * lp1p2 * lp1p2_ds0;
      const float detp_ds1 = float(2) * lp1p2 * lp1p2_ds1;
      const float a1_ds0 = ldrp2_ds0 * lp1p2 + ldrp2 * lp1p2_ds0 - ldrp1_ds0 * p22;
      const float a1_ds1 = ldrp2_ds1 * lp1p2 + ldrp2 * lp1p2_ds1 - ldrp1_ds1 * p22;
      const float a2_ds0 = ldrp2_ds0 * p12 - ldrp1_ds0 * lp1p2 - ldrp1 * lp1p2_ds0;
      const float a2_ds1 = ldrp2_ds1 * p12 - ldrp1_ds1 * lp1p2 - ldrp1 * lp1p2_ds1;

      const float dsl1ds0 = a1_ds0 / detp - a1 * detp_ds0 / (detp * detp);
      const float dsl1ds1 = a1_ds1 / detp - a1 * detp_ds1 / (detp * detp);
      const float dsl2ds0 = a2_ds0 / detp - a2 * detp_ds0 / (detp * detp);
      const float dsl2ds1 = a2_ds1 / detp - a2 * detp_ds1 / (detp * detp);

      float dsldr[4][6];
      for (int iP = 0; iP < 6; ++iP) {
        dsldr[0][iP] = dsl1ds0 * dsdr[0][iP] + dsl1ds1 * dsdr[2][iP];
        dsldr[1][iP] = dsl1ds0 * dsdr[1][iP] + dsl1ds1 * dsdr[3][iP];
        dsldr[2][iP] = dsl2ds0 * dsdr[0][iP] + dsl2ds1 * dsdr[2][iP];
        dsldr[3][iP] = dsl2ds0 * dsdr[1][iP] + dsl2ds1 * dsdr[3][iP];
      }

      for (int iDS = 0; iDS < 4; ++iDS) {
        for (int iP = 0; iP < 6; ++iP) {
          dsdr[iDS][iP] += dsldr[iDS][iP];
        }
      }

      const float lp1p2_dr0[6] = {float(0), float(0), float(0), ccc * ppx2 - ppy2 * sss, ccc * ppy2 + ppx2 * sss, pz2};
      const float lp1p2_dr1[6] = {float(0), float(0), float(0), ccc1 * ppx1 - ppy1 * sss1, ccc1 * ppy1 + ppx1 * sss1, pz1};
      const float ldrp1_dr0[6] = {-ppx1, -ppy1, -pz1, cB * ppy1 - ppx1 * sB + ccc * dx - sss * dy,
                              -cB * ppx1 - ppy1 * sB + sss * dx + ccc * dy, -dS[0] * pz1 + dz};
      const float ldrp1_dr1[6] = {ppx1, ppy1, ppz1, -cB1 * ppy1 + ppx1 * sB1, cB1 * ppx1 + ppy1 * sB1, dS[1] * pz1};
      const float ldrp2_dr0[6] = {-ppx2, -ppy2, -ppz2, cB * ppy2 - ppx2 * sB, -cB * ppx2 - ppy2 * sB, -dS[0] * pz2};
      const float ldrp2_dr1[6] = {ppx2, ppy2, ppz2, -cB1 * ppy2 + ppx2 * sB1 + ccc1 * dx - sss1 * dy,
                              cB1 * ppx2 + ppy2 * sB1 + sss1 * dx + ccc1 * dy, dz + dS[1] * pz2};
      const float p12_dr0[6] = {float(0), float(0), float(0), float(2) * px1, float(2) * py1, float(2) * pz1};
      const float p22_dr1[6] = {float(0), float(0), float(0), float(2) * px2, float(2) * py2, float(2) * pz2};

      float a1_dr0[6], a1_dr1[6], a2_dr0[6], a2_dr1[6], detp_dr0[6], detp_dr1[6];
      for (int iP = 0; iP < 6; ++iP) {
        a1_dr0[iP] = ldrp2_dr0[iP] * lp1p2 + ldrp2 * lp1p2_dr0[iP] - ldrp1_dr0[iP] * p22;
        a1_dr1[iP] = ldrp2_dr1[iP] * lp1p2 + ldrp2 * lp1p2_dr1[iP] - ldrp1_dr1[iP] * p22 - ldrp1 * p22_dr1[iP];
        a2_dr0[iP] = ldrp2_dr0[iP] * p12 + ldrp2 * p12_dr0[iP] - ldrp1_dr0[iP] * lp1p2 - ldrp1 * lp1p2_dr0[iP];
        a2_dr1[iP] = ldrp2_dr1[iP] * p12 - ldrp1_dr1[iP] * lp1p2 - ldrp1 * lp1p2_dr1[iP];
        detp_dr0[iP] = float(2) * lp1p2 * lp1p2_dr0[iP] - p12_dr0[iP] * p22;
        detp_dr1[iP] = float(2) * lp1p2 * lp1p2_dr1[iP] - p12 * p22_dr1[iP];

        dsdr[0][iP] += a1_dr0[iP] / detp - a1 * detp_dr0[iP] / (detp * detp);
        dsdr[1][iP] += a1_dr1[iP] / detp - a1 * detp_dr1[iP] / (detp * detp);
        dsdr[2][iP] += a2_dr0[iP] / detp - a2 * detp_dr0[iP] / (detp * detp);
        dsdr[3][iP] += a2_dr1[iP] / detp - a2 * detp_dr1[iP] / (detp * detp);
      }

      dS[0] += (ldrp2 * lp1p2 - ldrp1 * p22) / detp;
      dS[1] += (ldrp2 * p12 - ldrp1 * lp1p2) / detp;
      if (arithmeticTrace) {
        arithmeticTrace->finalDs[0] = dS[0];
        arithmeticTrace->finalDs[1] = dS[1];
      }
    }
  }

  KFPARTICLE_GPU_HOST_DEVICE inline void GetDStoParticleByCpuCompatibleDerivatives(
    const KFParticleGpuFitState& first,
    const KFParticleGpuFitState& second,
    float by,
    bool useMiddlePoint,
    float dS[2],
    float dsdr[4][6],
    KFParticleGpuDcaArithmeticTrace* arithmeticTrace = nullptr)
  {
    const float firstParameters[6] = {
      first.X(), -first.Z(), first.Y(), first.Px(), -first.Pz(), first.Py()};
    const float secondParameters[6] = {
      second.X(), -second.Z(), second.Y(), second.Px(), -second.Pz(), second.Py()};
    float rotatedDerivatives[4][6] = {};
    GetDStoParticleBzCpuCompatibleDerivatives(first,
                                               by,
                                               second,
                                               dS,
                                               rotatedDerivatives,
                                               firstParameters,
                                               secondParameters,
                                               useMiddlePoint,
                                               arithmeticTrace);
    for (int derivative = 0; derivative < 4; ++derivative) {
      dsdr[derivative][0] = rotatedDerivatives[derivative][0];
      dsdr[derivative][1] = rotatedDerivatives[derivative][2];
      dsdr[derivative][2] = -rotatedDerivatives[derivative][1];
      dsdr[derivative][3] = rotatedDerivatives[derivative][3];
      dsdr[derivative][4] = rotatedDerivatives[derivative][5];
      dsdr[derivative][5] = -rotatedDerivatives[derivative][4];
    }
  }


  KFPARTICLE_GPU_HOST_DEVICE inline void GetDStoParticleByCpuCompatible(
    const KFParticleGpuFitState& first,
    const KFParticleGpuFitState& second,
    float by,
    float dS[2],
    bool useMiddlePoint = false,
    float firstRoots[2] = nullptr,
    float secondRoots[2] = nullptr);

  KFPARTICLE_GPU_HOST_DEVICE inline bool BuildConstantByDcaKinematicSeed(
    const KFParticleGpuFitState& first,
    const KFParticleGpuFitState& second,
    float by,
    KFParticleGpuFitState& firstAtDca,
    KFParticleGpuFitState& secondAtDca)
  {
    const bool firstIsStraight = Abs(by * static_cast<float>(first.Q())) < 1.e-8f;
    const bool secondIsStraight = Abs(by * static_cast<float>(second.Q())) < 1.e-8f;
    float dS[2] = {0.f, 0.f};
    if (firstIsStraight && secondIsStraight) {
      GetDStoParticleLine(first, second, dS);
    }
    else {
      GetDStoParticleByCpuCompatible(first, second, by, dS);
    }

    if (Abs(dS[0] * first.Pz()) > 1000.f || Abs(dS[1] * second.Pz()) > 1000.f) {
      firstAtDca.Initialize();
      secondAtDca.Initialize();
      return false;
    }

    if (firstIsStraight && secondIsStraight) {
      TransportLine(first, dS[0], firstAtDca);
      TransportLine(second, dS[1], secondAtDca);
    }
    else {
      TransportConstantBy(first, dS[0], by, firstAtDca);
      TransportConstantBy(second, dS[1], by, secondAtDca);
    }
    return true;
  }

  KFPARTICLE_GPU_HOST_DEVICE inline bool BuildConstantByDcaKinematicMother(
    const KFParticleGpuFitState& first,
    const KFParticleGpuFitState& second,
    float by,
    KFParticleGpuFitState& mother)
  {
    KFParticleGpuFitState firstAtDca;
    KFParticleGpuFitState secondAtDca;
    if (!BuildConstantByDcaKinematicSeed(first, second, by, firstAtDca, secondAtDca)) {
      mother.Initialize();
      return false;
    }
    BuildKinematicMother(firstAtDca, secondAtDca, mother);
    return true;
  }

  KFPARTICLE_GPU_HOST_DEVICE inline void GetDStoParticleBzCpuCompatible(
    const KFParticleGpuFitState& first,
    const KFParticleGpuFitState& second,
    float field,
    const float firstParameters[6],
    const float secondParameters[6],
    float dS[2],
    bool useMiddlePoint,
    float outputFirstRoots[2],
    float outputSecondRoots[2])
  {
#if defined(__clang__)
#pragma clang fp contract(off)
#endif
    const float kCLight = 0.000299792458f;
    const float kOvSqr6 = 0.4082482904638631f;
    const float bq1 = field * static_cast<float>(first.Q()) * kCLight;
    const float bq2 = field * static_cast<float>(second.Q()) * kCLight;
    const bool straight1 = Abs(bq1) < 1.e-8f;
    const bool straight2 = Abs(bq2) < 1.e-8f;
    if (straight1 && straight2) {
      GetDStoParticleLine(first, second, dS);
      return;
    }

    const float px1 = firstParameters[3];
    const float py1 = firstParameters[4];
    const float pz1 = firstParameters[5];
    const float px2 = secondParameters[3];
    const float py2 = secondParameters[4];
    const float pz2 = secondParameters[5];
    const float pt12 = px1 * px1 + py1 * py1;
    const float pt22 = px2 * px2 + py2 * py2;
    const float x01 = firstParameters[0];
    const float y01 = firstParameters[1];
    const float z01 = firstParameters[2];
    const float x02 = secondParameters[0];
    const float y02 = secondParameters[1];
    const float z02 = secondParameters[2];
    const float dx0 = x01 - x02;
    const float dy0 = y01 - y02;
    const float dr02 = dx0 * dx0 + dy0 * dy0;
    const float drp1 = dx0 * px1 + dy0 * py1;
    const float dxyp1 = dx0 * py1 - dy0 * px1;
    const float drp2 = dx0 * px2 + dy0 * py2;
    const float dxyp2 = dx0 * py2 - dy0 * px2;
    const float p1p2 = px1 * px2 + py1 * py2;
    const float dp1p2 = px1 * py2 - px2 * py1;
    const float k11 = bq2 * drp1 - dp1p2;
    const float k21 = bq1 * (bq2 * dxyp1 - p1p2) + bq2 * pt12;
    const float k12 = bq1 * drp2 - dp1p2;
    const float k22 = bq2 * (bq1 * dxyp2 + p1p2) - bq1 * pt22;
    const float kp = dxyp1 * bq2 - dxyp2 * bq1 - p1p2;
    const float kd = 0.5f * dr02 * bq1 * bq2 + kp;
    const float c1 = -(bq1 * kd + pt12 * bq2);
    const float c2 = bq2 * kd + pt22 * bq1;
    const float discriminant = pt12 * pt22 - kd * kd;
    const float root = KFParticleGpuSqrt(discriminant > 0.f ? discriminant : 0.f);

    float firstRoots[2] = {0.f, 0.f};
    float secondRoots[2] = {0.f, 0.f};
    if (!straight1) {
      firstRoots[0] = CpuCompatibleAtan2(bq1 * k11 * c1 + k21 * root * bq1,
                                         bq1 * k11 * root * bq1 - k21 * c1) / bq1;
      firstRoots[1] = CpuCompatibleAtan2(bq1 * k11 * c1 - k21 * root * bq1,
                                         -bq1 * k11 * root * bq1 - k21 * c1) / bq1;
    }
    else if (pt12 > 0.f) {
      const float denominator = -k21 * c1;
      if (Abs(denominator) > 1.e-20f) {
        firstRoots[0] = (k11 * c1 + k21 * root) / denominator;
        firstRoots[1] = (k11 * c1 - k21 * root) / denominator;
      }
    }

    if (!straight2) {
      secondRoots[0] = CpuCompatibleAtan2(bq2 * k12 * c2 + k22 * root * bq2,
                                          bq2 * k12 * root * bq2 - k22 * c2) / bq2;
      secondRoots[1] = CpuCompatibleAtan2(bq2 * k12 * c2 - k22 * root * bq2,
                                          -bq2 * k12 * root * bq2 - k22 * c2) / bq2;
    }
    else if (pt22 > 0.f) {
      const float denominator = -k22 * c2;
      if (Abs(denominator) > 1.e-20f) {
        secondRoots[0] = (k12 * c2 + k22 * root) / denominator;
        secondRoots[1] = (k12 * c2 - k22 * root) / denominator;
      }
    }

    if (outputFirstRoots) {
      outputFirstRoots[0] = firstRoots[0];
      outputFirstRoots[1] = firstRoots[1];
    }
    if (outputSecondRoots) {
      outputSecondRoots[0] = secondRoots[0];
      outputSecondRoots[1] = secondRoots[1];
    }

    float rootDistance2[2] = {0.f, 0.f};
    for (int rootIndex = 0; rootIndex < 2; ++rootIndex) {
      const float bs1 = bq1 * firstRoots[rootIndex];
      const float bs2 = bq2 * secondRoots[rootIndex];
      float sin1 = 0.f;
      float cos1 = 0.f;
      float sin2 = 0.f;
      float cos2 = 0.f;
      CpuCompatibleSinCos(bs1, sin1, cos1);
      CpuCompatibleSinCos(bs2, sin2, cos2);
      const bool curved1 = Abs(bs1) > 1.e-8f;
      const bool curved2 = Abs(bs2) > 1.e-8f;
      const float sB1 = curved1 ? sin1 / bq1
                                : (1.f - bs1 * kOvSqr6) * (1.f + bs1 * kOvSqr6)
                                    * firstRoots[rootIndex];
      const float cB1 = curved1 ? (1.f - cos1) / bq1 : 0.5f * sB1 * bs1;
      const float sB2 = curved2 ? sin2 / bq2
                                : (1.f - bs2 * kOvSqr6) * (1.f + bs2 * kOvSqr6)
                                    * secondRoots[rootIndex];
      const float cB2 = curved2 ? (1.f - cos2) / bq2 : 0.5f * sB2 * bs2;
      const float x1 = x01 + sB1 * px1 + cB1 * py1;
      const float y1 = y01 - cB1 * px1 + sB1 * py1;
      const float z1 = z01 + firstRoots[rootIndex] * pz1;
      const float x2 = x02 + sB2 * px2 + cB2 * py2;
      const float y2 = y02 - cB2 * px2 + sB2 * py2;
      const float z2 = z02 + secondRoots[rootIndex] * pz2;
      const float dx = x1 - x2;
      const float dy = y1 - y2;
      const float dz = z1 - z2;
      rootDistance2[rootIndex] = dx * dx + dy * dy + dz * dz;
    }

    if (useMiddlePoint) {
      dS[0] = 0.5f * (firstRoots[0] + firstRoots[1]);
      dS[1] = 0.5f * (secondRoots[0] + secondRoots[1]);
    }
    else {
      const int selectedRoot = rootDistance2[0] < rootDistance2[1] ? 0 : 1;
      dS[0] = firstRoots[selectedRoot];
      dS[1] = secondRoots[selectedRoot];
    }

    const float bs1 = bq1 * dS[0];
    const float bs2 = bq2 * dS[1];
    float sin1 = 0.f;
    float cos1 = 0.f;
    float sin2 = 0.f;
    float cos2 = 0.f;
    CpuCompatibleSinCos(bs1, sin1, cos1);
    CpuCompatibleSinCos(bs2, sin2, cos2);
    const bool curved1 = Abs(bs1) > 1.e-8f;
    const bool curved2 = Abs(bs2) > 1.e-8f;
    const float sB1 = curved1 ? sin1 / bq1
                              : (1.f - bs1 * kOvSqr6) * (1.f + bs1 * kOvSqr6) * dS[0];
    const float cB1 = curved1 ? (1.f - cos1) / bq1 : 0.5f * sB1 * bs1;
    const float sB2 = curved2 ? sin2 / bq2
                              : (1.f - bs2 * kOvSqr6) * (1.f + bs2 * kOvSqr6) * dS[1];
    const float cB2 = curved2 ? (1.f - cos2) / bq2 : 0.5f * sB2 * bs2;
    const float x1 = x01 + sB1 * px1 + cB1 * py1;
    const float y1 = y01 - cB1 * px1 + sB1 * py1;
    const float z1 = z01 + dS[0] * pz1;
    const float ppx1 = cos1 * px1 + sin1 * py1;
    const float ppy1 = -sin1 * px1 + cos1 * py1;
    const float x2 = x02 + sB2 * px2 + cB2 * py2;
    const float y2 = y02 - cB2 * px2 + sB2 * py2;
    const float z2 = z02 + dS[1] * pz2;
    const float ppx2 = cos2 * px2 + sin2 * py2;
    const float ppy2 = -sin2 * px2 + cos2 * py2;
    const float p12 = ppx1 * ppx1 + ppy1 * ppy1 + pz1 * pz1;
    const float p22 = ppx2 * ppx2 + ppy2 * ppy2 + pz2 * pz2;
    const float p1p2AtDca = ppx1 * ppx2 + ppy1 * ppy2 + pz1 * pz2;
    const float dx = x2 - x1;
    const float dy = y2 - y1;
    const float dz = z2 - z1;
    const float drp1AtDca = ppx1 * dx + ppy1 * dy + pz1 * dz;
    const float drp2AtDca = ppx2 * dx + ppy2 * dy + pz2 * dz;
    float determinant = p1p2AtDca * p1p2AtDca - p12 * p22;
    if (Abs(determinant) < 1.e-4f) {
      determinant = 1.f;
    }
    dS[0] += (drp2AtDca * p1p2AtDca - drp1AtDca * p22) / determinant;
    dS[1] += (drp2AtDca * p12 - drp1AtDca * p1p2AtDca) / determinant;
  }

  KFPARTICLE_GPU_HOST_DEVICE inline void GetDStoParticleByCpuCompatible(
    const KFParticleGpuFitState& first,
    const KFParticleGpuFitState& second,
    float by,
    float dS[2],
    bool useMiddlePoint,
    float firstRoots[2],
    float secondRoots[2])
  {
    const float firstParameters[6] = {
      first.X(), -first.Z(), first.Y(), first.Px(), -first.Pz(), first.Py()};
    const float secondParameters[6] = {
      second.X(), -second.Z(), second.Y(), second.Px(), -second.Pz(), second.Py()};
    GetDStoParticleBzCpuCompatible(first,
                                   second,
                                   by,
                                   firstParameters,
                                   secondParameters,
                                   dS,
                                   useMiddlePoint,
                                   firstRoots,
                                   secondRoots);
  }

  KFPARTICLE_GPU_HOST_DEVICE inline bool GetCpuCompatibleMiddleRootSelfDerivatives(
    const KFParticleGpuFitState& first,
    const KFParticleGpuFitState& second,
    float field,
    float firstDsdr[6],
    float secondDsdr[6])
  {
    const float kCLight = 0.000299792458f;
    const float bq1 = field * static_cast<float>(first.Q()) * kCLight;
    const float bq2 = field * static_cast<float>(second.Q()) * kCLight;
    if (Abs(bq1) < 1.e-8f || Abs(bq2) < 1.e-8f) {
      return false;
    }
    const float p1[6] = {first.X(), -first.Z(), first.Y(), first.Px(), -first.Pz(), first.Py()};
    const float p2[6] = {second.X(), -second.Z(), second.Y(), second.Px(), -second.Pz(), second.Py()};
    const float px1 = p1[3], py1 = p1[4], px2 = p2[3], py2 = p2[4];
    const float dx0 = p1[0] - p2[0], dy0 = p1[1] - p2[1];
    const float pt12 = px1 * px1 + py1 * py1;
    const float pt22 = px2 * px2 + py2 * py2;
    const float dr02 = dx0 * dx0 + dy0 * dy0;
    const float drp1 = dx0 * px1 + dy0 * py1;
    const float dxyp1 = dx0 * py1 - dy0 * px1;
    const float drp2 = dx0 * px2 + dy0 * py2;
    const float dxyp2 = dx0 * py2 - dy0 * px2;
    const float p1p2 = px1 * px2 + py1 * py2;
    const float dp1p2 = px1 * py2 - px2 * py1;
    const float k11 = bq2 * drp1 - dp1p2;
    const float k21 = bq1 * (bq2 * dxyp1 - p1p2) + bq2 * pt12;
    const float k12 = bq1 * drp2 - dp1p2;
    const float k22 = bq2 * (bq1 * dxyp2 + p1p2) - bq1 * pt22;
    const float kd = 0.5f * dr02 * bq1 * bq2 + dxyp1 * bq2 - dxyp2 * bq1 - p1p2;
    const float c1 = -(bq1 * kd + pt12 * bq2);
    const float c2 = bq2 * kd + pt22 * bq1;
    const float discriminant = pt12 * pt22 - kd * kd;
    const float root = KFParticleGpuSqrt(discriminant > 0.f ? discriminant : 0.f);
    if (!(root > 0.f)) {
      return false;
    }

    const float dk11dr1[6] = {bq2 * px1, bq2 * py1, 0.f,
                               bq2 * dx0 - py2, bq2 * dy0 + px2, 0.f};
    const float dk12dr2[6] = {-bq1 * px2, -bq1 * py2, 0.f,
                               bq1 * dx0 + py1, bq1 * dy0 - px1, 0.f};
    const float dk21dr1[6] = {
      bq1 * bq2 * py1, -bq1 * bq2 * px1, 0.f,
      2.f * bq2 * px1 + bq1 * (-bq2 * dy0 - px2),
      2.f * bq2 * py1 + bq1 * (bq2 * dx0 - py2), 0.f};
    const float dk22dr2[6] = {
      -bq1 * bq2 * py2, bq1 * bq2 * px2, 0.f,
      bq2 * (-bq1 * dy0 + px1) - 2.f * bq1 * px2,
      bq2 * (bq1 * dx0 + py1) - 2.f * bq1 * py2, 0.f};
    const float dkddr1[6] = {
      bq1 * bq2 * dx0 + bq2 * py1 - bq1 * py2,
      bq1 * bq2 * dy0 - bq2 * px1 + bq1 * px2, 0.f,
      -bq2 * dy0 - px2, bq2 * dx0 - py2, 0.f};
    const float dkddr2[6] = {
      -bq1 * bq2 * dx0 - bq2 * py1 + bq1 * py2,
      -bq1 * bq2 * dy0 + bq2 * px1 - bq1 * px2, 0.f,
      bq1 * dy0 - px1, -bq1 * dx0 - py1, 0.f};
    const float dc1dr1[6] = {
      -bq1 * (bq1 * bq2 * dx0 + bq2 * py1 - bq1 * py2),
      -bq1 * (bq1 * bq2 * dy0 - bq2 * px1 + bq1 * px2), 0.f,
      -2.f * bq2 * px1 - bq1 * (-bq2 * dy0 - px2),
      -2.f * bq2 * py1 - bq1 * (bq2 * dx0 - py2), 0.f};
    const float dc2dr2[6] = {
      bq2 * (-bq1 * bq2 * dx0 - bq2 * py1 + bq1 * py2),
      bq2 * (-bq1 * bq2 * dy0 + bq2 * px1 - bq1 * px2), 0.f,
      bq2 * (bq1 * dy0 - px1) + 2.f * bq1 * px2,
      bq2 * (-bq1 * dx0 - py1) + 2.f * bq1 * py2, 0.f};

    float rootDr1[6] = {};
    float rootDr2[6] = {};
    for (int i = 0; i < 6; ++i) {
      rootDr1[i] = -kd / root * dkddr1[i];
      rootDr2[i] = -kd / root * dkddr2[i];
    }
    rootDr1[3] += px1 / root * pt22;
    rootDr1[4] += py1 / root * pt22;
    rootDr2[3] += px2 / root * pt12;
    rootDr2[4] += py2 / root * pt12;

    float firstRootDerivatives[2][6] = {};
    float secondRootDerivatives[2][6] = {};
    for (int rootIndex = 0; rootIndex < 2; ++rootIndex) {
      const float sign = rootIndex == 0 ? 1.f : -1.f;
      const float firstA = bq1 * (k11 * c1 + sign * k21 * root);
      const float firstB = rootIndex == 0
        ? bq1 * bq1 * k11 * root - k21 * c1
        : -bq1 * bq1 * k11 * root - k21 * c1;
      const float secondA = bq2 * (k12 * c2 + sign * k22 * root);
      const float secondB = rootIndex == 0
        ? bq2 * bq2 * k12 * root - k22 * c2
        : -bq2 * bq2 * k12 * root - k22 * c2;
      for (int i = 0; i < 6; ++i) {
        const float firstDa = bq1 * (dk11dr1[i] * c1 + k11 * dc1dr1[i]
                                      + sign * (dk21dr1[i] * root + k21 * rootDr1[i]));
        const float firstDb = (rootIndex == 0 ? 1.f : -1.f) * bq1 * bq1
                                * (dk11dr1[i] * root + k11 * rootDr1[i])
                              - dk21dr1[i] * c1 - k21 * dc1dr1[i];
        firstRootDerivatives[rootIndex][i] =
          (firstDa * firstB - firstDb * firstA)
          / (bq1 * (firstB * firstB + firstA * firstA));

        const float secondDa = bq2 * (dk12dr2[i] * c2 + k12 * dc2dr2[i]
                                       + sign * (dk22dr2[i] * root + k22 * rootDr2[i]));
        const float secondDb = (rootIndex == 0 ? 1.f : -1.f) * bq2 * bq2
                                 * (dk12dr2[i] * root + k12 * rootDr2[i])
                               - dk22dr2[i] * c2 - k22 * dc2dr2[i];
        secondRootDerivatives[rootIndex][i] =
          (secondDa * secondB - secondDb * secondA)
          / (bq2 * (secondB * secondB + secondA * secondA));
      }
    }
    for (int i = 0; i < 6; ++i) {
      firstDsdr[i] = 0.5f * (firstRootDerivatives[0][i] + firstRootDerivatives[1][i]);
      secondDsdr[i] = 0.5f * (secondRootDerivatives[0][i] + secondRootDerivatives[1][i]);
    }
    return true;
  }

  KFPARTICLE_GPU_HOST_DEVICE inline void GetCpuCompatibleSelectedDcaSelfDerivatives(
    const KFParticleGpuFitState& first,
    const KFParticleGpuFitState& second,
    float by,
    bool useMiddlePoint,
    float firstDsdr[6],
    float secondDsdr[6]);

  KFPARTICLE_GPU_HOST_DEVICE inline bool UseCpuCompatibleMiddleDcaPoint(
    const KFParticleGpuFitState& first,
    const KFParticleGpuFitState& second,
    const KFParticleGpuFieldRegion& firstField,
    const KFParticleGpuFieldRegion& secondField,
    float by,
    float* outputDistance2 = nullptr,
    float* outputProjectedCovariance = nullptr,
    float* outputFirstDsdr = nullptr,
    float* outputSecondDsdr = nullptr)
  {
    float firstRoots[2] = {0.f, 0.f};
    float secondRoots[2] = {0.f, 0.f};
    float unusedDs[2] = {0.f, 0.f};
    GetDStoParticleByCpuCompatible(
      first, second, by, unusedDs, false, firstRoots, secondRoots);

    float firstDsdr[6] = {};
    float secondDsdr[6] = {};
    if (!GetCpuCompatibleMiddleRootSelfDerivatives(
          first, second, by, firstDsdr, secondDsdr)) {
      GetCpuCompatibleSelectedDcaSelfDerivatives(
        first, second, by, true, firstDsdr, secondDsdr);
    }
    if (outputFirstDsdr || outputSecondDsdr) {
      for (int parameter = 0; parameter < 6; ++parameter) {
        if (outputFirstDsdr) {
          outputFirstDsdr[parameter] = firstDsdr[parameter];
        }
        if (outputSecondDsdr) {
          outputSecondDsdr[parameter] = secondDsdr[parameter];
        }
      }
    }

    KFParticleGpuFitState firstAtMiddle;
    KFParticleGpuFitState secondAtMiddle;
    unsigned int firstStatus = KFGpuFullFieldTransportInvalidInput;
    unsigned int secondStatus = KFGpuFullFieldTransportInvalidInput;
    if (!TransportFullField(first,
                            firstField,
                            0.5f * (firstRoots[0] + firstRoots[1]),
                            firstAtMiddle,
                            firstStatus,
                            firstDsdr)
        || !TransportFullField(second,
                               secondField,
                               0.5f * (secondRoots[0] + secondRoots[1]),
                               secondAtMiddle,
                               secondStatus,
                               secondDsdr)) {
      return false;
    }

    const float dx = firstAtMiddle.X() - secondAtMiddle.X();
    const float dy = firstAtMiddle.Y() - secondAtMiddle.Y();
    const float dz = firstAtMiddle.Z() - secondAtMiddle.Z();
    const float distance2 = dx * dx + dy * dy + dz * dz;
    const float c00 = firstAtMiddle.Covariance(0) + secondAtMiddle.Covariance(0);
    const float c10 = firstAtMiddle.Covariance(1) + secondAtMiddle.Covariance(1);
    const float c11 = firstAtMiddle.Covariance(2) + secondAtMiddle.Covariance(2);
    const float c20 = firstAtMiddle.Covariance(3) + secondAtMiddle.Covariance(3);
    const float c21 = firstAtMiddle.Covariance(4) + secondAtMiddle.Covariance(4);
    const float c22 = firstAtMiddle.Covariance(5) + secondAtMiddle.Covariance(5);
    const float projectedCovariance = dx * dx * c00 + dy * dy * c11 + dz * dz * c22
                                      + 2.f * (dx * dy * c10 + dx * dz * c20 + dy * dz * c21);
    if (outputDistance2) {
      *outputDistance2 = distance2;
    }
    if (outputProjectedCovariance) {
      *outputProjectedCovariance = projectedCovariance;
    }
    const bool covarianceAmbiguous =
      IsFinite(distance2) && IsFinite(projectedCovariance) && projectedCovariance > 0.f
      && distance2 * distance2 < 25.f * projectedCovariance;
    return covarianceAmbiguous;
  }

  KFPARTICLE_GPU_HOST_DEVICE inline void GetCpuCompatibleSelectedDcaSelfDerivatives(
    const KFParticleGpuFitState& first,
    const KFParticleGpuFitState& second,
    float by,
    bool useMiddlePoint,
    float firstDsdr[6],
    float secondDsdr[6])
  {
    float dS[2] = {};
    float dsdr[4][6] = {};
    GetDStoParticleByCpuCompatibleDerivatives(
      first, second, by, useMiddlePoint, dS, dsdr);
    for (int parameter = 0; parameter < 6; ++parameter) {
      firstDsdr[parameter] = dsdr[0][parameter];
      secondDsdr[parameter] = dsdr[3][parameter];
    }
  }

  KFPARTICLE_GPU_HOST_DEVICE inline bool BuildFullFieldDcaKinematicSeed(
    const KFParticleGpuFitState& first,
    const KFParticleGpuFitState& second,
    const KFParticleGpuFieldRegion& firstField,
    const KFParticleGpuFieldRegion& secondField,
    KFParticleGpuFitState& firstAtDca,
    KFParticleGpuFitState& secondAtDca)
  {
    float dS[2] = {0.f, 0.f};
    const float by = firstField.Get(first.Z()).y;
    const bool firstIsStraight = Abs(by * static_cast<float>(first.Q())) < 1.e-8f;
    const bool secondIsStraight = Abs(by * static_cast<float>(second.Q())) < 1.e-8f;
    if (firstIsStraight && secondIsStraight) {
      GetDStoParticleLine(first, second, dS);
    }
    else {
      const bool useMiddlePoint =
        UseCpuCompatibleMiddleDcaPoint(first, second, firstField, secondField, by);
      GetDStoParticleByCpuCompatible(first, second, by, dS, useMiddlePoint);
    }

    unsigned int firstStatus = KFGpuFullFieldTransportInvalidInput;
    unsigned int secondStatus = KFGpuFullFieldTransportInvalidInput;
    if (!TransportFullField(first, firstField, dS[0], firstAtDca, firstStatus)
        || !TransportFullField(second, secondField, dS[1], secondAtDca, secondStatus)
        || (firstStatus & KFGpuFullFieldTransportPathLimited)
        || (secondStatus & KFGpuFullFieldTransportPathLimited)) {
      firstAtDca.Initialize();
      secondAtDca.Initialize();
      return false;
    }
    return true;
  }

  KFPARTICLE_GPU_HOST_DEVICE inline float FullFieldDcaDerivativeStep(float value)
  {
    const float scale = Abs(value) > 1.f ? Abs(value) : 1.f;
    return 1.e-3f * scale;
  }

  KFPARTICLE_GPU_HOST_DEVICE inline bool StoreFullFieldDcaJacobianColumn(
    const KFParticleGpuFitState& firstPlus,
    const KFParticleGpuFitState& firstMinus,
    const KFParticleGpuFitState& secondPlus,
    const KFParticleGpuFitState& secondMinus,
    const KFParticleGpuFieldRegion& firstField,
    const KFParticleGpuFieldRegion& secondField,
    float step,
    int column,
    KFParticleGpuFullFieldDcaResult& result,
    unsigned int& status)
  {
    KFParticleGpuFitState firstAtPlus;
    KFParticleGpuFitState secondAtPlus;
    KFParticleGpuFitState firstAtMinus;
    KFParticleGpuFitState secondAtMinus;
    if (!BuildFullFieldDcaKinematicSeed(
          firstPlus, secondPlus, firstField, secondField, firstAtPlus, secondAtPlus)
        || !BuildFullFieldDcaKinematicSeed(
          firstMinus, secondMinus, firstField, secondField, firstAtMinus, secondAtMinus)) {
      result.Initialize();
      status |= KFGpuFullFieldDcaRejected;
      return false;
    }

    for (int row = 0; row < 6; ++row) {
      const float firstDerivative =
        (firstAtPlus.Parameter(row) - firstAtMinus.Parameter(row)) / (2.f * step);
      const float secondDerivative =
        (secondAtPlus.Parameter(row) - secondAtMinus.Parameter(row)) / (2.f * step);
      if (!IsFinite(firstDerivative) || !IsFinite(secondDerivative)) {
        result.Initialize();
        status |= KFGpuFullFieldDcaNonFinite;
        return false;
      }
      result.FirstJacobian(row, column) = firstDerivative;
      result.SecondJacobian(row, column) = secondDerivative;
    }

    return true;
  }

  KFPARTICLE_GPU_HOST_DEVICE inline bool BuildFullFieldDcaCoupledResult(
    const KFParticleGpuFitState& first,
    const KFParticleGpuFitState& second,
    const KFParticleGpuFieldRegion& firstField,
    const KFParticleGpuFieldRegion& secondField,
    KFParticleGpuFullFieldDcaResult& result,
    unsigned int& status)
  {
    result.Initialize();
    status = KFGpuFullFieldDcaSuccess;
    if (!IsFiniteState(first) || !IsFiniteState(second)) {
      status = KFGpuFullFieldDcaNonFinite;
      return false;
    }
    for (int component = 0; component < KFParticleGpuFieldRegion::NumberOfCoefficients; ++component) {
      if (!IsFinite(firstField.Coefficient(component)) || !IsFinite(secondField.Coefficient(component))) {
        status = KFGpuFullFieldDcaNonFinite;
        return false;
      }
    }
    if (first.Q() == 0 || second.Q() == 0) {
      status |= KFGpuFullFieldDcaNeutral;
    }
    if (!BuildFullFieldDcaKinematicSeed(
          first, second, firstField, secondField, result.first, result.second)) {
      status |= KFGpuFullFieldDcaRejected;
      return false;
    }

    for (int parameter = 0; parameter < 6; ++parameter) {
      KFParticleGpuFitState firstPlus = first;
      KFParticleGpuFitState firstMinus = first;
      const float step = FullFieldDcaDerivativeStep(first.Parameter(parameter));
      firstPlus.Parameter(parameter) += step;
      firstMinus.Parameter(parameter) -= step;
      if (!StoreFullFieldDcaJacobianColumn(firstPlus, firstMinus, second, second, firstField,
                                           secondField, step, parameter, result, status)) {
        return false;
      }
    }
    for (int parameter = 0; parameter < 6; ++parameter) {
      KFParticleGpuFitState secondPlus = second;
      KFParticleGpuFitState secondMinus = second;
      const float step = FullFieldDcaDerivativeStep(second.Parameter(parameter));
      secondPlus.Parameter(parameter) += step;
      secondMinus.Parameter(parameter) -= step;
      if (!StoreFullFieldDcaJacobianColumn(first, first, secondPlus, secondMinus, firstField,
                                           secondField, step, 6 + parameter, result, status)) {
        return false;
      }
    }

    for (int row = 0; row < 6; ++row) {
      for (int column = 0; column <= row; ++column) {
        float firstValue = 0.f;
        float secondValue = 0.f;
        for (int left = 0; left < 6; ++left) {
          for (int right = 0; right < 6; ++right) {
            const float firstCovariance = first.Covariance(left, right);
            firstValue += result.FirstJacobian(row, left) * firstCovariance
                          * result.FirstJacobian(column, right);
            secondValue += result.SecondJacobian(row, left) * firstCovariance
                           * result.SecondJacobian(column, right);
            const float secondCovariance = second.Covariance(left, right);
            firstValue += result.FirstJacobian(row, 6 + left) * secondCovariance
                          * result.FirstJacobian(column, 6 + right);
            secondValue += result.SecondJacobian(row, 6 + left) * secondCovariance
                           * result.SecondJacobian(column, 6 + right);
          }
        }
        result.FirstCovariance(row, column) = firstValue;
        result.SecondCovariance(row, column) = secondValue;
      }
    }

    // The DCA location depends on both six-parameter input states. The normal
    // full-field transport above uses a fixed dS, so its E-to-state covariance
    // misses that dependence. Rebuild these terms from the same total
    // numerical Jacobians as the 6x6 blocks. Energy itself is unchanged by
    // transport and each daughter input is independent of the other one.
    for (int row = 0; row < 6; ++row) {
      float firstEnergyCovariance = 0.f;
      float secondEnergyCovariance = 0.f;
      for (int sourceParameter = 0; sourceParameter < 6; ++sourceParameter) {
        firstEnergyCovariance += result.FirstJacobian(row, sourceParameter)
                                 * first.Covariance(sourceParameter, 6);
        secondEnergyCovariance += result.SecondJacobian(row, 6 + sourceParameter)
                                  * second.Covariance(sourceParameter, 6);
      }
      result.first.Covariance(6, row) = firstEnergyCovariance;
      result.second.Covariance(6, row) = secondEnergyCovariance;
    }
    result.first.Covariance(6, 6) = first.Covariance(6, 6);
    result.second.Covariance(6, 6) = second.Covariance(6, 6);
    for (int row = 0; row < 3; ++row) {
      for (int column = 0; column < 3; ++column) {
        float value = 0.f;
        for (int left = 0; left < 6; ++left) {
          for (int right = 0; right < 6; ++right) {
            value += result.FirstJacobian(row, left) * first.Covariance(left, right)
                     * result.SecondJacobian(column, right);
            value += result.FirstJacobian(row, 6 + left) * second.Covariance(left, right)
                     * result.SecondJacobian(column, 6 + right);
          }
        }
        if (!IsFinite(value)) {
          result.Initialize();
          status |= KFGpuFullFieldDcaNonFinite;
          return false;
        }
        result.Correlation(row, column) = value;
      }
    }
    return true;
  }
  KFPARTICLE_GPU_HOST_DEVICE inline bool BuildFullFieldDcaKinematicMother(
    const KFParticleGpuFitState& first,
    const KFParticleGpuFitState& second,
    const KFParticleGpuFieldRegion& firstField,
    const KFParticleGpuFieldRegion& secondField,
    KFParticleGpuFitState& mother)
  {
    KFParticleGpuFitState firstAtDca;
    KFParticleGpuFitState secondAtDca;
    if (!BuildFullFieldDcaKinematicSeed(
          first, second, firstField, secondField, firstAtDca, secondAtDca)) {
      mother.Initialize();
      return false;
    }
    BuildKinematicMother(firstAtDca, secondAtDca, mother);
    return true;
  }

  KFPARTICLE_GPU_HOST_DEVICE inline bool BuildLineDcaMeasurementSeed(
    const KFParticleGpuFitState& first,
    const KFParticleGpuFitState& second,
    KFParticleGpuFitState& currentAtDca,
    KFParticleGpuMeasurement& daughterMeasurement)
  {
    float dS[2] = {0.f, 0.f};
    float dsdr[4][6];
    GetDStoParticleLine(first, second, dS, dsdr);

    if (Abs(dS[0] * first.Pz()) > 1000.f || Abs(dS[1] * second.Pz()) > 1000.f) {
      currentAtDca.Initialize();
      daughterMeasurement.Initialize();
      return false;
    }

    float firstSelfDerivative[36];
    float firstOtherDerivative[36];
    float secondSelfDerivative[36];
    float secondOtherDerivative[36];
    for (int i = 0; i < 36; ++i) {
      firstSelfDerivative[i] = 0.f;
      firstOtherDerivative[i] = 0.f;
      secondSelfDerivative[i] = 0.f;
      secondOtherDerivative[i] = 0.f;
    }

    TransportLineWithJacobian(
      first, dS[0], dsdr[0], currentAtDca, dsdr[1], firstSelfDerivative, firstOtherDerivative);
    KFParticleGpuFitState secondAtDca;
    TransportLineWithJacobian(
      second, dS[1], dsdr[3], secondAtDca, dsdr[2], secondSelfDerivative, secondOtherDerivative);
    daughterMeasurement.StoreState(secondAtDca);

    float firstCovarianceFromSecond[36];
    float secondCovarianceFromFirst[36];
    MultQSQt<6>(firstOtherDerivative, second.Covariances(), firstCovarianceFromSecond);
    MultQSQt<6>(secondOtherDerivative, first.Covariances(), secondCovarianceFromFirst);

    for (int i = 0; i < 21; ++i) {
      currentAtDca.Covariance(i) += firstCovarianceFromSecond[i];
      daughterMeasurement.Covariance(i) += secondCovarianceFromFirst[i];
    }

    float firstCovFirstDerivativeT[6][6];
    for (int i = 0; i < 6; ++i) {
      for (int j = 0; j < 6; ++j) {
        firstCovFirstDerivativeT[i][j] = 0.f;
        for (int k = 0; k < 6; ++k) {
          firstCovFirstDerivativeT[i][j] +=
            first.Covariance(i, k) * firstSelfDerivative[j * 6 + k];
        }
      }
    }

    float secondOtherFirstCovFirstDerivativeT[6][6];
    for (int i = 0; i < 6; ++i) {
      for (int j = 0; j < 6; ++j) {
        secondOtherFirstCovFirstDerivativeT[i][j] = 0.f;
        for (int k = 0; k < 6; ++k) {
          secondOtherFirstCovFirstDerivativeT[i][j] +=
            secondOtherDerivative[i * 6 + k] * firstCovFirstDerivativeT[k][j];
        }
      }
    }

    float secondCovFirstOtherDerivativeT[6][6];
    for (int i = 0; i < 6; ++i) {
      for (int j = 0; j < 6; ++j) {
        secondCovFirstOtherDerivativeT[i][j] = 0.f;
        for (int k = 0; k < 6; ++k) {
          secondCovFirstOtherDerivativeT[i][j] +=
            second.Covariance(i, k) * firstOtherDerivative[j * 6 + k];
        }
      }
    }

    for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 3; ++j) {
        daughterMeasurement.Correlation(i, j) =
          secondOtherFirstCovFirstDerivativeT[i][j];
        for (int k = 0; k < 6; ++k) {
          daughterMeasurement.Correlation(i, j) +=
            secondSelfDerivative[i * 6 + k] * secondCovFirstOtherDerivativeT[k][j];
        }
      }
    }
    return true;
  }

  KFPARTICLE_GPU_HOST_DEVICE inline bool BuildConstantByDcaMeasurementSeedApprox(
    const KFParticleGpuFitState& first,
    const KFParticleGpuFitState& second,
    float by,
    KFParticleGpuFitState& currentAtDca,
    KFParticleGpuMeasurement& daughterMeasurement)
  {
    if (!BuildLineDcaMeasurementSeed(first, second, currentAtDca, daughterMeasurement)) {
      return false;
    }

    KFParticleGpuFitState firstAtDca;
    KFParticleGpuFitState secondAtDca;
    if (!BuildConstantByDcaKinematicSeed(first, second, by, firstAtDca, secondAtDca)) {
      currentAtDca.Initialize();
      daughterMeasurement.Initialize();
      return false;
    }

    // Until the full CBM transport Jacobian is ported, reuse the line-DCA
    // covariance/correlation terms and replace only the transported states.
    for (int i = 0; i < KFParticleGpuFitState::NumberOfParameters; ++i) {
      currentAtDca.Parameter(i) = firstAtDca.Parameter(i);
    }
    currentAtDca.S() = firstAtDca.S();
    currentAtDca.SFromDecay() = firstAtDca.SFromDecay();

    for (int i = 0; i < KFParticleGpuFitState::NumberOfParameters; ++i) {
      daughterMeasurement.Parameter(i) = secondAtDca.Parameter(i);
    }
    return true;
  }

  KFPARTICLE_GPU_HOST_DEVICE inline bool BuildFullFieldDcaMeasurementSeedAnalytic(
    const KFParticleGpuFitState& first,
    const KFParticleGpuFitState& second,
    const KFParticleGpuFieldRegion& firstField,
    const KFParticleGpuFieldRegion& secondField,
    KFParticleGpuFitState& currentAtDca,
    KFParticleGpuMeasurement& daughterMeasurement)
  {
#if defined(__clang__)
#pragma clang fp contract(off)
#endif
    const float by = firstField.Get(first.Z()).y;
    const bool useMiddlePoint =
      UseCpuCompatibleMiddleDcaPoint(first, second, firstField, secondField, by);
    float dS[2] = {};
    float dsdr[4][6] = {};
    GetDStoParticleByCpuCompatibleDerivatives(
      first, second, by, useMiddlePoint, dS, dsdr);

    float firstSelfDerivative[36] = {};
    float firstOtherDerivative[36] = {};
    float secondSelfDerivative[36] = {};
    float secondOtherDerivative[36] = {};
    unsigned int firstStatus = KFGpuFullFieldTransportInvalidInput;
    unsigned int secondStatus = KFGpuFullFieldTransportInvalidInput;
    KFParticleGpuFitState secondAtDca;
    if (!TransportFullField(first,
                            firstField,
                            dS[0],
                            currentAtDca,
                            firstStatus,
                            dsdr[0],
                            dsdr[1],
                            firstSelfDerivative,
                            firstOtherDerivative)
        || !TransportFullField(second,
                               secondField,
                               dS[1],
                               secondAtDca,
                               secondStatus,
                               dsdr[3],
                               dsdr[2],
                               secondSelfDerivative,
                               secondOtherDerivative)
        || (firstStatus & KFGpuFullFieldTransportPathLimited)
        || (secondStatus & KFGpuFullFieldTransportPathLimited)) {
      currentAtDca.Initialize();
      daughterMeasurement.Initialize();
      return false;
    }
    daughterMeasurement.StoreState(secondAtDca);

    float firstCovarianceFromSecond[36] = {};
    float secondCovarianceFromFirst[36] = {};
    MultQSQt<6>(firstOtherDerivative, second.Covariances(), firstCovarianceFromSecond);
    MultQSQt<6>(secondOtherDerivative, first.Covariances(), secondCovarianceFromFirst);
    for (int covariance = 0; covariance < 21; ++covariance) {
      currentAtDca.Covariance(covariance) += firstCovarianceFromSecond[covariance];
      daughterMeasurement.Covariance(covariance) += secondCovarianceFromFirst[covariance];
    }

    float firstCovFirstDerivativeT[6][6] = {};
    for (int row = 0; row < 6; ++row) {
      for (int column = 0; column < 6; ++column) {
        for (int source = 0; source < 6; ++source) {
          firstCovFirstDerivativeT[row][column] +=
            first.Covariance(row, source) * firstSelfDerivative[column * 6 + source];
        }
      }
    }
    float secondOtherFirstCovFirstDerivativeT[6][6] = {};
    for (int row = 0; row < 6; ++row) {
      for (int column = 0; column < 6; ++column) {
        for (int source = 0; source < 6; ++source) {
          secondOtherFirstCovFirstDerivativeT[row][column] +=
            secondOtherDerivative[row * 6 + source]
            * firstCovFirstDerivativeT[source][column];
        }
      }
    }
    float secondCovFirstOtherDerivativeT[6][6] = {};
    for (int row = 0; row < 6; ++row) {
      for (int column = 0; column < 6; ++column) {
        for (int source = 0; source < 6; ++source) {
          secondCovFirstOtherDerivativeT[row][column] +=
            second.Covariance(row, source) * firstOtherDerivative[column * 6 + source];
        }
      }
    }
    for (int row = 0; row < 3; ++row) {
      for (int column = 0; column < 3; ++column) {
        daughterMeasurement.Correlation(row, column) =
          secondOtherFirstCovFirstDerivativeT[row][column];
        for (int source = 0; source < 6; ++source) {
          daughterMeasurement.Correlation(row, column) +=
            secondSelfDerivative[row * 6 + source]
            * secondCovFirstOtherDerivativeT[source][column];
        }
      }
    }
    return true;
  }

  KFPARTICLE_GPU_HOST_DEVICE inline bool BuildFullFieldDcaMeasurementSeedApprox(
    const KFParticleGpuFitState& first,
    const KFParticleGpuFitState& second,
    const KFParticleGpuFieldRegion& firstField,
    const KFParticleGpuFieldRegion& secondField,
    KFParticleGpuFitState& currentAtDca,
    KFParticleGpuMeasurement& daughterMeasurement)
  {
    // Full-field transport supplies the fit states and their covariance. The
    // cross-daughter correlation remains the validated line-DCA approximation
    // until the coupled full-field derivative is implemented in a later step.
    if (!BuildLineDcaMeasurementSeed(first, second, currentAtDca, daughterMeasurement)) {
      return false;
    }

    KFParticleGpuFitState firstAtDca;
    KFParticleGpuFitState secondAtDca;
    if (!BuildFullFieldDcaKinematicSeed(
          first, second, firstField, secondField, firstAtDca, secondAtDca)) {
      currentAtDca.Initialize();
      daughterMeasurement.Initialize();
      return false;
    }

    currentAtDca = firstAtDca;
    daughterMeasurement.StoreState(secondAtDca);
    return true;
  }

  KFPARTICLE_GPU_HOST_DEVICE inline bool BuildFullFieldDcaMeasurementSeedCoupled(
    const KFParticleGpuFitState& first,
    const KFParticleGpuFitState& second,
    const KFParticleGpuFieldRegion& firstField,
    const KFParticleGpuFieldRegion& secondField,
    KFParticleGpuFitState& currentAtDca,
    KFParticleGpuMeasurement& daughterMeasurement)
  {
    KFParticleGpuFullFieldDcaResult coupled;
    unsigned int status = KFGpuFullFieldDcaRejected;
    if (!BuildFullFieldDcaCoupledResult(
          first, second, firstField, secondField, coupled, status)) {
      currentAtDca.Initialize();
      daughterMeasurement.Initialize();
      return false;
    }

    currentAtDca = coupled.first;
    daughterMeasurement.StoreState(coupled.second);
    // The coupled DCA Jacobians cover the six spatial/momentum coordinates
    // used by the DCA solve. Preserve energy/S transport terms from the
    // full-field state while replacing the affected covariance blocks.
    for (int row = 0; row < 6; ++row) {
      for (int column = 0; column <= row; ++column) {
        currentAtDca.Covariance(row, column) = coupled.FirstCovariance(row, column);
        daughterMeasurement.Covariance(row, column) = coupled.SecondCovariance(row, column);
      }
    }
    for (int row = 0; row < 3; ++row) {
      for (int column = 0; column < 3; ++column) {
        daughterMeasurement.Correlation(row, column) = coupled.Correlation(row, column);
      }
    }
    return true;
  }

  KFPARTICLE_GPU_HOST_DEVICE inline bool BuildFullFieldDcaIndependentTransportStates(
    const KFParticleGpuFitState& first,
    const KFParticleGpuFitState& second,
    const KFParticleGpuFieldRegion& firstField,
    const KFParticleGpuFieldRegion& secondField,
    KFParticleGpuFitState& firstAtDca,
    KFParticleGpuFitState& secondAtDca)
  {
    const float by = firstField.Get(first.Z()).y;
    const bool firstIsStraight = Abs(by * static_cast<float>(first.Q())) < 1.e-8f;
    const bool secondIsStraight = Abs(by * static_cast<float>(second.Q())) < 1.e-8f;
    float dS[2] = {};
    float firstDsdr[6] = {};
    float secondDsdr[6] = {};
    if (firstIsStraight && secondIsStraight) {
      GetDStoParticleLine(first, second, dS);
    }
    else {
      const bool useMiddlePoint =
        UseCpuCompatibleMiddleDcaPoint(first, second, firstField, secondField, by);
      float dsdr[4][6] = {};
      GetDStoParticleByCpuCompatibleDerivatives(
        first, second, by, useMiddlePoint, dS, dsdr);
      for (int parameter = 0; parameter < 6; ++parameter) {
        firstDsdr[parameter] = dsdr[0][parameter];
        secondDsdr[parameter] = dsdr[3][parameter];
      }
    }

    unsigned int firstStatus = KFGpuFullFieldTransportInvalidInput;
    unsigned int secondStatus = KFGpuFullFieldTransportInvalidInput;
    if (!TransportFullField(first, firstField, dS[0], firstAtDca, firstStatus, firstDsdr)
        || !TransportFullField(second, secondField, dS[1], secondAtDca, secondStatus, secondDsdr)
        || (firstStatus & KFGpuFullFieldTransportPathLimited)
        || (secondStatus & KFGpuFullFieldTransportPathLimited)) {
      firstAtDca.Initialize();
      secondAtDca.Initialize();
      return false;
    }
    return true;
  }

  KFPARTICLE_GPU_HOST_DEVICE inline bool BuildFullFieldDcaMeasurementSeedCpuCompatible(
    const KFParticleGpuFitState& first,
    const KFParticleGpuFitState& second,
    const KFParticleGpuFieldRegion& firstField,
    const KFParticleGpuFieldRegion& secondField,
    KFParticleGpuFitState& currentAtDca,
    KFParticleGpuMeasurement& daughterMeasurement)
  {
    KFParticleGpuFitState secondAtDca;
    if (!BuildFullFieldDcaIndependentTransportStates(
          first, second, firstField, secondField, currentAtDca, secondAtDca)) {
      daughterMeasurement.Initialize();
      return false;
    }
    daughterMeasurement.StoreState(secondAtDca);
    return true;
  }

  KFPARTICLE_GPU_HOST_DEVICE inline bool BuildFullFieldDcaMeasurementSeedCpuConstructV0(
    const KFParticleGpuFitState& first,
    const KFParticleGpuFitState& second,
    const KFParticleGpuFieldRegion& firstField,
    const KFParticleGpuFieldRegion& secondField,
    KFParticleGpuFitState& currentAtDca,
    KFParticleGpuMeasurement& daughterMeasurement)
  {
    // KFParticleFinder::ConstructV0 first transports both daughters
    // independently. KFParticleSIMD::Construct then invokes GetMeasurement,
    // which performs a second, coupled DCA transport before the Kalman update.
    KFParticleGpuFitState preliminaryFirst;
    KFParticleGpuFitState preliminarySecond;
    if (!BuildFullFieldDcaIndependentTransportStates(
          first, second, firstField, secondField, preliminaryFirst, preliminarySecond)) {
      currentAtDca.Initialize();
      daughterMeasurement.Initialize();
      return false;
    }
    return BuildFullFieldDcaMeasurementSeedAnalytic(
      preliminaryFirst, preliminarySecond, firstField, secondField,
      currentAtDca, daughterMeasurement);
  }

  KFPARTICLE_GPU_HOST_DEVICE inline float NumericalDerivativeStep(float value)
  {
    const float scale = Abs(value) > 1.f ? Abs(value) : 1.f;
    return 1.e-3f * scale;
  }

  KFPARTICLE_GPU_HOST_DEVICE inline void PerturbFitParameter(KFParticleGpuFitState& state,
                                                             int parameter,
                                                             float delta)
  {
    state.Parameter(parameter) += delta;
  }

  KFPARTICLE_GPU_HOST_DEVICE inline bool BuildConstantByDcaSeedJacobians(
    const KFParticleGpuFitState& first,
    const KFParticleGpuFitState& second,
    float by,
    const KFParticleGpuFitState& firstAtDca,
    const KFParticleGpuFitState& secondAtDca,
    float firstFirstJacobian[7][7],
    float firstSecondJacobian[7][7],
    float secondFirstJacobian[7][7],
    float secondSecondJacobian[7][7])
  {
    for (int row = 0; row < 7; ++row) {
      for (int column = 0; column < 7; ++column) {
        firstFirstJacobian[row][column] = 0.f;
        firstSecondJacobian[row][column] = 0.f;
        secondFirstJacobian[row][column] = 0.f;
        secondSecondJacobian[row][column] = 0.f;
      }
    }
    firstFirstJacobian[6][6] = 1.f;
    secondSecondJacobian[6][6] = 1.f;

    for (int parameter = 0; parameter < 6; ++parameter) {
      const float step = NumericalDerivativeStep(first.Parameter(parameter));
      KFParticleGpuFitState firstPlus = first;
      KFParticleGpuFitState firstMinus = first;
      PerturbFitParameter(firstPlus, parameter, step);
      PerturbFitParameter(firstMinus, parameter, -step);

      KFParticleGpuFitState firstPlusAtDca;
      KFParticleGpuFitState secondPlusAtDca;
      KFParticleGpuFitState firstMinusAtDca;
      KFParticleGpuFitState secondMinusAtDca;
      if (!BuildConstantByDcaKinematicSeed(
            firstPlus, second, by, firstPlusAtDca, secondPlusAtDca)
          || !BuildConstantByDcaKinematicSeed(
            firstMinus, second, by, firstMinusAtDca, secondMinusAtDca)) {
        return false;
      }

      const float inverseStep = 0.5f / step;
      for (int row = 0; row < 6; ++row) {
        firstFirstJacobian[row][parameter] =
          (firstPlusAtDca.Parameter(row) - firstMinusAtDca.Parameter(row)) * inverseStep;
        secondFirstJacobian[row][parameter] =
          (secondPlusAtDca.Parameter(row) - secondMinusAtDca.Parameter(row)) * inverseStep;
      }
    }

    for (int parameter = 0; parameter < 6; ++parameter) {
      const float step = NumericalDerivativeStep(second.Parameter(parameter));
      KFParticleGpuFitState secondPlus = second;
      KFParticleGpuFitState secondMinus = second;
      PerturbFitParameter(secondPlus, parameter, step);
      PerturbFitParameter(secondMinus, parameter, -step);

      KFParticleGpuFitState firstPlusAtDca;
      KFParticleGpuFitState secondPlusAtDca;
      KFParticleGpuFitState firstMinusAtDca;
      KFParticleGpuFitState secondMinusAtDca;
      if (!BuildConstantByDcaKinematicSeed(
            first, secondPlus, by, firstPlusAtDca, secondPlusAtDca)
          || !BuildConstantByDcaKinematicSeed(
            first, secondMinus, by, firstMinusAtDca, secondMinusAtDca)) {
        return false;
      }

      const float inverseStep = 0.5f / step;
      for (int row = 0; row < 6; ++row) {
        firstSecondJacobian[row][parameter] =
          (firstPlusAtDca.Parameter(row) - firstMinusAtDca.Parameter(row)) * inverseStep;
        secondSecondJacobian[row][parameter] =
          (secondPlusAtDca.Parameter(row) - secondMinusAtDca.Parameter(row)) * inverseStep;
      }
    }

    (void) firstAtDca;
    (void) secondAtDca;
    return true;
  }

  KFPARTICLE_GPU_HOST_DEVICE inline void PropagateTwoSourceCovariance7(
    const float firstJacobian[7][7],
    const KFParticleGpuFitState& first,
    const float secondJacobian[7][7],
    const KFParticleGpuFitState& second,
    KFParticleGpuFitState& destination)
  {
    for (int row = 0; row < 7; ++row) {
      for (int column = 0; column <= row; ++column) {
        float covariance = 0.f;
        for (int i = 0; i < 7; ++i) {
          for (int j = 0; j < 7; ++j) {
            covariance += firstJacobian[row][i] * first.Covariance(i, j) * firstJacobian[column][j];
            covariance += secondJacobian[row][i] * second.Covariance(i, j) * secondJacobian[column][j];
          }
        }
        destination.Covariance(row, column) = covariance;
      }
    }
  }

  KFPARTICLE_GPU_HOST_DEVICE inline void PropagateTwoSourceCovariance7(
    const float firstJacobian[7][7],
    const KFParticleGpuFitState& first,
    const float secondJacobian[7][7],
    const KFParticleGpuFitState& second,
    KFParticleGpuMeasurement& destination)
  {
    for (int row = 0; row < 7; ++row) {
      for (int column = 0; column <= row; ++column) {
        float covariance = 0.f;
        for (int i = 0; i < 7; ++i) {
          for (int j = 0; j < 7; ++j) {
            covariance += firstJacobian[row][i] * first.Covariance(i, j) * firstJacobian[column][j];
            covariance += secondJacobian[row][i] * second.Covariance(i, j) * secondJacobian[column][j];
          }
        }
        destination.Covariance(row, column) = covariance;
      }
    }
  }

  KFPARTICLE_GPU_HOST_DEVICE inline float CrossCovariance7(
    const float leftFirstJacobian[7][7],
    const float leftSecondJacobian[7][7],
    const float rightFirstJacobian[7][7],
    const float rightSecondJacobian[7][7],
    const KFParticleGpuFitState& first,
    const KFParticleGpuFitState& second,
    int leftRow,
    int rightRow)
  {
    float covariance = 0.f;
    for (int i = 0; i < 7; ++i) {
      for (int j = 0; j < 7; ++j) {
        covariance += leftFirstJacobian[leftRow][i] * first.Covariance(i, j)
                      * rightFirstJacobian[rightRow][j];
        covariance += leftSecondJacobian[leftRow][i] * second.Covariance(i, j)
                      * rightSecondJacobian[rightRow][j];
      }
    }
    return covariance;
  }

  KFPARTICLE_GPU_HOST_DEVICE inline bool BuildConstantByDcaMeasurementSeed(
    const KFParticleGpuFitState& first,
    const KFParticleGpuFitState& second,
    float by,
    KFParticleGpuFitState& currentAtDca,
    KFParticleGpuMeasurement& daughterMeasurement)
  {
    KFParticleGpuFitState firstAtDca;
    KFParticleGpuFitState secondAtDca;
    if (!BuildConstantByDcaKinematicSeed(first, second, by, firstAtDca, secondAtDca)) {
      currentAtDca.Initialize();
      daughterMeasurement.Initialize();
      return false;
    }

    float firstFirstJacobian[7][7];
    float firstSecondJacobian[7][7];
    float secondFirstJacobian[7][7];
    float secondSecondJacobian[7][7];
    if (!BuildConstantByDcaSeedJacobians(first,
                                         second,
                                         by,
                                         firstAtDca,
                                         secondAtDca,
                                         firstFirstJacobian,
                                         firstSecondJacobian,
                                         secondFirstJacobian,
                                         secondSecondJacobian)) {
      currentAtDca.Initialize();
      daughterMeasurement.Initialize();
      return false;
    }

    currentAtDca = firstAtDca;
    daughterMeasurement.StoreState(secondAtDca);
    PropagateTwoSourceCovariance7(firstFirstJacobian,
                                  first,
                                  firstSecondJacobian,
                                  second,
                                  currentAtDca);
    PropagateTwoSourceCovariance7(secondFirstJacobian,
                                  first,
                                  secondSecondJacobian,
                                  second,
                                  daughterMeasurement);

    for (int row = 0; row < 3; ++row) {
      for (int column = 0; column < 3; ++column) {
        daughterMeasurement.Correlation(row, column) =
          CrossCovariance7(secondFirstJacobian,
                           secondSecondJacobian,
                           firstFirstJacobian,
                           firstSecondJacobian,
                           first,
                           second,
                           row,
                           column);
      }
    }
    return true;
  }

  KFPARTICLE_GPU_HOST_DEVICE inline bool AddDaughterWithEnergyFit(
    KFParticleGpuFitState& particle,
    const KFParticleGpuMeasurement& measurement,
    int daughterCharge)
  {
#if defined(__clang__)
#pragma clang fp contract(off)
#endif
    float mS[6] = {
      particle.Covariance(0) + measurement.Covariance(0),
      particle.Covariance(1) + measurement.Covariance(1),
      particle.Covariance(2) + measurement.Covariance(2),
      particle.Covariance(3) + measurement.Covariance(3),
      particle.Covariance(4) + measurement.Covariance(4),
      particle.Covariance(5) + measurement.Covariance(5)};

    InvertCholetsky3(mS);

    const float zeta[3] = {
      measurement.Parameter(0) - particle.X(),
      measurement.Parameter(1) - particle.Y(),
      measurement.Parameter(2) - particle.Z()};

    const float dChi2 =
      (mS[0] * zeta[0] + mS[1] * zeta[1] + mS[3] * zeta[2]) * zeta[0]
      + (mS[1] * zeta[0] + mS[2] * zeta[1] + mS[4] * zeta[2]) * zeta[1]
      + (mS[3] * zeta[0] + mS[4] * zeta[1] + mS[5] * zeta[2]) * zeta[2];

    if (dChi2 > 1.e9f || dChi2 != dChi2) {
      return false;
    }

    float K[3][3];
    for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 3; ++j) {
        K[i][j] = 0.f;
        for (int k = 0; k < 3; ++k) {
          K[i][j] += particle.Covariance(i, k) * mS[KFParticleGpuFitState::CovarianceIndex(k, j)];
        }
      }
    }

    float mCHt0[7];
    float mCHt1[7];
    float mCHt2[7];

    mCHt0[0] = particle.Covariance(0);
    mCHt1[0] = particle.Covariance(1);
    mCHt2[0] = particle.Covariance(3);
    mCHt0[1] = particle.Covariance(1);
    mCHt1[1] = particle.Covariance(2);
    mCHt2[1] = particle.Covariance(4);
    mCHt0[2] = particle.Covariance(3);
    mCHt1[2] = particle.Covariance(4);
    mCHt2[2] = particle.Covariance(5);
    mCHt0[3] = particle.Covariance(6) - measurement.Covariance(6);
    mCHt1[3] = particle.Covariance(7) - measurement.Covariance(7);
    mCHt2[3] = particle.Covariance(8) - measurement.Covariance(8);
    mCHt0[4] = particle.Covariance(10) - measurement.Covariance(10);
    mCHt1[4] = particle.Covariance(11) - measurement.Covariance(11);
    mCHt2[4] = particle.Covariance(12) - measurement.Covariance(12);
    mCHt0[5] = particle.Covariance(15) - measurement.Covariance(15);
    mCHt1[5] = particle.Covariance(16) - measurement.Covariance(16);
    mCHt2[5] = particle.Covariance(17) - measurement.Covariance(17);
    mCHt0[6] = particle.Covariance(21) - measurement.Covariance(21);
    mCHt1[6] = particle.Covariance(22) - measurement.Covariance(22);
    mCHt2[6] = particle.Covariance(23) - measurement.Covariance(23);

    float k0[7];
    float k1[7];
    float k2[7];
    for (int i = 0; i < 7; ++i) {
      k0[i] = mCHt0[i] * mS[0] + mCHt1[i] * mS[1] + mCHt2[i] * mS[3];
      k1[i] = mCHt0[i] * mS[1] + mCHt1[i] * mS[2] + mCHt2[i] * mS[4];
      k2[i] = mCHt0[i] * mS[3] + mCHt1[i] * mS[4] + mCHt2[i] * mS[5];
    }

    particle.Px() += measurement.Parameter(3);
    particle.Py() += measurement.Parameter(4);
    particle.Pz() += measurement.Parameter(5);
    particle.E() += measurement.Parameter(6);

    particle.Covariance(9) += measurement.Covariance(9);
    particle.Covariance(13) += measurement.Covariance(13);
    particle.Covariance(14) += measurement.Covariance(14);
    particle.Covariance(18) += measurement.Covariance(18);
    particle.Covariance(19) += measurement.Covariance(19);
    particle.Covariance(20) += measurement.Covariance(20);
    particle.Covariance(24) += measurement.Covariance(24);
    particle.Covariance(25) += measurement.Covariance(25);
    particle.Covariance(26) += measurement.Covariance(26);
    particle.Covariance(27) += measurement.Covariance(27);

    for (int i = 0; i < 7; ++i) {
      particle.Parameter(i) += k0[i] * zeta[0] + k1[i] * zeta[1] + k2[i] * zeta[2];
    }

    for (int i = 0, k = 0; i < 7; ++i) {
      for (int j = 0; j <= i; ++j, ++k) {
        particle.Covariance(k) -=
          k0[i] * mCHt0[j] + k1[i] * mCHt1[j] + k2[i] * mCHt2[j];
      }
    }

    float K2[3][3];
    for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 3; ++j) {
        K2[i][j] = -K[j][i];
      }
      K2[i][i] += 1.f;
    }

    float A[3][3];
    for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 3; ++j) {
        A[i][j] = 0.f;
        for (int k = 0; k < 3; ++k) {
          A[i][j] += measurement.Correlation(i, k) * K2[k][j];
        }
      }
    }

    float M[3][3];
    for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 3; ++j) {
        M[i][j] = 0.f;
        for (int k = 0; k < 3; ++k) {
          M[i][j] += K[i][k] * A[k][j];
        }
      }
    }

    particle.Covariance(0) += 2.f * M[0][0];
    particle.Covariance(1) += M[0][1] + M[1][0];
    particle.Covariance(2) += 2.f * M[1][1];
    particle.Covariance(3) += M[0][2] + M[2][0];
    particle.Covariance(4) += M[1][2] + M[2][1];
    particle.Covariance(5) += 2.f * M[2][2];

    particle.NDF() += 2;
    particle.Q() += daughterCharge;
    particle.SFromDecay() = 0.f;
    particle.Chi2() += dChi2;
    return true;
  }

  /**
   * Missing-mass counterpart of AddDaughterWithEnergyFit().
   *
   * The spatial Kalman update is identical, while the measured four-momentum
   * is subtracted. This is the scalar KFParticle::SubtractDaughter() contract
   * used by the legacy CPU NeutralDaughterDecay implementation.
   */
  KFPARTICLE_GPU_HOST_DEVICE inline bool SubtractDaughterWithEnergyFit(
    KFParticleGpuFitState& particle,
    const KFParticleGpuMeasurement& measurement,
    int daughterCharge)
  {
    float mS[6] = {
      particle.Covariance(0) + measurement.Covariance(0),
      particle.Covariance(1) + measurement.Covariance(1),
      particle.Covariance(2) + measurement.Covariance(2),
      particle.Covariance(3) + measurement.Covariance(3),
      particle.Covariance(4) + measurement.Covariance(4),
      particle.Covariance(5) + measurement.Covariance(5)};
    InvertCholetsky3(mS);

    const float zeta[3] = {
      measurement.Parameter(0) - particle.X(),
      measurement.Parameter(1) - particle.Y(),
      measurement.Parameter(2) - particle.Z()};
    const float dChi2 =
      (mS[0] * zeta[0] + mS[1] * zeta[1] + mS[3] * zeta[2]) * zeta[0]
      + (mS[1] * zeta[0] + mS[2] * zeta[1] + mS[4] * zeta[2]) * zeta[1]
      + (mS[3] * zeta[0] + mS[4] * zeta[1] + mS[5] * zeta[2]) * zeta[2];
    if (dChi2 > 1.e9f || dChi2 != dChi2) {
      return false;
    }

    float K[3][3];
    for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 3; ++j) {
        K[i][j] = 0.f;
        for (int k = 0; k < 3; ++k) {
          K[i][j] += particle.Covariance(i, k)
                     * mS[KFParticleGpuFitState::CovarianceIndex(k, j)];
        }
      }
    }

    float mCHt0[7];
    float mCHt1[7];
    float mCHt2[7];
    mCHt0[0] = particle.Covariance(0);
    mCHt1[0] = particle.Covariance(1);
    mCHt2[0] = particle.Covariance(3);
    mCHt0[1] = particle.Covariance(1);
    mCHt1[1] = particle.Covariance(2);
    mCHt2[1] = particle.Covariance(4);
    mCHt0[2] = particle.Covariance(3);
    mCHt1[2] = particle.Covariance(4);
    mCHt2[2] = particle.Covariance(5);
    mCHt0[3] = particle.Covariance(6) + measurement.Covariance(6);
    mCHt1[3] = particle.Covariance(7) + measurement.Covariance(7);
    mCHt2[3] = particle.Covariance(8) + measurement.Covariance(8);
    mCHt0[4] = particle.Covariance(10) + measurement.Covariance(10);
    mCHt1[4] = particle.Covariance(11) + measurement.Covariance(11);
    mCHt2[4] = particle.Covariance(12) + measurement.Covariance(12);
    mCHt0[5] = particle.Covariance(15) + measurement.Covariance(15);
    mCHt1[5] = particle.Covariance(16) + measurement.Covariance(16);
    mCHt2[5] = particle.Covariance(17) + measurement.Covariance(17);
    mCHt0[6] = particle.Covariance(21) + measurement.Covariance(21);
    mCHt1[6] = particle.Covariance(22) + measurement.Covariance(22);
    mCHt2[6] = particle.Covariance(23) + measurement.Covariance(23);

    float k0[7];
    float k1[7];
    float k2[7];
    for (int i = 0; i < 7; ++i) {
      k0[i] = mCHt0[i] * mS[0] + mCHt1[i] * mS[1] + mCHt2[i] * mS[3];
      k1[i] = mCHt0[i] * mS[1] + mCHt1[i] * mS[2] + mCHt2[i] * mS[4];
      k2[i] = mCHt0[i] * mS[3] + mCHt1[i] * mS[4] + mCHt2[i] * mS[5];
    }

    particle.Px() -= measurement.Parameter(3);
    particle.Py() -= measurement.Parameter(4);
    particle.Pz() -= measurement.Parameter(5);
    particle.E() -= measurement.Parameter(6);
    particle.Covariance(9) += measurement.Covariance(9);
    particle.Covariance(13) += measurement.Covariance(13);
    particle.Covariance(14) += measurement.Covariance(14);
    particle.Covariance(18) += measurement.Covariance(18);
    particle.Covariance(19) += measurement.Covariance(19);
    particle.Covariance(20) += measurement.Covariance(20);
    particle.Covariance(24) += measurement.Covariance(24);
    particle.Covariance(25) += measurement.Covariance(25);
    particle.Covariance(26) += measurement.Covariance(26);
    particle.Covariance(27) += measurement.Covariance(27);

    for (int i = 0; i < 7; ++i) {
      particle.Parameter(i) += k0[i] * zeta[0] + k1[i] * zeta[1] + k2[i] * zeta[2];
    }
    for (int i = 0, packed = 0; i < 7; ++i) {
      for (int j = 0; j <= i; ++j, ++packed) {
        particle.Covariance(packed) -=
          k0[i] * mCHt0[j] + k1[i] * mCHt1[j] + k2[i] * mCHt2[j];
      }
    }

    float K2[3][3];
    for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 3; ++j) {
        K2[i][j] = -K[j][i];
      }
      K2[i][i] += 1.f;
    }
    float A[3][3];
    for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 3; ++j) {
        A[i][j] = 0.f;
        for (int k = 0; k < 3; ++k) {
          A[i][j] += measurement.Correlation(i, k) * K2[k][j];
        }
      }
    }
    float M[3][3];
    for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 3; ++j) {
        M[i][j] = 0.f;
        for (int k = 0; k < 3; ++k) {
          M[i][j] += K[i][k] * A[k][j];
        }
      }
    }
    particle.Covariance(0) += 2.f * M[0][0];
    particle.Covariance(1) += M[0][1] + M[1][0];
    particle.Covariance(2) += 2.f * M[1][1];
    particle.Covariance(3) += M[0][2] + M[2][0];
    particle.Covariance(4) += M[1][2] + M[2][1];
    particle.Covariance(5) += 2.f * M[2][2];

    particle.NDF() += 2;
    // CPU SubtractDaughter preserves this historical charge update.
    particle.Q() += daughterCharge;
    particle.SFromDecay() = 0.f;
    particle.Chi2() += dChi2;
    return true;
  }

  /** CPU-compatible filtered missing-mass reconstruction. */
  KFPARTICLE_GPU_HOST_DEVICE inline bool ReconstructMissingMassFiltered(
    const KFParticleGpuFitState& mother,
    const KFParticleGpuMeasurement& measurement,
    int daughterCharge,
    float motherMass,
    float daughterMass,
    float neutralMass,
    KFParticleGpuFitState& neutral,
    KFParticleGpuFitState& motherFiltered,
    KFParticleGpuFitState& daughterFiltered)
  {
    if (motherMass < 0.f || daughterMass < 0.f || neutralMass < 0.f) return false;
    float mS[6] = {
      mother.Covariance(0) + measurement.Covariance(0),
      mother.Covariance(1) + measurement.Covariance(1),
      mother.Covariance(2) + measurement.Covariance(2),
      mother.Covariance(3) + measurement.Covariance(3),
      mother.Covariance(4) + measurement.Covariance(4),
      mother.Covariance(5) + measurement.Covariance(5)};
    InvertCholetsky3(mS);
    const float zeta[3] = {
      measurement.Parameter(0) - mother.X(),
      measurement.Parameter(1) - mother.Y(),
      measurement.Parameter(2) - mother.Z()};
    const float dChi2 =
      (mS[0] * zeta[0] + mS[1] * zeta[1] + mS[3] * zeta[2]) * zeta[0]
      + (mS[1] * zeta[0] + mS[2] * zeta[1] + mS[4] * zeta[2]) * zeta[1]
      + (mS[3] * zeta[0] + mS[4] * zeta[1] + mS[5] * zeta[2]) * zeta[2];
    if (!IsFinite(dChi2) || dChi2 > 1.e9f) return false;

    float K[3][3] = {};
    for (int row = 0; row < 3; ++row) {
      for (int column = 0; column < 3; ++column) {
        for (int k = 0; k < 3; ++k) {
          K[row][column] += mother.Covariance(row, k)
            * mS[KFParticleGpuFitState::CovarianceIndex(k, column)];
        }
      }
    }
    float motherCHt[3][6] = {
      {mother.Covariance(0), mother.Covariance(1), mother.Covariance(3),
       mother.Covariance(6), mother.Covariance(10), mother.Covariance(15)},
      {mother.Covariance(1), mother.Covariance(2), mother.Covariance(4),
       mother.Covariance(7), mother.Covariance(11), mother.Covariance(16)},
      {mother.Covariance(3), mother.Covariance(4), mother.Covariance(5),
       mother.Covariance(8), mother.Covariance(12), mother.Covariance(17)}};
    float daughterCHt[3][6] = {
      {measurement.Covariance(0), measurement.Covariance(1), measurement.Covariance(3),
       measurement.Covariance(6), measurement.Covariance(10), measurement.Covariance(15)},
      {measurement.Covariance(1), measurement.Covariance(2), measurement.Covariance(4),
       measurement.Covariance(7), measurement.Covariance(11), measurement.Covariance(16)},
      {measurement.Covariance(3), measurement.Covariance(4), measurement.Covariance(5),
       measurement.Covariance(8), measurement.Covariance(12), measurement.Covariance(17)}};
    float motherGain[3][6];
    float daughterGain[3][6];
    for (int parameter = 0; parameter < 6; ++parameter) {
      for (int component = 0; component < 3; ++component) {
        motherGain[component][parameter] =
          motherCHt[0][parameter] * mS[component == 0 ? 0 : component == 1 ? 1 : 3]
          + motherCHt[1][parameter] * mS[component == 0 ? 1 : component == 1 ? 2 : 4]
          + motherCHt[2][parameter] * mS[component == 0 ? 3 : component == 1 ? 4 : 5];
        daughterGain[component][parameter] =
          daughterCHt[0][parameter] * mS[component == 0 ? 0 : component == 1 ? 1 : 3]
          + daughterCHt[1][parameter] * mS[component == 0 ? 1 : component == 1 ? 2 : 4]
          + daughterCHt[2][parameter] * mS[component == 0 ? 3 : component == 1 ? 4 : 5];
      }
    }

    motherFiltered = mother;
    daughterFiltered.Initialize();
    for (int parameter = 0; parameter < 8; ++parameter) {
      daughterFiltered.Parameter(parameter) = measurement.Parameter(parameter);
    }
    for (int covariance = 0; covariance < 36; ++covariance) {
      daughterFiltered.Covariance(covariance) = measurement.Covariance(covariance);
    }
    daughterFiltered.Q() = daughterCharge;
    for (int parameter = 0; parameter < 6; ++parameter) {
      motherFiltered.Parameter(parameter) = mother.Parameter(parameter)
        + motherGain[0][parameter] * zeta[0]
        + motherGain[1][parameter] * zeta[1]
        + motherGain[2][parameter] * zeta[2];
      daughterFiltered.Parameter(parameter) = measurement.Parameter(parameter)
        - daughterGain[0][parameter] * zeta[0]
        - daughterGain[1][parameter] * zeta[1]
        - daughterGain[2][parameter] * zeta[2];
    }
    for (int row = 0, packed = 0; row < 6; ++row) {
      for (int column = 0; column <= row; ++column, ++packed) {
        motherFiltered.Covariance(packed) = mother.Covariance(packed)
          - motherGain[0][row] * motherCHt[0][column]
          - motherGain[1][row] * motherCHt[1][column]
          - motherGain[2][row] * motherCHt[2][column];
        daughterFiltered.Covariance(packed) = measurement.Covariance(packed)
          - daughterGain[0][row] * daughterCHt[0][column]
          - daughterGain[1][row] * daughterCHt[1][column]
          - daughterGain[2][row] * daughterCHt[2][column];
      }
    }

    float K2[3][3];
    for (int row = 0; row < 3; ++row) {
      for (int column = 0; column < 3; ++column) {
        K2[row][column] = -K[column][row];
      }
      K2[row][row] += 1.f;
    }
    float A[3][3] = {};
    float M[3][3] = {};
    for (int row = 0; row < 3; ++row) {
      for (int column = 0; column < 3; ++column) {
        for (int k = 0; k < 3; ++k) {
          A[row][column] += measurement.Correlation(row, k) * K2[k][column];
        }
      }
    }
    for (int row = 0; row < 3; ++row) {
      for (int column = 0; column < 3; ++column) {
        for (int k = 0; k < 3; ++k) M[row][column] += K[row][k] * A[k][column];
      }
    }
    const float spatialCorrection[6] = {
      2.f * M[0][0], M[0][1] + M[1][0], 2.f * M[1][1],
      M[0][2] + M[2][0], M[1][2] + M[2][1], 2.f * M[2][2]};
    for (int covariance = 0; covariance < 6; ++covariance) {
      motherFiltered.Covariance(covariance) += spatialCorrection[covariance];
      daughterFiltered.Covariance(covariance) += spatialCorrection[covariance];
    }

    neutral = mother;
    if (!SubtractDaughterWithEnergyFit(neutral, measurement, daughterCharge)) return false;
    neutral.Q() = mother.Q() - daughterCharge;
    const float correctedChi2 = mother.Chi2() + dChi2;
    const int correctedNdf = mother.NDF() + 2;
    motherFiltered.Chi2() = correctedChi2;
    daughterFiltered.Chi2() = correctedChi2;
    motherFiltered.NDF() = correctedNdf;
    daughterFiltered.NDF() = correctedNdf;

    const float motherEnergy = KFParticleGpuSqrt(
      motherMass * motherMass + Momentum2(motherFiltered));
    const float daughterEnergy = KFParticleGpuSqrt(
      daughterMass * daughterMass + Momentum2(daughterFiltered));
    const float neutralEnergy = KFParticleGpuSqrt(
      neutralMass * neutralMass + Momentum2(neutral));
    neutral.E() = motherEnergy - daughterEnergy;
    motherFiltered.E() = daughterEnergy + neutralEnergy;
    daughterFiltered.E() = motherEnergy - neutralEnergy;
    motherFiltered.MassHypo() = motherMass;
    daughterFiltered.MassHypo() = daughterMass;
    neutral.MassHypo() = neutralMass;
    return IsFinite(neutral.E()) && IsFinite(motherFiltered.E())
      && IsFinite(daughterFiltered.E()) && neutral.E() > 0.f;
  }

  KFPARTICLE_GPU_HOST_DEVICE inline bool ApplyLinearMassConstraint(
    KFParticleGpuFitState& particle,
    float mass,
    float sigmaMass)
  {
    if (mass < 0.f || sigmaMass < 0.f) {
      return false;
    }
    const float mass2 = mass * mass;
    const float momentum2 = Momentum2(particle);
    const float constrainedEnergy = KFParticleGpuSqrt(mass2 + momentum2);
    const float H[8] = {
      0.f, 0.f, 0.f, -2.f * particle.Px(), -2.f * particle.Py(),
      -2.f * particle.Pz(), 2.f * particle.E(), 0.f};
    const float residual =
      mass2 - (particle.E() * particle.E() - momentum2);
    float CHt[8] = {};
    float variance = mass2 * sigmaMass * sigmaMass;
    for (int row = 0; row < 8; ++row) {
      for (int column = 0; column < 8; ++column) {
        CHt[row] += particle.Covariance(row, column) * H[column];
      }
      variance += H[row] * CHt[row];
    }
    if (!(variance > 1.e-20f) || variance != variance
        || constrainedEnergy != constrainedEnergy) {
      return false;
    }
    const float weight = 1.f / variance;
    particle.Chi2() += residual * residual * weight;
    particle.NDF() += 1;
    for (int row = 0; row < 8; ++row) {
      const float gain = CHt[row] * weight;
      particle.Parameter(row) += gain * residual;
      for (int column = 0; column <= row; ++column) {
        particle.Covariance(row, column) -= gain * CHt[column];
      }
    }
    particle.MassHypo() = mass;
    particle.SumDaughterMass() = mass;
    return true;
  }

  KFPARTICLE_GPU_HOST_DEVICE inline bool TransportLineToPoint(
    const KFParticleGpuFitState& particle,
    const KFParticleGpuVertexState& vertex,
    KFParticleGpuFitState& transported)
  {
    // Preserve the pre-transport state when input and output alias.
    const KFParticleGpuFitState source = particle;
    const float point[3] = {vertex.X(), vertex.Y(), vertex.Z()};
    float dsdr[6];
    const float dS = GetDStoPointLine(source, point, dsdr);
    if (dS != dS || Abs(dS) > 1.e6f) {
      return false;
    }
    TransportLine(source, dS, transported);
    // Fixed-step line covariance is the bounded projection baseline. The
    // subsequent production-vertex constraint includes vertex uncertainty.
    float jacobian[8][8] = {};
    for (int i = 0; i < 8; ++i) {
      jacobian[i][i] = 1.f;
    }
    for (int row = 0; row < 3; ++row) {
      const float momentum = source.Parameter(3 + row);
      for (int column = 0; column < 6; ++column) {
        jacobian[row][column] += momentum * dsdr[column];
      }
      jacobian[row][3 + row] += dS;
    }
    float covariance[36] = {};
    for (int row = 0; row < 8; ++row) {
      for (int column = 0; column <= row; ++column) {
        float value = 0.f;
        for (int left = 0; left < 8; ++left) {
          for (int right = 0; right < 8; ++right) {
            value += jacobian[row][left] * source.Covariance(left, right)
                     * jacobian[column][right];
          }
        }
        covariance[KFParticleGpuFitState::CovarianceIndex(row, column)] = value;
      }
    }
    for (int index = 0; index < 36; ++index) {
      transported.Covariance(index) = covariance[index];
    }
    return transported.X() == transported.X()
           && transported.Y() == transported.Y()
           && transported.Z() == transported.Z();
  }

  KFPARTICLE_GPU_HOST_DEVICE inline bool ConstrainToProductionVertex(
    KFParticleGpuFitState& particle,
    const KFParticleGpuVertexState& vertex)
  {
    KFParticleGpuMeasurement measurement;
    measurement.Initialize();
    measurement.Parameter(0) = vertex.X();
    measurement.Parameter(1) = vertex.Y();
    measurement.Parameter(2) = vertex.Z();
    for (int index = 0; index < KFParticleGpuVertexState::NumberOfCovarianceElements; ++index) {
      measurement.Covariance(index) = vertex.Covariance(index);
    }
    if (!AddDaughterWithEnergyFit(particle, measurement, 0)) {
      return false;
    }
    particle.AtProductionVertex() = 1;
    return true;
  }

  /**
   * Scalar neutral-particle counterpart of KFParticleSIMD::SetProductionVertex().
   *
   * Default V0 mothers are neutral, so their CBM transport to a primary vertex
   * is exactly the line transport regardless of the magnetic field.  Keeping
   * the vertex-induced correlations and the S covariance is essential: the CPU
   * finder applies its secondary L/dL cut only after this constraint.
   */
  KFPARTICLE_GPU_HOST_DEVICE inline bool SetNeutralProductionVertex(
    KFParticleGpuFitState& particle,
    const KFParticleGpuVertexState& vertex)
  {
    if (particle.Q() != 0 || !IsFiniteState(particle)) {
      return false;
    }

    const float decayPoint[3] = {particle.X(), particle.Y(), particle.Z()};
    const float decayPointCovariance[6] = {
      particle.Covariance(0), particle.Covariance(1), particle.Covariance(2),
      particle.Covariance(3), particle.Covariance(4), particle.Covariance(5)};
    const float point[3] = {vertex.X(), vertex.Y(), vertex.Z()};

    float dsdr[6];
    const float dS = GetDStoPointLine(particle, point, dsdr);
    if (!IsFinite(dS) || Abs(dS) > 1.e3f) {
      return false;
    }
    const float dsdp[6] = {-dsdr[0], -dsdr[1], -dsdr[2], 0.f, 0.f, 0.f};
    float selfDerivative[36] = {};
    float vertexDerivative[36] = {};
    KFParticleGpuFitState transported;
    TransportLineWithJacobian(particle,
                              dS,
                              dsdr,
                              transported,
                              dsdp,
                              selfDerivative,
                              vertexDerivative);
    particle = transported;

    float correlation[6][3] = {};
    for (int row = 0; row < 6; ++row) {
      for (int column = 0; column < 3; ++column) {
        for (int k = 0; k < 3; ++k) {
          correlation[row][column] +=
            vertex.Covariance(KFParticleGpuFitState::CovarianceIndex(column, k))
            * vertexDerivative[row * 6 + k];
        }
      }
    }
    for (int row = 0; row < 6; ++row) {
      for (int column = 0; column <= row; ++column) {
        float contribution = 0.f;
        for (int left = 0; left < 3; ++left) {
          for (int right = 0; right < 3; ++right) {
            contribution += vertexDerivative[row * 6 + left]
                            * vertex.Covariance(
                              KFParticleGpuFitState::CovarianceIndex(left, right))
                            * vertexDerivative[column * 6 + right];
          }
        }
        particle.Covariance(row, column) += contribution;
      }
    }

    float inverseResidualCovariance[6] = {
      particle.Covariance(0) + vertex.Covariance(0),
      particle.Covariance(1) + vertex.Covariance(1),
      particle.Covariance(2) + vertex.Covariance(2),
      particle.Covariance(3) + vertex.Covariance(3),
      particle.Covariance(4) + vertex.Covariance(4),
      particle.Covariance(5) + vertex.Covariance(5)};
    InvertCholetsky3(inverseResidualCovariance);

    const float residual[3] = {
      vertex.X() - particle.X(),
      vertex.Y() - particle.Y(),
      vertex.Z() - particle.Z()};
    float gain3[3][3] = {};
    for (int row = 0; row < 3; ++row) {
      for (int column = 0; column < 3; ++column) {
        for (int k = 0; k < 3; ++k) {
          gain3[row][column] += particle.Covariance(row, k)
                                      * inverseResidualCovariance[
                                        KFParticleGpuFitState::CovarianceIndex(k, column)];
        }
      }
    }

    const int covarianceColumns[3][7] = {
      {0, 1, 3, 6, 10, 15, 21},
      {1, 2, 4, 7, 11, 16, 22},
      {3, 4, 5, 8, 12, 17, 23}};
    float measurementColumns[3][7] = {};
    float gain[3][7] = {};
    for (int parameter = 0; parameter < 7; ++parameter) {
      const float c0 = particle.Covariance(covarianceColumns[0][parameter]);
      const float c1 = particle.Covariance(covarianceColumns[1][parameter]);
      const float c2 = particle.Covariance(covarianceColumns[2][parameter]);
      measurementColumns[0][parameter] = c0;
      measurementColumns[1][parameter] = c1;
      measurementColumns[2][parameter] = c2;
      gain[0][parameter] = c0 * inverseResidualCovariance[0]
                           + c1 * inverseResidualCovariance[1]
                           + c2 * inverseResidualCovariance[3];
      gain[1][parameter] = c0 * inverseResidualCovariance[1]
                           + c1 * inverseResidualCovariance[2]
                           + c2 * inverseResidualCovariance[4];
      gain[2][parameter] = c0 * inverseResidualCovariance[3]
                           + c1 * inverseResidualCovariance[4]
                           + c2 * inverseResidualCovariance[5];
    }
    for (int parameter = 0; parameter < 7; ++parameter) {
      particle.Parameter(parameter) += gain[0][parameter] * residual[0]
                                      + gain[1][parameter] * residual[1]
                                      + gain[2][parameter] * residual[2];
    }
    for (int row = 0, packed = 0; row < 7; ++row) {
      for (int column = 0; column <= row; ++column, ++packed) {
        particle.Covariance(packed) -=
          gain[0][row] * measurementColumns[0][column]
          + gain[1][row] * measurementColumns[1][column]
          + gain[2][row] * measurementColumns[2][column];
      }
    }

    float oneMinusGainTranspose[3][3];
    for (int row = 0; row < 3; ++row) {
      for (int column = 0; column < 3; ++column) {
        oneMinusGainTranspose[row][column] = -gain3[column][row];
      }
      oneMinusGainTranspose[row][row] += 1.f;
    }
    float intermediate[3][3] = {};
    for (int row = 0; row < 3; ++row) {
      for (int column = 0; column < 3; ++column) {
        for (int k = 0; k < 3; ++k) {
          intermediate[row][column] += correlation[k][row]
                                       * oneMinusGainTranspose[k][column];
        }
      }
    }
    float correction[3][3] = {};
    for (int row = 0; row < 3; ++row) {
      for (int column = 0; column < 3; ++column) {
        for (int k = 0; k < 3; ++k) {
          correction[row][column] += gain3[row][k] * intermediate[k][column];
        }
      }
    }
    particle.Covariance(0) += 2.f * correction[0][0];
    particle.Covariance(1) += correction[0][1] + correction[1][0];
    particle.Covariance(2) += 2.f * correction[1][1];
    particle.Covariance(3) += correction[0][2] + correction[2][0];
    particle.Covariance(4) += correction[1][2] + correction[2][1];
    particle.Covariance(5) += 2.f * correction[2][2];

    particle.Chi2() +=
      (inverseResidualCovariance[0] * residual[0]
       + inverseResidualCovariance[1] * residual[1]
       + inverseResidualCovariance[3] * residual[2]) * residual[0]
      + (inverseResidualCovariance[1] * residual[0]
         + inverseResidualCovariance[2] * residual[1]
         + inverseResidualCovariance[4] * residual[2]) * residual[1]
      + (inverseResidualCovariance[3] * residual[0]
         + inverseResidualCovariance[4] * residual[1]
         + inverseResidualCovariance[5] * residual[2]) * residual[2];
    particle.NDF() += 2;

    float decayDsdr[6];
    particle.S() = GetDStoPointLine(particle, decayPoint, decayDsdr);
    particle.Covariance(35) = 0.f;
    for (int column = 0; column < 6; ++column) {
      float dsdrCovariance = 0.f;
      for (int row = 0; row < 6; ++row) {
        dsdrCovariance += decayDsdr[row] * particle.Covariance(row, column);
      }
      particle.Covariance(28 + column) = dsdrCovariance;
      particle.Covariance(35) += dsdrCovariance * decayDsdr[column];
      if (column < 3) {
        float decayPointContribution = 0.f;
        for (int row = 0; row < 3; ++row) {
          decayPointContribution -= decayDsdr[row]
                                    * decayPointCovariance[
                                      KFParticleGpuFitState::CovarianceIndex(row, column)];
        }
        particle.Covariance(35) -= decayPointContribution * decayDsdr[column];
      }
    }
    particle.AtProductionVertex() = 1;
    return IsFiniteState(particle) && IsFinite(particle.Covariance(35));
  }

  KFPARTICLE_GPU_HOST_DEVICE inline bool GetDecayLength(
    const KFParticleGpuFitState& particle,
    float& length,
    float& error)
  {
    const float momentum2 = Momentum2(particle);
    if (!(momentum2 > 1.e-4f)) {
      length = 0.f;
      error = 1.e20f;
      return false;
    }
    const float s = particle.S();
    length = s * KFParticleGpuSqrt(momentum2);
    const float px = particle.Px();
    const float py = particle.Py();
    const float pz = particle.Pz();
    const float variance =
      momentum2 * particle.Covariance(35)
      + s * s / momentum2
          * (px * px * particle.Covariance(9)
             + py * py * particle.Covariance(14)
             + pz * pz * particle.Covariance(20)
             + 2.f * (px * py * particle.Covariance(13)
                      + px * pz * particle.Covariance(18)
                      + py * pz * particle.Covariance(19)))
      + 2.f * s * (px * particle.Covariance(31)
                   + py * particle.Covariance(32)
                   + pz * particle.Covariance(33));
    error = KFParticleGpuSqrt(Abs(variance));
    return IsFinite(length) && IsFinite(error);
  }

  KFPARTICLE_GPU_HOST_DEVICE inline void BuildKinematicMother(const KFParticleGpuFitState& first,
                                                              const KFParticleGpuFitState& second,
                                                              KFParticleGpuFitState& mother)
  {
    mother.Initialize();

    // This is deliberately only a kinematic seed. The real KF construction must
    // replace it with the AddDaughter/GetMeasurement Kalman update port.
    mother.X() = 0.5f * (first.X() + second.X());
    mother.Y() = 0.5f * (first.Y() + second.Y());
    mother.Z() = 0.5f * (first.Z() + second.Z());
    mother.Px() = first.Px() + second.Px();
    mother.Py() = first.Py() + second.Py();
    mother.Pz() = first.Pz() + second.Pz();
    mother.E() = first.E() + second.E();
    mother.S() = 0.f;

    // The seed treats daughters as independent until the full Kalman update is
    // ported, so only the momentum-energy block is propagated here.
    for (int i = 3; i <= 6; ++i) {
      for (int j = 3; j <= i; ++j) {
        mother.Covariance(i, j) = first.Covariance(i, j) + second.Covariance(i, j);
      }
    }

    mother.Chi2() = first.Chi2() + second.Chi2();
    mother.NDF() = first.NDF() + second.NDF();
    mother.Q() = first.Q() + second.Q();
    mother.SumDaughterMass() = first.SumDaughterMass() + second.SumDaughterMass();
    mother.MassHypo() = -1.f;
  }
}

#endif
