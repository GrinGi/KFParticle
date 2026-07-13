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

#include "KFParticleGpuFitState.h"
#include "KFParticleGpuMeasurement.h"

namespace KFParticleGpuMath
{
  KFPARTICLE_GPU_HOST_DEVICE inline float Abs(float value)
  {
    return value >= 0.f ? value : -value;
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
    transported.S() = particle.S() + dS;
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
    transported.S() = particle.S() + dS;
    transported.SFromDecay() = particle.SFromDecay() + dS;
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
    transported.S() = particle.S() + dS;
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

  KFPARTICLE_GPU_HOST_DEVICE inline bool BuildConstantByDcaKinematicSeed(
    const KFParticleGpuFitState& first,
    const KFParticleGpuFitState& second,
    float by,
    KFParticleGpuFitState& firstAtDca,
    KFParticleGpuFitState& secondAtDca)
  {
    const bool firstIsStraight = Abs(by * static_cast<float>(first.Q())) < 1.e-8f;
    const bool secondIsStraight = Abs(by * static_cast<float>(second.Q())) < 1.e-8f;
    if (firstIsStraight && secondIsStraight) {
      float dS[2] = {0.f, 0.f};
      GetDStoParticleLine(first, second, dS);
      if (Abs(dS[0] * first.Pz()) > 1000.f || Abs(dS[1] * second.Pz()) > 1000.f) {
        firstAtDca.Initialize();
        secondAtDca.Initialize();
        return false;
      }
      TransportLine(first, dS[0], firstAtDca);
      TransportLine(second, dS[1], secondAtDca);
      return true;
    }

    float dS[2] = {0.f, 0.f};
    GetDStoParticleLine(first, second, dS);

    for (int iteration = 0; iteration < 2; ++iteration) {
      if (Abs(dS[0] * first.Pz()) > 1000.f || Abs(dS[1] * second.Pz()) > 1000.f) {
        firstAtDca.Initialize();
        secondAtDca.Initialize();
        return false;
      }

      TransportConstantBy(first, dS[0], by, firstAtDca);
      TransportConstantBy(second, dS[1], by, secondAtDca);

      float correction[2] = {0.f, 0.f};
      GetDStoParticleLine(firstAtDca, secondAtDca, correction);
      dS[0] += correction[0];
      dS[1] += correction[1];
    }

    if (Abs(dS[0] * first.Pz()) > 1000.f || Abs(dS[1] * second.Pz()) > 1000.f) {
      firstAtDca.Initialize();
      secondAtDca.Initialize();
      return false;
    }

    TransportConstantBy(first, dS[0], by, firstAtDca);
    TransportConstantBy(second, dS[1], by, secondAtDca);
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
