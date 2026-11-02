/*
 * This file is part of KFParticle package
 * Copyright (C) 2007-2026
 *
 * KFParticle is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 */

/**
 * @file
 * @brief Value-based host/device mathematics for KFParticle fitting and transport.
 *
 * State order is (x,y,z,px,py,pz,E,S); a six-coordinate derivative omits E and S.
 * Symmetric covariances use packed lower-triangular storage; dense Jacobians are row-major.
 * dS is signed path divided by momentum magnitude, not a distance: for a line, delta r = p*dS.
 * S is the stored decay parameter; SFromDecay separately accumulates transport steps.
 * DCA means distance of closest approach; here DCA helpers estimate the two transport
 * parameters rather than returning a distance or necessarily fitting a common vertex.
 *
 * The file contains simple kinematic seeds, approximate and coupled covariance propagation,
 * Kalman updates, and CPU-compatibility helpers. Their contracts are intentionally distinct.
 * Correlation arrays contain cross covariances, not dimensionless correlation coefficients;
 * their row/column orientation is specified at the relevant producer.
 *
 * Helpers allocate no dynamic storage and perform no GPU launches or transfers. Raw array
 * sizes and index ranges are caller contracts. Unless explicitly supported, keep input and
 * output storage distinct. Numerical guards preserve the implemented CPU-compatible behavior;
 * a successful boolean result is not a general proof of physical or statistical validity.
 */

#ifndef KFPARTICLEGPUMATH_H
#define KFPARTICLEGPUMATH_H

#include "data/KFParticleGpuFitState.h"

/**
 * @brief Numerical primitives shared by host reference paths and device reconstruction.
 * Transport, DCA seed preparation, and fitting are separate operations; choose the covariance
 * and approximation contract required by the caller.
 */
namespace KFParticleGpuMath
{
  /**
   * @brief Optional arithmetic record of the CPU-compatible atan2 approximation.
   * Records intermediates for backend comparisons, not an error bound. Ratios can be non-finite
   * even when an axis special case produces a finite result.
   */
  struct KFParticleGpuAtan2ArithmeticTrace
  {
    float y; ///< Numerator coordinate supplied to atan2.
    float x; ///< Denominator coordinate supplied to atan2.
    float absY; ///< Magnitude of y before quadrant restoration.
    float absX; ///< Magnitude of x before quadrant restoration.
    float initialRatio; ///< absY/absX before angle reduction; may be Inf or NaN.
    float transformedRatio; ///< Reduced tangent argument of the polynomial.
    float ratio2; ///< Square of transformedRatio.
    float polynomialStage[4]; ///< Three Horner intermediates followed by the reduced-angle approximation.
    float result; ///< Final angle in radians after quadrant and zero handling.
    /// @brief Bits: 1=x zero, 2=y zero, 4=x negative, 8=y negative, 16=large ratio, 32=middle ratio.
    unsigned int branchFlags;
  };

  /**
   * @brief Optional arithmetic record of CPU-compatible sine/cosine evaluation.
   * Captures angle reduction, polynomial values, and quadrant restoration for numerical diagnosis.
   */
  struct KFParticleGpuSinCosArithmeticTrace
  {
    float value; ///< Original angle in radians.
    float scaled; ///< Angle multiplied by the stored approximation of 2/pi.
    float rounded; ///< Rounded scaled value before conversion to int.
    float reduced; ///< Angle after subtracting the split pi/2 multiple.
    float reduced2; ///< Square of the reduced angle.
    float sineSeries; ///< Reduced-angle sine polynomial before quadrant restoration.
    float cosineSeries; ///< Reduced-angle cosine polynomial before quadrant restoration.
    float sine; ///< Final sine approximation.
    float cosine; ///< Final cosine approximation.
    int quadrantCount; ///< Integer number of subtracted pi/2 steps.
    int quadrant; ///< quadrantCount modulo four, obtained from its low two bits.
  };

  /**
   * @brief Approximate atan2(y, x) in radians using the CPU-compatible polynomial and branch order.
   * Reduces the ratio around pi/8 and 3*pi/8, then restores the quadrant. At (0,0) it returns zero.
   * This is a compatibility primitive, not a promise of standard-library accuracy or IEEE signed-zero semantics.
   * @param trace Optional arithmetic snapshot; intermediate ratios can be non-finite on the axes.
   */
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

  /**
   * @brief Approximate sine and cosine together using CPU-compatible range reduction and polynomials.
   * Reduces value by a rounded multiple of pi/2, evaluates short series, and restores the quadrant.
   * @param value Angle in radians; its rounded 2/pi multiple must fit int.
   * @param[out] sine Approximate sine of value.
   * @param[out] cosine Approximate cosine of value.
   * @param trace Optional snapshot of reduction, series, and final values.
   */
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

  /**
   * @brief Transport outcome flags; zero is ordinary charged success, not a bit to test.
   * Neutral and PathLimited are nonzero outcomes that can accompany a true function return.
   */
  enum FullFieldTransportStatus : unsigned int
  {
    /// Charged transport completed without a special status.
    KFGpuFullFieldTransportSuccess = 0u,
    /// Neutral particle used straight-line transport.
    KFGpuFullFieldTransportNeutral = 1u << 0u,
    /// Charged |dS*pz| exceeded 1000; state retained, transport not applied.
    KFGpuFullFieldTransportPathLimited = 1u << 1u,
    /// Input state, step, field coefficients, or supplied self derivatives failed checks.
    KFGpuFullFieldTransportInvalidInput = 1u << 2u,
    /// Charged transported state failed IsFiniteState(); original state restored.
    KFGpuFullFieldTransportNonFiniteOutput = 1u << 3u
  };

  /**
   * @brief Flags reported by the numerical coupled full-field DCA builder.
   * Neutral is informational; Rejected and NonFinite identify detected failure categories.
   */
  enum FullFieldDcaStatus : unsigned int
  {
    /// No DCA failure or neutral-input flag.
    KFGpuFullFieldDcaSuccess = 0u,
    /// At least one daughter has zero charge.
    KFGpuFullFieldDcaNeutral = 1u << 0u,
    /// Nominal or perturbed DCA seed/transport could not be accepted.
    KFGpuFullFieldDcaRejected = 1u << 1u,
    /// Checked input, derivative, or position cross-covariance value was non-finite.
    KFGpuFullFieldDcaNonFinite = 1u << 2u
  };

  /**
   * @brief Diagnostic snapshot of the two-track Bz DCA arithmetic, including ill-conditioned intermediates.
   * Two-element particle/root axes are identified below. Coordinates belong to the solver frame;
   * By callers therefore record (x,-z,y,px,-pz,py). The routine does not clear the whole record.
   * A both-straight return leaves it untouched, and straight-particle subbranches do not populate
   * their atan records. Interpret only fields reached on the executed branch.
   */
  struct KFParticleGpuDcaArithmeticTrace
  {
    float parameters[2][6]; ///< [particle][coordinate], six solver-frame coordinates per input.
    float bq[2]; ///< Charge times field times 0.000299792458, for each particle.
    float pt2[2]; ///< Squared momentum transverse to the solver z axis, for each particle.
    float delta0[2]; ///< Initial (x1-x2,y1-y2) in the solver frame.
    float dr02; ///< Squared initial transverse separation.
    float drp[2]; ///< Initial transverse separation dotted with each particle's transverse momentum.
    float dxyp[2]; ///< dx*py - dy*px for each particle.
    float p1p2; ///< Transverse momentum dot product.
    float dp1p2; ///< Transverse momentum cross component px1*py2 - px2*py1.
    float k[4]; ///< Helix root coefficients in order k11,k21,k12,k22.
    float kp; ///< Combined transverse geometry coefficient before the separation-squared term.
    float kd; ///< kp + dr02*bq1*bq2/2.
    float c[2]; ///< Per-particle algebraic coefficients c1 and c2 in the root equations.
    float discriminant; ///< pt2[0]*pt2[1] - kd*kd before clipping.
    float root; ///< Square root of the discriminant clipped at zero.
    float atanNumerator[2][2]; ///< [particle][root] atan2 numerators in the charged-root formulas.
    float atanDenominator[2][2]; ///< [particle][root] atan2 denominators in the charged-root formulas.
    /// @brief [particle][root] arithmetic records, populated only for curved particles.
    KFParticleGpuAtan2ArithmeticTrace atan[2][2];
    float roots[2][2]; ///< [particle][root] snapshot before straight-particle roots are filled.
    float rootDistance2[2]; ///< Squared 3D separation for each completed root pair.
    float selectedDs[2]; ///< Selected or averaged transport parameters before tangent correction.
    float bs[2]; ///< Rotation angle bq*dS for each selected trajectory.
    float sine[2]; ///< Sine approximation of each selected rotation angle.
    float cosine[2]; ///< Cosine approximation of each selected rotation angle.
    KFParticleGpuSinCosArithmeticTrace sincos[2]; ///< Per-particle trigonometric trace for the selected root/mean.
    float position[2][3]; ///< [particle][xyz] positions before the tangent-line correction.
    float momentum[2][3]; ///< [particle][px,py,pz] momenta at those positions.
    float momentum2[2]; ///< Squared three-momentum of each transported particle.
    float momentumDot; ///< Dot product of the transported momenta.
    float separation[3]; ///< Transported second position minus first position, in xyz order.
    float separationDotMomentum[2]; ///< Separation dotted with each transported momentum.
    float determinantRaw; ///< momentumDot^2 - momentum2[0]*momentum2[1].
    float determinantUsed; ///< Denominator after the small-determinant compatibility guard.
    float correctionNumerator[2]; ///< Two numerators of the tangent-line step correction.
    float correction[2]; ///< Per-particle tangent-line correction added to selectedDs.
    float finalDs[2]; ///< Transport parameters after the tangent-line correction.
  };

  /**
   * @brief Transported daughter states plus numerical two-source DCA derivatives and covariance blocks.
   * Each row-major 6x12 Jacobian maps the two independent input states to one output's
   * (x,y,z,px,py,pz). Columns 0..5 belong to the first input, 6..11 to the second.
   * The separate packed 6x6 covariance arrays include both inputs' uncertainties; the state
   * objects retain their fixed-step 6x6 blocks until copied/replaced by a consumer.
   * Correlation is Cov(first position, second position), without normalization.
   * Call Initialize() before using an otherwise uninitialized record.
   */
  struct KFParticleGpuFullFieldDcaResult
  {
    KFParticleGpuFitState first; ///< First transported fit state, with separately rebuilt energy cross terms.
    KFParticleGpuFitState second; ///< Second transported fit state, with separately rebuilt energy cross terms.
    float firstCovariance[21]; ///< Packed 6x6 covariance of the first output, including both input sources.
    float secondCovariance[21]; ///< Packed 6x6 covariance of the second output, including both input sources.
    float correlation[9]; ///< Row-major 3x3 Cov(first position, second position).
    float firstJacobian[72]; ///< Row-major 6x12 derivative of the first output with respect to both inputs.
    float secondJacobian[72]; ///< Row-major 6x12 derivative of the second output with respect to both inputs.

    /**
     * @brief Initialize the two fit states and zero every separate Jacobian/covariance array.
     */
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

    /**
     * @brief Expose writable access to one first-output Jacobian element; requires row in [0,6), column in [0,12).
     */
    KFPARTICLE_GPU_HOST_DEVICE float& FirstJacobian(int row, int column)
    {
      return firstJacobian[row * 12 + column];
    }

    /**
     * @brief Read one first-output Jacobian element; requires row in [0,6), column in [0,12).
     */
    KFPARTICLE_GPU_HOST_DEVICE const float& FirstJacobian(int row, int column) const
    {
      return firstJacobian[row * 12 + column];
    }

    /**
     * @brief Expose writable access to one second-output Jacobian element; requires row in [0,6), column in [0,12).
     */
    KFPARTICLE_GPU_HOST_DEVICE float& SecondJacobian(int row, int column)
    {
      return secondJacobian[row * 12 + column];
    }

    /**
     * @brief Read one second-output Jacobian element; requires row in [0,6), column in [0,12).
     */
    KFPARTICLE_GPU_HOST_DEVICE const float& SecondJacobian(int row, int column) const
    {
      return secondJacobian[row * 12 + column];
    }

    /**
     * @brief Expose writable access to one symmetric first-output covariance element; requires row/column in [0,6).
     */
    KFPARTICLE_GPU_HOST_DEVICE float& FirstCovariance(int row, int column)
    {
      return firstCovariance[KFParticleGpuFitState::CovarianceIndex(row, column)];
    }

    /**
     * @brief Read one symmetric first-output covariance element; requires row/column in [0,6).
     */
    KFPARTICLE_GPU_HOST_DEVICE const float& FirstCovariance(int row, int column) const
    {
      return firstCovariance[KFParticleGpuFitState::CovarianceIndex(row, column)];
    }

    /**
     * @brief Expose writable access to one symmetric second-output covariance element; requires row/column in [0,6).
     */
    KFPARTICLE_GPU_HOST_DEVICE float& SecondCovariance(int row, int column)
    {
      return secondCovariance[KFParticleGpuFitState::CovarianceIndex(row, column)];
    }

    /**
     * @brief Read one symmetric second-output covariance element; requires row/column in [0,6).
     */
    KFPARTICLE_GPU_HOST_DEVICE const float& SecondCovariance(int row, int column) const
    {
      return secondCovariance[KFParticleGpuFitState::CovarianceIndex(row, column)];
    }

    /**
     * @brief Expose writable access to one Cov(first position, second position) element; requires row/column in [0,3).
     */
    KFPARTICLE_GPU_HOST_DEVICE float& Correlation(int row, int column)
    {
      return correlation[row * 3 + column];
    }

    /**
     * @brief Read one Cov(first position, second position) element; requires row/column in [0,3).
     */
    KFPARTICLE_GPU_HOST_DEVICE const float& Correlation(int row, int column) const
    {
      return correlation[row * 3 + column];
    }
  };

  /**
   * @brief Return the nonnegative magnitude by a sign comparison, without a library call.
   * No special normalization of signed zero or NaN is performed.
   */
  KFPARTICLE_GPU_HOST_DEVICE inline float Abs(float value)
  {
    return value >= 0.f ? value : -value;
  }

  /**
   * @brief Accept non-NaN values strictly between -3.4e38 and +3.4e38.
   * This portable guard also rejects the extreme finite float values outside that interval;
   * it is deliberately not identical to the standard isfinite predicate.
   */
  KFPARTICLE_GPU_HOST_DEVICE inline bool IsFinite(float value)
  {
    // This form is available on every XPU backend and rejects NaN and infinities
    // without depending on backend-specific stdlib overloads.
    return value == value && value > -3.4e38f && value < 3.4e38f;
  }

  /**
   * @brief Apply IsFinite() to all parameters, covariance entries, and stored floating-point fit scalars.
   * Does not test covariance positivity, physical kinematics, integer charge, or NDF validity.
   */
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

  /**
   * @brief Invert a packed symmetric 3x3 matrix in place using a signed Cholesky-like factorization.
   * Storage order is (00,10,11,20,21,22). The factorization uses A = U^T D U with diagonal signs D.
   * Pivots with magnitude below 1e-12 are replaced by positive 1e-12, so singular input yields a
   * regularized result rather than an error. This is not a positive-definiteness test.
   */
  KFPARTICLE_GPU_HOST_DEVICE inline void InvertCholetsky3(float a[6])
  {
    float d[3];
    float u[3][3];

    KFPARTICLE_GPU_UNROLL
    for (int i = 0; i < 3; ++i) {
      d[i] = 0.f;
      KFPARTICLE_GPU_UNROLL
      for (int j = 0; j < 3; ++j) {
        u[i][j] = 0.f;
      }
    }

    KFPARTICLE_GPU_UNROLL
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
    KFPARTICLE_GPU_UNROLL
    for (int i = 0; i < 3; ++i) {
      u1[i] = u[i][i];
      u[i][i] = 1.f / u[i][i];
    }

    KFPARTICLE_GPU_UNROLL
    for (int i = 0; i < 2; ++i) {
      u[i][i + 1] = -u[i][i + 1] * u[i][i] * u[i + 1][i + 1];
    }

    u[0][2] = u[0][1] * u1[1] * u[1][2] - u[0][2] * u[0][0] * u[2][2];

    KFPARTICLE_GPU_UNROLL
    for (int i = 0; i < 3; ++i) {
      a[i + 3] = u[i][2] * u[2][2] * d[2];
    }

    KFPARTICLE_GPU_UNROLL
    for (int i = 0; i < 2; ++i) {
      a[i + 1] = u[i][1] * u[1][1] * d[1] + u[i][2] * u[1][2] * d[2];
    }

    a[0] = u[0][0] * u[0][0] * d[0]
           + u[0][1] * u[0][1] * d[1]
           + u[0][2] * u[0][2] * d[2];
  }

  /**
   * @brief Transform a symmetric matrix as SOut = Q * S * Q^T.
   * @tparam N Matrix dimension.
   * @param Q Dense row-major N-by-N transformation matrix.
   * @param S Symmetric matrix in packed lower-triangular order, N*(N+1)/2 entries.
   * @param[out] SOut Packed symmetric result; may alias S, but must not overlap Q.
   */
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
        KFPARTICLE_GPU_UNROLL
        for (int k = 0; k < N; ++k) {
          mA[ij] += S[KFParticleGpuFitState::CovarianceIndex(i, k)] * Q[j * N + k];
        }
      }
    }

    for (int i = 0; i < N; ++i) {
      for (int j = 0; j <= i; ++j) {
        const int ij = KFParticleGpuFitState::CovarianceIndex(i, j);
        SOut[ij] = 0.f;
        KFPARTICLE_GPU_UNROLL
        for (int k = 0; k < N; ++k) {
          SOut[ij] += Q[i * N + k] * mA[k * N + j];
        }
      }
    }
  }

  /**
   * @brief Create a simple mother seed by averaging positions and summing daughter four-momenta.
   * Assumes independent daughters and sums only their momentum-energy covariance blocks;
   * other covariance entries retain Initialize() defaults. Charges, chi2, NDF, and daughter masses are summed.
   * This is not a fitted common vertex or a Kalman combination; no transport or quality check is performed.
   */
  KFPARTICLE_GPU_HOST_DEVICE inline void BuildKinematicMother(const KFParticleGpuFitState& first,
                                                              const KFParticleGpuFitState& second,
                                                              KFParticleGpuFitState& mother);

  /**
   * @brief Return squared three-momentum, px^2 + py^2 + pz^2.
   */
  KFPARTICLE_GPU_HOST_DEVICE inline float Momentum2(const KFParticleGpuFitState& particle)
  {
    return particle.Px() * particle.Px()
           + particle.Py() * particle.Py()
           + particle.Pz() * particle.Pz();
  }

  /**
   * @brief Return invariant mass squared, E^2 - |p|^2, without clipping negative values.
   */
  KFPARTICLE_GPU_HOST_DEVICE inline float Mass2(const KFParticleGpuFitState& particle)
  {
    return particle.E() * particle.E() - Momentum2(particle);
  }

  /**
   * @brief Return sqrt(mass squared), or minus sqrt(-mass squared) for a spacelike state.
   * The negative value preserves an unphysical mass-squared sign for diagnostics; it is not a physical mass.
   */
  KFPARTICLE_GPU_HOST_DEVICE inline float Mass(const KFParticleGpuFitState& particle)
  {
    const float mass2 = Mass2(particle);
    return mass2 > 0.f ? KFParticleGpuSqrt(mass2) : -KFParticleGpuSqrt(-mass2);
  }

  /**
   * @brief Compute signed invariant mass and its first-order covariance uncertainty.
   * With g = (-px,-py,-pz,E), uses error = sqrt(g^T C g)/mass for the momentum-energy block.
   * @param[out] mass Signed square root of E^2-|p|^2, also written on failure.
   * @param[out] error Propagated uncertainty on success; 1e8 on an invalid mass/variance.
   * @return True when mass squared and variance are nonnegative and mass exceeds 1e-8.
   * This predicate is not a complete finite-state validation.
   */
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

  /**
   * @brief Return px^2 + py^2 relative to the original z axis.
   */
  KFPARTICLE_GPU_HOST_DEVICE inline float TransverseMomentum2(const KFParticleGpuFitState& particle)
  {
    return particle.Px() * particle.Px() + particle.Py() * particle.Py();
  }

  /**
   * @brief Project a point onto the particle's straight trajectory and return the transport parameter dS.
   * Normally dS = p dot (point-r) / |p|^2; dsdr contains derivatives with respect to (x,y,z,px,py,pz).
   * For |p|^2 < 1e-4 the denominator is replaced by 1. The derivative expressions keep the
   * same algebra with this substituted value; the low-momentum branch is a compatibility fallback,
   * not an exact projection.
   */
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

  /**
   * @brief Move position along a straight trajectory while leaving momentum and covariance unchanged.
   * Uses r' = r + dS*p. Energy and S are preserved; SFromDecay accumulates dS.
   * Use TransportLineWithJacobian() when covariance propagation is required.
   */
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

  /**
   * @brief Move state parameters along a helix in a uniform y-directed magnetic field.
   * Rotates (px,pz), preserves py and energy, and updates position and SFromDecay.
   * Small curvature uses line transport; small rotation uses a series approximation.
   * The covariance is copied unchanged: this is a kinematic transport, not an uncertainty propagation.
   */
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

  /**
   * @brief Transport a straight trajectory and propagate its covariance, including a state-dependent dS.
   * Adds p*(partial dS/partial state) to the position rows of the transport Jacobian and computes J*C*J^T.
   * @param dsdr Six derivatives of dS with respect to this particle's (x,y,z,px,py,pz); required.
   * @param dsdrOther Optional six derivatives of dS with respect to another state.
   * @param selfDerivative Optional row-major 6x6 output Jacobian for this particle.
   * @param otherDerivative Optional row-major 6x6 cross-state Jacobian; written only with dsdrOther.
   * The other state's covariance is not added here; callers combine it separately. S is unchanged.
   */
  KFPARTICLE_GPU_HOST_DEVICE inline void TransportLineWithJacobian(
    const KFParticleGpuFitState& particle,
    float dS,
    const float dsdr[6],
    KFParticleGpuFitState& transported,
    const float dsdrOther[6],
    float* selfDerivative,
    float* otherDerivative);

  /**
   * @brief Transport state and covariance with the fixed-order CBM three-point magnetic-field approximation.
   * Samples the packed z-dependent field region, applies polynomial bending terms, and propagates
   * the 8x8 covariance with the implemented transport Jacobian. This is not an exact field integrator;
   * the Jacobian follows the CPU transport approximation rather than differentiating every field sample.
   * @param dS Signed path divided by momentum magnitude.
   * @param[out] status Transport flags; inspect PathLimited even when the function returns true.
   * @param dsdr Optional six self-state derivatives of dS; null means a fixed transport step.
   * @param dsdrOther Optional six derivatives of dS with respect to another state.
   * @param selfDerivative Optional row-major 6x6 self Jacobian.
   * @param otherDerivative Optional row-major 6x6 cross Jacobian; written only with dsdrOther.
   * @return False for rejected input or checked non-finite charged output.
   * Neutral input delegates to line transport. Charged |dS*pz| > 1000 returns true with PathLimited:
   * the state remains unchanged and derivative outputs are not written. Derivative outputs are not
   * rolled back on later failure. Cross-state covariance must be added by the caller.
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

  /**
   * @brief Transport a straight trajectory and propagate its covariance, including a state-dependent dS.
   * Adds p*(partial dS/partial state) to the position rows of the transport Jacobian and computes J*C*J^T.
   * @param dsdr Six derivatives of dS with respect to this particle's (x,y,z,px,py,pz); required.
   * @param dsdrOther Optional six derivatives of dS with respect to another state.
   * @param selfDerivative Optional row-major 6x6 output Jacobian for this particle.
   * @param otherDerivative Optional row-major 6x6 cross-state Jacobian; written only with dsdrOther.
   * The other state's covariance is not added here; callers combine it separately. S is unchanged.
   */
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
    KFPARTICLE_GPU_UNROLL
    for (int i = 0; i < 3; ++i) {
      KFPARTICLE_GPU_UNROLL
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

  /**
   * @brief Find straight-line closest-approach transport parameters for two particles.
   * Solves the two projection equations with determinant (p1 dot p2)^2 - |p1|^2*|p2|^2.
   * Near-parallel determinants with magnitude below 1e-8 are clamped to signed 1e-8
   * (positive at exact zero). No uniqueness, finite-output, or distance-quality check is made.
   * @param[out] dS Two path/momentum parameters, in first/second order.
   */
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

  /**
   * @brief Find straight-line closest approach and its four blocks of state derivatives.
   * dsdr rows are dS1/dr1, dS1/dr2, dS2/dr1, dS2/dr2; each r contains (x,y,z,px,py,pz).
   * Unlike the value-only overload, |determinant| < 1e-4 is replaced by 1. The derivative
   * algebra still includes determinant derivatives; near this guard it is a CPU-compatible
   * regularization, not the exact derivative of a smooth closest-approach solution.
   */
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

  /**
   * @brief Transport both daughters to line closest approach and create an unfitted mother seed.
   * Uses value-only line transport and BuildKinematicMother(); it does not propagate DCA uncertainty.
   */
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

  /**
   * @brief Estimate two-track DCA in constant Bz, with CPU-compatible analytic derivative expressions.
   * Forms two transverse root pairs, chooses the smaller 3D separation or averages the roots,
   * then applies a tangent-line correction. This is a local construction, not a global helix-distance minimizer.
   * @param param1 Optional six-parameter coordinate array. If null, both particles' own arrays are used.
   * @param param2 Required with non-null param1, in the same coordinate frame.
   * @param useMiddlePoint Average the two root pairs before the tangent-line correction.
   * @param[out] dS Corrected first/second transport parameters.
   * @param[out] dsdr Rows dS1/dr1, dS1/dr2, dS2/dr1, dS2/dr2, with six columns each.
   * @param arithmeticTrace Optional diagnostic intermediates; not all fields are written on every branch.
   * Both-straight input delegates to the derivative line solver on the original particles.
   * Branch choices and denominator guards are not differentiated; derivatives use ideal atan2
   * formulas with the compatibility trigonometric values.
   */
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

  /**
   * @brief Use the Bz DCA/derivative solver for a y-directed field by rotating the coordinate frame.
   * Maps (x,y,z) to (x,-z,y), and likewise momentum, then maps derivative columns back.
   * dsdr uses the original-frame row convention dS1/dr1, dS1/dr2, dS2/dr1, dS2/dr2.
   * The optional arithmetic trace stays in the rotated solver frame. The both-straight
   * fallback in the underlying solver uses original particle arrays before this wrapper's remapping.
   */
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


  /**
   * @brief Estimate constant-By closest approach via the coordinate rotation (x,y,z) -> (x,-z,y).
   * Returns the selected or averaged root pair after tangent-line correction.
   * Optional firstRoots/secondRoots expose the two uncorrected transverse roots per particle;
   * they are not written when both trajectories take the straight-line fallback.
   */
  KFPARTICLE_GPU_HOST_DEVICE inline void GetDStoParticleByCpuCompatible(
    const KFParticleGpuFitState& first,
    const KFParticleGpuFitState& second,
    float by,
    float dS[2],
    bool useMiddlePoint = false,
    float firstRoots[2] = nullptr,
    float secondRoots[2] = nullptr);

  /**
   * @brief Move both daughters to a constant-By DCA estimate without changing their covariances.
   * Uses a line solution when both by*charge magnitudes are below 1e-8, otherwise the helix root solver.
   * @return False if either |dS*pz| exceeds 1000; both outputs are then initialized.
   * A true return does not certify finite output or successful vertex fitting.
   */
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

  /**
   * @brief Form an unfitted mother from daughters transported to constant-By DCA.
   * Combines BuildConstantByDcaKinematicSeed() with BuildKinematicMother(); covariance transport is omitted.
   * @return False on seed path-limit rejection, with mother initialized; otherwise true.
   */
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

  /**
   * @brief Estimate constant-Bz DCA parameters without computing derivatives.
   * Uses two transverse roots, their 3D separation (or their average), and a tangent-line correction.
   * @param firstParameters Six coordinates/momenta in the solver frame.
   * @param secondParameters Six coordinates/momenta in the same frame.
   * @param outputFirstRoots Optional pair of first-particle roots before correction.
   * @param outputSecondRoots Optional pair of second-particle roots before correction.
   * The both-straight branch uses the original particle states and leaves root outputs untouched.
   * Small/negative discriminants and small denominators use the implementation's compatibility guards;
   * there is no convergence or finite-result status.
   */
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

  /**
   * @brief Estimate constant-By closest approach via the coordinate rotation (x,y,z) -> (x,-z,y).
   * Returns the selected or averaged root pair after tangent-line correction.
   * Optional firstRoots/secondRoots expose the two uncorrected transverse roots per particle;
   * they are not written when both trajectories take the straight-line fallback.
   */
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

  /**
   * @brief Differentiate the mean of the two uncorrected helix roots with respect to each particle itself.
   * Computes only dS1/dr1 and dS2/dr2, not cross-state derivatives or the tangent correction.
   * @param field Constant By used after rotating coordinates to the Bz solver frame.
   * @param[out] firstDsdr Six self derivatives in rotated order (x,-z,y,px,-pz,py).
   * @param[out] secondDsdr Corresponding rotated-frame derivatives for the second particle.
   * @return False for a straight trajectory or nonpositive discriminant; outputs are then untouched.
   * The successful path does not map columns back or explicitly check derivative finiteness.
   */
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

  /**
   * @brief Extract the self-state blocks from the corrected constant-By DCA derivatives.
   * Copies rows 0 and 3 of GetDStoParticleByCpuCompatibleDerivatives() into the two six-element outputs.
   * useMiddlePoint selects the averaged-root branch rather than the smaller-distance root.
   */
  KFPARTICLE_GPU_HOST_DEVICE inline void GetCpuCompatibleSelectedDcaSelfDerivatives(
    const KFParticleGpuFitState& first,
    const KFParticleGpuFitState& second,
    float by,
    bool useMiddlePoint,
    float firstDsdr[6],
    float secondDsdr[6]);

  /**
   * @brief Decide whether the two-root ambiguity should use the mean transport parameters.
   * Transports both states to their mean roots, sets delta to their position separation, and tests
   * |delta|^4 < 25 * delta^T*(C1+C2)*delta with a positive finite projected covariance.
   * This compares separation with positional uncertainty; false also covers transport failure.
   * Optional distance/covariance outputs are written only after both transports succeed;
   * optional derivative outputs are copied earlier and follow the chosen helper's convention.
   */
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

  /**
   * @brief Extract the self-state blocks from the corrected constant-By DCA derivatives.
   * Copies rows 0 and 3 of GetDStoParticleByCpuCompatibleDerivatives() into the two six-element outputs.
   * useMiddlePoint selects the averaged-root branch rather than the smaller-distance root.
   */
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

  /**
   * @brief Estimate DCA with local By, then transport both daughters through their own full field regions.
   * By is sampled at the first particle's z; root averaging follows UseCpuCompatibleMiddleDcaPoint().
   * Transport uses fixed-dS Jacobians, so the returned covariances omit the DCA parameter's input dependence.
   * @return False for transport failure or PathLimited, with both states initialized.
   */
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

  /**
   * @brief Choose the full-field finite-difference step: 1e-3 * max(|value|, 1).
   * This is a parameter-scale heuristic, not an adaptive error estimate.
   */
  KFPARTICLE_GPU_HOST_DEVICE inline float FullFieldDcaDerivativeStep(float value)
  {
    const float scale = Abs(value) > 1.f ? Abs(value) : 1.f;
    return 1.e-3f * scale;
  }

  /**
   * @brief Fill one column of both coupled DCA Jacobians using a central finite difference.
   * Re-evaluates the full-field seed for the supplied plus/minus input pairs and divides their
   * six output-parameter differences by 2*step. Field coefficients remain fixed.
   * @pre step is positive and column is in [0,12).
   * @return False on seed failure or non-finite derivatives; result is initialized and status gains a failure bit.
   * Other Jacobian columns remain unchanged on success.
   */
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

  /**
   * @brief Numerically propagate both independent input covariances through the full DCA seed construction.
   * Builds two 6x12 Jacobians by central differences, then J*diag(C1,C2)*J^T for each 6x6 block
   * and the first-position/second-position cross covariance. Energy cross terms are rebuilt separately.
   * The separate covariance arrays contain the coupled 6x6 blocks; result.first/second retain
   * their fixed-dS blocks until a consumer replaces them. S-related terms are not rebuilt here.
   * @param[out] status Flags for neutral input or detected numerical/seed failures.
   * @return Whether the implemented input, seed, derivative, and position-correlation checks passed.
   * Finite differences may cross root-selection branches; this is a numerical reference, not an exact derivative.
   */
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

  /**
   * @brief Build an unfitted mother after local-By DCA estimation and full-field daughter transport.
   * Uses fixed-step transported covariances and the simplified independent momentum-energy sum.
   * @return False on seed failure, with mother initialized; no Kalman vertex update is performed.
   */
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

  /**
   * @brief Prepare a current state and daughter measurement at straight-line closest approach.
   * Propagates each covariance with self derivatives, adds the other input's contribution to
   * the six-coordinate block, and stores Cov(second position, first position) in Correlation.
   * Assumes independent input particles; the common DCA calculation induces their output correlation.
   * @return False if either |dS*pz| exceeds 1000, initializing both outputs; no final finite-state check is made.
   */
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
        KFPARTICLE_GPU_UNROLL
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
        KFPARTICLE_GPU_UNROLL
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
        KFPARTICLE_GPU_UNROLL
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
        KFPARTICLE_GPU_UNROLL
        for (int k = 0; k < 6; ++k) {
          daughterMeasurement.Correlation(i, j) +=
            secondSelfDerivative[i * 6 + k] * secondCovFirstOtherDerivativeT[k][j];
        }
      }
    }
    return true;
  }

  /**
   * @brief Combine constant-By transported parameters with line-DCA uncertainty estimates.
   * First builds the line seed, then replaces only state parameters and the current state's SFromDecay.
   * Covariances and the position correlation remain those of the line approximation.
   * @return Whether both the line and constant-By seed builders succeeded.
   */
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

    // This approximation keeps line-DCA covariance/correlation and replaces only the parameters.
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

  /**
   * @brief Compute the induced daughter-position/current-position cross covariance at DCA.
   * With A=firstSelf, B=firstOther, C=secondSelf, D=secondOther, stores the position block
   * of D*C1*A^T + C*C2*B^T. C1 and C2 are independent input six-coordinate covariances.
   * Derivative matrices are row-major 6x6; Correlation is an unnormalized covariance, not a coefficient.
   * @tparam Compact Compute only intermediates contributing to the 3x3 output; false retains full
   * 6x6 intermediate products as a numerical reference.
   */
  template<bool Compact>
  KFPARTICLE_GPU_HOST_DEVICE inline void FullFieldDcaCorrelation(
    const KFParticleGpuFitState& first, const KFParticleGpuFitState& second,
    const float* firstSelfDerivative, const float* firstOtherDerivative,
    const float* secondSelfDerivative, const float* secondOtherDerivative,
    KFParticleGpuMeasurement& daughterMeasurement)
  {
#if defined(__clang__)
#pragma clang fp contract(off)
#endif
    constexpr int columns = Compact ? 3 : 6;
    constexpr int rows = Compact ? 3 : 6;
    float firstCovFirstDerivativeT[6][columns] = {};
    for (int row = 0; row < 6; ++row) {
      for (int column = 0; column < columns; ++column) {
        KFPARTICLE_GPU_UNROLL
        for (int source = 0; source < 6; ++source) {
          firstCovFirstDerivativeT[row][column] +=
            first.Covariance(row, source) * firstSelfDerivative[column * 6 + source];
        }
      }
    }
    float secondOtherFirstCovFirstDerivativeT[rows][columns] = {};
    for (int row = 0; row < rows; ++row) {
      for (int column = 0; column < columns; ++column) {
        KFPARTICLE_GPU_UNROLL
        for (int source = 0; source < 6; ++source) {
          secondOtherFirstCovFirstDerivativeT[row][column] +=
            secondOtherDerivative[row * 6 + source]
            * firstCovFirstDerivativeT[source][column];
        }
      }
    }
    float secondCovFirstOtherDerivativeT[6][columns] = {};
    for (int row = 0; row < 6; ++row) {
      for (int column = 0; column < columns; ++column) {
        KFPARTICLE_GPU_UNROLL
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
        KFPARTICLE_GPU_UNROLL
        for (int source = 0; source < 6; ++source) {
          daughterMeasurement.Correlation(row, column) +=
            secondSelfDerivative[row * 6 + source]
            * secondCovFirstOtherDerivativeT[source][column];
        }
      }
    }
  }

  /**
   * @brief Prepare a correlated full-field DCA measurement using analytic local-By path derivatives.
   * Combines the CPU-compatible root decision with CBM field transport, adds cross-source 6x6
   * covariance contributions, and forms Cov(daughter position, current position). Inputs are independent.
   * The derivatives follow the local-By/transport approximation, not an exact variable-field DCA solution.
   * @tparam CompactCorrelation Limit correlation intermediates to the required position block.
   * @return False for failed/path-limited transport, initializing both outputs; otherwise true.
   */
  template<bool CompactCorrelation = true>
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

    FullFieldDcaCorrelation<CompactCorrelation>(
      first, second, firstSelfDerivative, firstOtherDerivative,
      secondSelfDerivative, secondOtherDerivative, daughterMeasurement);
    return true;
  }

  /**
   * @brief Use full-field states and fixed-dS covariances with a line-DCA cross-correlation approximation.
   * Builds the line measurement first, then replaces its state/covariance and the current state
   * using BuildFullFieldDcaKinematicSeed(); StoreState() preserves the previously computed correlation.
   * @return Whether both seed constructions succeeded.
   */
  KFPARTICLE_GPU_HOST_DEVICE inline bool BuildFullFieldDcaMeasurementSeedApprox(
    const KFParticleGpuFitState& first,
    const KFParticleGpuFitState& second,
    const KFParticleGpuFieldRegion& firstField,
    const KFParticleGpuFieldRegion& secondField,
    KFParticleGpuFitState& currentAtDca,
    KFParticleGpuMeasurement& daughterMeasurement)
  {
    // This approximation uses full-field states/covariances but retains the line-DCA correlation.
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

  /**
   * @brief Create a measurement seed from finite-difference coupled full-field DCA propagation.
   * Replaces both six-coordinate covariance blocks with the coupled result; rebuilt energy cross terms
   * and the remaining transport terms are retained. Copies coupled.Correlation without transposing:
   * its stored orientation is Cov(first position, second position), unlike the analytic seed helper.
   * @return False on coupled-result failure, initializing both outputs.
   */
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

  /**
   * @brief Perform the preliminary full-field transport of each daughter with self-DCA derivatives only.
   * Uses local By to select the DCA branch, but omits the other particle's covariance contribution
   * and the induced cross-correlation. The both-straight branch uses zero path derivatives.
   * @return False on failed/path-limited transport, initializing both outputs.
   */
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

  /**
   * @brief Package independently transported daughters as a current state and measurement.
   * Does not construct a cross-correlation block: StoreState() leaves any existing Correlation unchanged.
   * Pass a freshly initialized measurement when zero correlation is intended.
   * @return Whether BuildFullFieldDcaIndependentTransportStates() succeeded.
   */
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

  /**
   * @brief Reproduce the two transport passes used by the CPU V0 construction sequence.
   * First transports daughters independently, then computes the analytic coupled measurement from
   * those preliminary states. This prepares the seed only; the daughter-addition Kalman update follows separately.
   * @return Whether both preparation passes succeeded; failed preparation initializes the outputs.
   */
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

  /**
   * @brief Choose the constant-field finite-difference step: 1e-3 * max(|value|, 1).
   * This heuristic does not adapt to branch changes or numerical conditioning.
   */
  KFPARTICLE_GPU_HOST_DEVICE inline float NumericalDerivativeStep(float value)
  {
    const float scale = Abs(value) > 1.f ? Abs(value) : 1.f;
    return 1.e-3f * scale;
  }

  /**
   * @brief Add delta to one state parameter without updating covariance or dependent quantities.
   * @pre parameter is in [0,8); callers use this for controlled numerical differentiation.
   */
  KFPARTICLE_GPU_HOST_DEVICE inline void PerturbFitParameter(KFParticleGpuFitState& state,
                                                             int parameter,
                                                             float delta)
  {
    state.Parameter(parameter) += delta;
  }

  /**
   * @brief Numerically differentiate both constant-By DCA output states with respect to both inputs.
   * Produces four 7x7 Jacobians in (x,y,z,px,py,pz,E) order. Six coordinates are perturbed
   * centrally; self-energy derivatives are one and cross-energy derivatives are zero.
   * firstAtDca and secondAtDca are retained interface arguments and are not used.
   * @return False when a perturbed seed fails; Jacobians can then be partially filled.
   * No explicit finite-derivative check or fixed-root constraint is applied.
   */
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

  /**
   * @brief Write C' = J1*C1*J1^T + J2*C2*J2^T for two independent seven-parameter inputs.
   * Jacobians are dense 7x7 matrices in (x,y,z,px,py,pz,E) order. Only the destination's
   * packed 7x7 covariance block is overwritten; parameters and S-related entries remain unchanged.
   * @pre Destination storage does not alias either input covariance.
   */
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

  /**
   * @brief Write C' = J1*C1*J1^T + J2*C2*J2^T for two independent seven-parameter inputs.
   * Jacobians are dense 7x7 matrices in (x,y,z,px,py,pz,E) order. Only the destination's
   * packed 7x7 covariance block is overwritten; parameters and S-related entries remain unchanged.
   * @pre Destination storage does not alias either input covariance.
   */
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

  /**
   * @brief Return one cross-covariance entry between two outputs depending on the same independent inputs.
   * Computes the selected entry of L1*C1*R1^T + L2*C2*R2^T using dense 7x7 Jacobians.
   * @pre leftRow and rightRow are in [0,7); input covariances have no mutual cross-correlation.
   */
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

  /**
   * @brief Prepare constant-By DCA states with numerical two-source covariance propagation.
   * Uses finite-difference 7x7 seed Jacobians, includes energy correlations, and stores
   * Cov(second position, first position) in the daughter measurement.
   * @return False if the nominal or any perturbed seed fails, initializing both outputs.
   */
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

  /**
   * @brief Fit a daughter to the current vertex while adding its momentum and energy to the mother.
   * Uses the three-position residual and summed position covariance for the Kalman update of
   * (x,y,z,px,py,pz,E), then adds the supplied position cross-correlation correction.
   * On success adds two NDF and daughterCharge, accumulates chi2, and resets SFromDecay.
   * No mass constraint, transport, or daughter-mass bookkeeping is performed here.
   * @return False, without changing particle, if the chi2 increment is NaN or exceeds 1e9.
   * A true return does not certify covariance positivity or finite output; no post-update check is made.
   */
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
    KFPARTICLE_GPU_UNROLL
    for (int i = 0; i < 3; ++i) {
      KFPARTICLE_GPU_UNROLL
      for (int j = 0; j < 3; ++j) {
        K[i][j] = 0.f;
        KFPARTICLE_GPU_UNROLL
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
    KFPARTICLE_GPU_UNROLL
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

    KFPARTICLE_GPU_UNROLL
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
    KFPARTICLE_GPU_UNROLL
    for (int i = 0; i < 3; ++i) {
      KFPARTICLE_GPU_UNROLL
      for (int j = 0; j < 3; ++j) {
        K2[i][j] = -K[j][i];
      }
      K2[i][i] += 1.f;
    }

    float A[3][3];
    KFPARTICLE_GPU_UNROLL
    for (int i = 0; i < 3; ++i) {
      KFPARTICLE_GPU_UNROLL
      for (int j = 0; j < 3; ++j) {
        A[i][j] = 0.f;
        KFPARTICLE_GPU_UNROLL
        for (int k = 0; k < 3; ++k) {
          A[i][j] += measurement.Correlation(i, k) * K2[k][j];
        }
      }
    }

    float M[3][3];
    KFPARTICLE_GPU_UNROLL
    for (int i = 0; i < 3; ++i) {
      KFPARTICLE_GPU_UNROLL
      for (int j = 0; j < 3; ++j) {
        M[i][j] = 0.f;
        KFPARTICLE_GPU_UNROLL
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
   * @brief Fit a measured daughter spatially while subtracting its four-momentum to form a missing state.
   * This is not the inverse of AddDaughterWithEnergyFit(): uncertainty is added before the
   * Kalman correction, and the historical CPU charge update still adds daughterCharge.
   * Adds two NDF and the fit chi2, resets SFromDecay, and applies the position-correlation correction.
   * @return False before mutation if the chi2 increment is NaN or exceeds 1e9; otherwise true.
   * Callers needing the physical difference charge must set it explicitly.
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
    KFPARTICLE_GPU_UNROLL
    for (int i = 0; i < 3; ++i) {
      KFPARTICLE_GPU_UNROLL
      for (int j = 0; j < 3; ++j) {
        K[i][j] = 0.f;
        KFPARTICLE_GPU_UNROLL
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
    KFPARTICLE_GPU_UNROLL
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

    KFPARTICLE_GPU_UNROLL
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
    KFPARTICLE_GPU_UNROLL
    for (int i = 0; i < 3; ++i) {
      KFPARTICLE_GPU_UNROLL
      for (int j = 0; j < 3; ++j) {
        K2[i][j] = -K[j][i];
      }
      K2[i][i] += 1.f;
    }
    float A[3][3];
    KFPARTICLE_GPU_UNROLL
    for (int i = 0; i < 3; ++i) {
      KFPARTICLE_GPU_UNROLL
      for (int j = 0; j < 3; ++j) {
        A[i][j] = 0.f;
        KFPARTICLE_GPU_UNROLL
        for (int k = 0; k < 3; ++k) {
          A[i][j] += measurement.Correlation(i, k) * K2[k][j];
        }
      }
    }
    float M[3][3];
    KFPARTICLE_GPU_UNROLL
    for (int i = 0; i < 3; ++i) {
      KFPARTICLE_GPU_UNROLL
      for (int j = 0; j < 3; ++j) {
        M[i][j] = 0.f;
        KFPARTICLE_GPU_UNROLL
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

  /**
   * @brief Filter mother/daughter spatial states and infer a missing particle with supplied mass hypotheses.
   * Builds a subtraction state, corrects its charge to mother.Q()-daughterCharge, and assigns energies
   * from combinations of mass-shell energies: Eneutral=Em-Ed, EmFiltered=Ed+En, EdFiltered=Em-En.
   * The three energies on the right use the filtered momenta and supplied masses.
   * This is not a simultaneous nonlinear mass fit; energy covariances are not recomputed after assignment.
   * @return Whether the implemented mass/chi2 checks and final finite-energy/positive-neutral-energy checks pass.
   * Outputs may already be modified on failure; this is not a complete finite-state validation.
   */
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

  /**
   * @brief Apply one linearized Kalman update toward E^2-|p|^2 = mass^2.
   * Uses H=(0,0,0,-2px,-2py,-2pz,2E,0) and variance H*C*H^T + mass^2*sigmaMass^2.
   * The last term is this implementation's noise convention, not (2*mass*sigmaMass)^2.
   * Updates all eight parameters/covariance entries, adds one NDF and residual chi2,
   * and sets MassHypo and SumDaughterMass. One linearization need not enforce the target mass exactly.
   * @return False before mutation for negative inputs or the implemented variance/NaN guards.
   * sigmaMass=0 removes the measurement-noise term; no post-update finite-state test is made.
   */
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

  /**
   * @brief Project a particle to a vertex position and propagate covariance through the line projection.
   * Includes the dependence of dS on the particle's state, but not vertex uncertainty.
   * Input and output may alias because the original state is copied before transport.
   * @return False for NaN or |dS|>1e6, or NaN output coordinates; this is not a full finiteness check.
   * Output can be modified when the final coordinate check fails.
   */
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
    // Include the particle dependence of the projection step; vertex uncertainty is handled separately.
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

  /**
   * @brief Apply a position-only vertex measurement using the daughter-addition Kalman update.
   * The measurement has vertex coordinates/covariance but zero four-momentum and charge.
   * On success sets AtProductionVertex; it does not first transport to the vertex or rebuild S and its covariance.
   * Use SetNeutralProductionVertex() for the neutral-particle transport and decay-length treatment.
   * @return The update outcome; particle is unchanged if that update rejects its chi2 increment.
   */
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
   * @brief Transport a neutral particle to a production vertex, constrain it, and rebuild its decay parameter.
   * Includes vertex-induced covariance/correlation, performs the position update, then estimates
   * S back toward the saved decay position and fills S-state covariances for the six coordinates
   * and Var(S), including the saved decay-position uncertainty. The existing Cov(S,E) is retained.
   * Adds two NDF, accumulates chi2, and marks AtProductionVertex; only charge-zero input is supported.
   * @return Whether input/path checks and final IsFiniteState() checks pass.
   * Early rejection leaves particle unchanged; a final failure can leave it modified.
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

  /**
   * @brief Compute the signed decay length L = S*|p| and its first-order propagated uncertainty.
   * Uses the S/momentum covariance including cross terms; error is sqrt(abs(variance)),
   * so a negative variance is not itself rejected.
   * @return False for |p|^2 <= 1e-4 (length=0, error=1e20), or non-finite result values.
   * Requires a meaningful S and S covariance, normally established by a production-vertex constraint.
   */
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

  /**
   * @brief Create a simple mother seed by averaging positions and summing daughter four-momenta.
   * Assumes independent daughters and sums only their momentum-energy covariance blocks;
   * other covariance entries retain Initialize() defaults. Charges, chi2, NDF, and daughter masses are summed.
   * This is not a fitted common vertex or a Kalman combination; no transport or quality check is performed.
   */
  KFPARTICLE_GPU_HOST_DEVICE inline void BuildKinematicMother(const KFParticleGpuFitState& first,
                                                              const KFParticleGpuFitState& second,
                                                              KFParticleGpuFitState& mother)
  {
    mother.Initialize();

    // This simple seed is distinct from the measurement preparation and Kalman-fit path.
    mother.X() = 0.5f * (first.X() + second.X());
    mother.Y() = 0.5f * (first.Y() + second.Y());
    mother.Z() = 0.5f * (first.Z() + second.Z());
    mother.Px() = first.Px() + second.Px();
    mother.Py() = first.Py() + second.Py();
    mother.Pz() = first.Pz() + second.Pz();
    mother.E() = first.E() + second.E();
    mother.S() = 0.f;

    // Independent daughter inputs: this seed propagates only the momentum-energy covariance block.
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
