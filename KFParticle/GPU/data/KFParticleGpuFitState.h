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
 * @brief Fixed-size numerical track, vertex, field, particle, and measurement values.
 *
 * Host/device helpers own inline arrays, not dynamic allocations. Indexed access
 * is unchecked; callers supply valid indices and readable input arrays. Returned
 * pointers/references borrow the containing object's storage. These values do not
 * manage event identity, buffer transfers, or synchronization.
 */

#ifndef KFPARTICLEGPUFITSTATE_H
#define KFPARTICLEGPUFITSTATE_H

#include "KFParticleGpuPlatform.h"

/**
 * @brief Six-parameter track state before applying a particle mass hypothesis.
 *
 * Selection metadata is deliberately stored elsewhere so lightweight kernels
 * do not have to load track parameters and covariance. Parameters are ordered
 * as (x, y, z, px, py, pz); covariance stores the lower triangle of the symmetric
 * 6x6 matrix, row by row. Initialization copies values without validating them.
 */
class KFParticleGpuTrackState
{
 public:
  static const int NumberOfParameters = 6;  ///< Number of coordinates in this state's parameter vector.
  static const int NumberOfCovarianceElements = 21;  ///< Number of entries in the packed symmetric covariance.

  /// @brief Zeroes all track parameters and covariance elements.
  KFPARTICLE_GPU_HOST_DEVICE KFParticleGpuTrackState() { Reset(); }

  /// @brief Zeroes all six parameters and 21 covariance entries.
  KFPARTICLE_GPU_HOST_DEVICE void Reset()
  {
    for (int i = 0; i < NumberOfParameters; ++i) {
      fParameters[i] = 0.f;
    }
    for (int i = 0; i < NumberOfCovarianceElements; ++i) {
      fCovariance[i] = 0.f;
    }
  }

  /// @brief Copies six parameters and 21 covariance entries; both input arrays must be readable.
  KFPARTICLE_GPU_HOST_DEVICE void Initialize(const float parameters[NumberOfParameters],
                                             const float covariance[NumberOfCovarianceElements])
  {
    for (int i = 0; i < NumberOfParameters; ++i) {
      fParameters[i] = parameters[i];
    }
    for (int i = 0; i < NumberOfCovarianceElements; ++i) {
      fCovariance[i] = covariance[i];
    }
  }

  /// @brief Returns the symmetric packed index for matrix coordinates; both indices must be in [0, NumberOfParameters).
  KFPARTICLE_GPU_HOST_DEVICE static int CovarianceIndex(int i, int j)
  {
    return (j <= i) ? i * (i + 1) / 2 + j : j * (j + 1) / 2 + i;
  }

  /// @brief Exposes writable access to one parameter; requires 0 <= i < NumberOfParameters.
  KFPARTICLE_GPU_HOST_DEVICE float& Parameter(int i) { return fParameters[i]; }

  /// @brief Reads one parameter; requires 0 <= i < NumberOfParameters.
  KFPARTICLE_GPU_HOST_DEVICE const float& Parameter(int i) const { return fParameters[i]; }

  /// @brief Borrows writable contiguous parameter storage; ownership remains with this state.
  KFPARTICLE_GPU_HOST_DEVICE float* Parameters() { return fParameters; }

  /// @brief Borrows read-only contiguous parameter storage; ownership remains with this state.
  KFPARTICLE_GPU_HOST_DEVICE const float* Parameters() const { return fParameters; }

  /// @brief Exposes writable access to one packed covariance entry; requires a valid flattened index.
  KFPARTICLE_GPU_HOST_DEVICE float& Covariance(int i) { return fCovariance[i]; }

  /// @brief Reads one packed covariance entry; requires a valid flattened index.
  KFPARTICLE_GPU_HOST_DEVICE const float& Covariance(int i) const { return fCovariance[i]; }

  /// @brief Borrows writable packed covariance storage; ownership remains with this state.
  KFPARTICLE_GPU_HOST_DEVICE float* Covariances() { return fCovariance; }

  /// @brief Borrows read-only packed covariance storage; ownership remains with this state.
  KFPARTICLE_GPU_HOST_DEVICE const float* Covariances() const { return fCovariance; }

  /// @brief Exposes writable access to a symmetric covariance element; both indices must be valid parameter indices.
  KFPARTICLE_GPU_HOST_DEVICE float& Covariance(int i, int j) { return fCovariance[CovarianceIndex(i, j)]; }

  /// @brief Reads a symmetric covariance element; both indices must be valid parameter indices.
  KFPARTICLE_GPU_HOST_DEVICE const float& Covariance(int i, int j) const
  {
    return fCovariance[CovarianceIndex(i, j)];
  }

  /// @brief Exposes writable access to the stored x position.
  KFPARTICLE_GPU_HOST_DEVICE float& X() { return fParameters[0]; }

  /// @brief Exposes writable access to the stored y position.
  KFPARTICLE_GPU_HOST_DEVICE float& Y() { return fParameters[1]; }

  /// @brief Exposes writable access to the stored z position.
  KFPARTICLE_GPU_HOST_DEVICE float& Z() { return fParameters[2]; }

  /// @brief Exposes writable access to the stored x momentum component.
  KFPARTICLE_GPU_HOST_DEVICE float& Px() { return fParameters[3]; }

  /// @brief Exposes writable access to the stored y momentum component.
  KFPARTICLE_GPU_HOST_DEVICE float& Py() { return fParameters[4]; }

  /// @brief Exposes writable access to the stored z momentum component.
  KFPARTICLE_GPU_HOST_DEVICE float& Pz() { return fParameters[5]; }

  /// @brief Reads the stored x position.
  KFPARTICLE_GPU_HOST_DEVICE const float& X() const { return fParameters[0]; }

  /// @brief Reads the stored y position.
  KFPARTICLE_GPU_HOST_DEVICE const float& Y() const { return fParameters[1]; }

  /// @brief Reads the stored z position.
  KFPARTICLE_GPU_HOST_DEVICE const float& Z() const { return fParameters[2]; }

  /// @brief Reads the stored x momentum component.
  KFPARTICLE_GPU_HOST_DEVICE const float& Px() const { return fParameters[3]; }

  /// @brief Reads the stored y momentum component.
  KFPARTICLE_GPU_HOST_DEVICE const float& Py() const { return fParameters[4]; }

  /// @brief Reads the stored z momentum component.
  KFPARTICLE_GPU_HOST_DEVICE const float& Pz() const { return fParameters[5]; }

 private:
  float fParameters[NumberOfParameters];  ///< Owned parameter array in the order documented for this state.
  float fCovariance[NumberOfCovarianceElements];  ///< Owned symmetric covariance in packed lower-triangular order.
};

/**
 * @brief Numerical primary vertex with position, covariance, and fit-quality counters.
 *
 * Parameters are (x, y, z); the symmetric covariance is packed as
 * (xx, yx, yy, zx, zy, zz). No event ID or track membership is stored.
 * Initialization copies values without validating covariance or fit quality.
 */
class KFParticleGpuVertexState
{
 public:
  static const int NumberOfParameters = 3;  ///< Number of coordinates in this state's parameter vector.
  static const int NumberOfCovarianceElements = 6;  ///< Number of entries in the packed symmetric covariance.

  /// @brief Zeroes position, covariance, fit quality, and contributor count.
  KFPARTICLE_GPU_HOST_DEVICE KFParticleGpuVertexState() { Initialize(); }

  /// @brief Resets all vertex values and counters to zero.
  KFPARTICLE_GPU_HOST_DEVICE void Initialize()
  {
    for (int i = 0; i < NumberOfParameters; ++i) {
      fParameters[i] = 0.f;
    }
    for (int i = 0; i < NumberOfCovarianceElements; ++i) {
      fCovariance[i] = 0.f;
    }
    fChi2 = 0.f;
    fNDF = 0;
    fNContributors = 0;
  }

  /// @brief Copies position, covariance, chi2, NDF, and contributor count without validation.
  KFPARTICLE_GPU_HOST_DEVICE void Initialize(const float parameters[NumberOfParameters],
                                             const float covariance[NumberOfCovarianceElements],
                                             float chi2,
                                             int ndf,
                                             int nContributors)
  {
    for (int i = 0; i < NumberOfParameters; ++i) {
      fParameters[i] = parameters[i];
    }
    for (int i = 0; i < NumberOfCovarianceElements; ++i) {
      fCovariance[i] = covariance[i];
    }
    fChi2 = chi2;
    fNDF = ndf;
    fNContributors = nContributors;
  }

  /// @brief Exposes writable access to one parameter; requires 0 <= i < NumberOfParameters.
  KFPARTICLE_GPU_HOST_DEVICE float& Parameter(int i) { return fParameters[i]; }

  /// @brief Reads one parameter; requires 0 <= i < NumberOfParameters.
  KFPARTICLE_GPU_HOST_DEVICE const float& Parameter(int i) const { return fParameters[i]; }

  /// @brief Exposes writable access to one packed covariance entry; requires a valid flattened index.
  KFPARTICLE_GPU_HOST_DEVICE float& Covariance(int i) { return fCovariance[i]; }

  /// @brief Reads one packed covariance entry; requires a valid flattened index.
  KFPARTICLE_GPU_HOST_DEVICE const float& Covariance(int i) const { return fCovariance[i]; }

  /// @brief Exposes writable access to the stored x position.
  KFPARTICLE_GPU_HOST_DEVICE float& X() { return fParameters[0]; }

  /// @brief Exposes writable access to the stored y position.
  KFPARTICLE_GPU_HOST_DEVICE float& Y() { return fParameters[1]; }

  /// @brief Exposes writable access to the stored z position.
  KFPARTICLE_GPU_HOST_DEVICE float& Z() { return fParameters[2]; }

  /// @brief Reads the stored x position.
  KFPARTICLE_GPU_HOST_DEVICE const float& X() const { return fParameters[0]; }

  /// @brief Reads the stored y position.
  KFPARTICLE_GPU_HOST_DEVICE const float& Y() const { return fParameters[1]; }

  /// @brief Reads the stored z position.
  KFPARTICLE_GPU_HOST_DEVICE const float& Z() const { return fParameters[2]; }

  /// @brief Exposes writable access to the stored fit chi2.
  KFPARTICLE_GPU_HOST_DEVICE float& Chi2() { return fChi2; }

  /// @brief Reads the stored fit chi2.
  KFPARTICLE_GPU_HOST_DEVICE const float& Chi2() const { return fChi2; }

  /// @brief Exposes writable access to the stored number of fit degrees of freedom.
  KFPARTICLE_GPU_HOST_DEVICE int& NDF() { return fNDF; }

  /// @brief Reads the stored number of fit degrees of freedom.
  KFPARTICLE_GPU_HOST_DEVICE const int& NDF() const { return fNDF; }

  /// @brief Exposes writable access to the stored number of vertex contributors.
  KFPARTICLE_GPU_HOST_DEVICE int& NContributors() { return fNContributors; }

  /// @brief Reads the stored number of vertex contributors.
  KFPARTICLE_GPU_HOST_DEVICE const int& NContributors() const { return fNContributors; }

 private:
  float fParameters[NumberOfParameters];  ///< Owned parameter array in the order documented for this state.
  float fCovariance[NumberOfCovarianceElements];  ///< Owned symmetric covariance in packed lower-triangular order.
  float fChi2;  ///< Stored fit chi2.
  int fNDF;  ///< Stored vertex degrees of freedom.
  int fNContributors;  ///< Stored vertex contributor count.
};

/**
 * @brief Three Cartesian magnetic-field components in the transport code's field units.
 * @note A plain value with no position, field map, ownership, or finite-value checks.
 */
struct KFParticleGpuFieldValue
{
  /// @brief Creates a zero magnetic-field value.
  KFPARTICLE_GPU_HOST_DEVICE KFParticleGpuFieldValue() : x(0.f), y(0.f), z(0.f) {}

  /// @brief Stores the supplied Cartesian field components without validation.
  KFPARTICLE_GPU_HOST_DEVICE KFParticleGpuFieldValue(float bx, float by, float bz) : x(bx), y(by), z(bz) {}

  /**
   * @brief Updates each component as this += weight * (other - this).
   * @note Weights in [0, 1] interpolate; values outside that interval extrapolate and are not clamped.
   */
  KFPARTICLE_GPU_HOST_DEVICE void Combine(const KFParticleGpuFieldValue& other, float weight)
  {
    x += weight * (other.x - x);
    y += weight * (other.y - y);
    z += weight * (other.z - z);
  }

  float x;  ///< Cartesian Bx component.
  float y;  ///< Cartesian By component.
  float z;  ///< Cartesian Bz component.
};

/**
 * @brief Local magnetic field represented by three quadratic polynomials in z.
 *
 * Separating it avoids carrying ten coefficients through kernels that do not
 * transport particles in a nonhomogeneous field. Coefficients 0..2 describe Bx,
 * 3..5 describe By, and 6..8 describe Bz, each as a + b*dz + c*dz^2.
 * Coefficient 9 is the reference z coordinate, so dz = z - coefficients[9].
 * This is a stored approximation, not a field-map lookup or a polynomial fitter.
 */
class KFParticleGpuFieldRegion
{
 public:
  static const int NumberOfCoefficients = 10;  ///< Nine polynomial coefficients plus the reference z coordinate.

  /// @brief Creates a zero field with reference z equal to zero.
  KFPARTICLE_GPU_HOST_DEVICE KFParticleGpuFieldRegion() { Reset(); }

  /// @brief Copies ten coefficients from a readable input array without validation.
  KFPARTICLE_GPU_HOST_DEVICE explicit KFParticleGpuFieldRegion(
    const float coefficients[NumberOfCoefficients])
  {
    SetCoefficients(coefficients);
  }

  /// @brief Zeroes all ten coefficients, including the reference z coordinate.
  KFPARTICLE_GPU_HOST_DEVICE void Reset()
  {
    for (int i = 0; i < NumberOfCoefficients; ++i) {
      fCoefficients[i] = 0.f;
    }
  }

  /// @brief Copies ten coefficients from a readable array; performs no fitting or validity checks.
  KFPARTICLE_GPU_HOST_DEVICE void SetCoefficients(const float coefficients[NumberOfCoefficients])
  {
    for (int i = 0; i < NumberOfCoefficients; ++i) {
      fCoefficients[i] = coefficients[i];
    }
  }

  /// @brief Exposes writable access to a field coefficient; requires 0 <= i < NumberOfCoefficients.
  KFPARTICLE_GPU_HOST_DEVICE float& Coefficient(int i) { return fCoefficients[i]; }

  /// @brief Reads a field coefficient; requires 0 <= i < NumberOfCoefficients.
  KFPARTICLE_GPU_HOST_DEVICE const float& Coefficient(int i) const { return fCoefficients[i]; }

  /// @brief Borrows writable coefficient storage; ownership remains with this region.
  KFPARTICLE_GPU_HOST_DEVICE float* Coefficients() { return fCoefficients; }

  /// @brief Borrows read-only coefficient storage; ownership remains with this region.
  KFPARTICLE_GPU_HOST_DEVICE const float* Coefficients() const { return fCoefficients; }

  /// @brief Evaluates all three field polynomials at z; no validity-range or finite-value checks are applied.
  KFPARTICLE_GPU_HOST_DEVICE KFParticleGpuFieldValue Get(float z) const
  {
#if defined(__clang__)
#pragma clang fp contract(off)
#endif
    const float dz = z - fCoefficients[9];
    const float dz2 = dz * dz;
    return KFParticleGpuFieldValue(fCoefficients[0] + fCoefficients[1] * dz + fCoefficients[2] * dz2,
                                   fCoefficients[3] + fCoefficients[4] * dz + fCoefficients[5] * dz2,
                                   fCoefficients[6] + fCoefficients[7] * dz + fCoefficients[8] * dz2);
  }

 private:
  float fCoefficients[NumberOfCoefficients];  ///< Owned polynomial coefficients and reference z coordinate.
};

/**
 * @brief Fixed-size particle state used as a local value by KFParticle fit operations.
 *
 * Field coefficients and reconstruction metadata remain separate because they
 * are not needed by every mathematical operation. Parameters are
 * (x, y, z, px, py, pz, E, S), with the symmetric 8x8 covariance packed by
 * lower-triangular rows. S is the KFParticle decay-length/transport parameter,
 * not a spatial distance by itself. Additional scalars record fit and transport history.
 *
 * Mutable accessors expose the stored values directly and maintain no derived
 * quantities automatically. Default initialization prepares a construction seed,
 * not a fitted or validated particle. The class itself provides no thread isolation.
 */
class KFParticleGpuFitState
{
 public:
  static const int NumberOfParameters = 8;  ///< Number of coordinates in this state's parameter vector.
  static const int NumberOfCovarianceElements = 36;  ///< Number of entries in the packed symmetric covariance.

  /// @brief Creates the default construction seed via Initialize().
  KFPARTICLE_GPU_HOST_DEVICE KFParticleGpuFitState() { Initialize(); }

  /**
   * @brief Resets the construction seed: zero parameters, position variances 100, Var(S) = 1, and NDF = -3.
   * @note Other covariance entries, chi2, transport history, charge, and flags are zero;
   * the mass hypothesis is -1 and the summed daughter mass is zero.
   */
  KFPARTICLE_GPU_HOST_DEVICE void Initialize()
  {
    for (int i = 0; i < NumberOfParameters; ++i) {
      fParameters[i] = 0.f;
    }
    for (int i = 0; i < NumberOfCovarianceElements; ++i) {
      fCovariance[i] = 0.f;
    }

    fCovariance[0] = 100.f;
    fCovariance[2] = 100.f;
    fCovariance[5] = 100.f;
    fCovariance[35] = 1.f;

    fChi2 = 0.f;
    fSFromDecay = 0.f;
    fSumDaughterMass = 0.f;
    fMassHypothesis = -1.f;
    fNdf = -3;
    fCharge = 0;
    fAtProductionVertex = 0;
    fConstructMethod = 0;
  }

  /**
   * @brief Copies the track, computes E = sqrt(mass^2 + p^2), and propagates covariance to energy.
   * @param track Six-parameter input track; values are not checked for physical validity.
   * @param charge Electric charge stored in the fit state.
   * @param mass Mass hypothesis used for energy and daughter-mass bookkeeping.
   * @note Resets transport history and flags, sets NDF to zero and Var(S) to one.
   * Energy covariance is evaluated only for positive energy; zero energy avoids division by zero.
   */
  KFPARTICLE_GPU_HOST_DEVICE void Initialize(const KFParticleGpuTrackState& track, int charge, float mass)
  {
    for (int i = 0; i < NumberOfParameters; ++i) {
      fParameters[i] = 0.f;
    }
    for (int i = 0; i < NumberOfCovarianceElements; ++i) {
      fCovariance[i] = 0.f;
    }
    for (int i = 0; i < KFParticleGpuTrackState::NumberOfParameters; ++i) {
      fParameters[i] = track.Parameter(i);
    }
    for (int i = 0; i < KFParticleGpuTrackState::NumberOfCovarianceElements; ++i) {
      fCovariance[i] = track.Covariance(i);
    }

    const float energy = KFParticleGpuSqrt(mass * mass + Px() * Px() + Py() * Py() + Pz() * Pz());
    E() = energy;
    S() = 0.f;

    // Invalid zero-energy tracks stay finite and can be rejected by selection.
    if (energy > 0.f) {
      const float energyInv = 1.f / energy;
      const float h0 = Px() * energyInv;
      const float h1 = Py() * energyInv;
      const float h2 = Pz() * energyInv;

      fCovariance[21] = h0 * fCovariance[6] + h1 * fCovariance[10] + h2 * fCovariance[15];
      fCovariance[22] = h0 * fCovariance[7] + h1 * fCovariance[11] + h2 * fCovariance[16];
      fCovariance[23] = h0 * fCovariance[8] + h1 * fCovariance[12] + h2 * fCovariance[17];
      fCovariance[24] = h0 * fCovariance[9] + h1 * fCovariance[13] + h2 * fCovariance[18];
      fCovariance[25] = h0 * fCovariance[13] + h1 * fCovariance[14] + h2 * fCovariance[19];
      fCovariance[26] = h0 * fCovariance[18] + h1 * fCovariance[19] + h2 * fCovariance[20];
      fCovariance[27] = h0 * h0 * fCovariance[9] + h1 * h1 * fCovariance[14]
                        + h2 * h2 * fCovariance[20]
                        + 2.f * (h0 * h1 * fCovariance[13] + h0 * h2 * fCovariance[18]
                                   + h1 * h2 * fCovariance[19]);
    }
    fCovariance[35] = 1.f;

    fChi2 = 0.f;
    fSFromDecay = 0.f;
    fSumDaughterMass = mass;
    fMassHypothesis = mass;
    fNdf = 0;
    fCharge = charge;
    fAtProductionVertex = 0;
    fConstructMethod = 0;
  }

  /// @brief Returns the symmetric packed index for matrix coordinates; both indices must be in [0, NumberOfParameters).
  KFPARTICLE_GPU_HOST_DEVICE static int CovarianceIndex(int i, int j)
  {
    return (j <= i) ? i * (i + 1) / 2 + j : j * (j + 1) / 2 + i;
  }

  /// @brief Exposes writable access to one parameter; requires 0 <= i < NumberOfParameters.
  KFPARTICLE_GPU_HOST_DEVICE float& Parameter(int i) { return fParameters[i]; }

  /// @brief Reads one parameter; requires 0 <= i < NumberOfParameters.
  KFPARTICLE_GPU_HOST_DEVICE const float& Parameter(int i) const { return fParameters[i]; }
  // Matrix routines need contiguous arrays, but storage remains owned by this state.

  /// @brief Borrows writable contiguous parameter storage; ownership remains with this state.
  KFPARTICLE_GPU_HOST_DEVICE float* Parameters() { return fParameters; }

  /// @brief Borrows read-only contiguous parameter storage; ownership remains with this state.
  KFPARTICLE_GPU_HOST_DEVICE const float* Parameters() const { return fParameters; }

  /// @brief Exposes writable access to one packed covariance entry; requires a valid flattened index.
  KFPARTICLE_GPU_HOST_DEVICE float& Covariance(int i) { return fCovariance[i]; }

  /// @brief Reads one packed covariance entry; requires a valid flattened index.
  KFPARTICLE_GPU_HOST_DEVICE const float& Covariance(int i) const { return fCovariance[i]; }

  /// @brief Borrows writable packed covariance storage; ownership remains with this state.
  KFPARTICLE_GPU_HOST_DEVICE float* Covariances() { return fCovariance; }

  /// @brief Borrows read-only packed covariance storage; ownership remains with this state.
  KFPARTICLE_GPU_HOST_DEVICE const float* Covariances() const { return fCovariance; }

  /// @brief Exposes writable access to a symmetric covariance element; both indices must be valid parameter indices.
  KFPARTICLE_GPU_HOST_DEVICE float& Covariance(int i, int j) { return fCovariance[CovarianceIndex(i, j)]; }

  /// @brief Reads a symmetric covariance element; both indices must be valid parameter indices.
  KFPARTICLE_GPU_HOST_DEVICE const float& Covariance(int i, int j) const
  {
    return fCovariance[CovarianceIndex(i, j)];
  }

  /// @brief Exposes writable access to the stored x position.
  KFPARTICLE_GPU_HOST_DEVICE float& X() { return fParameters[0]; }

  /// @brief Exposes writable access to the stored y position.
  KFPARTICLE_GPU_HOST_DEVICE float& Y() { return fParameters[1]; }

  /// @brief Exposes writable access to the stored z position.
  KFPARTICLE_GPU_HOST_DEVICE float& Z() { return fParameters[2]; }

  /// @brief Exposes writable access to the stored x momentum component.
  KFPARTICLE_GPU_HOST_DEVICE float& Px() { return fParameters[3]; }

  /// @brief Exposes writable access to the stored y momentum component.
  KFPARTICLE_GPU_HOST_DEVICE float& Py() { return fParameters[4]; }

  /// @brief Exposes writable access to the stored z momentum component.
  KFPARTICLE_GPU_HOST_DEVICE float& Pz() { return fParameters[5]; }

  /// @brief Exposes writable access to the stored energy.
  KFPARTICLE_GPU_HOST_DEVICE float& E() { return fParameters[6]; }

  /// @brief Exposes writable access to the stored KFParticle decay-length parameter S.
  KFPARTICLE_GPU_HOST_DEVICE float& S() { return fParameters[7]; }

  /// @brief Reads the stored x position.
  KFPARTICLE_GPU_HOST_DEVICE const float& X() const { return fParameters[0]; }

  /// @brief Reads the stored y position.
  KFPARTICLE_GPU_HOST_DEVICE const float& Y() const { return fParameters[1]; }

  /// @brief Reads the stored z position.
  KFPARTICLE_GPU_HOST_DEVICE const float& Z() const { return fParameters[2]; }

  /// @brief Reads the stored x momentum component.
  KFPARTICLE_GPU_HOST_DEVICE const float& Px() const { return fParameters[3]; }

  /// @brief Reads the stored y momentum component.
  KFPARTICLE_GPU_HOST_DEVICE const float& Py() const { return fParameters[4]; }

  /// @brief Reads the stored z momentum component.
  KFPARTICLE_GPU_HOST_DEVICE const float& Pz() const { return fParameters[5]; }

  /// @brief Reads the stored energy.
  KFPARTICLE_GPU_HOST_DEVICE const float& E() const { return fParameters[6]; }

  /// @brief Reads the stored KFParticle decay-length parameter S.
  KFPARTICLE_GPU_HOST_DEVICE const float& S() const { return fParameters[7]; }

  /// @brief Exposes writable access to the stored fit chi2.
  KFPARTICLE_GPU_HOST_DEVICE float& Chi2() { return fChi2; }

  /// @brief Reads the stored fit chi2.
  KFPARTICLE_GPU_HOST_DEVICE const float& Chi2() const { return fChi2; }

  /// @brief Exposes writable access to the stored accumulated transport parameter relative to the decay point.
  KFPARTICLE_GPU_HOST_DEVICE float& SFromDecay() { return fSFromDecay; }

  /// @brief Reads the stored accumulated transport parameter relative to the decay point.
  KFPARTICLE_GPU_HOST_DEVICE const float& SFromDecay() const { return fSFromDecay; }

  /// @brief Exposes writable access to the stored sum of daughter mass hypotheses.
  KFPARTICLE_GPU_HOST_DEVICE float& SumDaughterMass() { return fSumDaughterMass; }

  /// @brief Reads the stored sum of daughter mass hypotheses.
  KFPARTICLE_GPU_HOST_DEVICE const float& SumDaughterMass() const { return fSumDaughterMass; }

  /// @brief Exposes writable access to the stored mass hypothesis; a negative value denotes an unspecified hypothesis.
  KFPARTICLE_GPU_HOST_DEVICE float& MassHypo() { return fMassHypothesis; }

  /// @brief Reads the stored mass hypothesis; a negative value denotes an unspecified hypothesis.
  KFPARTICLE_GPU_HOST_DEVICE const float& MassHypo() const { return fMassHypothesis; }

  /// @brief Exposes writable access to the stored number of fit degrees of freedom.
  KFPARTICLE_GPU_HOST_DEVICE int& NDF() { return fNdf; }

  /// @brief Reads the stored number of fit degrees of freedom.
  KFPARTICLE_GPU_HOST_DEVICE const int& NDF() const { return fNdf; }

  /// @brief Exposes writable access to the stored electric charge.
  KFPARTICLE_GPU_HOST_DEVICE int& Q() { return fCharge; }

  /// @brief Reads the stored electric charge.
  KFPARTICLE_GPU_HOST_DEVICE const int& Q() const { return fCharge; }

  /// @brief Exposes writable access to the stored production-vertex state flag.
  KFPARTICLE_GPU_HOST_DEVICE int& AtProductionVertex() { return fAtProductionVertex; }

  /// @brief Reads the stored production-vertex state flag.
  KFPARTICLE_GPU_HOST_DEVICE const int& AtProductionVertex() const { return fAtProductionVertex; }

  /// @brief Exposes writable access to the stored construction-method selector.
  KFPARTICLE_GPU_HOST_DEVICE int& ConstructMethod() { return fConstructMethod; }

  /// @brief Reads the stored construction-method selector.
  KFPARTICLE_GPU_HOST_DEVICE const int& ConstructMethod() const { return fConstructMethod; }

 private:
  float fParameters[NumberOfParameters];  ///< Owned parameter array in the order documented for this state.
  float fCovariance[NumberOfCovarianceElements];  ///< Owned symmetric covariance in packed lower-triangular order.
  float fChi2;  ///< Stored fit chi2.
  float fSFromDecay;  ///< Accumulated transport parameter from the decay point.
  float fSumDaughterMass;  ///< Sum of daughter mass hypotheses.
  float fMassHypothesis;  ///< Assigned mass hypothesis; negative when unspecified.
  int fNdf;  ///< Stored particle degrees of freedom; -3 for the default construction seed.
  int fCharge;  ///< Stored electric charge.
  int fAtProductionVertex;  ///< Flag indicating the production-vertex representation.
  int fConstructMethod;  ///< Selector interpreted by construction routines.
};

/**
 * @brief Transported daughter measurement and correlation block used by a Kalman update.
 *
 * The arrays intentionally mirror the scalar KFParticle `GetMeasurement`
 * contract: transported daughter parameters, transported covariance, and the
 * 3x3 correlation block between the current particle and the daughter.
 * All arrays are owned inline. StoreState() updates parameters and covariance
 * only; the row-major correlation block must be maintained separately.
 */
class KFParticleGpuMeasurement
{
 public:
  /// @brief Zeroes parameters, covariance, and the correlation block.
  KFPARTICLE_GPU_HOST_DEVICE KFParticleGpuMeasurement() { Initialize(); }

  /// @brief Zeroes all measurement arrays, including the correlation block.
  KFPARTICLE_GPU_HOST_DEVICE void Initialize()
  {
    for (int i = 0; i < KFParticleGpuFitState::NumberOfParameters; ++i) {
      fParameters[i] = 0.f;
    }
    for (int i = 0; i < KFParticleGpuFitState::NumberOfCovarianceElements; ++i) {
      fCovariance[i] = 0.f;
    }
    for (int i = 0; i < 9; ++i) {
      fCorrelation[i] = 0.f;
    }
  }

  /**
   * @brief Copies fit parameters and covariance, leaving the existing correlation block unchanged.
   * @note Does not transport the state or copy its fit-quality scalars and flags.
   */
  KFPARTICLE_GPU_HOST_DEVICE void StoreState(const KFParticleGpuFitState& state)
  {
    for (int i = 0; i < KFParticleGpuFitState::NumberOfParameters; ++i) {
      fParameters[i] = state.Parameter(i);
    }
    for (int i = 0; i < KFParticleGpuFitState::NumberOfCovarianceElements; ++i) {
      fCovariance[i] = state.Covariance(i);
    }
  }

  /// @brief Exposes writable access to one parameter; requires 0 <= i < 8.
  KFPARTICLE_GPU_HOST_DEVICE float& Parameter(int i) { return fParameters[i]; }

  /// @brief Reads one parameter; requires 0 <= i < 8.
  KFPARTICLE_GPU_HOST_DEVICE const float& Parameter(int i) const { return fParameters[i]; }

  /// @brief Exposes writable access to one packed covariance entry; requires a valid flattened index.
  KFPARTICLE_GPU_HOST_DEVICE float& Covariance(int i) { return fCovariance[i]; }

  /// @brief Reads one packed covariance entry; requires a valid flattened index.
  KFPARTICLE_GPU_HOST_DEVICE const float& Covariance(int i) const { return fCovariance[i]; }

  /// @brief Exposes writable access to a symmetric covariance element; both indices must be valid parameter indices.
  KFPARTICLE_GPU_HOST_DEVICE float& Covariance(int i, int j)
  {
    return fCovariance[KFParticleGpuFitState::CovarianceIndex(i, j)];
  }

  /// @brief Reads a symmetric covariance element; both indices must be valid parameter indices.
  KFPARTICLE_GPU_HOST_DEVICE const float& Covariance(int i, int j) const
  {
    return fCovariance[KFParticleGpuFitState::CovarianceIndex(i, j)];
  }

  /// @brief Exposes writable access to a row-major correlation entry; requires 0 <= i, j < 3.
  KFPARTICLE_GPU_HOST_DEVICE float& Correlation(int i, int j) { return fCorrelation[i * 3 + j]; }

  /// @brief Reads a row-major correlation entry; requires 0 <= i, j < 3.
  KFPARTICLE_GPU_HOST_DEVICE const float& Correlation(int i, int j) const
  {
    return fCorrelation[i * 3 + j];
  }

 private:
  /// Owned daughter parameters in KFParticleGpuFitState order.
  float fParameters[KFParticleGpuFitState::NumberOfParameters];
  /// Owned daughter covariance in packed lower-triangular order.
  float fCovariance[KFParticleGpuFitState::NumberOfCovarianceElements];
  float fCorrelation[9];  ///< Owned row-major 3x3 particle/daughter correlation block.
};

#endif
