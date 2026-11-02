/*
 * This file is part of KFParticle package
 * Copyright (C) 2007-2026
 *
 * KFParticle is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 */

#ifndef KFPARTICLEGPUSOAVIEW_H
#define KFPARTICLEGPUSOAVIEW_H

/**
 * @file
 * @brief Non-owning structure-of-arrays views and single-particle load/store helpers.
 *
 * Each numerical component occupies a contiguous row of particle values. Access uses
 * component * stride + particle, where stride is measured in elements, not bytes.
 * Views neither allocate nor free storage; copying a view copies only pointers and layout.
 * Load/store helpers copy one state within the current execution context, not between CPU
 * and GPU memory. Storage lifetime, accessibility, bounds, and synchronization belong to the caller.
 */

#include "data/KFParticleGpuFitState.h"

/**
 * @brief Component-major view of six track parameters and 21 packed covariance components.
 * @tparam Value float for writable data or const float for read-only data.
 *
 * Parameters follow (x,y,z,px,py,pz); covariance components follow the packed lower
 * triangle of the symmetric 6x6 matrix. Keeping stride independent of logical size
 * allows buffers to retain capacity between events. Normally size <= stride.
 * No accessor checks pointers, indices, allocation size, or index-arithmetic overflow.
 * A const view object still permits writes when Value is nonconst.
 */
template<typename Value>
class KFParticleGpuTrackSoAViewBase
{
 public:
  /// @brief Default constructor; plain default-initialization leaves members unset. Bind storage before use.
  KFPARTICLE_GPU_HOST_DEVICE KFParticleGpuTrackSoAViewBase() = default;

  /**
   * @brief Bind external parameter/covariance arrays with a shared logical size and row stride.
   * Storage must cover every accessed component row and particle slot; no validation is performed.
   */
  KFPARTICLE_GPU_HOST_DEVICE KFParticleGpuTrackSoAViewBase(Value* parameters,
                                                        Value* covariances,
                                                        unsigned int size,
                                                        unsigned int stride)
    : fParameters(parameters), fCovariances(covariances), fSize(size), fStride(stride)
  {
  }

  /// @brief Access one parameter row (component 0..5) at a caller-validated particle slot.
  KFPARTICLE_GPU_HOST_DEVICE Value& Parameter(unsigned int component, unsigned int particle) const
  {
    return fParameters[component * fStride + particle];
  }

  /// @brief Access one packed covariance row (component 0..20), not a matrix row/column pair.
  KFPARTICLE_GPU_HOST_DEVICE Value& Covariance(unsigned int component, unsigned int particle) const
  {
    return fCovariances[component * fStride + particle];
  }

  /// @brief Borrow the base pointer of the parameter-component array.
  KFPARTICLE_GPU_HOST_DEVICE Value* ParametersData() const { return fParameters; }

  /// @brief Borrow the base pointer of the packed-covariance-component array.
  KFPARTICLE_GPU_HOST_DEVICE Value* CovariancesData() const { return fCovariances; }

  /// @brief Return the declared logical track count; accessors do not enforce it.
  KFPARTICLE_GPU_HOST_DEVICE unsigned int Size() const { return fSize; }

  /// @brief Return the element spacing between component rows, independent of logical size.
  KFPARTICLE_GPU_HOST_DEVICE unsigned int Stride() const { return fStride; }

 private:
  Value* fParameters; ///< Borrowed six-row parameter storage.
  Value* fCovariances; ///< Borrowed 21-row packed covariance storage.
  unsigned int fSize; ///< Logical number of tracks.
  unsigned int fStride; ///< Component-row spacing in elements.
};

/// Writable view of externally owned input-track numerical data.
typedef KFParticleGpuTrackSoAViewBase<float> KFParticleGpuTrackSoAView;

/// Read-only view of externally owned input-track numerical data.
typedef KFParticleGpuTrackSoAViewBase<const float> KFParticleGpuConstTrackSoAView;

/// @brief Create a read-only alias with the same pointers, size, and stride; no numerical data is copied.
KFPARTICLE_GPU_HOST_DEVICE inline KFParticleGpuConstTrackSoAView
MakeConstView(const KFParticleGpuTrackSoAView& view)
{
  return KFParticleGpuConstTrackSoAView(
    view.ParametersData(), view.CovariancesData(), view.Size(), view.Stride());
}

/**
 * @brief Component indices for fit metadata stored separately from parameters and covariance.
 * Floating-point and integer indices address different arrays, each using the view's common stride.
 */
struct KFParticleGpuFitSoALayout
{
  /// @brief Row indices in the floating-point fit-metadata array.
  enum FloatComponent
  {
    Chi2 = 0, ///< Stored fit chi-square.
    SFromDecay, ///< Accumulated transport parameter from the decay point, not a geometric length.
    SumDaughterMass, ///< Stored sum of daughter mass hypotheses.
    MassHypothesis, ///< Assigned mass hypothesis; negative when unspecified.
    NumberOfFloatComponents ///< Row count, not a valid component index.
  };

  /// @brief Row indices in the integer fit-metadata array.
  enum IntegerComponent
  {
    Ndf = 0, ///< Fit degrees of freedom.
    Charge, ///< Electric charge.
    AtProductionVertex, ///< Stored production-vertex representation flag.
    ConstructMethod, ///< Construction-method selector.
    NumberOfIntegerComponents ///< Row count, not a valid component index.
  };
};

/**
 * @brief Component-major view of a full KFParticle fit and its floating-point/integer metadata.
 * @tparam FloatValue float or const float, controlling numerical-array mutability.
 * @tparam IntegerValue int or const int, controlling integer-metadata mutability.
 *
 * Parameters are (x,y,z,px,py,pz,E,S); covariance has 36 packed lower-triangle entries
 * of the symmetric 8x8 matrix. Separate metadata arrays use KFParticleGpuFitSoALayout.
 * All four arrays share one element stride, letting kernels load only the needed categories.
 * Normally size <= stride; pointers, component/particle bounds, and address arithmetic are unchecked.
 * Constness of the view object does not change the mutability of its pointed-to arrays.
 */
template<typename FloatValue, typename IntegerValue>
class KFParticleGpuFitSoAViewBase
{
 public:
  /// @brief Default constructor; plain default-initialization leaves members unset. Bind storage before use.
  KFPARTICLE_GPU_HOST_DEVICE KFParticleGpuFitSoAViewBase() = default;

  /**
   * @brief Bind four externally owned component arrays with a shared logical size and element stride.
   * Does not allocate, initialize array elements, or check the supplied layout.
   */
  KFPARTICLE_GPU_HOST_DEVICE KFParticleGpuFitSoAViewBase(FloatValue* parameters,
                                                      FloatValue* covariances,
                                                      FloatValue* fitScalars,
                                                      IntegerValue* fitIntegers,
                                                      unsigned int size,
                                                      unsigned int stride)
    : fParameters(parameters)
    , fCovariances(covariances)
    , fFitScalars(fitScalars)
    , fFitIntegers(fitIntegers)
    , fSize(size)
    , fStride(stride)
  {
  }

  /// @brief Access one of eight parameter rows at a caller-validated particle slot.
  KFPARTICLE_GPU_HOST_DEVICE FloatValue& Parameter(unsigned int component, unsigned int particle) const
  {
    return fParameters[component * fStride + particle];
  }

  /// @brief Access one of 36 packed covariance rows; component is a packed index, not a matrix coordinate.
  KFPARTICLE_GPU_HOST_DEVICE FloatValue& Covariance(unsigned int component, unsigned int particle) const
  {
    return fCovariances[component * fStride + particle];
  }

  /// @brief Access a floating-point metadata row indexed by KFParticleGpuFitSoALayout::FloatComponent.
  KFPARTICLE_GPU_HOST_DEVICE FloatValue& FitScalar(unsigned int component, unsigned int particle) const
  {
    return fFitScalars[component * fStride + particle];
  }

  /// @brief Access an integer metadata row indexed by KFParticleGpuFitSoALayout::IntegerComponent.
  KFPARTICLE_GPU_HOST_DEVICE IntegerValue& FitInteger(unsigned int component, unsigned int particle) const
  {
    return fFitIntegers[component * fStride + particle];
  }

  /// @brief Borrow the base pointer of the eight-row parameter array.
  KFPARTICLE_GPU_HOST_DEVICE FloatValue* ParametersData() const { return fParameters; }

  /// @brief Borrow the base pointer of the 36-row packed covariance array.
  KFPARTICLE_GPU_HOST_DEVICE FloatValue* CovariancesData() const { return fCovariances; }

  /// @brief Borrow the base pointer of the floating-point fit-metadata array.
  KFPARTICLE_GPU_HOST_DEVICE FloatValue* FitScalarsData() const { return fFitScalars; }

  /// @brief Borrow the base pointer of the integer fit-metadata array.
  KFPARTICLE_GPU_HOST_DEVICE IntegerValue* FitIntegersData() const { return fFitIntegers; }

  /// @brief Return the declared logical particle count; accessors do not enforce it.
  KFPARTICLE_GPU_HOST_DEVICE unsigned int Size() const { return fSize; }

  /// @brief Return the spacing in elements between adjacent component rows in every array.
  KFPARTICLE_GPU_HOST_DEVICE unsigned int Stride() const { return fStride; }

 private:
  FloatValue* fParameters; ///< Borrowed eight-row fit-parameter storage.
  FloatValue* fCovariances; ///< Borrowed 36-row packed covariance storage.
  FloatValue* fFitScalars; ///< Borrowed floating-point fit-metadata rows.
  IntegerValue* fFitIntegers; ///< Borrowed integer fit-metadata rows.
  unsigned int fSize; ///< Logical number of fitted particles.
  unsigned int fStride; ///< Common component-row spacing in elements, not bytes.
};

/// Writable view of externally owned full-fit arrays.
typedef KFParticleGpuFitSoAViewBase<float, int> KFParticleGpuFitSoAView;

/// Read-only view of externally owned full-fit arrays.
typedef KFParticleGpuFitSoAViewBase<const float, const int> KFParticleGpuConstFitSoAView;

/// @brief Create a read-only alias of all four fit arrays without copying their contents.
KFPARTICLE_GPU_HOST_DEVICE inline KFParticleGpuConstFitSoAView
MakeConstView(const KFParticleGpuFitSoAView& view)
{
  return KFParticleGpuConstFitSoAView(view.ParametersData(),
                                    view.CovariancesData(),
                                    view.FitScalarsData(),
                                    view.FitIntegersData(),
                                    view.Size(),
                                    view.Stride());
}

/**
 * @brief Gather one track's six parameters and 21 covariance entries into a scalar state.
 * @pre The source particle slot and all component rows are readable; destination is writable.
 * Copies values as stored, without validation, fitting, or any CPU/GPU transfer operation.
 */
KFPARTICLE_GPU_HOST_DEVICE inline void LoadTrackState(const KFParticleGpuConstTrackSoAView& source,
                                                    unsigned int particle,
                                                    KFParticleGpuTrackState& destination)
{
  for (unsigned int component = 0; component < KFParticleGpuTrackState::NumberOfParameters; ++component) {
    destination.Parameter(component) = source.Parameter(component, particle);
  }
  for (unsigned int component = 0;
       component < KFParticleGpuTrackState::NumberOfCovarianceElements;
       ++component) {
    destination.Covariance(component) = source.Covariance(component, particle);
  }
}

/**
 * @brief Gather a track from a writable view through a read-only alias; source storage is unchanged.
 * Has the same unchecked storage and index requirements as the const-view overload.
 */
KFPARTICLE_GPU_HOST_DEVICE inline void LoadTrackState(const KFParticleGpuTrackSoAView& source,
                                                    unsigned int particle,
                                                    KFParticleGpuTrackState& destination)
{
  LoadTrackState(MakeConstView(source), particle, destination);
}

/**
 * @brief Scatter a scalar track's six parameters and 21 covariance entries into one SoA slot.
 * @pre The destination slot is writable in every component row and source is readable.
 * Does not update logical size, resize storage, validate values, or synchronize other writers.
 */
KFPARTICLE_GPU_HOST_DEVICE inline void StoreTrackState(const KFParticleGpuTrackState& source,
                                                     KFParticleGpuTrackSoAView& destination,
                                                     unsigned int particle)
{
  for (unsigned int component = 0; component < KFParticleGpuTrackState::NumberOfParameters; ++component) {
    destination.Parameter(component, particle) = source.Parameter(component);
  }
  for (unsigned int component = 0;
       component < KFParticleGpuTrackState::NumberOfCovarianceElements;
       ++component) {
    destination.Covariance(component, particle) = source.Covariance(component);
  }
}

/**
 * @brief Gather eight parameters, 36 covariance entries, and all eight metadata values into a scalar fit.
 * @pre All four source arrays contain the readable particle slot; destination is writable.
 * Overwrites the complete numerical fit state without validation or recalculation.
 * Candidate identity and daughter ancestry are separate data and are not copied here.
 */
KFPARTICLE_GPU_HOST_DEVICE inline void LoadFitState(const KFParticleGpuConstFitSoAView& source,
                                                  unsigned int particle,
                                                  KFParticleGpuFitState& destination)
{
  for (unsigned int component = 0; component < KFParticleGpuFitState::NumberOfParameters; ++component) {
    destination.Parameter(component) = source.Parameter(component, particle);
  }
  for (unsigned int component = 0;
       component < KFParticleGpuFitState::NumberOfCovarianceElements;
       ++component) {
    destination.Covariance(component) = source.Covariance(component, particle);
  }

  destination.Chi2() =
    source.FitScalar(KFParticleGpuFitSoALayout::Chi2, particle);
  destination.SFromDecay() =
    source.FitScalar(KFParticleGpuFitSoALayout::SFromDecay, particle);
  destination.SumDaughterMass() =
    source.FitScalar(KFParticleGpuFitSoALayout::SumDaughterMass, particle);
  destination.MassHypo() =
    source.FitScalar(KFParticleGpuFitSoALayout::MassHypothesis, particle);

  destination.NDF() =
    source.FitInteger(KFParticleGpuFitSoALayout::Ndf, particle);
  destination.Q() =
    source.FitInteger(KFParticleGpuFitSoALayout::Charge, particle);
  destination.AtProductionVertex() =
    source.FitInteger(KFParticleGpuFitSoALayout::AtProductionVertex, particle);
  destination.ConstructMethod() =
    source.FitInteger(KFParticleGpuFitSoALayout::ConstructMethod, particle);
}

/**
 * @brief Gather a full fit from a writable view through a read-only alias, leaving source arrays unchanged.
 * Has the same unchecked storage and index requirements as the const-view overload.
 */
KFPARTICLE_GPU_HOST_DEVICE inline void LoadFitState(const KFParticleGpuFitSoAView& source,
                                                  unsigned int particle,
                                                  KFParticleGpuFitState& destination)
{
  LoadFitState(MakeConstView(source), particle, destination);
}

/**
 * @brief Scatter the complete numerical fit and metadata into one slot of the four SoA arrays.
 * @pre The source is readable and the destination slot is writable in every component row.
 * The const view reference fixes the view, not its pointed-to values. Does not update size,
 * copy candidate identity/ancestry, validate the fit, or synchronize concurrent accesses.
 */
KFPARTICLE_GPU_HOST_DEVICE inline void StoreFitState(const KFParticleGpuFitState& source,
                                                   const KFParticleGpuFitSoAView& destination,
                                                   unsigned int particle)
{
  for (unsigned int component = 0; component < KFParticleGpuFitState::NumberOfParameters; ++component) {
    destination.Parameter(component, particle) = source.Parameter(component);
  }
  for (unsigned int component = 0;
       component < KFParticleGpuFitState::NumberOfCovarianceElements;
       ++component) {
    destination.Covariance(component, particle) = source.Covariance(component);
  }

  destination.FitScalar(KFParticleGpuFitSoALayout::Chi2, particle) = source.Chi2();
  destination.FitScalar(KFParticleGpuFitSoALayout::SFromDecay, particle) = source.SFromDecay();
  destination.FitScalar(KFParticleGpuFitSoALayout::SumDaughterMass, particle) = source.SumDaughterMass();
  destination.FitScalar(KFParticleGpuFitSoALayout::MassHypothesis, particle) = source.MassHypo();

  destination.FitInteger(KFParticleGpuFitSoALayout::Ndf, particle) = source.NDF();
  destination.FitInteger(KFParticleGpuFitSoALayout::Charge, particle) = source.Q();
  destination.FitInteger(KFParticleGpuFitSoALayout::AtProductionVertex, particle) =
    source.AtProductionVertex();
  destination.FitInteger(KFParticleGpuFitSoALayout::ConstructMethod, particle) =
    source.ConstructMethod();
}

#endif
