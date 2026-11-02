/**
 * @file
 * @brief Host-side allocation, validation, and compact-output planning measurements.
 */

#ifndef KFPARTICLEGPUHOSTTIMING_H
#define KFPARTICLEGPUHOSTTIMING_H

#include <cstdint>

/**
 * @brief Accumulates host-measured durations, allocation counts, and counting-readback bytes.
 *
 * A passive value record: it starts no timers and performs no GPU synchronization.
 * Durations are milliseconds measured by callers on the host, not device kernel
 * times; readback timing can include waiting for queued work. Subintervals overlap
 * broader pipeline/queue measurements and must not be added as independent stages.
 * Counters are ordinary, non-atomic values; concurrent updates require external coordination.
 */
struct KFParticleGpuHostTiming
{
  std::uint64_t initialAllocations = 0;      ///< Nonzero allocations made into previously empty buffer handles.
  std::uint64_t replacementAllocations = 0;  ///< Nonzero allocations replacing existing storage, not first allocations.
  double initialAllocationMilliseconds = 0.;      ///< Host time spent in initial buffer allocations.
  double replacementAllocationMilliseconds = 0.;  ///< Host time spent replacing existing buffer allocations.
  std::uint64_t countingReadbackBytes = 0;  ///< Status payload bytes read back to determine compact output sizes.
  double countingReadbackMilliseconds = 0.;  ///< Host elapsed time for counting-status readback, including waits.
  double denseOutputPlanningMilliseconds = 0.;  ///< Host interval for dense output sizing and associated preparation.
  double packingValidationMilliseconds = 0.;  ///< Host time checking packed-input contracts.
  double initialTransactionValidationMilliseconds = 0.;  ///< Host time validating the initial transaction layout.
  double compactOutputValidationMilliseconds = 0.;  ///< Host time validating the compact-output transaction layout.

  /**
   * @brief Adds all counts, byte totals, and durations from another measurement record.
   * @note Does not detect overlapping intervals, counter overflow, or incompatible measurement scopes.
   */
  void Add(const KFParticleGpuHostTiming& other)
  {
    initialAllocations += other.initialAllocations;
    replacementAllocations += other.replacementAllocations;
    initialAllocationMilliseconds += other.initialAllocationMilliseconds;
    replacementAllocationMilliseconds += other.replacementAllocationMilliseconds;
    countingReadbackBytes += other.countingReadbackBytes;
    countingReadbackMilliseconds += other.countingReadbackMilliseconds;
    denseOutputPlanningMilliseconds += other.denseOutputPlanningMilliseconds;
    packingValidationMilliseconds += other.packingValidationMilliseconds;
    initialTransactionValidationMilliseconds += other.initialTransactionValidationMilliseconds;
    compactOutputValidationMilliseconds += other.compactOutputValidationMilliseconds;
  }

  /**
   * @brief Returns a field-by-field difference from an earlier cumulative snapshot without changing this record.
   * @param before Earlier snapshot of the same counters, with no intervening reset.
   * @return Current totals minus the supplied snapshot.
   * @note No ordering checks or clamping are performed. Unsigned counters wrap on
   * underflow, and duration differences can be negative for incompatible snapshots.
   */
  KFParticleGpuHostTiming Since(const KFParticleGpuHostTiming& before) const
  {
    auto delta = *this;
    delta.initialAllocations -= before.initialAllocations;
    delta.replacementAllocations -= before.replacementAllocations;
    delta.initialAllocationMilliseconds -= before.initialAllocationMilliseconds;
    delta.replacementAllocationMilliseconds -= before.replacementAllocationMilliseconds;
    delta.countingReadbackBytes -= before.countingReadbackBytes;
    delta.countingReadbackMilliseconds -= before.countingReadbackMilliseconds;
    delta.denseOutputPlanningMilliseconds -= before.denseOutputPlanningMilliseconds;
    delta.packingValidationMilliseconds -= before.packingValidationMilliseconds;
    delta.initialTransactionValidationMilliseconds -= before.initialTransactionValidationMilliseconds;
    delta.compactOutputValidationMilliseconds -= before.compactOutputValidationMilliseconds;
    return delta;
  }
};

#endif
