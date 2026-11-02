/* Copyright (C) 2026 GSI Helmholtzzentrum fuer Schwerionenforschung, Darmstadt
   SPDX-License-Identifier: GPL-3.0-only */

#ifndef KFPARTICLEGPUMATERIALIZATIONSCHEDULE_H
#define KFPARTICLEGPUMATERIALIZATIONSCHEDULE_H

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace kfparticle::gpu
{
  enum class GpuMaterializationScheduleMode {
    WeightedContiguous,
    Static,
    Dynamic,
    Guided
  };

  inline const char* GpuMaterializationScheduleName(
    GpuMaterializationScheduleMode mode)
  {
    switch (mode) {
      case GpuMaterializationScheduleMode::WeightedContiguous:
        return "weighted-contiguous";
      case GpuMaterializationScheduleMode::Static: return "static";
      case GpuMaterializationScheduleMode::Dynamic: return "dynamic";
      case GpuMaterializationScheduleMode::Guided: return "guided";
    }
    throw std::invalid_argument("unknown KFParticle materialization schedule");
  }

  inline GpuMaterializationScheduleMode ParseGpuMaterializationSchedule(
    const std::string& value)
  {
    if (value.empty() || value == "guided") {
      return GpuMaterializationScheduleMode::Guided;
    }
    if (value == "weighted-contiguous") {
      return GpuMaterializationScheduleMode::WeightedContiguous;
    }
    if (value == "static") { return GpuMaterializationScheduleMode::Static; }
    if (value == "dynamic") { return GpuMaterializationScheduleMode::Dynamic; }
    throw std::invalid_argument(
      "KFPARTICLE_GPU_MATERIALIZATION_SCHEDULE must be "
      "weighted-contiguous, static, dynamic, or guided");
  }

  inline GpuMaterializationScheduleMode ResolveGpuMaterializationSchedule(
    const std::string& value, bool performanceMonitoring)
  {
    const auto mode = ParseGpuMaterializationSchedule(value);
    if (mode != GpuMaterializationScheduleMode::Guided
        && !performanceMonitoring) {
      throw std::invalid_argument(
        "non-default KFParticle materialization schedules are benchmark-only; "
        "set KFPARTICLE_GPU_PERFORMANCE_MONITORING=1");
    }
    return mode;
  }

  struct GpuMaterializationWorkerRange {
    std::size_t begin = 0u;
    std::size_t end = 0u;
    std::uint64_t estimatedWeight = 0u;
  };

  inline std::uint64_t GpuMaterializationSaturatingAdd(
    std::uint64_t left, std::uint64_t right)
  {
    const auto maximum = std::numeric_limits<std::uint64_t>::max();
    return right > maximum - left ? maximum : left + right;
  }

  inline std::uint64_t GpuMaterializationSaturatingMultiply(
    std::uint64_t value, std::uint64_t factor)
  {
    const auto maximum = std::numeric_limits<std::uint64_t>::max();
    return value != 0u && factor > maximum / value
             ? maximum : value * factor;
  }

  inline std::uint64_t GpuMaterializationEventWeight(
    unsigned int physicalTracks,
    unsigned int twoDaughterCandidates,
    unsigned int cascadeCandidates,
    unsigned int graphCandidates)
  {
    const std::uint64_t candidates = GpuMaterializationSaturatingAdd(
      twoDaughterCandidates,
      GpuMaterializationSaturatingAdd(cascadeCandidates, graphCandidates));
    const std::uint64_t trackWork = GpuMaterializationSaturatingMultiply(
      physicalTracks, 3u);
    const std::uint64_t candidateWork = GpuMaterializationSaturatingMultiply(
      candidates, 4u);
    return GpuMaterializationSaturatingAdd(
      16u, GpuMaterializationSaturatingAdd(trackWork, candidateWork));
  }

  /**
   * Partitions source-ordered events into non-empty contiguous worker ranges.
   *
   * The planner is deliberately deterministic and linear. It repeatedly
   * chooses the closest boundary to the average remaining work while leaving
   * at least one event for each remaining worker. A single heavy event remains
   * indivisible, but cannot cause overlap, omission, or arithmetic wraparound.
   */
  inline std::vector<GpuMaterializationWorkerRange>
  PlanGpuMaterializationWorkers(const std::vector<std::uint64_t>& eventWeights,
                                unsigned int requestedWorkers)
  {
    if (requestedWorkers == 0u) {
      throw std::invalid_argument(
        "KFParticle materialization schedule requires at least one worker");
    }
    if (eventWeights.empty()) { return {}; }

    const std::size_t workerCount = std::min<std::size_t>(
      requestedWorkers, eventWeights.size());
    std::uint64_t remainingWeight = 0u;
    for (std::uint64_t weight : eventWeights) {
      remainingWeight = GpuMaterializationSaturatingAdd(
        remainingWeight, std::max<std::uint64_t>(1u, weight));
    }

    std::vector<GpuMaterializationWorkerRange> ranges;
    ranges.reserve(workerCount);
    std::size_t begin = 0u;
    for (std::size_t worker = 0u; worker < workerCount; ++worker) {
      const std::size_t workersIncludingCurrent = workerCount - worker;
      const std::size_t workersAfterCurrent = workersIncludingCurrent - 1u;
      const std::size_t maximumEnd = eventWeights.size() - workersAfterCurrent;
      std::size_t end = begin;
      std::uint64_t rangeWeight = 0u;

      if (workersAfterCurrent == 0u) {
        end = eventWeights.size();
        for (std::size_t event = begin; event < end; ++event) {
          rangeWeight = GpuMaterializationSaturatingAdd(
            rangeWeight, std::max<std::uint64_t>(1u, eventWeights[event]));
        }
      }
      else {
        const std::uint64_t target =
          remainingWeight / workersIncludingCurrent
          + (remainingWeight % workersIncludingCurrent != 0u ? 1u : 0u);
        while (end < maximumEnd) {
          const std::uint64_t nextWeight =
            std::max<std::uint64_t>(1u, eventWeights[end]);
          const std::uint64_t candidateWeight =
            GpuMaterializationSaturatingAdd(rangeWeight, nextWeight);
          if (end != begin) {
            const std::uint64_t currentDistance = rangeWeight < target
                                                    ? target - rangeWeight
                                                    : rangeWeight - target;
            const std::uint64_t candidateDistance = candidateWeight < target
                                                      ? target - candidateWeight
                                                      : candidateWeight - target;
            if (rangeWeight >= target
                || candidateDistance >= currentDistance) {
              break;
            }
          }
          rangeWeight = candidateWeight;
          ++end;
        }
        if (end == begin) {
          rangeWeight = std::max<std::uint64_t>(1u, eventWeights[end]);
          ++end;
        }
      }

      ranges.push_back({begin, end, rangeWeight});
      remainingWeight = remainingWeight > rangeWeight
                          ? remainingWeight - rangeWeight : 0u;
      begin = end;
    }
    return ranges;
  }
}  // namespace kfparticle::gpu

#endif  // KFPARTICLEGPUMATERIALIZATIONSCHEDULE_H
