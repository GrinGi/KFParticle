/* Copyright (C) 2026 GSI Helmholtzzentrum fuer Schwerionenforschung, Darmstadt
   SPDX-License-Identifier: GPL-3.0-only */

#ifndef KFPARTICLEGPUFINDER_H
#define KFPARTICLEGPUFINDER_H

#include "execution/KFParticleGpuExecutor.h"
#include "KFPTrackVector.h"
#include "KFVertex.h"

#include <optional>
#include <thread>

/**
 * @brief Synchronous event/batch reconstruction without a CBMRoot dependency.
 *
 * Borrows host input until a call returns and produces independently owned scalar
 * KFParticle output. Uses the existing packer, batch planner and GPU executor;
 * one event is a batch of one. ReconstructBatch never silently splits a batch or
 * substitutes the legacy CPU Finder. XPU's "cpu" device runs the GPU algorithm.
 *
 * All calls, including Initialize/ReleaseResources, belong to the constructing
 * host thread. Instances share one process-wide runtime, not independent devices.
 * Buffers remain cached when an instance is destroyed; ReleaseResources explicitly
 * releases shared KFParticle buffers/queue, never global XPU or returned results.
 * A later call may recreate these resources. Coordinate with other runtime users.
 *
 * The current host-input route requires a FIXTARGET/NonhomogeneousField build.
 * Physics uses the supported catalogue defaults; arbitrary CPU Finder cuts and
 * device-resident input are not exposed. Low-level core diagnostic environment
 * switches remain process-wide. Regular event/batch calls ignore benchmark and
 * host-result overrides; prepared integration may explicitly inherit them.
 *
 * @code
 * KFParticleGpuFinder::Options options;
 * options.device = "hip1";
 * KFParticleGpuFinder finder(options);
 * auto result = finder.Reconstruct({42, first, last, chiToPV, vertices});
 * finder.ReleaseResources();
 * @endcode
 */
class KFParticleGpuFinder {
 public:
  using Status = kfparticle::gpu::GpuExecutionStatus;

  /** @brief Immutable configuration, copied and validated at construction. */
  struct Options {
    std::string device; ///< Empty adopts XPU's active/default device; otherwise require this selector.
    bool initializeXpuIfNeeded = true; ///< False requires application-initialized XPU.
    bool verbose = false; ///< XPU verbosity, applied only when this interface initializes XPU.
    std::vector<int> motherPdgs; ///< Empty selects the complete supported catalogue.
    kfparticle::gpu::GpuBatchLimits limits = kfparticle::gpu::DefaultGpuBatchLimits(64u);
    unsigned int packingThreads = 1u; ///< Positive bound, not the number of installed CPUs.
    unsigned int materializationThreads = 1u; ///< Positive bound on event-local CPU workers.
    bool diagnostics = false; ///< Retain full candidate/selection snapshots in addition to particles.
    bool timings = false; ///< Return batch timings without printing a benchmark report.
  };

  /**
   * @brief Borrowed data for one event; sourceEventId retains all 64 bits.
   *
   * First/last states must pair unique nonnegative source track IDs and supply
   * charge, PDG, six parameters, covariance and ten field coefficients. chiToPV
   * has one entry per physical track. PVIndex is event-local, or negative when
   * unassigned. Vertices carry position, covariance and fit metadata. Events in
   * a batch are disjoint; equal track IDs in different events remain independent.
   */
  struct Event {
    std::uint64_t sourceEventId;
    const KFPTrackVector& first;
    const KFPTrackVector& last;
    const std::vector<float>& chiToPV;
    const std::vector<KFVertex>& vertices;
    float minSecondaryTrackChiToPV = -1.f; ///< Existing packer cut; negative keeps catalogue behavior.
  };

  /** @brief Optional, owned candidate snapshots; particles are stored only in Result. */
  struct Diagnostics {
    std::vector<kfparticle::gpu::GpuChannelSummary> channels;
    std::vector<kfparticle::gpu::GpuRawCandidate> candidates;
  };

  /**
   * @brief Millisecond batch totals, returned only on the first Result when enabled.
   * Wall phases include required waits; GPU pipeline includes transfers and kernels.
   * Parent phases overlap their children and must not be summed indiscriminately.
   * On a rejected batch, only total is meaningful.
   */
  struct Timing {
    double total = 0.;
    double initialization = 0.;
    double inputPreparation = 0.;
    double capacityPlanning = 0.;
    double queueAndCapacity = 0.;
    double packing = 0.;
    double hostToDevice = 0.;
    double construction = 0.;
    double graph = 0.;
    double selection = 0.;
    double deviceToHost = 0.;
    double pipeline = 0.;
    double extraction = 0.;
    double materialization = 0.;
  };

  /** @brief Owned event output, valid after later calls, input destruction or cleanup. */
  struct Result {
    std::uint64_t sourceEventId = 0u;
    Status status = Status::InputRejected;
    unsigned int tracks = 0u; ///< Physical tracks, not doubled first/last rows.
    unsigned int candidates = 0u;
    unsigned int selectedCandidates = 0u;
    unsigned int overflowFlags = 0u;
    std::vector<KFParticle> particles; ///< Scalar particles with event-local daughter links.
    std::optional<Diagnostics> diagnostics;
    std::optional<Timing> timing; ///< Batch total on the first event only; otherwise disengaged.
    std::string message;

    /// True for completed reconstruction and legitimate empty/no-pair input.
    bool Succeeded() const;
  };

  /**
   * @brief Explicit controls for integration with an already admitted batch.
   * Environment-based monitoring/reference controls and benchmark replay are
   * separate opt-ins, disabled by default. Snapshot-only diagnostics may skip
   * scalar materialization; compact publication must materialize the full event.
   */
  struct PreparedOptions {
    kfparticle::gpu::GpuOutputMode outputMode = kfparticle::gpu::GpuOutputMode::PublishableOutput;
    bool materializeOutput = true;
    bool inheritEnvironment = false;
    bool replay = false;
  };

  /// Construct with compact output, bounded host workers and no eager XPU initialization.
  KFParticleGpuFinder();

  /// Validate static options without GPU work; device compatibility is checked at initialization.
  explicit KFParticleGpuFinder(const Options& options);

  /// Destroy this handle only; shared runtime cleanup is explicit via ReleaseResources.
  ~KFParticleGpuFinder() = default;

  KFParticleGpuFinder(const KFParticleGpuFinder&) = delete;
  KFParticleGpuFinder& operator=(const KFParticleGpuFinder&) = delete;
  KFParticleGpuFinder(KFParticleGpuFinder&&) = delete;
  KFParticleGpuFinder& operator=(KFParticleGpuFinder&&) = delete;

  /// Return the immutable options actually used by this instance.
  const Options& GetOptions() const { return fOptions; }

  /// Optionally initialize eagerly; normal reconstruction initializes lazily when needed.
  void Initialize() const;

  /// Release shared KFParticle storage on its submission thread; leave global XPU alive.
  void ReleaseResources() const;

  /// Reconstruct through the same atomic batch-of-one path as ReconstructBatch.
  Result Reconstruct(const Event& event) const;

  /**
   * @brief Execute exactly one bounded batch, returning one result per input in order.
   * Empty input returns an empty vector. Structural or resource-admission failure
   * rejects the complete batch without GPU work. Execution failures are statuses;
   * no partial particle vector is published. Wrong-thread calls throw logic_error.
   */
  std::vector<Result> ReconstructBatch(const std::vector<Event>& events) const;

  /**
   * @brief Execute an admitted library batch without preparing or sorting its input again.
   *
   * The batch owns the channel request and reservations; Options::motherPdgs and
   * limits are not reapplied. Caller-owned source arrays must remain alive.
   * Returns the library's full result metadata directly, with no particle copies,
   * for integration reporting, diagnostics and event-atomic publication/fallback.
   * Runtime failures are statuses; malformed explicit run controls throw.
   */
  std::vector<kfparticle::gpu::GpuEventResult> ReconstructPreparedBatch(
    const kfparticle::gpu::GpuPreparedBatch& batch,
    const PreparedOptions& run) const;

  /**
   * @brief One-shot single-event integration with full metadata and ABI-only admission.
   * Retains the original individual-event route: collection resource ceilings
   * do not reject a lone event. Options::motherPdgs still applies. Replay must
   * be disabled. Use Reconstruct for the regular bounded standalone interface.
   */
  kfparticle::gpu::GpuEventResult ReconstructDetailedEvent(
    const Event& event, const PreparedOptions& run) const;

 private:
  /// Require the host submission thread that constructed this handle.
  void CheckThread() const;

  /// Derive runtime attachment settings from the immutable Finder options.
  KFParticleGpuRuntimeSettings RuntimeSettings() const;

  Options fOptions;
  std::thread::id fOwnerThread;
};

#endif
