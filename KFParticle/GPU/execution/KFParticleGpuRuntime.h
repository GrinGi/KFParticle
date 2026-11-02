/*
 * This file is part of KFParticle package
 * Copyright (C) 2007-2026
 *
 * KFParticle is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 */

#ifndef KFPARTICLEGPURUNTIME_H
#define KFPARTICLEGPURUNTIME_H

#include <memory>
#include <string>

class KFParticleGpuSteering;

#ifdef KFPARTICLE_USE_XPU
namespace xpu
{
  class queue;
}
#endif

/**
 * @brief Options used when KFParticle must initialize the process-wide XPU runtime.
 *
 * Device, profiling, and verbosity are forwarded only when XPU's device list is empty.
 * An already initialized XPU environment is reused without reconfiguration. In builds
 * without KFPARTICLE_USE_XPU, all settings are ignored and only host steering is created.
 */
struct KFParticleGpuRuntimeSettings
{
  std::string device = "cpu"; ///< Device selector passed to xpu::initialize(), for example "cpu" or "hip1".
  bool profile = false; ///< Enable XPU profiling if this runtime initializes XPU.
  bool verbose = false; ///< Enable XPU verbose output if this runtime initializes XPU.
  bool initializeXpuIfNeeded = true; ///< Allow XPU initialization; otherwise require application initialization.
};

/**
 * @brief Process-wide owner of KFParticle steering and, in XPU builds, its execution queue.
 *
 * Instance() provides a single noncopyable object. Initialize() creates its resources;
 * Finalize() releases steering-owned buffers before the queue. The hidden Impl keeps
 * XPU types and lifecycle bookkeeping out of this public interface.
 *
 * The application may initialize XPU first. This class can initialize it if permitted,
 * but never shuts down the global XPU runtime. Repeated initialization while ready is
 * a no-op, not a device switch. A completed Finalize() allows later reinitialization.
 *
 * Only Initialize()/Finalize() are protected by an internal mutex. Status queries,
 * returned references, and work submitted through steering/queue are not protected by
 * that mutex. Callers must coordinate resource use with lifecycle changes and keep XPU
 * usable until KFParticle cleanup is complete.
 */
class KFParticleGpuRuntime
{
 public:
  /**
   * @brief Return the process-wide runtime, constructing it on first use without initializing resources.
   */
  static KFParticleGpuRuntime& Instance();

  /**
   * @brief Initialize using default settings, including the "cpu" selector if XPU must be started.
   */
  void Initialize();

  /**
   * @brief Initialize steering and an XPU queue, reusing an existing global XPU environment.
   *
   * Serializes lifecycle changes. If already initialized, returns without applying settings.
   * In XPU builds, preloads the KFParticle device image and constructs a default XPU queue;
   * settings.device is forwarded to XPU initialization, not to an explicit queue constructor.
   * @throws std::logic_error If XPU appears uninitialized and initializing it is forbidden.
   * @note Backend/allocation exceptions propagate. Failure is not rolled back: XPU may remain
   * initialized and some owned resources may remain allocated even though IsInitialized() is false.
   */
  void Initialize(const KFParticleGpuRuntimeSettings& settings);

  /**
   * @brief Finalize steering, release it and the queue, then reset lifecycle flags.
   *
   * Steering waits for its XPU queue and clears published device pointers before releasing buffers.
   * Does not shut down global XPU. Returns immediately if initialization never completed,
   * so it is not a cleanup path for partially failed Initialize(). Backend exceptions propagate.
   * @pre No caller is concurrently using borrowed steering/queue references.
   */
  void Finalize();

  /**
   * @brief Report whether KFParticle initialization completed; this is an unlocked status read.
   */
  bool IsInitialized() const;

  /**
   * @brief Report whether this object initialized XPU since its last completed Finalize().
   * Not a global XPU status or ownership flag. May be true after a later initialization step
   * fails, or false when the application initialized XPU. This is an unlocked status read.
   */
  bool InitializedXpu() const;

  /**
   * @brief Borrow the owned steering object; the reference must not outlive its runtime resources.
   * @throws std::logic_error If KFParticle initialization has not completed.
   * No lock is held during access or subsequent steering operations.
   */
  KFParticleGpuSteering& GetSteering();

  /**
   * @brief Borrow read-only steering access, with the same lifetime and synchronization requirements.
   * @throws std::logic_error If KFParticle initialization has not completed.
   */
  const KFParticleGpuSteering& GetSteering() const;

#ifdef KFPARTICLE_USE_XPU
  /**
   * @brief Borrow the runtime-owned XPU queue shared with steering and its buffer manager.
   * @throws std::logic_error If KFParticle initialization has not completed.
   * The caller must coordinate submissions and must not retain the reference past Finalize().
   */
  xpu::queue& GetQueue();

  /**
   * @brief Borrow read-only access to the same queue, without locking or extending its lifetime.
   * @throws std::logic_error If KFParticle initialization has not completed.
   */
  const xpu::queue& GetQueue() const;
#endif

 private:
  /// @brief Allocate empty private state; Instance() is the only public construction path.
  KFParticleGpuRuntime();

  /// @brief Call Finalize() at singleton destruction; orderly cleanup should occur before XPU teardown.
  ~KFParticleGpuRuntime();

  /// @brief Intentionally unavailable: the process-wide execution resources cannot be copied.
  KFParticleGpuRuntime(const KFParticleGpuRuntime&);

  /// @brief Intentionally unavailable: assigning runtime ownership is not supported.
  KFParticleGpuRuntime& operator=(const KFParticleGpuRuntime&);

  /// @brief Private resource and synchronization state, defined in the implementation file.
  struct Impl;
  std::unique_ptr<Impl> fImpl; ///< Sole owner of the hidden state and its execution resources.
};

#endif
