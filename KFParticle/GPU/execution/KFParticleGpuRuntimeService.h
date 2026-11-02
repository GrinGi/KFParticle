/* Copyright (C) 2026 GSI Helmholtzzentrum fuer Schwerionenforschung, Darmstadt
   SPDX-License-Identifier: GPL-3.0-only */

#ifndef KFPARTICLEGPURUNTIMESERVICE_H
#define KFPARTICLEGPURUNTIMESERVICE_H

#include <mutex>
#include <thread>

class KFParticleGpuSteering;
struct KFParticleGpuRuntimeSettings;

namespace kfparticle::gpu
{
  /**
   * Serializes access to the process-wide KFParticle GPU queue and buffers.
   *
   * Acquire attaches to application-initialized XPU, as used by CBMRoot.
   * Initialize additionally supports standalone startup and checks the submission
   * thread/device. Complete reconstruction transactions remain exclusive;
   * global XPU shutdown always belongs to the application.
   */
  class GpuRuntimeService {
   public:
    class Transaction {
     public:
      Transaction(Transaction&&) noexcept;
      Transaction& operator=(Transaction&&) noexcept;
      ~Transaction();

      Transaction(const Transaction&)            = delete;
      Transaction& operator=(const Transaction&) = delete;

      KFParticleGpuSteering& Steering();

     private:
      friend class GpuRuntimeService;
      explicit Transaction(std::unique_lock<std::mutex> lock);
      std::unique_lock<std::mutex> fRuntimeLock;
    };

    static GpuRuntimeService& Instance();

    /** Initialize or attach on the submission thread; an empty device adopts active XPU. */
    void Initialize(const KFParticleGpuRuntimeSettings& settings);

    Transaction Acquire();

    /// Initialize/validate and acquire exclusively under the same service lock.
    Transaction Acquire(const KFParticleGpuRuntimeSettings& settings);

    /// Releases the KFParticle queue and device state while the application's XPU
    /// runtime is still alive. Safe to call when no attachment was made.
    void Finalize();

    /// With requireOwnerThread=true, reject cleanup from another submission thread.
    void Finalize(bool requireOwnerThread);

   private:
    GpuRuntimeService();
    ~GpuRuntimeService();

    GpuRuntimeService(const GpuRuntimeService&)            = delete;
    GpuRuntimeService& operator=(const GpuRuntimeService&) = delete;

    void InitializeLocked(const KFParticleGpuRuntimeSettings& settings);

    std::mutex fRuntimeMutex;
    std::thread::id fOwnerThread;
  };
}  // namespace kfparticle::gpu

#endif  // KFPARTICLEGPURUNTIMESERVICE_H
