/**
 * @file
 * @brief Optional host-side XPU timer scopes for ordered steps and transaction profiling.
 */

#ifndef KFPARTICLEGPUSTEPTRACE_H
#define KFPARTICLEGPUSTEPTRACE_H

#include <xpu/host.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

/**
 * @brief Records successive named intervals inside one XPU timer scope.
 *
 * Enabled only by KFPARTICLE_GPU_CONSTRUCTION_TIMERS=1 at construction. Next()
 * closes the previous interval and opens the next; nested scopes appear as children.
 * Finish() closes the scope and optionally prints its timing tree to stderr.
 * The destructor also calls Finish(), suppressing exceptions during cleanup.
 *
 * Wall times include child scopes; self time subtracts their wall times, not GPU
 * durations. Device details require XPU_PROFILE=1 before XPU initialization.
 * HIP profiling synchronizes timed operations and changes execution performance:
 * these diagnostic measurements are not ordinary asynchronous run timings.
 *
 * @note Scopes must be closed in stack order, including externally created XPU
 * timers. This class provides no synchronization for concurrent timer-stack access.
 */
class KFParticleGpuStepTrace {
 public:
  /** @brief Tests whether the construction-timer environment switch is exactly "1". */
  static bool Enabled() {
    const char* value = std::getenv("KFPARTICLE_GPU_CONSTRUCTION_TIMERS");
    return value && std::strcmp(value, "1") == 0;
  }

  /**
   * @brief Opens the named parent timer if tracing is enabled.
   * @param name Non-null timer label identifying the measured scope.
   * @param report Whether Finish() should print the tree, rather than only close it.
   */
  explicit KFParticleGpuStepTrace(const char* name, bool report = false)
    : fActive(Enabled()), fReport(report) {
    if (fActive) { xpu::push_timer(name); }
  }
  /** @brief Finishes an active scope without propagating cleanup exceptions. */
  ~KFParticleGpuStepTrace() {
    try { Finish(); } catch (...) {}
  }
  /** @brief Prevents copying ownership of an open timer scope. */
  KFParticleGpuStepTrace(const KFParticleGpuStepTrace&) = delete;

  /** @brief Prevents assigning ownership of an open timer scope. */
  KFParticleGpuStepTrace& operator=(const KFParticleGpuStepTrace&) = delete;

  /**
   * @brief Ends the current interval, if any, and starts a named sibling interval.
   * @param name Non-null label for the next interval.
   * @note Does nothing when disabled or finished. Nested timers must already be closed.
   */
  void Next(const char* name) {
    if (!fActive) { return; }
    CloseInterval();
    xpu::push_timer(name);
    fInterval = true;
  }

  /**
   * @brief Closes the interval and parent timer, optionally printing the timing tree.
   *
   * Output uses milliseconds and includes inclusive wall/device times, derived self
   * wall time, and per-kernel details at leaf scopes. After successful timer closure,
   * subsequent calls do nothing. Unlike the destructor, this call propagates errors.
   */
  void Finish() {
    if (!fActive) { return; }
    CloseInterval();
    const auto timing = xpu::pop_timer();
    fActive = false;
    if (!fReport) { return; }
    const auto name = std::string(timing.name());
    std::fprintf(stderr,
      "KFP_STEP_TRACE BEGIN contract=1 section=%s details=%u intrusive=1 units=ms parent_times_include_children=1\n",
      name.c_str(), timing.has_details() ? 1u : 0u);
    unsigned int sequence = 0u;
    Print(timing, 0u, sequence);
    std::fprintf(stderr, "KFP_STEP_TRACE END section=%s rows=%u\n", name.c_str(), sequence);
    std::fflush(stderr);
  }

 private:
  /** @brief Pops the current interval timer, leaving its parent scope open. */
  void CloseInterval() {
    if (fInterval) {
      xpu::pop_timer();
      fInterval = false;
    }
  }

  /**
   * @brief Prints a timing node followed by its children in recorded order.
   * @param timing Completed XPU timing tree or subtree.
   * @param depth Indentation level, with zero denoting the root.
   * @param[in,out] sequence Shared row counter, incremented for each timing node.
   * @note Self wall time is max(0, wall minus direct-child wall times), not CPU-only
   * execution time. Parent totals include children and must not be summed with them.
   */
  static void Print(const xpu::timings& timing, unsigned int depth, unsigned int& sequence) {
    const auto children = timing.children();
    double childWall = 0.;
    for (const auto& child : children) { childWall += child.wall(); }
    const auto name = std::string(timing.name());
    const unsigned int row = ++sequence;
    std::fprintf(stderr,
      "KFP_STEP %03u depth=%u wall_ms=%.6f self_ms=%.6f kernel_ms=%.6f h2d_ms=%.6f d2h_ms=%.6f memset_ms=%.6f | %*s%s\n",
      row, depth, timing.wall(), std::max(0., timing.wall() - childWall),
      timing.kernel_time(), timing.copy(xpu::h2d), timing.copy(xpu::d2h), timing.memset(),
      static_cast<int>(depth * 2u), "", name.c_str());
    if (children.empty()) {
      for (const auto& kernel : timing.kernels()) {
        const auto kernelName = std::string(kernel.name());
        std::fprintf(stderr,
          "KFP_STEP_KERNEL row=%u calls=%zu device_ms=%.6f name=%s\n",
          row, kernel.times().size(), kernel.total(), kernelName.c_str());
      }
    }
    for (const auto& child : children) { Print(child, depth + 1u, sequence); }
  }

  bool fActive = false;    ///< Parent timer is open and has not been finished.
  bool fReport = false;    ///< Print the tree when the parent timer is closed.
  bool fInterval = false;  ///< A Next() interval is open inside the parent timer.
};

/**
 * @brief Collects aggregate device timings inside one optional transaction timer.
 *
 * KFPARTICLE_GPU_DEVICE_PROFILE=1 opens the "KFParticle transaction" scope.
 * Explicit Finish() prints transfer, kernel, and memset totals plus per-kernel
 * call counts to stderr. Destruction without Finish() closes the timer silently.
 *
 * Device details additionally require XPU_PROFILE=1 before XPU initialization;
 * this class does not enable profiling itself. The current HIP profiler synchronizes
 * timed operations, so this mode is diagnostic only, not a production speed benchmark.
 * @note All nested timers must be closed before this scope. Timer-stack access
 * requires external serialization, as for KFParticleGpuStepTrace.
 */
class KFParticleGpuProfileScope {
 public:
  /** @brief Opens the transaction timer only when the profile switch is exactly "1". */
  KFParticleGpuProfileScope() {
    const char* value = std::getenv("KFPARTICLE_GPU_DEVICE_PROFILE");
    fActive = value && std::string(value) == "1";
    if (fActive) { xpu::push_timer("KFParticle transaction"); }
  }
  /** @brief Closes an unfinished timer without reporting or propagating exceptions. */
  ~KFParticleGpuProfileScope() {
    if (fActive) {
      try { xpu::pop_timer(); } catch (...) {}
    }
  }
  /** @brief Prevents copying ownership of the transaction timer. */
  KFParticleGpuProfileScope(const KFParticleGpuProfileScope&) = delete;

  /** @brief Prevents assigning ownership of the transaction timer. */
  KFParticleGpuProfileScope& operator=(const KFParticleGpuProfileScope&) = delete;

  /**
   * @brief Closes the transaction timer and prints inclusive device totals in milliseconds.
   *
   * Does nothing when disabled or already finished. Marks the scope inactive before
   * popping the timer; a pop failure propagates and is not retried by the destructor.
   * The details flag indicates whether XPU collected profiling details.
   */
  void Finish() {
    if (!fActive) { return; }
    fActive = false;
    const auto t = xpu::pop_timer();
    std::fprintf(stderr,
      "KFP_DEVICE_PROFILE details=%u intrusive=1 backend=xpu synchronous_operations=1 kernel_ms=%.6f h2d_ms=%.6f d2h_ms=%.6f memset_ms=%.6f\n",
      t.has_details() ? 1u : 0u, t.kernel_time(), t.copy(xpu::h2d),
      t.copy(xpu::d2h), t.memset());
    for (const auto& kernel : t.kernels()) {
      const auto name = std::string(kernel.name());
      std::fprintf(stderr, "KFP_DEVICE_KERNEL name=%s calls=%zu device_ms=%.6f\n",
        name.c_str(), kernel.times().size(), kernel.total());
    }
    std::fflush(stderr);
  }

 private:
  bool fActive = false;  ///< Timer was opened and has not been handed to Finish().
};

#endif
