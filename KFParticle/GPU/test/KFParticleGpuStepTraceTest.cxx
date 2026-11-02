#include "diagnostics/KFParticleGpuStepTrace.h"
#include <iostream>
#include <stdexcept>

namespace {
void Require(bool condition, const char* message) {
  if (!condition) { throw std::runtime_error(message); }
}
}

int main() {
  try {
    for (const char* value : {"0", "ALL", ""}) {
      setenv("KFPARTICLE_GPU_CONSTRUCTION_TIMERS", value, 1);
      xpu::push_timer("outer-disabled");
      {
        KFParticleGpuStepTrace trace("disabled", true);
        trace.Next("must-not-appear");
        trace.Finish();
      }
      Require(xpu::pop_timer().children().empty(), "disabled trace changed timer stack");
    }
    setenv("KFPARTICLE_GPU_CONSTRUCTION_TIMERS", "1", 1);
    xpu::push_timer("outer-enabled");
    {
      KFParticleGpuStepTrace trace("test-ordered-trace", true);
      trace.Next("CPU plan");
      {
        KFParticleGpuStepTrace allocation("ALLOC test type=device action=initial bytes=64");
      }
      trace.Next("KERNEL test-boundary");
      trace.Next("D2H test-boundary");
      trace.Finish();
      trace.Finish();
    }
    const auto outer = xpu::pop_timer();
    const auto roots = outer.children();
    Require(roots.size() == 1u, "Finish/destructor produced duplicate or missing root");
    const auto steps = roots[0].children();
    Require(steps.size() == 3u, "missing interval");
    Require(steps[0].name() == "CPU plan" && steps[1].name() == "KERNEL test-boundary"
              && steps[2].name() == "D2H test-boundary", "interval order changed");
    Require(steps[0].children().size() == 1u, "allocation was not nested under planning");
    xpu::push_timer("outer-unwind");
    try {
      KFParticleGpuStepTrace trace("exception-scope");
      trace.Next("CPU failure");
      throw std::runtime_error("expected");
    } catch (const std::runtime_error&) {}
    const auto unwind = xpu::pop_timer();
    Require(unwind.name() == "outer-unwind" && unwind.children().size() == 1u,
            "exception left an open timer");
    unsetenv("KFPARTICLE_GPU_CONSTRUCTION_TIMERS");
    for (const char* value : {"0", "ALL", ""}) {
      setenv("KFPARTICLE_GPU_DEVICE_PROFILE", value, 1);
      xpu::push_timer("profile-disabled");
      { KFParticleGpuProfileScope profile; profile.Finish(); }
      Require(xpu::pop_timer().children().empty(), "disabled profile changed timer stack");
    }
    setenv("KFPARTICLE_GPU_DEVICE_PROFILE", "1", 1);
    setenv("KFPARTICLE_GPU_CONSTRUCTION_TIMERS", "1", 1);
    xpu::push_timer("profile-with-trace");
    {
      KFParticleGpuProfileScope profile;
      { KFParticleGpuStepTrace trace("nested-trace"); trace.Next("CPU plan"); }
      profile.Finish();
      profile.Finish();
    }
    const auto profile = xpu::pop_timer();
    Require(profile.children().size() == 1u
              && profile.children()[0].children().size() == 1u,
            "profile and trace nesting changed");
    xpu::push_timer("profile-unwind");
    try {
      KFParticleGpuProfileScope profile;
      throw std::runtime_error("expected");
    } catch (const std::runtime_error&) {}
    const auto profileUnwind = xpu::pop_timer();
    Require(profileUnwind.name() == "profile-unwind" && profileUnwind.children().size() == 1u,
            "exception left an open profile timer");
    unsetenv("KFPARTICLE_GPU_CONSTRUCTION_TIMERS");
    unsetenv("KFPARTICLE_GPU_DEVICE_PROFILE");
    std::cout << "PASS construction timer nesting, order, disabled mode and exception cleanup\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "FAIL construction timers: " << error.what() << '\n';
    return 1;
  }
}
