# KFParticle GPU

KFParticle GPU reconstructs short-lived particles from fitted tracks and primary
vertices using Kalman-filter fits. It processes independent events together on
an AMD GPU through XPU and returns ordinary `KFParticle` objects with daughter
relationships. It can be used as a standalone C++ library or through CBMRoot.

The library receives reconstructed tracks, not detector hits. Event finding,
tracking, PID, input-file reading and output-file writing belong to the calling
application. Online and offline applications use the same reconstruction API.

## Requirements

- Linux x86-64 with SSE4.1, a C++17 compiler and CMake 3.20 or newer.
- Installed CERN ROOT and Vc, including their CMake package files.
- Installed, unmodified [XPU master](https://github.com/xpu-devel/xpu), with
  headers and shared libraries. For AMD GPU execution its HIP backend is required.
- A compatible AMD GPU and ROCm toolchain, including HIP, rocPRIM and hipCUB.
- Bash and AWK for the optional integration tests; GTest for shared host tests.

Install these dependencies according to their own documentation or use the
versions provided by your computing environment. Administrator privileges are
not required to install KFParticle into a user-owned directory. CPU-XPU execution
is also available without a GPU; it runs the GPU algorithm on the CPU, not the
original SIMD CPU Finder.

## Standalone installation

Replace the paths below with your KFParticle checkout, build/install directories
of your choice and existing dependencies. `XPU_PREFIX` is an installation prefix,
not an XPU source directory.
KFParticle neither downloads nor modifies XPU and does not require a special
XPU fork or a fixed commit.

```bash
export KF_SOURCE=/path/to/KFParticle
export KF_BUILD=/path/to/kfparticle-build
export KF_PREFIX=/path/to/kfparticle-install
export XPU_PREFIX=/path/to/installed/xpu
export ROOT_PREFIX=/path/to/installed/root
export VC_PREFIX=/path/to/installed/vc
export ROCM_ROOT=/opt/rocm

source "$ROOT_PREFIX/bin/thisroot.sh"
cmake -S "$KF_SOURCE" -B "$KF_BUILD" \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$KF_PREFIX" \
  -DCMAKE_INSTALL_LIBDIR=lib -DCMAKE_PREFIX_PATH="$ROOT_PREFIX;$VC_PREFIX" \
  -DKFPARTICLE_USE_XPU=ON -DFIXTARGET=ON -DKFPARTICLE_INSTALL_SDK=ON \
  -DKFPARTICLE_XPU_ROOT="$XPU_PREFIX" -DXPU_ENABLE_HIP=ON -DXPU_ENABLE_OPENMP=ON \
  -DXPU_HIP_ARCH=native -DXPU_ROCM_ROOT="$ROCM_ROOT"
cmake --build "$KF_BUILD" --parallel 8
cmake --install "$KF_BUILD"
```

`native` automatically detects the architectures of GPUs visible during
compilation; no architecture names need to be entered. Build with the intended
GPUs accessible and use XPU built for those architectures as well. For a build
targeting other machines, set `XPU_HIP_ARCH` to their architecture list instead.
`FIXTARGET=ON` selects the nonhomogeneous field model required by the current GPU
input interface. For a CPU-XPU build use `-DXPU_ENABLE_HIP=OFF` and omit the ROCm
options. Use compatible compiler, ROOT/Vc and XPU builds throughout.

The installation provides headers under `include/`, the host library and enabled
kernel images under `lib/`, and a CMake package under `lib/cmake/KFParticle/`.
XPU remains an external dependency and is not copied into the KFParticle prefix.
The libraries needed for HIP are `libKFParticle.so`, `libKFParticle_Hip.so` and
the existing `libxpu.so`, `libxpu_Hip.so`. Do not mix different XPU installations.
Build settings are recorded in `share/KFParticle/KFParticleBuildInfo.txt`.

For execution, make these library directories available through your environment
or application runtime paths. Adjust `XPU_LIB_DIR` if XPU uses `lib64`:

```bash
export XPU_LIB_DIR="$XPU_PREFIX/lib"
export LD_LIBRARY_PATH="$KF_PREFIX/lib:$XPU_LIB_DIR:$(root-config --libdir):$VC_PREFIX/lib:$ROCM_ROOT/lib:$ROCM_ROOT/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
```

## Using the C++ library

Link the installed package from your application's `CMakeLists.txt`:

```cmake
find_package(KFParticle CONFIG REQUIRED COMPONENTS GPU HIP)
add_executable(my_analysis main.cxx)
target_link_libraries(my_analysis PRIVATE KFParticle::KFParticle)
```

```bash
cmake -S /path/to/my-analysis -B /path/to/my-analysis-build \
  -DCMAKE_PREFIX_PATH="$KF_PREFIX;$ROOT_PREFIX;$VC_PREFIX" \
  -DKFPARTICLE_XPU_ROOT="$XPU_PREFIX"
cmake --build /path/to/my-analysis-build --parallel 8
```

For a CPU-XPU application require only the `GPU` component. The imported target
provides the include paths, C++ standard, field/SIMD definitions and dependency
linkage; do not duplicate these definitions in the application.

### Input and output

`KFParticleGpuFinder::Event` borrows the following data for one event:

| Input | Contents |
| --- | --- |
| `sourceEventId` | A 64-bit ID, unique within the submitted batch |
| First/last `KFPTrackVector` | Matching track IDs, six fitted parameters, 21 covariance elements, charge, PDG, ten field coefficients and PV assignments |
| `chiToPV` array | Track-to-PV compatibility for primary/secondary cuts, one value per physical track |
| `KFVertex` array | Primary vertices with position, covariance and fit metadata |

Track IDs must be unique and nonnegative within an event. `PVIndex` is event-local;
a negative value means unassigned. Tracks in different events are independent,
even if their IDs coincide. Use the scalar KFParticle units and field convention.
Keep the input arrays alive and unchanged until the synchronous call returns.

Each result owns its particle vector. Compact output contains accepted candidates
and their required track/intermediate-particle dependencies; daughter indices
refer to that event's output vector. Check `Succeeded()` before using the output.
An event that fails does not publish a partial vector. Empty output can be valid.

### Reconstruction

For an application-provided `std::vector<KFParticleGpuFinder::Event> events`:

```cpp
#include "execution/KFParticleGpuFinder.h"
#include <stdexcept>
#include <vector>

KFParticleGpuFinder::Options options;
options.device = "hip0";
options.limits = kfparticle::gpu::DefaultGpuBatchLimits(512u);
options.packingThreads = 4u;
options.materializationThreads = 4u;
KFParticleGpuFinder finder(options);

std::vector<KFParticleGpuFinder::Result> results = finder.ReconstructBatch(events);
finder.ReleaseResources();
for (const KFParticleGpuFinder::Result& result : results) {
  if (!result.Succeeded()) { throw std::runtime_error(result.message); }
  // Write or analyse result.particles.
}
```

Use `Reconstruct(event)` for one event. Select `hipN` for the corresponding XPU
device or `cpu` for CPU-XPU execution. Submit bounded batches: the standalone API
does not silently split an oversized batch or substitute the original CPU Finder.

In a streaming or file-reading loop, keep one Finder outside the loop, accumulate
events to the chosen size/latency limit and flush the final partial batch. Calls
must remain on the constructing thread; packing and materialization can use their
separately bounded worker counts. Release cached resources before XPU shutdown,
including on an error path. Results remain valid after release. If the application
already initializes XPU, set `initializeXpuIfNeeded=false` and use its active device.

Framework integrations can submit an admitted `GpuPreparedBatch` through
`ReconstructPreparedBatch` without repeating input preparation. See the public
header for prepared-input, timing and diagnostic options. Diagnostic snapshots
are optional and are not needed for ordinary output.

## Example application

`example/KFParticleGpuExample.cxx` demonstrates the complete basic workflow in
one commented `main`: generate tracks/vertices, prepare a batch, configure/create
`KFParticleGpuFinder`, reconstruct, access the returned particles and fill the
existing `KFTopoPerformance` QA. No external input files, MC or CBMRoot data are
needed. The example is not a benchmark or a physics-validation sample.

### Build with the installed SDK

Use the dependency paths and runtime library environment from
[Standalone installation](#standalone-installation):

```bash
export EXAMPLE_BUILD=/path/to/kfparticle-example-build
cmake -S "$KF_PREFIX/share/KFParticle/example" -B "$EXAMPLE_BUILD" \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="$KF_PREFIX;$ROOT_PREFIX;$VC_PREFIX" \
  -DKFPARTICLE_XPU_ROOT="$XPU_PREFIX"
cmake --build "$EXAMPLE_BUILD" --parallel 8
export KFP_EXAMPLE="$EXAMPLE_BUILD/KFParticleGpuExample"
```

### Build within CBMRoot

Alternatively, reuse an already configured, GPU-enabled CBMRoot build:

```bash
export CBM_SOURCE=/path/to/cbmroot
export CBM_BUILD=/path/to/configured-cbmroot-build
cmake -S "$CBM_SOURCE" -B "$CBM_BUILD" \
  -DKFPARTICLE_BUILD_GPU_EXAMPLE=ON
cmake --build "$CBM_BUILD" --target KFParticleGpuExample --parallel 8
source "$CBM_BUILD/config.sh"
export KFP_EXAMPLE="$CBM_BUILD/bin/KFParticleGpuExample"
```

### Run and inspect the output

Run from a writable directory. No reconstruction configuration or input dataset
is required. For the standalone installation, `"$XPU_PREFIX/bin/xpuinfo"` lists
the available devices:

```bash
"$KFP_EXAMPLE" hip0
```

The optional argument selects the device: `hip0` (default), `hip1`, or `cpu`
for CPU-XPU execution. If `XPU_DEVICE` is set, it must agree with this argument.
The program prints particle/candidate counts for each event and finishes with
`Wrote kfparticle-qa.root` on success; errors return a nonzero exit code.

Eight toy events are generated in memory, each containing a primary vertex,
a displaced pion pair and two primary tracks. `ExampleData.cxx` prepares the
arrays and borrowed event views without copying track arrays into the batch.
The comments in `main` explain their layout, ownership and lifetime.

Open `kfparticle-qa.root` with ROOT to inspect the reconstructed-distribution
histograms under `GPU/`. The file is written in the current directory and is
replaced on each run. `ExampleQa.cxx` adapts the results to the existing QA;
it does not run the CPU Finder. MC, CPU constraint-cache and ZR histogram
families are disabled.

## CBMRoot integration

CBMRoot provides input conversion, PID/refitting, event collection and output
selection. Its adapters call the same Finder interface. Online reconstruction
uses `cbmreco`; offline reconstruction uses the FairRunAna selector chain.
Both currently pass fitted tracks through host memory to KFParticle.

Enable the integration in an already configured CBMRoot build:

```bash
export CBM_SOURCE=/path/to/cbmroot
export CBM_BUILD=/path/to/configured-cbmroot-build
cmake -S "$CBM_SOURCE" -B "$CBM_BUILD" \
  -DCBM_KFPARTICLE_USE_XPU=ON -DCBM_KFPARTICLE_GPU_DIAGNOSTICS=ON
cmake --build "$CBM_BUILD" --target cbmreco CbmRecoTasks --parallel 8
```

This build reuses CBMRoot's XPU target and disables standalone SDK installation
within the embedded subproject. It does not require a second standalone XPU
installation. Use the normal CBMRoot input/configuration workflow and matching
reconstruction parameters. The selector configuration must explicitly specify
`kfp.selector.doRefitTracks`.

| Mode | Behaviour |
| --- | --- |
| `cpu-only` | Original CPU Finder and CPU output |
| `diagnostic` | CPU output plus GPU reconstruction/reference comparison |
| `qualified-gpu` | Explicit GPU publication with framework eligibility checks and whole-event CPU fallback |

CPU publication remains the default. GPU publication requires the
`CBM_KFPARTICLE_GPU_QUALIFICATION_TRIAL=ON` build option and an enabled adaptive
policy; building GPU support alone does not enable it. The adapters publish
event-selection masks rather than a persistent branch containing every particle.
Offline qualified batches stay within one FairRunAna input entry, so a larger
batch limit does not combine separate single-event entries.

## Tests

These commands verify correctness and integration; they are not analysis launch
recipes. Synthetic API tests do not establish complete physics equivalence.

### Installed standalone API

After the standalone installation above, build only the small test application:

```bash
export KF_CHECK=/path/to/kfparticle-check-build
cmake -S "$KF_PREFIX/share/KFParticle/standalone" -B "$KF_CHECK" \
  -DKFPARTICLE_GPU_TEST_INSTALLED=ON -DKFPARTICLE_GPU_TEST_DEVICE=hip0 \
  -DCMAKE_PREFIX_PATH="$KF_PREFIX;$ROOT_PREFIX;$VC_PREFIX" \
  -DKFPARTICLE_XPU_ROOT="$XPU_PREFIX" -DCMAKE_BUILD_TYPE=Release
cmake --build "$KF_CHECK" --parallel 8
ctest --test-dir "$KF_CHECK" --output-on-failure --no-tests=error
```

This tests CPU-XPU and the selected HIP device with nonempty input, event/batch
agreement, PV isolation, owned results and resource cleanup. Use device `cpu`
for a CPU-only installation. It does not rebuild KFParticle or perform tracking.

### Development and CBMRoot checks

The example smoke test generates toy input, reconstructs one batch and writes
the reconstructed-distribution QA file. It runs on CPU-XPU and optionally the
device selected by `KFPARTICLE_EXAMPLE_TEST_DEVICE`. After building it, run
`ctest --test-dir "$EXAMPLE_BUILD" --output-on-failure --no-tests=error`.
This is a basic usage check, not a benchmark or physics-equivalence gate.

| Suite | Purpose |
| --- | --- |
| Host data/shared interface | Layout, views, input packing, batch bounds and invalid input |
| XPU Lifecycle | Fit/transport references, routing, event isolation, decay graphs, overflow and materialization |
| Standalone Finder | Public single-event, batch and prepared-input APIs |
| CBMRoot adapter | Framework policy, fallback and result integration |
| CPU/GPU V0 regression | Independent original-CPU versus GPU fits and daughter lineage |
| Bash contracts | Build registration, preflight failures, calibration and timing-report parsing |

For a configured CBMRoot build with `BUILD_TESTING=ON` and GTest:

```bash
export KFP_TEST="$CBM_SOURCE/external/KFParticle/KFParticle/GPU/test"
cmake --build "$CBM_BUILD" --parallel 8 --target \
  _GTestKfpGpuAdapter KFParticleGpuHostInterfaceTest KFParticleCpuGpuV0Equivalence
ctest --test-dir "$CBM_BUILD" -L kfp --output-on-failure --no-tests=error
export KFPARTICLE_CBMROOT_BUILD_DIR="$CBM_BUILD"
export KFPARTICLE_CBMROOT_DEVICE=hip0
export KFPARTICLE_CBMROOT_EQUIVALENCE_SKIP_BUILD=1
bash "$KFP_TEST/cbmroot/run_cbmroot_regression.sh" all
```

The CBMRoot build-registration contract checks embedded targets, public ABI
settings and reuse of CBMRoot's XPU runtime. It does not require an installed
SDK or SDK package metadata. Standalone SDK export/import checks are separate:

```bash
bash "$KF_SOURCE/KFParticle/GPU/test/cbmroot/test_kfp_build_registration.sh" sdk
```

This configuration-only check needs the KFParticle source tree, Bash, CMake
and a C++ compiler, not CBMRoot or an accelerator. It does not replace compiled
GPU regression tests.

`run_xpu_tests.sh` provides optional source-tree `lifecycle`, `ctest`, `benchmark`
and `matrix` modes. Supply the XPU source path and backend options for that
development build; `--run-only` reuses a matching build without compiling again.

Real-data integration tests are opt-in: `run_cbmroot_kfp_online_preflight.sh`
wraps `cbmreco`, and `run_cbmroot_kfp_offline_preflight.sh` wraps FairRunAna.
They need explicit input/configuration paths and `KFPARTICLE_CBMROOT_PREFLIGHT_EXECUTE=1`.
Online uses `ONLINE_CONFIG` and `ONLINE_INPUT_LOCATOR`; offline uses `OFFLINE_CONFIG`,
`OFFLINE_INPUT_PREFIX` and `OFFLINE_PARAMETER_PREFIX`, all prefixed by
`KFPARTICLE_CBMROOT_`. Online needs its matching `reco.par.bin`; offline needs
the matching `.par.root`, not that binary archive.

For a whole-timeslice online check, use `ONLINE_NUM_TS=1`, `ONLINE_MAX_EVENTS=0`
and, for example, `EVENT_BATCH_SIZE=512` with `DIAGNOSTIC_BATCH_EVENTS=512`.
All these names have the same prefix. Both batch aliases must agree if set.
Published-trial checks additionally require all three adaptive limits:
`ADAPTIVE_MINIMUM_WORK`, `ADAPTIVE_TARGET_WORK` and `ADAPTIVE_MAXIMUM_BYTES`.
Their values must match the dataset and memory budget, not a universal preset.

### Performance measurements

The synthetic V0 benchmark compares serial and batched execution of the GPU
algorithm, not GPU versus the original CPU Finder. Real-data
`run_cbmroot_kfp_optimized_replay.sh` performs tracking once and repeats KFParticle
on the same prepared input. Each repetition uploads input again; allocations are
retained, so distinguish the cold first sample from subsequent samples.
Compare equal event/track/pair counts and report total/per-event time. Keep
transaction wall time, GPU intervals and summed CPU worker time separate.
Full snapshots and detailed tracing/profiling are opt-in and affect timings.

## Source layout

| Directory | Contents |
| --- | --- |
| `data/` | Track/vertex states, transfer views and result materialization |
| `planning/` | Channel catalogue, decay plans, routing and task tiling |
| `execution/` | Public Finder, runtime, buffers, steering and kernels |
| `algorithms/` | Fit mathematics, candidate construction and selection |
| `diagnostics/` | Optional comparison, timing and tracing helpers |
| `example/` | Minimal reconstruction/QA application and toy-input generator |
| `test/` | Host/device tests, integration checks and benchmarks |
