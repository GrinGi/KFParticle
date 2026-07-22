#!/usr/bin/env bash
set -euo pipefail

# Reuse the hermetic standalone setup and its normal regressions, then append
# the opt-in evidence tool.  The caller keeps the same device/XPU variables as
# run_xpu_lifecycle_test.sh; only benchmark size and iteration count are new.
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

KFPARTICLE_GPU_RUN_BATCH_BENCHMARK=1 \
  bash "${script_dir}/run_xpu_lifecycle_test.sh" "$@"
