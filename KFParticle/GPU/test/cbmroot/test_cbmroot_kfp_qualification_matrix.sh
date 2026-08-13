#!/usr/bin/env bash
set -euo pipefail

# Keep mock matrices isolated from retained target-server qualification data.
while IFS= read -r variable; do
  unset "${variable}"
done < <(compgen -A variable KFPARTICLE_ || true)

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
tmp_dir="$(mktemp -d "${TMPDIR:-/tmp}/kfparticle-qualification-matrix.XXXXXX")"
trap 'rm -rf "${tmp_dir}"' EXIT
mkdir -p "${tmp_dir}/build"

cat >"${tmp_dir}/mock-campaign.sh" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
output="${KFPARTICLE_CBMROOT_QUALIFICATION_LOG_DIR}/qualification-${KFPARTICLE_CBMROOT_QUALIFICATION_MODE}-${KFPARTICLE_CBMROOT_QUALIFICATION_WORKLOAD}.evidence.txt"
mkdir -p "$(dirname "${output}")"
printf 'decision: qualified\n' >"${output}"
EOF

KFPARTICLE_CBMROOT_BUILD_DIR="${tmp_dir}/build" \
KFPARTICLE_CBMROOT_DEVICE=hip1 \
KFPARTICLE_CBMROOT_QUALIFICATION_MODE=online \
KFPARTICLE_CBMROOT_QUALIFICATION_WORKLOADS='low:1 typical:2 high:3' \
KFPARTICLE_CBMROOT_QUALIFICATION_LOG_DIR="${tmp_dir}/logs" \
KFPARTICLE_CBMROOT_QUALIFICATION_CAMPAIGN_LAUNCHER="${tmp_dir}/mock-campaign.sh" \
  bash "${script_dir}/run_cbmroot_kfp_qualification_matrix.sh" >"${tmp_dir}/matrix.log"

grep -q '^PASS cbmroot-kfp-qualification-matrix' "${tmp_dir}/matrix.log"
grep -q '^decision: qualified$' "${tmp_dir}/logs/qualification-online-matrix.evidence.txt"
for workload in low typical high; do
  test -s "${tmp_dir}/logs/online-${workload}/qualification-online-${workload}.evidence.txt"
done

if KFPARTICLE_CBMROOT_BUILD_DIR="${tmp_dir}/build" \
   KFPARTICLE_CBMROOT_QUALIFICATION_MODE=offline \
   KFPARTICLE_CBMROOT_QUALIFICATION_WORKLOADS='low:1 typical:2' \
   KFPARTICLE_CBMROOT_QUALIFICATION_CAMPAIGN_LAUNCHER="${tmp_dir}/mock-campaign.sh" \
     bash "${script_dir}/run_cbmroot_kfp_qualification_matrix.sh" >"${tmp_dir}/incomplete.log" 2>&1; then
  echo "FAIL cbmroot-kfp-qualification-matrix-contract - incomplete workload set was accepted" >&2
  exit 1
fi
grep -q 'missing required high workload' "${tmp_dir}/incomplete.log"

cat >"${tmp_dir}/mock-rejecting-campaign.sh" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
output="${KFPARTICLE_CBMROOT_QUALIFICATION_LOG_DIR}/qualification-${KFPARTICLE_CBMROOT_QUALIFICATION_MODE}-${KFPARTICLE_CBMROOT_QUALIFICATION_WORKLOAD}.evidence.txt"
mkdir -p "$(dirname "${output}")"
if [[ "${KFPARTICLE_CBMROOT_QUALIFICATION_WORKLOAD}" == "typical" ]]; then
  printf 'decision: rejected\nreason: fixture rejection\n' >"${output}"
  exit 2
fi
printf 'decision: qualified\n' >"${output}"
EOF
if KFPARTICLE_CBMROOT_BUILD_DIR="${tmp_dir}/build" \
   KFPARTICLE_CBMROOT_QUALIFICATION_MODE=offline \
   KFPARTICLE_CBMROOT_QUALIFICATION_WORKLOADS='low:1 typical:2 high:3' \
   KFPARTICLE_CBMROOT_QUALIFICATION_LOG_DIR="${tmp_dir}/rejected" \
   KFPARTICLE_CBMROOT_QUALIFICATION_CAMPAIGN_LAUNCHER="${tmp_dir}/mock-rejecting-campaign.sh" \
     bash "${script_dir}/run_cbmroot_kfp_qualification_matrix.sh" >"${tmp_dir}/rejected.log" 2>&1; then
  echo "FAIL cbmroot-kfp-qualification-matrix-contract - rejected workload produced a passing matrix" >&2
  exit 1
fi
grep -q '^decision: rejected$' "${tmp_dir}/rejected/qualification-offline-matrix.evidence.txt"
test -s "${tmp_dir}/rejected/offline-high/qualification-offline-high.evidence.txt"

cat >"${tmp_dir}/mock-aborting-campaign.sh" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
output="${KFPARTICLE_CBMROOT_QUALIFICATION_LOG_DIR}/qualification-${KFPARTICLE_CBMROOT_QUALIFICATION_MODE}-${KFPARTICLE_CBMROOT_QUALIFICATION_WORKLOAD}.evidence.txt"
mkdir -p "$(dirname "${output}")"
if [[ "${KFPARTICLE_CBMROOT_QUALIFICATION_WORKLOAD}" == "typical" ]]; then
  exit 7
fi
printf 'decision: qualified\n' >"${output}"
EOF
mkdir -p "${tmp_dir}/aborted/offline-typical"
printf 'decision: qualified\nreason: stale fixture\n' \
  >"${tmp_dir}/aborted/offline-typical/qualification-offline-typical.evidence.txt"
if KFPARTICLE_CBMROOT_BUILD_DIR="${tmp_dir}/build" \
   KFPARTICLE_CBMROOT_QUALIFICATION_MODE=offline \
   KFPARTICLE_CBMROOT_QUALIFICATION_WORKLOADS='low:1 typical:2 high:3' \
   KFPARTICLE_CBMROOT_QUALIFICATION_LOG_DIR="${tmp_dir}/aborted" \
   KFPARTICLE_CBMROOT_QUALIFICATION_CAMPAIGN_LAUNCHER="${tmp_dir}/mock-aborting-campaign.sh" \
     bash "${script_dir}/run_cbmroot_kfp_qualification_matrix.sh" >"${tmp_dir}/aborted.log" 2>&1; then
  echo "FAIL cbmroot-kfp-qualification-matrix-contract - pre-gate campaign failure produced a passing matrix" >&2
  exit 1
fi
grep -q '^decision: rejected$' \
  "${tmp_dir}/aborted/offline-typical/qualification-offline-typical.evidence.txt"
grep -q 'campaign stopped before the qualification gate (status 7)' \
  "${tmp_dir}/aborted/offline-typical/qualification-offline-typical.evidence.txt"
test -s "${tmp_dir}/aborted/offline-high/qualification-offline-high.evidence.txt"

echo "PASS cbmroot-kfp-qualification-matrix-contract - three bounded workloads are isolated, stale evidence is removed, and pre-gate failures are retained"
