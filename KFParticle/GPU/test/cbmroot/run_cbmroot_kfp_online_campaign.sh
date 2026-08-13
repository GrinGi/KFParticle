#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
build_dir="${KFPARTICLE_CBMROOT_BUILD_DIR:-}"
device="${KFPARTICLE_CBMROOT_DEVICE:-hip1}"
repeat="${KFPARTICLE_CBMROOT_ONLINE_REPEAT:-2}"

if [[ -z "${build_dir}" || ! -d "${build_dir}" ]]; then
  echo "FAIL cbmroot-kfp-online-campaign - set KFPARTICLE_CBMROOT_BUILD_DIR" >&2
  exit 2
fi

log_dir="${KFPARTICLE_CBMROOT_PREFLIGHT_LOG_DIR:-${build_dir}/kfparticle-cbmroot-preflight}"
cpu_evidence="${log_dir}/online-cpu-only-${device}.evidence.txt"
diagnostic_evidence="${log_dir}/online-diagnostic-${device}.evidence.txt"
campaign_evidence="${log_dir}/online-cpu-diagnostic-${device}.evidence.txt"

run_mode()
{
  local mode="$1"
  local evidence="$2"
  KFPARTICLE_CBMROOT_KFP_MODE="${mode}" \
  KFPARTICLE_CBMROOT_ONLINE_REPEAT="${repeat}" \
  KFPARTICLE_CBMROOT_PREFLIGHT_EXECUTE=1 \
  KFPARTICLE_CBMROOT_PREFLIGHT_EVIDENCE_FILE="${evidence}" \
    bash "${script_dir}/run_cbmroot_kfp_online_preflight.sh"
}

evidence_result()
{
  sed -n 's/^result: //p' "$1" | tail -1
}

handoff_identity()
{
  local summary="$1"
  local base
  local pid_assigned
  local selected_tracks
  local cpu_particles
  local cpu_candidates
  local decay_channels
  local pid_policy
  local refit_policy
  base="$(
    sed -nE \
      's/.* events_available=([0-9]+) events=([0-9]+) processed=([0-9]+) no_pv=([0-9]+) conversion_failed=([0-9]+) tracks=([0-9]+) sts_hit_links=([0-9]+) vertices=([0-9]+) pid_slots=([0-9]+).*/events_available=\1 events=\2 processed=\3 no_pv=\4 conversion_failed=\5 tracks=\6 sts_hit_links=\7 vertices=\8 pid_slots=\9/p' \
      <<<"${summary}"
  )"
  pid_assigned="$(sed -nE 's/.* pid_assigned=([0-9]+).*/\1/p' <<<"${summary}")"
  selected_tracks="$(sed -nE 's/.* selected_tracks=([0-9]+).*/\1/p' <<<"${summary}")"
  cpu_particles="$(sed -nE 's/.* cpu_particles=([0-9]+).*/\1/p' <<<"${summary}")"
  cpu_candidates="$(sed -nE 's/.* cpu_candidates=([0-9]+).*/\1/p' <<<"${summary}")"
  decay_channels="$(sed -nE 's/.* decay_channels=([0-9]+).*/\1/p' <<<"${summary}")"
  pid_policy="$(sed -nE 's/.* pid_policy=([^ ]+).*/\1/p' <<<"${summary}")"
  refit_policy="$(sed -nE 's/.* refit_policy=([^ ]+).*/\1/p' <<<"${summary}")"
  if [[ -n "${base}" && -n "${pid_assigned}" && -n "${selected_tracks}" \
        && -n "${cpu_particles}" && -n "${cpu_candidates}" && -n "${decay_channels}" \
        && -n "${pid_policy}" && -n "${refit_policy}" ]]; then
    printf '%s pid_assigned=%s selected_tracks=%s cpu_particles=%s cpu_candidates=%s decay_channels=%s pid_policy=%s refit_policy=%s\n' \
      "${base}" "${pid_assigned}" "${selected_tracks}" "${cpu_particles}" "${cpu_candidates}" \
      "${decay_channels}" "${pid_policy}" "${refit_policy}"
  fi
}

echo "KFParticle CBMRoot online CPU/diagnostic campaign"
echo "  build dir  : ${build_dir}"
echo "  device     : ${device}"
echo "  repetitions: ${repeat}"

run_mode cpu-only "${cpu_evidence}"
run_mode diagnostic "${diagnostic_evidence}"

cpu_result="$(evidence_result "${cpu_evidence}")"
diagnostic_result="$(evidence_result "${diagnostic_evidence}")"
cpu_handoff="$(handoff_identity "${cpu_result}")"
diagnostic_handoff="$(handoff_identity "${diagnostic_result}")"

if [[ -z "${cpu_handoff}" || -z "${diagnostic_handoff}" ]]; then
  echo "FAIL cbmroot-kfp-online-campaign - evidence is missing an online handoff result" >&2
  exit 2
fi
if [[ "${cpu_handoff}" != "${diagnostic_handoff}" ]]; then
  echo "CPU handoff       : ${cpu_handoff}" >&2
  echo "diagnostic handoff: ${diagnostic_handoff}" >&2
  echo "FAIL cbmroot-kfp-online-campaign - CPU and diagnostic modes changed the online input handoff" >&2
  exit 2
fi

{
  echo "KFParticle CBMRoot online campaign evidence"
  echo "generated: $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
  echo "device: ${device}"
  echo "repetitions: ${repeat}"
  echo "handoff: ${cpu_handoff}"
  echo "cpu evidence: ${cpu_evidence}"
  echo "diagnostic evidence: ${diagnostic_evidence}"
  echo "cpu result: ${cpu_result}"
  echo "diagnostic result: ${diagnostic_result}"
} >"${campaign_evidence}"

echo "PASS cbmroot-kfp-online-campaign - repeated CPU and diagnostic runs preserve the online handoff and satisfy both contracts"
echo "  evidence   : ${campaign_evidence}"
