#!/usr/bin/env bash
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$here/cbmroot_kfp_preflight_common.sh"
temporary="$(mktemp -d)"
trap 'rm -rf "$temporary"' EXIT
count=0
pass() { count=$((count+1)); echo "PASS [$count] $*"; }
expect_fail() {
  if "$@" > "$temporary/failure.log" 2>&1; then
    echo "FAIL expected rejection: $*" >&2; exit 1
  fi
}
mkdir -p "$temporary/hash-source"/{algo,reco,cmake,external/KFParticle,external/xpu}
printf 'project(test)\n' > "$temporary/hash-source/CMakeLists.txt"
first="$(kfp_preflight_source_fingerprint "$temporary/hash-source")"
second="$(kfp_preflight_source_fingerprint "$temporary/hash-source")"
[[ "$first" == "$second" && ${#first} == 64 ]]
pass 'stable source fingerprint without Python'
printf 'changed\n' > "$temporary/hash-source/algo/test.h"
[[ "$first" != "$(kfp_preflight_source_fingerprint "$temporary/hash-source")" ]]
pass 'source changes invalidate fingerprint'
current="$(kfp_preflight_source_fingerprint "$temporary/hash-source")"
mkdir -p "$temporary/hash-source/external/KFParticle/.local/history"
printf 'archived source\n' > "$temporary/hash-source/external/KFParticle/.local/history/old.cxx"
[[ "$current" == "$(kfp_preflight_source_fingerprint "$temporary/hash-source")" ]]
pass 'local development history is excluded from source identity'
printf 'lifecycle fragment\n' > "$temporary/hash-source/algo/test.inc"
[[ "$current" != "$(kfp_preflight_source_fingerprint "$temporary/hash-source")" ]]
pass 'lifecycle fragment changes invalidate source identity'

replay_fixture="$temporary/replay.log"
awk 'BEGIN {
  for (batch = 1; batch <= 2; ++batch) {
    for (sample = 1; sample <= 3; ++sample) {
      printf "KFP_IN_PROCESS_SAMPLE contract=1 sample=%d/3 mode=optimized events=%d succeeded=%d candidates=%d selected=0 overflow=0 source_xor=%d source_sum=%d signature_count=%d signature_xor=%d signature_sum=%d transaction_ms=1 pipeline_ms=1 queue_wait_ms=1 queue_capacity_ms=1 packing_ms=1 packing_threads_requested=2 packing_threads_used=2 h2d_ms=1 construction_ms=1 graph_ms=1 selection_ms=1 d2h_ms=1 extraction_ms=1 materialization_ms=1 materialization_cpu_work_ms=3 materialization_threads_requested=4 materialization_threads_used=3 pair_plan_ms=1 counting_readback_ms=1 counting_readback_bytes=64 dense_plan_ms=1 validation_packing_ms=1 validation_initial_ms=1 validation_compact_ms=1 initial_allocations=0 initial_allocation_ms=0 replacement_allocations=0 replacement_allocation_ms=0 launches=1 queue_waits=1 real_pairs=1 h2d_bytes=1 d2h_bytes=1 output_mode=compact host_assembly_ms=1 status_bytes=1 retained_snapshot_bytes=1\n",
             sample, batch, batch, batch, batch, batch, batch, batch, batch
    }
  }
}' > "$replay_fixture"
awk -v samples=3 -f "$here/optimized_replay_aggregate.awk" \
  "$replay_fixture" > "$temporary/replay.aggregate"
grep -q 'physical_batches=2 records=6 samples=3' "$temporary/replay.aggregate"
[[ "$(grep -c '^KFP_IN_PROCESS_AGGREGATE ' "$temporary/replay.aggregate")" == 3 ]]
grep -q 'sample=2/3 .* events=3 .* transaction_ms=2.000000000' "$temporary/replay.aggregate"
grep -q 'sample=2/3 .* overflow=0' "$temporary/replay.aggregate"
grep -q 'sample=2/3 .* materialization_cpu_work_ms=6.000000000 .* materialization_threads_requested=4 materialization_threads_used=3' "$temporary/replay.aggregate"
pass 'optimized replay aggregates each repetition and overflow evidence across physical batches'

awk 'NR == 2 { sub("signature_xor=1", "signature_xor=99") } { print }' \
  "$replay_fixture" > "$temporary/replay.identity-drift"
expect_fail awk -v samples=3 -f "$here/optimized_replay_aggregate.awk" \
  "$temporary/replay.identity-drift"
grep -q 'result identity changed in physical batch 1 at sample 2/3' "$temporary/failure.log"
pass 'optimized replay rejects per-batch identity drift'

sed '2d' "$replay_fixture" > "$temporary/replay.sequence-drift"
expect_fail awk -v samples=3 -f "$here/optimized_replay_aggregate.awk" \
  "$temporary/replay.sequence-drift"
grep -q 'sample sequence broke at record 2' "$temporary/failure.log"
pass 'optimized replay rejects missing or reordered samples'

# Keep safety checks on the maintained reader, not on retired comparison tools.
while IFS='|' read -r old new; do
  sed "s/$old/$new/g" "$replay_fixture" > "$temporary/replay.invalid"
  expect_fail awk -v samples=3 -f "$here/optimized_replay_aggregate.awk" \
    "$temporary/replay.invalid"
  pass "optimized replay rejects $new"
done <<'CASES'
overflow=0|overflow=1
succeeded=1|succeeded=0
transaction_ms=1|transaction_ms=nan
pipeline_ms=1|pipeline_ms=1e999
h2d_ms=1|h2d_ms=-1
real_pairs=1|real_pairs=invalid
packing_threads_used=2|packing_threads_used=3
materialization_threads_used=3|materialization_threads_used=5
output_mode=compact|output_mode=full
CASES

sed 's/signature_sum=1 /signature_sum=18446744073709551614 /g' \
  "$replay_fixture" > "$temporary/replay.uint64"
awk -v samples=3 -f "$here/optimized_replay_aggregate.awk" \
  "$temporary/replay.uint64" > /dev/null
sed '2s/signature_sum=18446744073709551614/signature_sum=18446744073709551615/' \
  "$temporary/replay.uint64" > "$temporary/replay.uint64-drift"
expect_fail awk -v samples=3 -f "$here/optimized_replay_aggregate.awk" \
  "$temporary/replay.uint64-drift"
pass 'optimized replay preserves exact uint64 result identity'

host_fixture="$temporary/replay.host.log"
awk 'BEGIN {
  for (batch = 1; batch <= 2; ++batch) {
    for (sample = 1; sample <= 3; ++sample) {
      print "KFP_HOST_RESULT contract=1 reference=0 event_local_maps=1 validated_publication_input=1 repeated_track_validation=0 materialization_schedule=guided graph_ms=1 leaves_ms=2 candidates_ms=3 commit_ms=4 scratch_initialized=5 scratch_reset=6 scratch_capacity_bytes=7 candidate_map_capacity_entries=8 track_map_capacity_entries=9 worker_estimated_weight_min=10 worker_estimated_weight_max=11 worker_estimated_weight_total=21 worker_events_min=12 worker_events_max=13 worker_tracks_min=14 worker_tracks_max=15 worker_candidates_min=16 worker_candidates_max=17 worker_cpu_min_ms=18 worker_cpu_max_ms=19"
    }
  }
}' > "$host_fixture"
awk -v samples=3 -v batches=2 \
  -v kind=host -f "$here/optimized_replay_aggregate.awk" \
  "$host_fixture" > "$temporary/replay.host.aggregate"
[[ "$(grep -c '^KFP_HOST_RESULT_AGGREGATE ' "$temporary/replay.host.aggregate")" == 3 ]]
grep -q 'sample=2/3 validated_publication_input=1 repeated_track_validation=0 materialization_schedule=guided graph_ms=2.000000000 .*scratch_capacity_bytes=14 .*candidate_map_capacity_entries=16 track_map_capacity_entries=18 .*worker_estimated_weight_min=20 worker_estimated_weight_max=22 worker_estimated_weight_total=42 worker_events_min=24 worker_events_max=26 worker_tracks_min=28 worker_tracks_max=30 worker_candidates_min=32 worker_candidates_max=34 worker_cpu_min_ms=36.000000000 worker_cpu_max_ms=38.000000000' \
  "$temporary/replay.host.aggregate"
pass 'optimized replay aggregates event-local host workspace evidence'

sed '1s/event_local_maps=1/event_local_maps=0/' "$host_fixture" \
  > "$temporary/replay.host.invalid"
expect_fail awk -v samples=3 -v batches=2 \
  -v kind=host -f "$here/optimized_replay_aggregate.awk" \
  "$temporary/replay.host.invalid"
grep -q 'record 1 is not the optimized trusted-input contract' "$temporary/failure.log"
pass 'optimized replay rejects invalid host workspace contract'

sed '1s/validated_publication_input=1/validated_publication_input=0/' \
  "$host_fixture" > "$temporary/replay.host.untrusted"
expect_fail awk -v samples=3 -v batches=2 \
  -v kind=host -f "$here/optimized_replay_aggregate.awk" \
  "$temporary/replay.host.untrusted"
grep -q 'record 1 is not the optimized trusted-input contract' "$temporary/failure.log"
pass 'optimized replay rejects an untrusted production materializer record'

sed 's/materialization_schedule=guided/materialization_schedule=static/' \
  "$host_fixture" > "$temporary/replay.host.static-schedule"
expect_fail awk -v samples=3 -v batches=2 \
  -v kind=host -f "$here/optimized_replay_aggregate.awk" \
  "$temporary/replay.host.static-schedule"
grep -q 'record 1 is not the optimized trusted-input contract' "$temporary/failure.log"
pass 'optimized replay rejects a schedule that differs from the requested mode'

awk -v samples=3 -v batches=2 -v expected_schedule=static \
  -v kind=host -f "$here/optimized_replay_aggregate.awk" \
  "$temporary/replay.host.static-schedule" > "$temporary/replay.host.static.aggregate"
grep -q 'materialization_schedule=static' "$temporary/replay.host.static.aggregate"
pass 'optimized replay accepts an explicitly requested benchmark schedule'

expect_fail awk -v samples=3 -v batches=2 -v expected_schedule=auto \
  -v kind=host -f "$here/optimized_replay_aggregate.awk" "$host_fixture"
grep -q 'invalid expected materialization schedule auto' "$temporary/failure.log"
pass 'optimized replay rejects an unknown benchmark schedule'

for invalid in -1 nan 1e999; do
  sed "s/leaves_ms=2/leaves_ms=$invalid/" "$host_fixture" > "$temporary/host.invalid"
  expect_fail awk -v samples=3 -v batches=2 -v kind=host \
    -f "$here/optimized_replay_aggregate.awk" "$temporary/host.invalid"
  pass "host timing reader rejects $invalid"
done
expect_fail awk -v samples=3 -v batches=3 -v kind=host \
  -f "$here/optimized_replay_aggregate.awk" "$host_fixture"
pass 'host timing reader rejects missing physical batches'
expect_fail awk -v samples=3 -v kind=unknown \
  -f "$here/optimized_replay_aggregate.awk" "$host_fixture"
pass 'replay reader rejects an unknown record mode'

echo "PASS kfp-timing Bash contracts: $count"
