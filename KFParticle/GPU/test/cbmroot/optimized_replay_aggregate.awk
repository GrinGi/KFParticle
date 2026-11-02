BEGIN {
  if (kind == "") { kind = "replay" }
  if (kind != "replay" && kind != "host") { fail("invalid aggregation kind " kind) }
  if (samples !~ /^[1-9][0-9]*$/) { fail("samples must be a positive integer") }
  if (kind == "host") {
    if (batches !~ /^[1-9][0-9]*$/) { fail("batches must be a positive integer") }
    if (expected_schedule == "") { expected_schedule = "guided" }
    if (expected_schedule != "weighted-contiguous" && expected_schedule != "static" &&
        expected_schedule != "dynamic" && expected_schedule != "guided") {
      fail("invalid expected materialization schedule " expected_schedule)
    }
  }
  host_key_count = split("graph_ms leaves_ms candidates_ms commit_ms scratch_initialized scratch_reset scratch_capacity_bytes candidate_map_capacity_entries track_map_capacity_entries worker_estimated_weight_min worker_estimated_weight_max worker_estimated_weight_total worker_events_min worker_events_max worker_tracks_min worker_tracks_max worker_candidates_min worker_candidates_max worker_cpu_min_ms worker_cpu_max_ms", host_keys)
  count_key_count = split("events succeeded candidates selected overflow", count_keys)
  timing_key_count = split("transaction_ms pipeline_ms queue_wait_ms queue_capacity_ms packing_ms h2d_ms construction_ms graph_ms selection_ms d2h_ms extraction_ms materialization_ms materialization_cpu_work_ms pair_plan_ms counting_readback_ms dense_plan_ms validation_packing_ms validation_initial_ms validation_compact_ms initial_allocation_ms replacement_allocation_ms host_assembly_ms", timing_keys)
  maximum_key_count = split("packing_threads_requested packing_threads_used materialization_threads_requested materialization_threads_used", maximum_keys)
  work_key_count = split("initial_allocations replacement_allocations launches queue_waits real_pairs h2d_bytes d2h_bytes counting_readback_bytes status_bytes retained_snapshot_bytes", work_keys)
  split("events succeeded candidates selected overflow source_xor source_sum signature_count signature_xor signature_sum", identity_keys)
  expected_sample = 1
}

function fail(message) {
  print "FAIL optimized-replay - " message > "/dev/stderr"
  failed = 1
  exit 1
}

function require(key) {
  if (!(key in field) || field[key] == "") {
    fail("sample record " records " is missing " key)
  }
}

function require_number(key, integer, value) {
  require(key)
  value = field[key]
  if ((integer && value !~ /^[0-9]+$/) ||
      (!integer && value !~ /^[0-9]+([.][0-9]+)?([eE][+-]?[0-9]+)?$/)) {
    fail("invalid numeric " key " at record " records)
  }
  if (sprintf("%.17g", value + 0) ~ /[Ii][Nn][Ff]|[Nn][Aa][Nn]/) {
    fail("non-finite " key " at record " records)
  }
}

(kind == "replay" && $1 == "KFP_IN_PROCESS_SAMPLE") ||
(kind == "host" && $1 == "KFP_HOST_RESULT") {
  ++records
  delete field
  for (i = 2; i <= NF; ++i) {
    separator = index($i, "=")
    if (separator > 1) {
      key = substr($i, 1, separator - 1)
      field[key] = substr($i, separator + 1)
    }
  }
  if (kind == "host") {
    if (field["reference"] != "0" || field["event_local_maps"] != "1" ||
        field["validated_publication_input"] != "1" ||
        field["repeated_track_validation"] != "0" ||
        field["materialization_schedule"] != expected_schedule) {
      fail("record " records " is not the optimized trusted-input contract")
    }
    sample = ((records - 1) % samples) + 1
    for (i = 1; i <= host_key_count; ++i) {
      key = host_keys[i]
      require_number(key, key !~ /_ms$/)
      total[sample, key] += field[key] + 0
    }
    next
  }

  require("sample")
  split(field["sample"], sample_parts, "/")
  sample = sample_parts[1] + 0
  denominator = sample_parts[2] + 0
  if (denominator != samples) {
    fail("sample denominator " denominator " does not match requested " samples)
  }
  if (sample != expected_sample) {
    fail("sample sequence broke at record " records ": expected " expected_sample "/" samples ", got " field["sample"])
  }

  require("mode")
  if (field["mode"] != "optimized") {
    fail("non-optimized sample detected at record " records)
  }
  require("output_mode")
  if (field["output_mode"] != "compact") {
    fail("non-compact publication detected at record " records)
  }
  require_number("events", 1)
  require_number("succeeded", 1)
  require_number("overflow", 1)
  if ((field["events"] + 0) != (field["succeeded"] + 0)) {
    fail("incomplete GPU result at record " records ": events=" field["events"] ", succeeded=" field["succeeded"])
  }
  if ((field["overflow"] + 0) != 0) {
    fail("GPU overflow at record " records ": flags=" field["overflow"])
  }

  identity = ""
  for (i in identity_keys) {
    key = identity_keys[i]
    require(key)
    identity = identity SUBSEP key "=" field[key]
  }
  if (sample == 1) {
    ++batches
    batch_identity = identity
  }
  else if (identity != batch_identity) {
    fail("result identity changed in physical batch " batches " at sample " field["sample"])
  }

  for (i in count_keys) {
    key = count_keys[i]
    require_number(key, 1)
    total[sample, key] += field[key] + 0
  }
  for (i in timing_keys) {
    key = timing_keys[i]
    require_number(key, 0)
    total[sample, key] += field[key] + 0
  }
  for (i in work_keys) {
    key = work_keys[i]
    require_number(key, 1)
    total[sample, key] += field[key] + 0
  }
  for (i in maximum_keys) {
    key = maximum_keys[i]
    require_number(key, 1)
    if ((field[key] + 0) > maximum[sample, key]) {
      maximum[sample, key] = field[key] + 0
    }
  }
  if ((field["materialization_threads_used"] + 0) < 1 || (field["materialization_threads_used"] + 0) > (field["materialization_threads_requested"] + 0)) {
    fail("invalid materialization worker count at record " records)
  }
  if ((field["packing_threads_used"] + 0) < 1 || (field["packing_threads_used"] + 0) > (field["packing_threads_requested"] + 0)) {
    fail("invalid packing worker count at record " records)
  }

  ++expected_sample
  if (expected_sample > samples) { expected_sample = 1 }
}

END {
  if (failed) { exit 1 }
  if (kind == "host") {
    if (records != samples * batches) {
      fail("expected " samples * batches " records, found " records)
    }
    for (sample = 1; sample <= samples; ++sample) {
      printf "KFP_HOST_RESULT_AGGREGATE sample=%d/%d validated_publication_input=1 repeated_track_validation=0 materialization_schedule=%s", sample, samples, expected_schedule
      for (i = 1; i <= host_key_count; ++i) {
        key = host_keys[i]
        printf key ~ /_ms$/ ? " %s=%.9f" : " %s=%.0f", key, total[sample, key]
      }
      printf "\n"
    }
    exit 0
  }
  if (records == 0) { fail("no KFP_IN_PROCESS_SAMPLE records found") }
  if (expected_sample != 1) { fail("last physical batch has an incomplete replay sequence") }

  printf "KFP_REPLAY_AGGREGATE_META contract=1 physical_batches=%d records=%d samples=%d\n", batches, records, samples
  for (sample = 1; sample <= samples; ++sample) {
    printf "KFP_IN_PROCESS_AGGREGATE contract=1 sample=%d/%d mode=optimized output_mode=compact physical_batches=%d", sample, samples, batches
    for (i = 1; i <= count_key_count; ++i) {
      key = count_keys[i]
      printf " %s=%.0f", key, total[sample, key]
    }
    for (i = 1; i <= timing_key_count; ++i) {
      key = timing_keys[i]
      printf " %s=%.9f", key, total[sample, key]
    }
    for (i = 1; i <= maximum_key_count; ++i) {
      key = maximum_keys[i]
      printf " %s=%.0f", key, maximum[sample, key]
    }
    for (i = 1; i <= work_key_count; ++i) {
      key = work_keys[i]
      printf " %s=%.0f", key, total[sample, key]
    }
    printf "\n"
  }
}
