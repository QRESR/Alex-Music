#!/usr/bin/env bash
# Runs the measurement matrix for the audio-embedding pilot.
set -u
PY=/root/.openclaw/workspace/youtube-music-proxy/.venv/bin/python
CASE=/root/.openclaw/workspace/youtube-music-proxy/research/pilot/embed_case.py
A=/root/.hermes/profiles/gera/cache/scratch/audio-emb
MUSIC=laion/larger_clap_music
UNFUSED=laion/clap-htsat-unfused
RES=/root/.openclaw/workspace/youtube-music-proxy/research/pilot/results.jsonl
OUT=$A/emb
mkdir -p "$OUT"
: > "$RES"

run() { # tag model file mode out extra...
  local tag=$1 model=$2 file=$3 mode=$4 out=$5; shift 5
  local line
  line=$($PY "$CASE" --model "$model" --file "$A/$file" --mode "$mode" --out "$out" "$@" 2>/dev/null | tail -1)
  echo "{\"tag\":\"$tag\", \"case\": $line}" >> "$RES"
  echo "[$tag] $line"
}

# A. 'full' mode twice on the same track -> nondeterminism check (rand_trunc)
run full_a $MUSIC MJ_PIA_Solarity_AcidTechno.ogg full $OUT/techno_full1.npy
run full_b $MUSIC MJ_PIA_Solarity_AcidTechno.ogg full $OUT/techno_full2.npy

# B. excerpt(30s) on remaining tracks
run ex_chopin $MUSIC Chopin_Mazurka_in_A_flat_major_B_85.flac excerpt $OUT/chopin_ex.npy --excerpt 30
run ex_blues  $MUSIC Matt_s_Blues.mp3 excerpt $OUT/blues_ex.npy --excerpt 30
run ex_rock   $MUSIC MokkaMusic_Exotic_Jam.ogg excerpt $OUT/rock_ex.npy --excerpt 30

# C. chunked 10s windows, batch=1 -> production strategy (deterministic, full track)
run ck_chopin $MUSIC Chopin_Mazurka_in_A_flat_major_B_85.flac chunked $OUT/chopin_ck.npy
run ck_blues  $MUSIC Matt_s_Blues.mp3 chunked $OUT/blues_ck.npy
run ck_techno $MUSIC MJ_PIA_Solarity_AcidTechno.ogg chunked $OUT/techno_ck.npy
run ck_rock   $MUSIC MokkaMusic_Exotic_Jam.ogg chunked $OUT/rock_ck.npy

# D. chunked with batch=6 -> throughput optimisation
run ckb_rock  $MUSIC MokkaMusic_Exotic_Jam.ogg chunked $OUT/rock_ckb6.npy --batch 6

# E. re-run chunked on one track -> determinism check
run ck_techno2 $MUSIC MJ_PIA_Solarity_AcidTechno.ogg chunked $OUT/techno_ck2.npy

# F. secondary model (general-audio CLAP)
run uf_techno $UNFUSED MJ_PIA_Solarity_AcidTechno.ogg chunked $OUT/techno_ck_unfused.npy
run uf_blues  $UNFUSED Matt_s_Blues.mp3 chunked $OUT/blues_ck_unfused.npy
run uf_full   $UNFUSED Matt_s_Blues.mp3 full $OUT/blues_full_unfused.npy

echo "DONE"
