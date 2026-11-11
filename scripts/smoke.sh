#!/usr/bin/env bash
# Run the full pipeline with fake models and the real ffmpeg renderer.
# Proves the install, the CLI and ffmpeg work before any model is downloaded.
set -euo pipefail

cd "$(dirname "$0")/.."
workdir="$(mktemp -d)"
trap 'rm -rf "$workdir"' EXIT

run() { python -m shortgen --config configs/smoke.yaml "$@"; }

run doctor
run run examples/story.md --project "$workdir/smoke"
run status "$workdir/smoke"

video="$workdir/smoke/output/final.mp4"
test -s "$video"
ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 "$video" \
  | awk '{ printf "smoke test passed: %.1f s video rendered\n", $1 }'
