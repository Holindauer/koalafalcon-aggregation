#!/usr/bin/env bash

set -euo pipefail

export RUSTFLAGS="-C target-cpu=native"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$ROOT"

LOG="koalafalcon/scripts/koalafalcon-benchmarks-$(date +%Y%m%d-%H%M%S).log"
BIN="$ROOT/target/opt-runtime/examples/signature"
NS=(512 1024)
SIGNATURES=131072

if command -v nproc >/dev/null 2>&1; then
  HW_THREADS="$(nproc)"
elif command -v sysctl >/dev/null 2>&1; then
  HW_THREADS="$(sysctl -n hw.ncpu)"
else
  HW_THREADS=1
fi

{

echo
echo "--------------------------------"
echo "Running signature benchmarks"
echo "  crate=$ROOT"
echo "  profile=opt-runtime  RUSTFLAGS=$RUSTFLAGS"
echo "  hw_threads=$HW_THREADS"
echo "  n=${NS[*]}"
echo "--------------------------------"

echo
echo "Building signature..."
cargo build -p koalafalcon --profile opt-runtime --example signature

for n in "${NS[@]}"; do
  echo
  echo "================================"
  echo "Falcon-$n"
  echo "================================"

  echo
  echo "single-threaded signature performance (n=$n):"
  "$BIN" --n "$n" --threads 1 --signatures "$SIGNATURES"

  echo
  echo "multi-threaded signature performance (n=$n):"
  "$BIN" --n "$n" --threads "$HW_THREADS" --signatures "$SIGNATURES"
done

} 2>&1 | tee "$LOG"
