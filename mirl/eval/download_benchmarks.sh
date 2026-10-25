#!/usr/bin/env bash
# Download TB-Lite and TB-2.1 with harbor, and check that the task sets are the ones we evaluated on.
#
# Usage:  TMAX_DIR=/path/to/tmax bash mirl/eval/download_benchmarks.sh <out_dir>
# Then:   export TBLITE_DIR=<out_dir>/openthoughts-tblite/openthoughts-tblite
#         export TB21_DIR=<out_dir>/terminal-bench-2-1/terminal-bench-2-1
#
# The check hashes every task.toml and environment/Dockerfile. TB-2.1 on the harbor hub can change over time; we used
# the snapshot of 2026-09-17. If a hash differs, the scores may differ from the paper.
set -uo pipefail
[ $# -eq 1 ] || { sed -n '4,6p' "$0"; exit 2; }
: "${TMAX_DIR:?set TMAX_DIR}"
OUT=$(mkdir -p "$1" && cd "$1" && pwd)
TBLITE_SHA=9236647053faa54df5e9453ca86abfbffefeb0ab82277f9c865139c6b46b70ba
TB21_SHA=8c39ada8bf9ecff15953f5628062b94dbcacb3ae2675d5d313852d9678e87877

taskset_sha(){ python3 - "$1" <<'PY'
import glob, hashlib, os, sys
d = sys.argv[1]; h = hashlib.sha256()
for p in sorted(glob.glob(d + "/*/task.toml")) + sorted(glob.glob(d + "/*/environment/Dockerfile")):
    h.update(os.path.relpath(p, d).encode()); h.update(open(p, "rb").read())
print(h.hexdigest())
PY
}

check(){ local name=$1 dir=$2 want=$3 n
  n=$(find "$dir" -name task.toml 2>/dev/null | wc -l | tr -d ' ')
  if [ "$(taskset_sha "$dir")" = "$want" ]; then echo "$name: $n tasks, same task set as in the paper"
  else echo "WARNING: $name: $n tasks, the task set differs from the one in the paper; scores may differ"; fi; }

cd "$TMAX_DIR"
[ -d "$OUT/openthoughts-tblite/openthoughts-tblite" ] || uv run harbor datasets download openthoughts-tblite@2.0 -o "$OUT/openthoughts-tblite"
[ -d "$OUT/terminal-bench-2-1/terminal-bench-2-1" ] || uv run harbor datasets download terminal-bench/terminal-bench-2-1 -o "$OUT/terminal-bench-2-1"
check TB-Lite "$OUT/openthoughts-tblite/openthoughts-tblite" $TBLITE_SHA
check TB-2.1 "$OUT/terminal-bench-2-1/terminal-bench-2-1" $TB21_SHA
echo "export TBLITE_DIR=$OUT/openthoughts-tblite/openthoughts-tblite"
echo "export TB21_DIR=$OUT/terminal-bench-2-1/terminal-bench-2-1"
