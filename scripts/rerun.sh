#!/usr/bin/env bash
# Rerun check of the misbehavior prefixes (paper §3.4) with mirl/prefixes/rerun.py. Each sandbox gets its own network
# (Apptainer bridge network), so that parallel sandboxes cannot collide on ports; this needs root (`sudo -n`) and patches
# TMax's env.py once (mirl/prefixes/net_bridge.py). If the bridge network does not work on your machine, add
# --no-net-bridge; the sandboxes then share the host network, and some prefixes may fail the check because of port
# collisions (lower --workers to reduce them).
#
# Usage:
#   bash scripts/rerun.sh --prefixes <dir>/prefixes.jsonl --tasks-dir <dir>/tasks --out <dir> [options]
#
# Options:
#   --prefixes FILE    prefixes from mirl/prefixes/sample.py (required)
#   --tasks-dir DIR    TMax task directories built by mirl/profiling/prepare_tasks.sh (required)
#   --out DIR          output directory (required)
#   --tries N          tries per prefix (default: 4)
#   --min-pass N       keep a prefix if at least this many tries pass (default: 2)
#   --no-exact         compare outputs after masking times, PIDs, addresses and similar parts (default: exact match)
#   --workers N        prefixes checked at once (default: 24)
#   --no-net-bridge    share the host network instead of one network per sandbox
#   --nested           run the sandboxes without --fakeroot and --userns, as in scripts/profile.sh
#
# Environment: TMAX_DIR (TMax checkout at 7387d2f9, with its uv environment).
set -uo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
ARGS=() BRIDGE=1 NESTED=0
while [ $# -gt 0 ]; do
  case $1 in
    --prefixes|--tasks-dir|--out|--tries|--min-pass|--workers) ARGS+=("$1" "$2"); shift 2 ;;
    --no-exact) ARGS+=("$1"); shift ;;
    --no-net-bridge) BRIDGE=0; shift ;;
    --nested) NESTED=1; shift ;;
    -h|--help) sed -n '2,22p' "$0"; exit 0 ;;
    *) echo "unknown option: $1 (see --help)" >&2; exit 2 ;;
  esac
done
: "${TMAX_DIR:?set TMAX_DIR}"
SUDO=""; sudo -n true 2>/dev/null && SUDO="sudo -nE"
[ $BRIDGE = 0 ] || [ -n "$SUDO" ] || { echo "the bridge network needs sudo -n; add --no-net-bridge to run without it" >&2; exit 2; }
[ $NESTED = 0 ] || [ -n "$SUDO" ] || { echo "--nested needs sudo -n" >&2; exit 2; }

ENVS="APPTAINER_NET_BRIDGE=$BRIDGE"
[ $BRIDGE = 1 ] && python3 "$ROOT/mirl/prefixes/net_bridge.py" "$TMAX_DIR"
[ $NESTED = 1 ] && ENVS="$ENVS APPTAINER_NO_FAKEROOT=1 APPTAINER_NO_USERNS=1"
$SUDO env $ENVS TMAX_DIR="$TMAX_DIR" PATH="$PATH" \
  "$TMAX_DIR/.venv/bin/python" "$ROOT/mirl/prefixes/rerun.py" "${ARGS[@]}"
