#!/usr/bin/env bash
# Evaluate one model on TB-Lite or TB-2.1 with several runs in parallel, as in the paper: one vLLM server per GPU
# (1 GPU each) and one run of <tasks per GPU> tasks at a time per server. Each GPU takes the next run that no other GPU
# has taken. A run that already has score.json is skipped.
#
# Usage:
#   bash scripts/eval.sh --model <path or HF id> --bench <tblite|tb21> [options]
#
# Options:
#   --model PATH     model checkpoint, a local directory or a Hugging Face id (required)
#   --bench NAME     tblite (100 tasks) or tb21 (89 tasks) (required)
#   --name NAME      served model name and prefix of the run names (default: basename of --model)
#   --runs N         number of runs (seeds) (default: 3)
#   --gpus LIST      GPUs to use, comma-separated (default: 0,1,2,3,4,5,6,7)
#   --nc N           tasks at once per GPU, i.e. harbor --n-concurrent per server (default: 8; see the warning in
#                    mirl/eval/run_eval.sh before you raise it)
#   --out DIR        output directory (default: runs)
#   --port N         port of the server on the first GPU; GPU k uses port + k (default: 8700)
#
# Environment: TMAX_DIR, TBLITE_DIR, TB21_DIR, APPTAINER_EVAL_DIR (see mirl/eval/run_eval.sh); VLLM (default: vllm).
# Results: <out>/<name>_<bench>_run<k>/score.json and score.txt; log: <out>/eval_<name>_<bench>.log
set -uo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
MODEL="" BENCH="" NAME="" RUNS=3 GPUS="0,1,2,3,4,5,6,7" NC=8 OUT=runs PORT=8700
while [ $# -gt 0 ]; do
  case $1 in
    --model) MODEL=$2; shift 2 ;;  --bench) BENCH=$2; shift 2 ;;  --name) NAME=$2; shift 2 ;;
    --runs) RUNS=$2; shift 2 ;;    --gpus) GPUS=$2; shift 2 ;;    --nc) NC=$2; shift 2 ;;
    --out) OUT=$2; shift 2 ;;      --port) PORT=$2; shift 2 ;;
    -h|--help) sed -n '2,22p' "$0"; exit 0 ;;
    *) echo "unknown option: $1 (see --help)" >&2; exit 2 ;;
  esac
done
[ -n "$MODEL" ] && [ -n "$BENCH" ] || { echo "--model and --bench are required (see --help)" >&2; exit 2; }
case $BENCH in tblite|tb21) ;; *) echo "--bench must be tblite or tb21" >&2; exit 2 ;; esac
NAME=${NAME:-$(basename "$MODEL")}
VLLM=${VLLM:-vllm}
mkdir -p "$OUT"; OUT=$(cd "$OUT" && pwd)
LOG=$OUT/eval_${NAME}_${BENCH}.log CLAIMS=$OUT/.claims_${NAME}_${BENCH}
rm -rf "$CLAIMS"                                             # claims of an earlier, stopped call would skip those runs
mkdir -p "$CLAIMS" "$OUT/servers"
# vLLM settings used in the paper (one GPU per server)
FLAGS="--dtype bfloat16 --max-model-len 262144 --language-model-only --enable-auto-tool-choice --tool-call-parser qwen3_xml --enable-prefix-caching"
say(){ echo "$(date '+%F %T') $*" | tee -a "$LOG"; }

worker(){ local g=$1 port=$2 sp=""
  stop_server(){ [ -n "$sp" ] && { kill -- -"$sp" 2>/dev/null; sleep 20; kill -9 -- -"$sp" 2>/dev/null; }; sp=""; }
  trap stop_server EXIT
  for k in $(seq 1 "$RUNS"); do
    local run=${NAME}_${BENCH}_run$k
    [ -f "$OUT/$run/score.json" ] && continue
    mkdir "$CLAIMS/$run" 2>/dev/null || continue            # taken by another GPU
    if [ -z "$sp" ]; then
      CUDA_VISIBLE_DEVICES=$g setsid "$VLLM" serve "$MODEL" --served-model-name "$NAME" --port "$port" $FLAGS \
        > "$OUT/servers/vllm_gpu$g.log" 2>&1 < /dev/null &
      sp=$!
      local ok=0; for t in $(seq 150); do curl -s -m5 "localhost:$port/v1/models" | grep -q "\"$NAME\"" && { ok=1; break; }; sleep 8; done
      [ $ok = 1 ] || { say "GPU $g: the vLLM server did not start (see $OUT/servers/vllm_gpu$g.log); $run given back"; rmdir "$CLAIMS/$run"; stop_server; return 1; }
      say "GPU $g: server up on port $port"
    fi
    local t0=$(date +%s)
    say "GPU $g: start $run ($NC tasks at once)"
    if MIRL_EVAL_NO_PAUSE=1 bash "$ROOT/mirl/eval/run_eval.sh" "$NAME" "http://localhost:$port/v1" "$BENCH" "$NC" "$OUT/$run" > /dev/null 2>&1; then
      say "GPU $g: $run: $(head -1 "$OUT/$run/score.txt" 2>/dev/null)"
    else
      say "GPU $g: $run FAILED: harbor produced no results, no score written (see $OUT/$run.log)"
    fi
    echo $(( $(date +%s) - t0 )) > "$OUT/$run/wall_seconds"
  done
  stop_server; }

# print the concurrency warning once, then start one worker per GPU
MIRL_EVAL_NO_PAUSE=${MIRL_EVAL_NO_PAUSE:-0} bash "$ROOT/mirl/eval/run_eval.sh" --warning-only "$NC"
say "start: $MODEL on $BENCH, $RUNS runs, GPUs $GPUS, $NC tasks per GPU"
i=0; PIDS=()
for g in ${GPUS//,/ }; do worker "$g" $((PORT + i)) & PIDS+=($!); i=$((i + 1)); sleep 3; done
for p in "${PIDS[@]}"; do wait "$p"; done
say "done: $(ls -d "$OUT"/${NAME}_${BENCH}_run*/score.json 2>/dev/null | wc -l) of $RUNS runs have a score"
python3 - "$OUT" "$NAME" "$BENCH" <<'PY'
import glob, json, statistics, sys
out, name, bench = sys.argv[1:]
s = [json.load(open(p)) for p in sorted(glob.glob(f"{out}/{name}_{bench}_run*/score.json"))]
if s:
    x = [r["score"] for r in s]
    se = statistics.stdev(x) / len(x) ** 0.5 if len(x) > 1 else 0.0
    print(f"{name} {bench}: {statistics.mean(x):.2f} +- {se:.2f} (mean +- SE over {len(x)} runs); "
          f"runs over the load limit: {sum(not r['load_ok'] for r in s)}")
PY
