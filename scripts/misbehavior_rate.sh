#!/usr/bin/env bash
# Misbehavior rate (paper §4.5) with mirl/analysis/rate.py: sample each policy on the unseen prefixes (one vLLM server
# per GPU), judge the samples with Qwen3.8-27B (vLLM) and Gemini 3.8 Flash (API), and write <out>/rates.md.
#
# Usage:
#   bash scripts/misbehavior_rate.sh --set <unseen.jsonl> --taxonomy <dir> --out <dir> [options]
#
# Options:
#   --set FILE         unseen prefixes from mirl/analysis/unseen.py (required)
#   --taxonomy DIR     output directory of mirl/profiling/taxonomy.py (required)
#   --out DIR          output directory (required)
#   --policies LIST    name=model pairs, space-separated
#                      (default: "qwen3.5-4b=Qwen/Qwen3.5-4B tmax-4b=allenai/tmax-4b mirl=Rtian/MiRL-Qwen3.5-4B")
#   --samples N        samples per prefix (default: 16)
#   --judge-model M    Qwen judge (default: Qwen/Qwen3.8-27B)
#   --judge-tp N       tensor parallel size of the Qwen judge (default: 1)
#   --gpus LIST        GPUs, comma-separated (default: 0,1,2,3,4,5,6,7)
#   --phases LIST      any of sample, qwen, gemini, report, space-separated (default: all, in this order)
#
# Environment: TMAX_DIR (for the python of rate.py, which needs transformers: the TMax training environment, or set PY);
#              VLLM (default: vllm); GEMINI_API_KEY (gemini judge).
set -uo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
SET="" TAX="" OUT="" SAMPLES=16 JUDGE=Qwen/Qwen3.8-27B JTP=1 GPUS=0,1,2,3,4,5,6,7 PHASES="sample qwen gemini report"
POLICIES="qwen3.5-4b=Qwen/Qwen3.5-4B tmax-4b=allenai/tmax-4b mirl=Rtian/MiRL-Qwen3.5-4B"
while [ $# -gt 0 ]; do
  case $1 in
    --set) SET=$2; shift 2 ;;  --taxonomy) TAX=$2; shift 2 ;;  --out) OUT=$2; shift 2 ;;
    --policies) POLICIES=$2; shift 2 ;;  --samples) SAMPLES=$2; shift 2 ;;
    --judge-model) JUDGE=$2; shift 2 ;;  --judge-tp) JTP=$2; shift 2 ;;  --gpus) GPUS=$2; shift 2 ;;
    --phases) PHASES=$2; shift 2 ;;
    -h|--help) sed -n '2,21p' "$0"; exit 0 ;;
    *) echo "unknown option: $1 (see --help)" >&2; exit 2 ;;
  esac
done
[ -n "$SET" ] && [ -n "$TAX" ] && [ -n "$OUT" ] || { echo "--set, --taxonomy and --out are required (see --help)" >&2; exit 2; }
PY=${PY:-${TMAX_DIR:?set TMAX_DIR or PY}/training/open-instruct/.venv/bin/python} VLLM=${VLLM:-vllm}
mkdir -p "$OUT/servers"
IFS=, read -ra G <<< "$GPUS"
PIDS=() URLS=""
serve(){  # serve <model> <served name> <tp> <first port>: one server per <tp> GPUs -> URLS
  local model=$1 name=$2 tp=$3 port=$4 i=0 g
  URLS="" PIDS=()
  for ((g = 0; g + tp <= ${#G[@]}; g += tp)); do
    local devs=$(IFS=,; echo "${G[*]:$g:$tp}") p=$((port + i))
    CUDA_VISIBLE_DEVICES=$devs setsid "$VLLM" serve "$model" --served-model-name $name \
      --port $p --tensor-parallel-size $tp --dtype bfloat16 --max-model-len 65536 --language-model-only \
      --gpu-memory-utilization 0.90 --enable-prefix-caching > "$OUT/servers/vllm_${name}_$p.log" 2>&1 < /dev/null &
    PIDS+=($!); URLS+="${URLS:+,}http://127.0.0.1:$p"; i=$((i + 1))
  done
  for u in ${URLS//,/ }; do
    for t in $(seq 180); do curl -sf $u/health > /dev/null && break; sleep 10
      [ $t = 180 ] && { echo "server $u did not start (see $OUT/servers)" >&2; stop; exit 1; }; done
  done
}
stop(){ for p in "${PIDS[@]}"; do kill -- -$p 2>/dev/null; done; sleep 15
        for p in "${PIDS[@]}"; do kill -9 -- -$p 2>/dev/null; done; PIDS=(); }
trap stop EXIT
R="$PY $ROOT/mirl/analysis/rate.py"
for ph in $PHASES; do case $ph in
  sample) for pm in $POLICIES; do
            serve "${pm#*=}" policy 1 8700
            $R sample --set "$SET" --out "$OUT" --name "${pm%%=*}" --model "${pm#*=}" --urls "$URLS" --samples $SAMPLES
            stop
          done ;;
  qwen)   serve "$JUDGE" judge $JTP 8800
          for pm in $POLICIES; do $R judge --set "$SET" --out "$OUT" --taxonomy "$TAX" --name "${pm%%=*}" --judge qwen --urls "$URLS"; done
          stop ;;
  gemini) for pm in $POLICIES; do $R judge --set "$SET" --out "$OUT" --taxonomy "$TAX" --name "${pm%%=*}" --judge gemini; done ;;
  report) $R report --out "$OUT" ;;
  *) echo "unknown phase: $ph" >&2; exit 2 ;;
esac; done
