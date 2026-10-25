#!/usr/bin/env bash
# Error profiling rollouts (paper §3.2): serve the policy and the teacher (error localizer + teacher) with vLLM, then
# run one rollout per environment with mirl/profiling/run_profiling.py. The paper runs this twice on the 2,017
# environments in mirl/profiling/environments.json.
#
# Usage:
#   bash scripts/profile.sh --tasks-dir <dir>/tasks --seed 1 --out <dir> [options]
#
# Options:
#   --tasks-dir DIR    TMax task directories built by mirl/profiling/prepare_tasks.sh (required)
#   --seed N           rollout seed; the policy request seed is 100000 * N + row (required)
#   --out DIR          output directory (required)
#   --envs FILE        environment list (default: mirl/profiling/environments.json)
#   --limit N          only the first N environments, e.g. for a quick test (default: all)
#   --workers N        rollouts at once (default: 48)
#   --policy MODEL     policy (default: Qwen/Qwen3.5-4B)
#   --teacher MODEL    error localizer and teacher (default: Qwen/Qwen3.8-27B)
#   --policy-gpus L    GPUs of the policy server, comma-separated (default: 0,1)
#   --teacher-gpus L   GPUs of the teacher server, comma-separated (default: 2,3,4,5,6,7)
#   --teacher-tp N     tensor parallel size of the teacher (default: 2)
#   --resume           skip environments that already have a trajectory
#   --nested           run the sandboxes without --fakeroot and --userns, for machines that are themselves containers
#                      (needs mirl/profiling/prepare_tasks.sh --nested)
#
# Environment: TMAX_DIR (TMax checkout at 7387d2f9, with its uv environment); VLLM (default: vllm).
# If `sudo -n` works, the rollouts run as root, as in our runs; otherwise they run as this user (Apptainer --fakeroot).
# --nested needs `sudo -n`.
set -uo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
TASKS="" SEED="" OUT="" ENVS=$ROOT/mirl/profiling/environments.json LIMIT=0 WORKERS=48 RESUME="" NESTED=0
POLICY=Qwen/Qwen3.5-4B TEACHER=Qwen/Qwen3.8-27B PGPUS=0,1 TGPUS=2,3,4,5,6,7 TTP=2
while [ $# -gt 0 ]; do
  case $1 in
    --tasks-dir) TASKS=$2; shift 2 ;;  --seed) SEED=$2; shift 2 ;;  --out) OUT=$2; shift 2 ;;
    --envs) ENVS=$2; shift 2 ;;        --limit) LIMIT=$2; shift 2 ;; --workers) WORKERS=$2; shift 2 ;;
    --policy) POLICY=$2; shift 2 ;;    --teacher) TEACHER=$2; shift 2 ;;
    --policy-gpus) PGPUS=$2; shift 2 ;; --teacher-gpus) TGPUS=$2; shift 2 ;; --teacher-tp) TTP=$2; shift 2 ;;
    --resume) RESUME=--resume; shift ;;
    --nested) NESTED=1; shift ;;
    -h|--help) sed -n '2,27p' "$0"; exit 0 ;;
    *) echo "unknown option: $1 (see --help)" >&2; exit 2 ;;
  esac
done
[ -n "$TASKS" ] && [ -n "$SEED" ] && [ -n "$OUT" ] || { echo "--tasks-dir, --seed and --out are required (see --help)" >&2; exit 2; }
: "${TMAX_DIR:?set TMAX_DIR}"
SUDO=""; sudo -n true 2>/dev/null && SUDO="sudo -nE"
[ $NESTED = 0 ] || [ -n "$SUDO" ] || { echo "--nested needs sudo -n" >&2; exit 2; }
VLLM=${VLLM:-vllm}
mkdir -p "$OUT/servers"; OUT=$(cd "$OUT" && pwd)
ndev(){ echo "$1" | tr ',' '\n' | grep -c .; }

# Generation settings of the paper. Policy: Qwen3.5-4B with its stock chat template, temperature 1.0, 65,536-token
# window. Teacher: Qwen3.8-27B, temperature 0.7 (in rollout.py), prefix caching (consecutive judge prompts of one
# rollout share the history). The repetition notice for the judge is off (REPEAT_NOTICE=1 turns it on).
POLICY_FLAGS="--dtype bfloat16 --max-model-len 65536 --language-model-only --enable-auto-tool-choice --tool-call-parser qwen3_xml --gpu-memory-utilization 0.90"
TEACHER_FLAGS="--dtype bfloat16 --max-model-len 65536 --enable-auto-tool-choice --tool-call-parser qwen3_xml --enable-prefix-caching --max-num-batched-tokens 8192 --gpu-memory-utilization 0.90"
export VLLM_HOST_IP=127.0.0.1
PIDS=()
cleanup(){ for p in "${PIDS[@]}"; do kill -- -"$p" 2>/dev/null; done; }
trap cleanup EXIT
CUDA_VISIBLE_DEVICES=$PGPUS setsid "$VLLM" serve "$POLICY" --served-model-name policy --port 8700 \
  -dp "$(ndev "$PGPUS")" -tp 1 $POLICY_FLAGS > "$OUT/servers/vllm_policy.log" 2>&1 < /dev/null &
PIDS+=($!)
CUDA_VISIBLE_DEVICES=$TGPUS setsid "$VLLM" serve "$TEACHER" --served-model-name teacher --port 8800 \
  -dp $(( $(ndev "$TGPUS") / TTP )) -tp "$TTP" $TEACHER_FLAGS > "$OUT/servers/vllm_teacher.log" 2>&1 < /dev/null &
PIDS+=($!)
for s in "8700 policy" "8800 teacher"; do set -- $s
  ok=0; for t in $(seq 160); do curl -s -m5 "127.0.0.1:$1/v1/models" | grep -q "\"$2\"" && { ok=1; break; }; sleep 15; done
  [ $ok = 1 ] || { echo "the $2 server did not start (see $OUT/servers/vllm_$2.log)" >&2; exit 1; }
done
echo "servers up: policy $POLICY on GPUs $PGPUS, teacher $TEACHER on GPUs $TGPUS"

NESTED_ENV=""; [ $NESTED = 1 ] && NESTED_ENV="APPTAINER_NO_FAKEROOT=1 APPTAINER_NO_USERNS=1"
$SUDO env $NESTED_ENV OPENAI_API_KEY=local HOSTED_VLLM_API_KEY=local \
  TMAX_DIR="$TMAX_DIR" ROLLOUT_SEED="$SEED" POLICY_TEMP=1.0 REPEAT_NOTICE=0 PATH="$PATH" \
  "$TMAX_DIR/.venv/bin/python" "$ROOT/mirl/profiling/run_profiling.py" \
  --tasks-dir "$TASKS" --out "$OUT" --envs "$ENVS" --limit "$LIMIT" --workers "$WORKERS" $RESUME \
  2>&1 | tee -a "$OUT/profile.log"
