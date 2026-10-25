#!/usr/bin/env bash
# One TB-Lite or TB-2.1 run of a model served by vLLM, with harbor + the TMax agent (Vanillux2Agent) and our Apptainer
# sandbox backend (apptainer_env.py). The vLLM server is started by the caller (see scripts/eval.sh).
#
# Usage:  mirl/eval/run_eval.sh <served_model_name> <api_base> <tblite|tb21> <n_concurrent> <out_dir>
#   e.g.  mirl/eval/run_eval.sh mirl http://localhost:8700/v1 tb21 8 runs/mirl_tb21_run1
#
# Environment:
#   TMAX_DIR            checkout of github.com/hamishivi/tmax at 7387d2f9, with its uv environment (harbor 0.6.6)
#   TBLITE_DIR          harbor dataset dir of TB-Lite (100 tasks)
#   TB21_DIR            harbor dataset dir of TB-2.1 (89 tasks)
#   APPTAINER_EVAL_DIR  work dir with the task sandboxes built by build_sandboxes.sh
#   APPTAINER_EVAL_ROOTLESS  1 = rootless sandboxes (no sudo); starts sandbox_guard.py (see apptainer_env.py)
#   K                   seeds in one harbor job (default 1; the paper runs separate jobs, one per seed)
#   MIRL_EVAL_NO_PAUSE  set to 1 to skip the 10 s pause after the warning
#
# Output: <out_dir>/ (harbor job), <out_dir>/score.json and score.txt (from score.py).
#         mirl/eval/run_eval.sh --warning-only <n_concurrent>   (only print the warning below; used by scripts/eval.sh)
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)

# Evaluation settings of the paper (Table 1). These are the defaults of Vanillux2Agent in TMax; we pass them
# explicitly so that they are visible here and easy to change.
TEMPERATURE=0.7          # sampling temperature
TOP_P=0.95               # top-p
MAX_STEPS=64             # agent steps per task
MAX_TOKENS=16384         # output tokens per step
COMMAND_TIMEOUT=120      # seconds per command
MAX_FORMAT_ERRORS=64     # steps without a valid tool call before the run stops

warning(){ cat >&2 <<EOF
================================================================================
WARNING: the number of tasks that run at the same time changes the score.

Each task has a time limit for the whole agent run, and each command has a limit
of 120 s. If too many tasks run at once, each sandbox gets less CPU and each model
step takes longer. More tasks then time out, and a task that times out scores 0.
The score goes down, and the cause is the machine, not the model.

Setting used in the paper: one vLLM server per GPU (1 GPU each) and at most
8 tasks at once per server (--n-concurrent 8), on a machine with 96 CPU threads
for 8 GPUs. Use fewer tasks per server if your machine has fewer CPU threads.

After the run, check the numbers that score.py prints:
  - time-outs (AgentTimeoutError) per run: 24 or fewer
  - model time per agent step: 15 s or less
If a run is above these limits, lower <n_concurrent> and run it again.

This run: --n-concurrent $1
================================================================================
EOF
[ "${MIRL_EVAL_NO_PAUSE:-0}" = 1 ] || sleep 10; }

if [ "${1:-}" = --warning-only ]; then warning "${2:-?}"; exit 0; fi
[ $# -eq 5 ] || { sed -n '5,6p' "$0"; exit 2; }
M=$1 API=$2 SUITE=$3 NC=$4 O=$5
warning "$NC"

case $SUITE in
  tblite) DS=${TBLITE_DIR:?set TBLITE_DIR}; N=100 ;;
  tb21)   DS=${TB21_DIR:?set TB21_DIR};     N=89 ;;
  *) echo "suite must be tblite or tb21" >&2; exit 2 ;;
esac
: "${TMAX_DIR:?set TMAX_DIR}" "${APPTAINER_EVAL_DIR:?set APPTAINER_EVAL_DIR}"

export OPENAI_API_KEY=local HOSTED_VLLM_API_KEY=local
export PYTHONPATH=$HERE${PYTHONPATH:+:$PYTHONPATH}
O=$(mkdir -p "$O" && cd "$O" && pwd)
rm -rf "$O"/*
if [ "${APPTAINER_EVAL_ROOTLESS:-0}" = 1 ]; then   # rootless mode: user-level process and memory guards (one per machine)
  setsid nohup flock -n "$APPTAINER_EVAL_DIR/sandbox_guard.lock" python3 "$HERE/sandbox_guard.py" \
    >> "$APPTAINER_EVAL_DIR/sandbox_guard.out" 2>&1 < /dev/null &
fi
cd "$TMAX_DIR"
uv run harbor run --path "$DS" --model "hosted_vllm/$M" \
  --environment-import-path apptainer_env:ApptainerRolloutEnvironment \
  --agent-import-path Vanillux2Agent:Vanillux2Agent \
  --agent-kwarg "api_base=$API" --agent-kwarg "max_format_errors=$MAX_FORMAT_ERRORS" \
  --agent-kwarg "temperature=$TEMPERATURE" --agent-kwarg "top_p=$TOP_P" --agent-kwarg "max_steps=$MAX_STEPS" \
  --agent-kwarg "max_tokens=$MAX_TOKENS" --agent-kwarg "command_timeout=$COMMAND_TIMEOUT" \
  --n-concurrent "$NC" -k "${K:-1}" -y -o "$O" > "${O%/}.log" 2>&1
rc=$?

# A harbor crash leaves no trial results; then write no score.json, so that the run is not averaged in as 0.
python3 "$HERE/score.py" --tasks "$N" --json "$O" > "$O/score.json.tmp"
trials=$(python3 -c 'import json, sys; print(json.load(open(sys.argv[1]))["trials"])' "$O/score.json.tmp" 2>/dev/null || echo 0)
if [ "$trials" = 0 ]; then
  rm -f "$O/score.json.tmp"
  echo "ERROR: harbor produced no trial results (exit code $rc); no score written. See ${O%/}.log" >&2
  exit 1
fi
[ $rc -eq 0 ] || echo "WARNING: harbor exited with code $rc; the score uses the $trials trial results it wrote. See ${O%/}.log" >&2
mv "$O/score.json.tmp" "$O/score.json"
python3 "$HERE/score.py" --tasks "$N" "$O" | tee "$O/score.txt"
