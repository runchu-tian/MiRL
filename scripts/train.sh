#!/usr/bin/env bash
# MiRL training (paper §3.4-3.5, Table 1): GRPO-style RL with open-instruct's grpo_fast and the TMax recipe, rollouts that
# start from the misbehavior prefixes, reward r_o AND r_e. Starts the sandbox service (mirl/train/sandbox_server.py),
# then the trainer: 4 learners (GPUs 0-3, DeepSpeed ZeRO-3, all weights) and 4 vLLM engines (GPUs 4-7).
# Run mirl/train/install.sh once before.
#
# If `sudo -n` works, the sandbox service runs as root, as in our runs, with the sandbox guards of the TMax env.py patch
# (own /dev, read-only /dev and /sys, 8 GB writable layer and 16 GB of RAM per sandbox) and one network per sandbox.
# Otherwise it runs as this user (Apptainer --fakeroot and --userns), on the host network and without these guards,
# which a sandbox in a user namespace does not need.
#
# Usage:
#   bash scripts/train.sh --data <dir with train.jsonl> --tasks-dir <dir>/tasks --out <dir> [options]
#
# Options:
#   --data DIR         directory with train.jsonl from mirl/train/build_rows.py (required)
#   --tasks-dir DIR    TMax task directories built by mirl/profiling/prepare_tasks.sh (required)
#   --out DIR          output directory: checkpoints in <out>/model, logs (required)
#   --steps N          training steps (default: 160)
#   --no-exact         rerun the prefix with the masked match rule of mirl/prefixes/rerun.py (default: exact)
#   --no-net-bridge    share the host network instead of one network per sandbox (as root)
#   --nested           run the sandboxes without --fakeroot and --userns, as in scripts/profile.sh (needs sudo -n)
#
# Environment: TMAX_DIR (TMax checkout at 7387d2f9 with both uv environments), GEMINI_API_KEY (behavior judge; not needed for the
#              ablation rows of build_rows.py --reward outcome).
set -uo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
DATA="" TASKS="" OUT="" STEPS=160 MATCH=exact BRIDGE=1 NESTED=0
while [ $# -gt 0 ]; do
  case $1 in
    --data) DATA=$2; shift 2 ;;  --tasks-dir) TASKS=$2; shift 2 ;;  --out) OUT=$2; shift 2 ;;  --steps) STEPS=$2; shift 2 ;;
    --no-exact) MATCH=masked; shift ;;
    --no-net-bridge) BRIDGE=0; shift ;;
    --nested) NESTED=1; shift ;;
    -h|--help) sed -n '2,25p' "$0"; exit 0 ;;
    *) echo "unknown option: $1 (see --help)" >&2; exit 2 ;;
  esac
done
[ -n "$DATA" ] && [ -n "$TASKS" ] && [ -n "$OUT" ] || { echo "--data, --tasks-dir and --out are required (see --help)" >&2; exit 2; }
: "${TMAX_DIR:?set TMAX_DIR}"
# the key is needed only for rows with the MiRL reward (the ablation rows make no judge call); the quotes are escaped
# because ground_truth is a JSON string inside the row
if grep -qF '\"reward\": \"mirl\"' "$DATA/train.jsonl"; then : "${GEMINI_API_KEY:?set GEMINI_API_KEY}"; fi
SUDO=""; sudo -n true 2>/dev/null && SUDO="sudo -nE"
[ $NESTED = 0 ] || [ -n "$SUDO" ] || { echo "--nested needs sudo -n" >&2; exit 2; }
mkdir -p "$OUT"; OUT=$(cd "$OUT" && pwd); DATA=$(cd "$DATA" && pwd); TASKS=$(cd "$TASKS" && pwd)

# Sandbox service: as root with the sandbox guards and, by default, one network per sandbox; else as this user.
if [ -n "$SUDO" ]; then
  SBX_ENV="APPTAINER_NET_BRIDGE=$BRIDGE APPTAINER_CNI_SLOTS=8 APPTAINER_PRIVATE_DEV=1 APPTAINER_DEV_RO=1 APPTAINER_SYS_RO=1
           APPTAINER_OVERLAY_MB=8192 APPTAINER_MEM_MB=16384"
  [ $NESTED = 1 ] && SBX_ENV="$SBX_ENV APPTAINER_NO_FAKEROOT=1 APPTAINER_NO_USERNS=1"
  LIMIT="prlimit --nofile=1048576:1048576"     # sandbox PTYs run out of the default 1024 open files under sudo
else
  SBX_ENV="APPTAINER_NET_BRIDGE=0 APPTAINER_PRIVATE_DEV=1" LIMIT=""
fi
$SUDO $LIMIT env $SBX_ENV TASKS_DIR="$TASKS" TMAX_DIR="$TMAX_DIR" SANDBOX_PORT=8650 \
  SANDBOX_MATCH=$MATCH START_CONCURRENCY=16 SANDBOX_LOG="$OUT/sandbox_server.jsonl" OPENAI_API_KEY=local PATH="$PATH" \
  "$TMAX_DIR/.venv/bin/python" "$ROOT/mirl/train/sandbox_server.py" > "$OUT/sandbox_server.log" 2>&1 < /dev/null &
SBX=$!
cleanup() {
  curl -s -m 600 -X POST -H "Content-Type: application/json" -d "{}" 127.0.0.1:8650/close_all > /dev/null 2>&1
  $SUDO kill $SBX 2>/dev/null
}
trap cleanup EXIT
for t in $(seq 60); do curl -s -m5 127.0.0.1:8650/stats > /dev/null && break; sleep 2; done
curl -s -m5 127.0.0.1:8650/stats > /dev/null || { echo "the sandbox service did not start (see $OUT/sandbox_server.log)" >&2; exit 1; }

# Trainer: the TMax RL recipe (scripts/tmax/RL/qwen35_4b.sh) with group size 16, 8 prefixes per step and 160 steps
# (Table 1): DPPO (TV, 0.1), no KL, centered advantages, lr 1e-6 constant, temperature 1.0, async 4, active sampling
# (groups where all samples get the same reward are dropped and more prefixes are drawn), at most 64 steps per rollout.
NP=8 G=16
MODEL=Qwen/Qwen3.5-4B REV=851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a
cd "$TMAX_DIR/training/open-instruct"
export VLLM_ALLOW_INSECURE_SERIALIZATION=1 VLLM_DISABLE_COMPILE_CACHE=1 VLLM_USE_V1=1 PYTORCH_ALLOC_CONF=expandable_segments:True \
       SWERL_RESET_FAILURE_ZERO_REWARD=1 PREFIX_NO_TOOL_CALL_AS_TOOL=1 PREFIX_DROP_RESET_FAILED=1 LMHEAD_MASKED_ONLY=1 \
       PREFIX_JUDGE_LOG="$OUT/judge.jsonl" PREFIX_ENV_LOG="$OUT/env.jsonl"
.venv/bin/python open_instruct/grpo_fast.py \
    --dataset_mixer_list "$DATA" 1.0 --dataset_mixer_list_splits train \
    --max_prompt_token_length 49152 --per_turn_max_tokens 16384 --response_length 65536 --pack_length 116736 \
    --per_device_train_batch_size 1 \
    --num_unique_prompts_rollout $NP --num_samples_per_prompt_rollout $G --total_episodes $((NP * G * STEPS)) \
    --async_steps 4 --active_sampling \
    --model_name_or_path $MODEL --model_revision $REV --tokenizer_name_or_path $MODEL --tokenizer_revision $REV \
    --temperature 1.0 --learning_rate 1e-6 --lr_scheduler_type constant \
    --deepspeed_stage 3 --sequence_parallel_size 4 --num_epochs 1 \
    --num_learners_per_node 4 --vllm_num_engines 4 --vllm_tensor_parallel_size 1 --vllm_gpu_memory_utilization 0.85 \
    --vllm_inference_batch_size 64 \
    --beta 0.0 --load_ref_policy false --use_vllm_logprobs true --truncated_importance_sampling_ratio_cap 0.0 --seed 42 \
    --gradient_checkpointing --vllm_enable_prefix_caching --push_to_hub false \
    --save_traces --save_trainer_logprobs false --rollouts_save_path "$OUT/rollouts" \
    --tools prefix_replay_sandbox --tool_configs '{"server_url": "http://127.0.0.1:8650"}' \
    --pool_size $((4 * NP * G)) --max_steps 64 --verification_reward 1.0 --reward_aggregator last --mask_tool_use true \
    --tool_parser_type vllm_qwen3_xml --backend_timeout 3600 \
    --vllm_gdn_prefill_backend triton --inflight_updates true \
    --lm_head_fp32 true --use_liger_grpo_loss --liger_grpo_loss_chunk_size 8 --advantage_normalization_type centered \
    --loss_fn dppo --dppo_divergence_type tv --dppo_divergence_threshold 0.1 \
    --output_dir "$OUT/model" --exp_name mirl_$(basename "$OUT") \
    --save_freq 5 --checkpoint_state_freq 5 --checkpoint_state_dir "$OUT/state" --local_eval_every -1 \
    2>&1 | tee "$OUT/train.log"
