#!/usr/bin/env bash
# Install the MiRL training pieces into a TMax checkout at 7387d2f9 (idempotent):
#   patches/tmax_env.patch            rl_data/generator/env.py: sandbox switches, all off by default (no --fakeroot /
#                                     --userns for nested machines, one network per sandbox, read-only /dev and /sys,
#                                     memory and overlay caps, no host firewall edits, a PTY leak fix)
#   patches/tmax_open_instruct.patch  open-instruct: a sample whose prefix rerun failed leaves its group and the loss;
#                                     a reply without a tool call is answered with a tool message, as in error profiling;
#                                     the lm head runs on the loss tokens only (same loss, less memory); registration of
#                                     the environment and the reward
#   prefix_replay_sandbox.py          -> open_instruct/environments/
#   prefix_judge.py, prompts/behavior_judge.txt -> open_instruct/
# The patched files are first restored from git, so that the env.py patches of mirl/profiling/prepare_tasks.sh --nested
# and mirl/prefixes/net_bridge.py (both contained in patches/tmax_env.patch) do not conflict.
#
# Usage: bash mirl/train/install.sh        (TMAX_DIR = the TMax checkout)
set -euo pipefail
D=$(cd "$(dirname "$0")" && pwd)
: "${TMAX_DIR:?set TMAX_DIR}"
OI=training/open-instruct/open_instruct
FILES="rl_data/generator/env.py $OI/data_loader.py $OI/vllm_utils.py $OI/grpo_utils.py $OI/ground_truth_utils.py
       $OI/environments/__init__.py $OI/environments/tools/tools.py"
cd "$TMAX_DIR"
[ "$(git rev-parse HEAD)" = 7387d2f9142397a458dc39f0827a2ab0b4c03cda ] || { echo "TMAX_DIR is not at 7387d2f9" >&2; exit 1; }
if git apply --reverse --check "$D/patches/tmax_env.patch" "$D/patches/tmax_open_instruct.patch" 2>/dev/null; then
  echo "patches already applied"
else
  git checkout -- $FILES
  git apply "$D/patches/tmax_env.patch" "$D/patches/tmax_open_instruct.patch"
  echo "patches applied"
fi
cp "$D/prefix_replay_sandbox.py" "$OI/environments/prefix_replay_sandbox.py"
cp "$D/prefix_judge.py" "$OI/prefix_judge.py"
cp "$D/prompts/behavior_judge.txt" "$OI/mirl_behavior_judge.txt"
echo "MiRL training files installed into $TMAX_DIR/$OI"
