#!/usr/bin/env bash
# Prepare the TMax environments for error profiling: download the task pack, unpack it, and build one Apptainer
# sandbox directory per environment in the list (<task>/container.sif, a directory: some kernels cannot mount squashfs).
#
# Usage:  bash mirl/profiling/prepare_tasks.sh <dir> [--envs environments.json] [--nested]
# Output: <dir>/tasks/<task>/ (task.json, container.def, tests, container.sif/); log: <dir>/build.log
# Environment: PAR (parallel builds, default 24); TMAX_DIR (with --nested). Needs `hf` (huggingface_hub CLI), unzip and
#              apptainer. If `sudo -n` works, apptainer runs as root, as in our runs; otherwise it runs as this user
#              (unprivileged builds use --fakeroot). --nested needs `sudo -n`.
#
# --nested: use it when the machine is itself a container, where Apptainer's --fakeroot and --userns cannot mount /proc
# (this was our case). It patches TMax's rl_data/generator/env.py so that its sandboxes run without the two flags when
# APPTAINER_NO_FAKEROOT=1 and APPTAINER_NO_USERNS=1 are set (scripts/profile.sh --nested sets them). Without the two
# variables env.py behaves as before. On a normal machine, leave it out.
#
# Two changes to the task pack, as in our runs (the task content is unchanged):
#   - container.def files point to fixture files under a /gpfs/... path of the TMax authors; they are repointed to
#     <dir>/tasks/.
#   - every container.def starts from ubuntu:22.04 on Docker Hub. To avoid Docker Hub's pull limit, ubuntu:22.04 is
#     pulled once as a local sandbox and the defs use it (Bootstrap: localimage). The originals are kept as
#     container.def.dockerboot.
set -uo pipefail
[ $# -ge 1 ] || { sed -n '5,6p' "$0"; exit 2; }
DIR=$(mkdir -p "$1" && cd "$1" && pwd); shift
ENVS=$(cd "$(dirname "$0")" && pwd)/environments.json NESTED=0
while [ $# -gt 0 ]; do
  case $1 in --envs) ENVS=$2; shift 2 ;; --nested) NESTED=1; shift ;; *) echo "unknown option: $1" >&2; exit 2 ;; esac
done
PAR=${PAR:-24}
SUDO=""; sudo -n true 2>/dev/null && SUDO="sudo -nE"
[ $NESTED = 0 ] || [ -n "$SUDO" ] || { echo "--nested needs sudo -n" >&2; exit 2; }
T=$DIR/tasks

if [ $NESTED = 1 ]; then
  : "${TMAX_DIR:?set TMAX_DIR for --nested}"
  python3 - "$TMAX_DIR" <<'PY'
import sys
from pathlib import Path
p = Path(sys.argv[1]) / "rl_data" / "generator" / "env.py"
s = p.read_text()
if "_userns_flags" in s:
    print("env.py already patched")
    sys.exit(0)
old = '''    if os.environ.get("APPTAINER_IGNORE_FAKEROOT_COMMAND", "0") == "1":
        return ["--fakeroot", "--ignore-fakeroot-command"]
    return ["--fakeroot"]'''
new = '''    if os.environ.get("APPTAINER_NO_FAKEROOT", "0") == "1":
        return []
    if os.environ.get("APPTAINER_IGNORE_FAKEROOT_COMMAND", "0") == "1":
        return ["--fakeroot", "--ignore-fakeroot-command"]
    return ["--fakeroot"]


def _userns_flags():
    if os.environ.get("APPTAINER_NO_USERNS", "0") == "1":
        return []
    return ["--userns"]'''
if old not in s:
    sys.exit("env.py does not look like TMax at 7387d2f9; not patched")
s = s.replace(old, new)
s = s.replace('*_fakeroot_flags(), "--userns",', '*_fakeroot_flags(), *_userns_flags(),')
s = s.replace('            *_fakeroot_flags(),\n            "--userns",', '            *_fakeroot_flags(),\n            *_userns_flags(),')
p.write_text(s)
print("env.py patched")
PY
fi

if [ ! -d "$T" ]; then
  hf download allenai/TMax-SFT-16.5K tasks.zip --repo-type dataset --local-dir "$DIR/dl" > /dev/null
  unzip -q -o "$DIR/dl/tasks.zip" -d "$T"
  sed -i "s#/gpfs/scrubbed/osey/tmax/rl_data/output/tasks_skill_tax_v2_20260505_1k/#$T/#g" "$T"/*/container.def
fi

B=$DIR/base_ubuntu_22.04.sandbox
if [ ! -d "$B" ]; then
  (cd /tmp && $SUDO apptainer pull "$DIR/base_ubuntu_22.04.sif" docker://ubuntu:22.04)
  $SUDO apptainer build --force --sandbox "$B" "$DIR/base_ubuntu_22.04.sif"
fi
for f in "$T"/*/container.def; do [ -f "$f.dockerboot" ] || cp "$f" "$f.dockerboot"; done
sed -i "s#^Bootstrap: docker#Bootstrap: localimage#; s#^From: ubuntu:22.04#From: $B#" "$T"/*/container.def

python3 - "$ENVS" > "$DIR/todo.txt" <<'PY'
import json, sys
print("\n".join(sorted({e["task"] for e in json.load(open(sys.argv[1]))})))
PY
build_one(){ t=$1; d=$T/$t
  [ -d "$d/container.sif" ] && { echo "SKIP $t"; return 0; }
  if $SUDO apptainer build --force --sandbox "$d/container.sif" "$d/container.def" > "$DIR/logs/$t.log" 2>&1; then
    echo "OK $t"
  else
    echo "FAIL $t"; $SUDO rm -rf "$d/container.sif"
  fi; }
export -f build_one; export T DIR SUDO
mkdir -p "$DIR/logs"
xargs -P "$PAR" -I{} bash -c 'build_one {}' < "$DIR/todo.txt" >> "$DIR/build.log" 2>&1
echo "built: $(grep -c '^OK' "$DIR/build.log") ok, $(grep -c '^FAIL' "$DIR/build.log") failed (see $DIR/build.log)"
