#!/usr/bin/env bash
# Build one Apptainer sandbox directory per TB-Lite / TB-2.1 task for apptainer_env.py (directories, not .sif files:
# some kernels cannot mount squashfs). TB-Lite: `docker build` of the task's environment/ dir, as harbor's Docker
# environment does, tagged aeval/tblite__<task>:v1. TB-2.1: the task's prebuilt docker_image (pulled if missing). Then
# `apptainer build --sandbox <dir> docker-daemon://<tag>`, plus <dir>.json = `docker inspect` Config (WorkingDir, User,
# Env), which apptainer_env.py uses for the cwd, user and environment of each command. Tasks already built are skipped.
#
# Usage:  bash mirl/eval/build_sandboxes.sh [tblite|tb21|all]
# Environment: TBLITE_DIR, TB21_DIR (from download_benchmarks.sh), APPTAINER_EVAL_DIR (work dir), PAR (parallel
#              builds, default 8). Needs Docker. If `sudo -n` works, apptainer builds as root, as in the paper; otherwise
#              it builds as this user (then use APPTAINER_EVAL_ROOTLESS=1 for the evaluation, see apptainer_env.py).
# Output: $APPTAINER_EVAL_DIR/sbx/{tblite,tb21}/<task>/ ; log: $APPTAINER_EVAL_DIR/logs/build.log (one OK/FAIL line
#         per task). In the paper, all 189 tasks built with no failure (about 105 GB in total).
SUITE=${1:-all}; PAR=${PAR:-8}
: "${APPTAINER_EVAL_DIR:?set APPTAINER_EVAL_DIR}"
W=$APPTAINER_EVAL_DIR; mkdir -p "$W/sbx/tblite" "$W/sbx/tb21" "$W/logs" "$W/tmp"
[ "$SUITE" = tb21 ] || : "${TBLITE_DIR:?set TBLITE_DIR}"
[ "$SUITE" = tblite ] || : "${TB21_DIR:?set TB21_DIR}"
SUDO=""; sudo -n true 2>/dev/null && SUDO="sudo -n"
# without sudo the sandbox belongs to this user, and some image dirs have no write permission: add it before rm
rmtree(){ [ -n "$SUDO" ] || { [ -e "$1" ] && chmod -R u+rwX "$1" 2>/dev/null; }; $SUDO rm -rf "$1"; }
one(){ suite=$1; d=${2%/}; t=$(basename "$d"); out=$W/sbx/$suite/$t
  [ -d "$out" ] && [ -s "$out.json" ] && { echo "SKIP $suite/$t"; return 0; }
  t0=$(date +%s)
  if [ "$suite" = tblite ]; then tag=aeval/tblite__$(echo "$t" | tr 'A-Z' 'a-z'):v1
    docker build -q -t "$tag" "$d/environment" > "$W/logs/${suite}__$t.build" 2>&1 \
      || { echo "FAIL $suite/$t docker build ($(tail -c 200 "$W/logs/${suite}__$t.build" | tr '\n' ' '))"; return 1; }
  else tag=$(grep -oP '^docker_image\s*=\s*"\K[^"]+' "$d/task.toml")
    docker image inspect "$tag" > /dev/null 2>&1 || docker pull -q "$tag" > "$W/logs/${suite}__$t.pull" 2>&1 \
      || { echo "FAIL $suite/$t docker pull $tag ($(tail -c 200 "$W/logs/${suite}__$t.pull" | tr '\n' ' '))"; return 1; }
  fi
  docker inspect --format '{{json .Config}}' "$tag" > "$out.json.tmp" 2>/dev/null || { echo "FAIL $suite/$t inspect $tag"; return 1; }
  rmtree "$out.tmp"
  $SUDO env APPTAINER_TMPDIR="$W/tmp" APPTAINER_CACHEDIR="$W/tmp/cache" apptainer build -F --sandbox "$out.tmp" "docker-daemon://$tag" \
    > "$W/logs/${suite}__$t.sif.log" 2>&1 \
    || { echo "FAIL $suite/$t apptainer build ($(tail -c 200 "$W/logs/${suite}__$t.sif.log" | tr '\n' ' '))"; rmtree "$out.tmp"; rm -f "$out.json.tmp"; return 1; }
  $SUDO mv "$out.tmp" "$out"; mv "$out.json.tmp" "$out.json"
  echo "OK $suite/$t $(( $(date +%s) - t0 ))s $($SUDO du -sh "$out" | cut -f1)"; }
export -f one rmtree; export W SUDO
{ [ "$SUITE" = tb21 ] || for d in "$TBLITE_DIR"/*/; do echo "tblite $d"; done
  [ "$SUITE" = tblite ] || for d in "$TB21_DIR"/*/; do echo "tb21 $d"; done; } | xargs -P "$PAR" -L 1 bash -c 'one $0 $1' >> "$W/logs/build.log" 2>&1
echo "BUILD DONE $(date '+%F %T')" >> "$W/logs/build.log"
echo "built: $(grep -c '^OK' "$W/logs/build.log") ok, $(grep -c '^FAIL' "$W/logs/build.log") failed (see $W/logs/build.log)"
