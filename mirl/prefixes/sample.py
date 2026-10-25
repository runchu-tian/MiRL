"""Sample misbehavior prefixes (paper §3.4): for each misbehavior, N random error steps of that type, each with its
prefix rebuilt from the error-profiling output.

A misbehavior prefix is the prefix of an error step: what the policy saw before it proposed the step (system prompt,
task, earlier steps and their outputs). For each sampled step we write
  messages  the message list the policy saw, taken from the rollout's traj file
  replay    the executed commands of the prefix with their recorded outputs, in order, for the rerun check (rerun.py)
Rollouts in which no correction passed within the teacher's tries (a k_cap_hit record) are discarded (paper §3.2).

Usage:
  python sample.py --taxonomy <dir> --io runs/profiling_seed1/io runs/profiling_seed2/io --out <dir> [--per-type 80]
--io must list the same io/ directories, in the same order, as `taxonomy.py errors`, since the error ids carry the
run index. Output: <out>/prefixes.jsonl
"""
from __future__ import annotations
import argparse, collections, json, random, re, sys
from pathlib import Path

ID_RE = re.compile(r"^(?:run(\d+):)?(.+)#(\d+)$")     # [run<k>:]<task>_r<row>#<step>, as written by taxonomy.py


def locate(eid, io_dirs):
    """io file and step of an error id."""
    m = ID_RE.match(eid)
    if not m:
        sys.exit(f"cannot parse error id {eid!r}")
    run, stem, step = m.groups()
    return Path(io_dirs[int(run) - 1 if run else 0]) / f"{stem}.json", int(step)


def build(eid, mode, io_dirs):
    io_path, step = locate(eid, io_dirs)
    recs = json.loads(io_path.read_text())
    if any(r.get("kind") == "k_cap_hit" for r in recs):
        return None
    # Every step that got past the policy call adds two messages (the accepted step and its output), so the prefix of
    # step S is the system prompt, the task, and the message pairs of the proposed steps before S.
    n = sum(1 for r in recs if r.get("kind") == "propose" and r["step"] < step)
    traj = json.loads((io_path.parent.parent / "traj" / io_path.name).read_text())
    ex = sorted((r for r in recs if r.get("kind") == "execute" and r["step"] < step), key=lambda r: r["step"])
    return {"id": eid, "mode": mode, "task": io_path.stem.rsplit("_r", 1)[0], "step": step,
            "messages": traj["messages"][:2 + 2 * n],
            "replay": [{"command": r["command"], "output": r["output"], "step": r["step"]} for r in ex]}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--taxonomy", type=Path, required=True, help="output directory of taxonomy.py")
    ap.add_argument("--io", nargs="+", required=True, help="io/ directories of the profiling runs, as for taxonomy.py")
    ap.add_argument("--out", type=Path, required=True, help="output directory")
    ap.add_argument("--per-type", type=int, default=80, help="prefixes per misbehavior (default: 80)")
    ap.add_argument("--seed", type=int, default=0, help="sampling seed (default: 0)")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    modes = [t["name"] for t in json.loads((a.taxonomy / "misbehaviors.json").read_text())]
    by_mode = collections.defaultdict(list)
    for l in open(a.taxonomy / "labels.jsonl"):
        r = json.loads(l)
        for t in r["types"]:
            by_mode[t].append(r["id"])
    rng = random.Random(a.seed)
    n = 0
    with open(a.out / "prefixes.jsonl", "w") as fh:
        for mode in modes:
            pool = by_mode[mode][:]
            rng.shuffle(pool)
            got = 0
            for eid in pool:                 # in random order, skipping discarded rollouts, until --per-type
                if got == a.per_type:
                    break
                p = build(eid, mode, a.io)
                if p is not None:
                    fh.write(json.dumps(p, ensure_ascii=False) + "\n")
                    got += 1
            print(f"  {got:4d}  {mode}")
            n += got
    print(f"{n} prefixes -> {a.out / 'prefixes.jsonl'}")


if __name__ == "__main__":
    main()
