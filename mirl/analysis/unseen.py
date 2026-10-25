"""Unseen misbehavior prefixes for the misbehavior rate (paper §4.5).

From the sampled prefixes (mirl/prefixes/sample.py), keep those that are not training prefixes and whose environment is
not a training environment (mirl/prefixes/rerun.py keeps the training prefixes), and draw --per-type of them for each
misbehavior in --modes. The policy only writes one step after each prefix, so these prefixes need no rerun check.

One JSON line per prefix:
  id, mode, task, messages  the message list the policy saw before the error step
  judge_context             the task and the last k prefix steps, as the {task} of the behavior judge in training

Usage: python unseen.py --prefixes <dir>/prefixes.jsonl --train <dir>/prefixes_rerun.jsonl --modes "<name>" ... --out <file>
"""
import argparse, json, random, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "train"))
from build_rows import judge_task, template_ready


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--prefixes", type=Path, required=True, help="prefixes.jsonl from mirl/prefixes/sample.py")
    ap.add_argument("--train", type=Path, required=True, help="the training prefixes, prefixes_rerun.jsonl")
    ap.add_argument("--modes", nargs="+", required=True, help="misbehaviors to measure")
    ap.add_argument("--out", type=Path, required=True, help="output file")
    ap.add_argument("--per-type", type=int, default=20, help="unseen prefixes per misbehavior (default: 20)")
    ap.add_argument("--k", type=int, default=5, help="prefix steps in the judge context (default: 5)")
    ap.add_argument("--seed", type=int, default=260929, help="sampling seed (default: 260929)")
    a = ap.parse_args()
    train = [json.loads(l) for l in open(a.train)]
    train_ids, train_envs = {p["id"] for p in train}, {p["task"] for p in train}
    prefixes = [json.loads(l) for l in open(a.prefixes)]
    rng = random.Random(a.seed)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(a.out, "w") as fh:
        for mode in a.modes:
            pool = sorted((p for p in prefixes if p["mode"] == mode and p["id"] not in train_ids
                           and p["task"] not in train_envs), key=lambda p: p["id"])
            if not pool:
                sys.exit(f"no unseen prefixes for {mode!r} (is it a name in misbehaviors.json?)")
            rng.shuffle(pool)
            for p in pool[:a.per_type]:
                fh.write(json.dumps(dict(id=p["id"], mode=mode, task=p["task"], messages=template_ready(p["messages"]),
                                         judge_context=judge_task(p["messages"], a.k)), ensure_ascii=False) + "\n")
            print(f"  {min(len(pool), a.per_type):4d} of {len(pool):4d}  {mode}")
            n += min(len(pool), a.per_type)
    print(f"{n} unseen prefixes -> {a.out}")


if __name__ == "__main__":
    main()
