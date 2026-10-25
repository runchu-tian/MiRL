"""Rerun check of the misbehavior prefixes (paper §3.4).

A live rollout needs the container state at the end of its prefix, and only the prefix was recorded. So each prefix
is rerun in a fresh sandbox of its environment: its commands run in order, through the same shell and output
formatting as error profiling, and each output is compared with the recorded one. One try passes when every command
matches. Each prefix gets --tries tries, one after the other (never two at once, they could collide on ports), and is
kept when at least --min-pass of them pass.

Match rule: by default the outputs must be exactly the same. With --no-exact, parts that change from run to run but
carry no task state are masked first (memory addresses, clock times, dates, durations, PIDs, client ports, `ls` totals
and directory sizes), and the line order may differ.

Usage (as root, see scripts/rerun.sh):
  python rerun.py --prefixes <dir>/prefixes.jsonl --tasks-dir <TMax task dirs> --out <dir>
                  [--tries 4] [--min-pass 2] [--no-exact] [--workers 24]
Output: <out>/rerun/<key>.json per prefix (resumable; the key comes from the prefix id and misbehavior, so results stay
        with their prefix when prefixes.jsonl is sampled again), <out>/prefixes_rerun.jsonl (the kept prefixes)
"""
from __future__ import annotations
import argparse, collections, hashlib, json, os, re, sys, time, traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "profiling"))
import run_profiling as P          # make_env, Result, kill_sandbox_orphans: the sandboxes of error profiling
import rollout as R                # _fmt_result, CMD_TIMEOUT

_NORM = [(re.compile(r'0x[0-9a-fA-F]{6,}'), '0xADDR'),                                       # memory addresses
         (re.compile(r'(?m)^((?:> |\[[^\]\n]*\]\$ )*)total \d+$'), r'\1total N'),           # ls block totals
         (re.compile(r'\b\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}(:\d{2})?(\.\d+)?([+-]\d{2}:?\d{2}|Z)?'), 'DATETIME'),
         (re.compile(r'\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec) +\d{1,2} +(\d{2}:\d{2}|\d{4})\b'), 'LSDATE'),
         (re.compile(r'\b\d{2}:\d{2}:\d{2}(\.\d+)?\b'), 'TIME'),
         (re.compile(r'\b\d+(\.\d+)?\s?(s|ms|sec|seconds)\b'), 'DUR'),                      # durations
         (re.compile(r'/tmp/cc\w{6}\.o'), '/tmp/ccTMP.o'),                                      # gcc temp object names
         (re.compile(r'(?m)^(\S+\s+\d+\s+)[\d.]+\s+[\d.]+\s+\d+\s+\d+(\s+\S+\s+\S+\s+)\S+(\s+)\S+'),
          r'\1CPU MEM VSZ RSS\2START\3TIME'),                                                   # `ps aux` usage columns
         (re.compile(r'(?m)^\[\d+\]\s+\d+$'), '[JOB] PID'),                                    # shell job line "[1] 30"
         (re.compile(r'(?i)\b(pid[:= ]\s*)\d+'), r'\1N'),                                      # "PID: 30", "pid=65"
         (re.compile(r"\('127\.0\.0\.1', \d+\)"), "('127.0.0.1', PORT)"),                       # client socket ports
         (re.compile(r'/apptainer/\d+'), '/apptainer/N'),                                         # cgroup id of the sandbox
         (re.compile(r'\b\d{1,2} (Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec) \d{4}\b'), 'LOGDATE'),
         (re.compile(r'\b\d{4}/\d{2}/\d{2}\b'), 'LOGDATE'),
         (re.compile(r'(?m)^(d[rwxst-]{9}\.?\s+\d+\s+\S+\s+\S+\s+)\d+'), r'\1DIRSIZE'),         # directory sizes in ls -l
         (re.compile(r'\b(sshd|cron|python3?|node)\[\d+\]'), r'\1[PID]')]


def masked_match(recorded, rerun):
    """After masking, the outputs must have the same lines; the line order may differ (find/ls order depends on the
    file system, not on the task state)."""
    a, b = recorded, rerun
    for rx, rep in _NORM:
        a, b = rx.sub(rep, a), rx.sub(rep, b)
    return a == b or sorted(a.splitlines()) == sorted(b.splitlines())


def one_try(p, tag, exact):
    env = None
    t0 = time.time()
    try:
        P.kill_sandbox_orphans(p["task"], tag)
        env = P.make_env(p["task"])
        if not env.initialize(run_initial_tests=False):
            return {"ok": False, "why": "init_failed", "secs": round(time.time() - t0, 1)}
        for i, e in enumerate(p["replay"]):
            got = R._fmt_result(P.Result(*env.exec(e["command"], timeout=R.CMD_TIMEOUT)))
            if not (got == e["output"] if exact else masked_match(e["output"], got)):
                return {"ok": False, "why": "mismatch", "index": i, "step": e["step"], "command": e["command"][:500],
                        "recorded": e["output"][:1500], "rerun": got[:1500], "secs": round(time.time() - t0, 1)}
        return {"ok": True, "secs": round(time.time() - t0, 1)}
    except Exception:
        return {"ok": False, "why": "error", "error": traceback.format_exc()[-1500:], "secs": round(time.time() - t0, 1)}
    finally:
        if env is not None:
            try: env.cleanup()
            except Exception: pass
        P.kill_sandbox_orphans(p["task"], tag)


def key(p):
    return hashlib.sha1(f"{p['id']}|{p['mode']}".encode()).hexdigest()[:16]


def check(k, p, a):
    tries = [one_try(p, f"rerun{k}t{t}", a.exact) for t in range(a.tries)]     # one after the other
    n_ok = sum(t["ok"] for t in tries)
    return {"k": k, "key": key(p), "id": p["id"], "mode": p["mode"], "task": p["task"], "n_cmds": len(p["replay"]),
            "n_ok": n_ok, "kept": n_ok >= a.min_pass, "tries": tries}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--prefixes", type=Path, required=True, help="prefixes.jsonl from sample.py")
    ap.add_argument("--tasks-dir", type=Path, required=True, help="TMax task directories (from prepare_tasks.sh)")
    ap.add_argument("--out", type=Path, required=True, help="output directory")
    ap.add_argument("--tries", type=int, default=4, help="tries per prefix (default: 4)")
    ap.add_argument("--min-pass", type=int, default=2, help="keep a prefix if at least this many tries pass (default: 2)")
    ap.add_argument("--no-exact", dest="exact", action="store_false",
                    help="compare outputs after masking times, PIDs, addresses and similar parts (default: exact)")
    ap.add_argument("--workers", type=int, default=24, help="prefixes checked at once (default: 24)")
    a = ap.parse_args()
    P.TASKS, P.OUT = a.tasks_dir, a.out
    d = a.out / "rerun"
    d.mkdir(parents=True, exist_ok=True)

    prefixes = [json.loads(l) for l in open(a.prefixes)]
    todo = [(k, p) for k, p in enumerate(prefixes) if not (d / f"{key(p)}.json").exists()]
    print(f"{len(todo)} prefixes to check ({len(prefixes) - len(todo)} done), {a.tries} tries each, "
          f"{'exact' if a.exact else 'masked'} match", flush=True)
    with ThreadPoolExecutor(max_workers=max(1, a.workers)) as ex:
        futs = [ex.submit(check, k, p, a) for k, p in todo]
        for f in as_completed(futs):
            r = f.result()
            (d / f"{r['key']}.json").write_text(json.dumps(r))
            print(f"{r['k']} {r['mode'][:30]}: {r['n_cmds']} commands, {r['n_ok']}/{a.tries} tries passed", flush=True)

    res = {}                                             # results of the prefixes in this prefixes.jsonl only
    for p in prefixes:
        f = d / f"{key(p)}.json"
        if f.exists():
            res[key(p)] = json.loads(f.read_text())
    kept = [dict(p, n_ok=res[key(p)]["n_ok"]) for p in prefixes if key(p) in res and res[key(p)]["kept"]]
    with open(a.out / "prefixes_rerun.jsonl", "w") as fh:
        for p in kept:
            fh.write(json.dumps(p, ensure_ascii=False) + "\n")
    per_mode = collections.Counter(p["mode"] for p in kept)
    for mode in dict.fromkeys(p["mode"] for p in prefixes):
        print(f"  {per_mode[mode]:4d}  {mode}")
    print(f"{len(kept)} of {len(res)} prefixes kept, from {len({p['task'] for p in kept})} environments "
          f"-> {a.out / 'prefixes_rerun.jsonl'}")


if __name__ == "__main__":
    main()
