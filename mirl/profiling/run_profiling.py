"""Error profiling rollouts (paper §3.2) over a list of TMax environments, with full I/O capture.

For each environment: start its Apptainer sandbox, run one rollout (rollout.py), run the task's tests, and write
  <out>/io/<task>_r<row>.json    every policy, judge and teacher call and every executed command (rollout.Recorder)
  <out>/traj/<task>_r<row>.json  the final messages, the rollout statistics and the test result
and at the end <out>/stats.json. The policy and the teacher are served by vLLM (see scripts/profile.sh).

Usage (see scripts/profile.sh):
  python run_profiling.py --tasks-dir <TMax task dirs> --out <dir> [--envs environments.json] [--workers 48]
                          [--limit N] [--resume]
Environment: TMAX_DIR (TMax checkout at 7387d2f9), ROLLOUT_SEED, REPEAT_NOTICE, and the settings in rollout.py.
"""
from __future__ import annotations
import json, os, sys, time, argparse, traceback, signal, subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, os.environ.get("TMAX_DIR", ""))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rl_data.generator.env import InteractiveContainerEnvironment
import rollout as R

TASKS = Path(".")      # set from --tasks-dir: one directory per task, as unpacked by prepare_tasks.sh
OUT = Path(".")        # set from --out


class Result:
    """Adapter: env.exec returns (ok, text); rollout expects .stdout/.stderr/.return_code."""
    def __init__(self, ok, text):
        self.stdout = text or ""
        self.stderr = ""
        self.return_code = 0 if ok else 1


def load_task(task_id):
    d = TASKS / task_id
    tj = json.loads((d / "task.json").read_text())
    instr = tj.get("description") or tj.get("instruction") or tj.get("task") or ""
    return d, instr


def make_env(task_id):
    d = TASKS / task_id
    return InteractiveContainerEnvironment(
        container_sif_path=str(d / "container.sif"),      # a sandbox directory built by prepare_tasks.sh
        initial_test_path=str(d / "test_initial_state.py"),
        final_test_path=str(d / "test_final_state.py"),
        def_path=str(d / "container.def"),
        verbose=False,
    )


def stem(item):
    """File stem of one rollout: task id and its row in the environment list."""
    return f"{item['task']}_r{item['row']}"


def kill_sandbox_orphans(task_id, name=""):
    """Kill processes that an earlier run of this task left behind. Sandboxes share the host network, so a leftover
    server would hold its port for later tasks. A process is killed when it has APPTAINER_CONTAINER=<this sandbox> in
    its environment, is re-parented to pid 1, is not a live instance's starter or init, and is not in a live
    instance's mount namespace. Each kill is logged to <out>/orphans_killed.log."""
    sif = str(TASKS / task_id / "container.sif")
    try:
        out = subprocess.run(["apptainer", "instance", "list", "--json"],
                             capture_output=True, text=True, timeout=30).stdout
        inst = json.loads(out or '{"instances":[]}').get("instances", [])
    except Exception:
        return 0                      # cannot see live instances: do not kill anything
    live_pids, live_ns = set(), set()
    for i in inst:
        pid = int(i["pid"]); live_pids.add(pid)
        try:
            live_ns.add(os.readlink(f"/proc/{pid}/ns/mnt"))
            live_pids.add(int(open(f"/proc/{pid}/stat").read().rsplit(")", 1)[1].split()[1]))
        except Exception:
            pass
    key = f"APPTAINER_CONTAINER={sif}".encode()
    killed = []
    for pid in os.listdir("/proc"):
        if not pid.isdigit() or int(pid) in live_pids or int(pid) == os.getpid():
            continue
        try:
            ppid = int(open(f"/proc/{pid}/stat").read().rsplit(")", 1)[1].split()[1])
            if ppid != 1:
                continue
            with open(f"/proc/{pid}/environ", "rb") as f:
                if key not in f.read():
                    continue
            if os.readlink(f"/proc/{pid}/ns/mnt") in live_ns:
                continue
            with open(f"/proc/{pid}/cmdline", "rb") as f:
                cmd = f.read().replace(b"\0", b" ").decode(errors="replace").strip()[:120]
            os.kill(int(pid), signal.SIGKILL); killed.append((int(pid), cmd))
        except Exception:
            continue
    if killed:
        try:
            with open(OUT / "orphans_killed.log", "a") as f:
                for pid, cmd in killed:
                    f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')}\t{name or task_id}\t{pid}\t{cmd}\n")
        except Exception:
            pass
    return len(killed)


def run_one(item, args, idx, total):
    task_id, name = item["task"], stem(item)
    d, instr = load_task(task_id)
    rec_path = OUT / "io" / f"{name}.json"
    rec_path.parent.mkdir(parents=True, exist_ok=True)
    rec = R.Recorder(str(rec_path))
    env = None
    t0 = time.time()
    try:
        k0 = kill_sandbox_orphans(task_id, name)      # leftovers of an earlier, crashed run
        if k0:
            print(f"    {name}: killed {k0} leftover process(es) before start", flush=True)
        env = make_env(task_id)
        if not env.initialize(run_initial_tests=False):
            return {"task": task_id, "row": item["row"], "error": "initialize failed"}
        env_exec = lambda cmd, to: Result(*env.exec(cmd, timeout=to))
        messages, stats = R.run_trajectory(
            task_id, instr, env_exec, args.policy_model, args.teacher_model,
            args.policy_base, args.teacher_base, rec, row=item["row"])
        try:
            passed, vtext = env.run_final_tests()
        except Exception as e:
            passed, vtext = False, f"verifier error: {e}"
        stats["verifier_passed"] = bool(passed)
        stats["wall_s"] = round(time.time() - t0, 1)
        rec.add(kind="verifier", passed=bool(passed), output=str(vtext)[:4000])
        stats["row"] = item["row"]
        traj_path = OUT / "traj" / f"{name}.json"
        traj_path.parent.mkdir(parents=True, exist_ok=True)
        traj_path.write_text(json.dumps({"task": task_id, "row": item["row"],
                                         "messages": messages, "stats": stats}, indent=2, default=str))
        print(f"[{idx}/{total}] {name}: steps={stats['steps']} bad={stats['n_bad']} "
              f"rw={stats['n_rewrites']} pass={passed} {stats['wall_s']}s", flush=True)
        return stats
    except Exception as e:
        traceback.print_exc()
        return {"task": task_id, "row": item["row"], "error": f"{type(e).__name__}: {e}"[:300],
                "wall_s": round(time.time() - t0, 1)}
    finally:
        rec.save()
        if env is not None:
            try: env.cleanup()
            except Exception: pass
        k = kill_sandbox_orphans(task_id, name)
        if k:
            print(f"    {name}: killed {k} leftover sandbox process(es)", flush=True)


def main():
    global TASKS, OUT
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks-dir", required=True, help="TMax task directories (from prepare_tasks.sh)")
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--envs", default=str(here / "environments.json"),
                    help="environment list: [{task, row}, ...]; default: the 2,017 environments of the paper")
    ap.add_argument("--limit", type=int, default=0, help="only the first N environments (0 = all)")
    ap.add_argument("--workers", type=int, default=48, help="rollouts at once")
    ap.add_argument("--resume", action="store_true", help="skip environments whose traj file already exists")
    ap.add_argument("--policy-model", default="hosted_vllm/policy")
    ap.add_argument("--teacher-model", default="hosted_vllm/teacher")
    # 127.0.0.1, not localhost: under sudo, localhost may resolve to ::1 first while vLLM listens on IPv4 only
    ap.add_argument("--policy-base", default="http://127.0.0.1:8700/v1")
    ap.add_argument("--teacher-base", default="http://127.0.0.1:8800/v1")
    args = ap.parse_args()
    TASKS, OUT = Path(args.tasks_dir), Path(args.out)
    OUT.mkdir(parents=True, exist_ok=True)

    envs = json.loads(Path(args.envs).read_text())
    if args.limit:
        envs = envs[:args.limit]
    print(f"{len(envs)} environments; seed {R.ROLLOUT_SEED}; policy temperature {R.POLICY_TEMP}; "
          f"repetition notice {'on' if R.REPEAT_NOTICE else 'off'}", flush=True)

    allstats = []
    t0 = time.time()
    todo = envs
    if args.resume:
        done = [t for t in envs if (OUT / "traj" / f"{stem(t)}.json").exists()]
        todo = [t for t in envs if t not in done]
        print(f"resume: {len(done)} done, {len(todo)} to run", flush=True)
        for t in done:
            try:
                allstats.append(json.loads((OUT / "traj" / f"{stem(t)}.json").read_text())["stats"])
            except Exception as e:
                allstats.append({"task": t["task"], "row": t["row"], "error": f"unreadable traj: {e}"})
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as ex:
        futs = {ex.submit(run_one, t, args, i, len(todo)): t for i, t in enumerate(todo, 1)}
        for f in as_completed(futs):
            try:
                allstats.append(f.result())
            except Exception as e:
                allstats.append({"task": futs[f]["task"], "row": futs[f]["row"],
                                 "error": f"{type(e).__name__}: {e}"[:300]})
    (OUT / "stats.json").write_text(json.dumps(
        {"n_envs": len(envs), "wall_s": round(time.time() - t0, 1), "per_env": allstats}, indent=2, default=str))
    print(f"=== done in {(time.time() - t0) / 60:.1f} min ===", flush=True)


if __name__ == "__main__":
    main()
