#!/usr/bin/env python3
"""Score one harbor run of TB-Lite or TB-2.1, as in the paper.

Score = 100 * (sum of verifier rewards) / (number of tasks in the suite). Partial rewards count; a trial that errored
or is missing counts 0. "solved" counts the trials with reward >= 1.

Also prints the two load checks of the run (see the warning in run_eval.sh): the number of agent time-outs
(AgentTimeoutError) and the mean model time per agent step (llm_s in the agent's timing.json). A run is valid only if
time-outs <= 24 and model time per step <= 15 s; otherwise the machine was overloaded and the score is too low.

Usage:  score.py --tasks 89 RUN_DIR          (one line for people)
        score.py --tasks 89 --json RUN_DIR   (one JSON object)
"""
import argparse
import collections
import glob
import json
import os
import statistics

MAX_TIMEOUTS, MAX_STEP_S, MAX_STEPS = 24, 15.0, 64


def score_run(run_dir, n_tasks):
    reward, solved, trials = 0.0, 0, 0
    exc, steps, llm_s = collections.Counter(), [], []
    for p in glob.glob(f"{run_dir}/*/result.json") + glob.glob(f"{run_dir}/*/*/result.json"):
        r = json.load(open(p))
        if "task_name" not in r:          # the job-level result.json
            continue
        trials += 1
        x = float((((r.get("verifier_result") or {}).get("rewards") or {}).get("reward")) or 0.0)
        reward += x
        solved += x >= 1.0
        e = (r.get("exception_info") or {}).get("exception_type")
        if e:
            exc[e] += 1
        tj = os.path.join(os.path.dirname(p), "agent", "timing.json")   # one entry per agent step
        if os.path.exists(tj):
            try:
                t = json.load(open(tj))
                steps.append(len(t))
                llm_s += [s["llm_s"] for s in t if "llm_s" in s]
            except (ValueError, KeyError):
                pass
    timeouts = exc.get("AgentTimeoutError", 0)
    step_s = statistics.mean(llm_s) if llm_s else None
    return {
        "dir": os.path.abspath(run_dir), "tasks": n_tasks, "trials": trials,
        "score": 100 * reward / n_tasks, "solved": solved, "exceptions": dict(exc),
        "agent_timeouts": timeouts, "model_s_per_step": step_s,
        "mean_steps": statistics.mean(steps) if steps else None,
        "hit_step_limit": sum(s >= MAX_STEPS for s in steps) / len(steps) if steps else None,
        "load_ok": timeouts <= MAX_TIMEOUTS and (step_s is None or step_s <= MAX_STEP_S),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", type=int, required=True, help="tasks in the suite: 100 for TB-Lite, 89 for TB-2.1")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("run_dir")
    a = ap.parse_args()
    r = score_run(a.run_dir, a.tasks)
    if a.json:
        print(json.dumps(r))
        return
    sps = "n/a" if r["model_s_per_step"] is None else f"{r['model_s_per_step']:.1f}"
    print(f"score {r['score']:.2f}% | solved {r['solved']}/{r['tasks']} | trials {r['trials']} | "
          f"agent time-outs {r['agent_timeouts']} | model s/step {sps} | exceptions {r['exceptions']}")
    if r["trials"] < r["tasks"]:
        print(f"WARNING: only {r['trials']} of {r['tasks']} trials have a result; the missing ones count 0.")
    if not r["load_ok"]:
        print(f"WARNING: overloaded run (time-outs > {MAX_TIMEOUTS} or model s/step > {MAX_STEP_S:.0f}). "
              "The score is too low; lower --n-concurrent and run again.")


if __name__ == "__main__":
    main()
