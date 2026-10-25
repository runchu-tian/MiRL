#!/usr/bin/env python3
"""User-level guards for the ROOTLESS evaluation sandboxes (APPTAINER_EVAL_ROOTLESS=1). Rootless Apptainer cannot
set pids.max or --memory/--cpus (fakeroot mode has no cgroups), so this loop gives the same protection from outside,
as a normal user (no root). Like the kernel limits of the root mode, the process and memory guards do not end the
sandbox; they kill single processes and the sandbox lives on:

  (a) process guard: processes in the instance's PID namespace > PIDS_MAX (4096 = the root mode's pids.max) -> SIGKILL the
      newest killable processes until the count is at the limit (root mode: a fork beyond pids.max fails with EAGAIN);
  (b) memory guard:  summed Pss_Anon + Pss_Shmem (shared pages split, file-backed pages left out, ~ what the cgroup OOM killer cannot reclaim) of those processes > the task's memory_mb (the root mode's --memory) -> SIGKILL the killable
      process with the largest PSS, repeat until below (the kernel OOM killer's rule; the agent sees "Killed");
  (c) leak guard:    our instances (name prefix aev_) older than MAX_AGE_H (12 h) -> stop the instance.
Never killed: the instance's init (the pid `apptainer instance list` reports), Apptainer's starter/init processes
(comm in GUARD_PROTECT_COMM), and any pid listed in <trials>/<instance>/protect_pids (e.g. an RL persistent shell).
Known differences: enforced by polling (every GUARD_INTERVAL_S = 2 s, 1 s while a limit is exceeded), not by the
kernel, so a fast spike can pass between two polls;
no per-task CPU limit.

The memory limit is read from <APPTAINER_EVAL_DIR>/trials/<instance>/limits.json, written by
apptainer_env.py at sandbox start. Every kill/stop is logged (instance, pid, comm, RSS or count).
Use:  python sandbox_guard.py [--once]      env: APPTAINER_EVAL_DIR, GUARD_INTERVAL_S (2), GUARD_LOG
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

W = Path(os.environ.get("APPTAINER_EVAL_DIR", os.path.expanduser("~/.cache/mirl/apptainer_eval")))
PIDS_MAX = int(os.environ.get("APPTAINER_EVAL_PIDS_MAX", "4096"))
MAX_AGE_S = float(os.environ.get("GUARD_MAX_AGE_H", "12")) * 3600
INTERVAL = float(os.environ.get("GUARD_INTERVAL_S", "2"))
PROTECT_COMM = set(os.environ.get("GUARD_PROTECT_COMM", "starter,starter-suid,appinit,sinit").split(","))
LOG = Path(os.environ.get("GUARD_LOG", str(W / "sandbox_guard.log")))
PAGE = os.sysconf("SC_PAGE_SIZE")
CLK = os.sysconf("SC_CLK_TCK")
PREFIX = "aev_"


def log(msg: str):
    line = f"{time.strftime('%F %T')} {msg}"
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")


def instances() -> dict[str, int]:
    r = subprocess.run(["apptainer", "instance", "list", "--json"], capture_output=True, text=True, timeout=60)
    try:
        return {i["instance"]: int(i["pid"]) for i in json.loads(r.stdout).get("instances", [])
                if i.get("instance", "").startswith(PREFIX)}
    except (ValueError, KeyError):
        return {}


def pidns(pid: int) -> str | None:
    try:
        return os.readlink(f"/proc/{pid}/ns/pid")
    except OSError:
        return None


def uptime_s() -> float:
    return float(Path("/proc/uptime").read_text().split()[0])


def age_s(pid: int) -> float | None:
    try:
        start_ticks = int(Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[19])
    except (OSError, IndexError, ValueError):
        return None
    return uptime_s() - start_ticks / CLK


def mem_limit_mb(inst: str) -> float | None:
    try:
        return json.loads((W / "trials" / inst / "limits.json").read_text()).get("memory_mb")
    except (OSError, ValueError):
        return None


def stop(inst: str, why: str):
    log(f"STOP {inst}: {why}")
    subprocess.run(["apptainer", "instance", "stop", "-F", inst], capture_output=True, timeout=300)


def comm(pid: int) -> str:
    try:
        return Path(f"/proc/{pid}/comm").read_text().strip()
    except OSError:
        return "?"


def rss_b(pid: int) -> int:
    try:
        return int(Path(f"/proc/{pid}/statm").read_text().split()[1]) * PAGE
    except (OSError, IndexError, ValueError):
        return 0


def pss_b(pid: int) -> int:
    """Non-reclaimable memory of a process: Pss_Anon + Pss_Shmem (smaps_rollup). Shared pages are split between the
    processes that share them, and file-backed pages (libraries, mapped files) are left out, because the cgroup OOM
    killer reclaims those first (summed RSS counted shared chromium pages ~50x -> false kills; then full
    PSS still counted file-backed pages)."""
    try:
        v = 0
        for line in Path(f"/proc/{pid}/smaps_rollup").read_text().splitlines():
            if line.startswith(("Pss_Anon:", "Pss_Shmem:")):
                v += int(line.split()[1]) * 1024
        return v
    except (OSError, IndexError, ValueError):
        return 0


def start_ticks(pid: int) -> int:
    try:
        return int(Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[19])
    except (OSError, IndexError, ValueError):
        return 0


def protected(inst: str, init_pid: int) -> set[int]:
    s = {init_pid}
    try:
        s |= {int(x) for x in (W / "trials" / inst / "protect_pids").read_text().split()}
    except (OSError, ValueError):
        pass
    return s


def ns_procs(ns: str) -> list[int]:
    return [int(p) for p in os.listdir("/proc") if p.isdigit() and pidns(int(p)) == ns]


def kill(inst: str, pid: int, why: str):
    log(f"KILL {inst} pid={pid} comm={comm(pid)} rss={rss_b(pid) / 2**20:.0f}MB: {why}")
    try:
        os.kill(pid, 9)
    except (ProcessLookupError, PermissionError):
        pass


def check_once() -> bool:
    """One pass over all instances. Returns True if a limit was exceeded (the caller then polls again sooner)."""
    inst = instances()
    exceeded = False
    for name, init_pid in inst.items():
        a = age_s(init_pid)
        if a is not None and a > MAX_AGE_S:
            stop(name, f"age {a / 3600:.1f} h > {MAX_AGE_S / 3600:.0f} h (leak guard)")
            continue
        ns = pidns(init_pid)
        if not ns:
            continue
        keep = protected(name, init_pid)
        killable = lambda ps: [p for p in ps if p not in keep and comm(p) not in PROTECT_COMM]  # noqa: E731
        procs = ns_procs(ns)
        if len(procs) > PIDS_MAX:                   # (a) newest first, down to the limit
            exceeded = True
            n_over = len(procs) - PIDS_MAX
            for p in sorted(killable(procs), key=start_ticks, reverse=True)[:n_over]:
                kill(name, p, f"{len(procs)} processes > {PIDS_MAX} (process guard)")
            procs = ns_procs(ns)
        lim = mem_limit_mb(name)
        if lim:                                     # (b) largest PSS first, until below the limit
            for _ in range(64):
                if sum(rss_b(p) for p in procs) / 2**20 <= lim:   # cheap upper bound first (RSS >= PSS)
                    break
                total = sum(pss_b(p) for p in procs) / 2**20
                if total <= lim:
                    break
                exceeded = True
                cand = killable(procs)
                if not cand:
                    log(f"WARN {name}: anon+shmem PSS {total:.0f} MB > memory_mb {lim:.0f}, but only protected processes are left")
                    break
                big = max(cand, key=pss_b)
                kill(name, big, f"anon+shmem PSS {total:.0f} MB > memory_mb {lim:.0f} (memory guard)")
                time.sleep(0.2)
                procs = ns_procs(ns)
    return exceeded


def main():
    LOG.parent.mkdir(parents=True, exist_ok=True)
    if "--once" in sys.argv:
        check_once()
        return
    log(f"guard start: pids>{PIDS_MAX}, rss>memory_mb, age>{MAX_AGE_S / 3600:.0f} h, every {INTERVAL:.0f} s")
    while True:
        exceeded = False
        try:
            exceeded = check_once()
        except Exception as e:  # noqa: BLE001  (never die; log and go on)
            log(f"guard error: {e!r}")
        time.sleep(1.0 if exceeded else INTERVAL)


if __name__ == "__main__":
    main()
