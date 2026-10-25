"""Harbor environment that runs each trial in an Apptainer sandbox instead of Docker.

We use it for all TB-Lite and TB-2.1 evaluations in the paper. Docker creates one network per trial and our machines
allowed only 24 of them at once; Apptainer has no such limit, and these sandboxes have the same settings as the
sandboxes of our RL rollouts.

Use:  PYTHONPATH=<this dir> harbor run ... --environment-import-path apptainer_env:ApptainerRolloutEnvironment
Images: one Apptainer sandbox DIRECTORY per task (some kernels cannot mount squashfs .sif files), built by
build_sandboxes.sh into $APPTAINER_EVAL_DIR/sbx/{tblite,tb21}/<task>/, with <task>.json = the Docker image Config
(WorkingDir, User, Env).

The same as harbor's Docker environment (protocol unchanged): the agent, prompts, time limits, verifier and scoring;
exec runs `bash -c <command>` with stderr merged into stdout, cwd = exec cwd or task workdir or the image WORKDIR,
user = exec user or the image USER, env = image ENV + harbor's env; on a command time-out the client is stopped and
RuntimeError("Command timed out after N seconds") is raised (the command itself keeps running, as with docker exec,
because it runs in its own session); /logs/{agent,verifier,artifacts} are bind mounts of the trial dirs (mounted=True);
CPU and memory limits are the task's (cpus, memory_mb), as docker compose's deploy limits.

The same as the sandboxes of our RL rollouts (TMax rl_data/generator/env.py): root instance (sudo -n, no fakeroot),
own network per sandbox (--net --network bridge, CNI), --no-home, --no-mount tmp, --contain --no-mount cwd, the same
--drop-caps list (at start and on every exec), read-only remount of the bind-mounted /dev nodes (fail closed),
pids.max 4096 on the sandbox cgroup.

Two differences from the rollout sandboxes, both to match Docker (found by running the oracle agent):
- Writable layer: APPTAINER_EVAL_WRITABLE=copy (default) runs on a private copy of the image dir (cp -a, --writable,
  no overlay). Apptainer's overlay (dir, tmpfs or --writable-tmpfs) drops the setgid bit on
  `chgrp X dir; chmod 2770 dir` (acl-permissions-inheritance: oracle 0 vs 1 with Docker). =overlay keeps the rollout
  behaviour (directory overlay on local disk).
- --no-mount bind-paths: Apptainer binds the host /etc/localtime and /etc/hosts by default; tzdata's postinst then
  fails ("mv: cannot move /etc/localtime.dpkg-new: Device or resource busy"), so `apt-get install r-base` breaks
  (adaptive-rejection-sampler: oracle 0 vs 1). /etc/resolv.conf is still provided (DNS works).
"""
from __future__ import annotations

import asyncio
import contextlib
import fcntl
import hashlib
import json
import os
import shlex
import shutil
import subprocess
import time
from pathlib import Path

from harbor.environments.base import BaseEnvironment, ExecResult
from harbor.environments.capabilities import EnvironmentCapabilities
from harbor.models.environment_type import EnvironmentType

W = Path(os.environ.get("APPTAINER_EVAL_DIR", os.path.expanduser("~/.cache/mirl/apptainer_eval")))
DROP_CAPS = "CAP_NET_ADMIN,CAP_SYS_ADMIN,CAP_SYS_BOOT,CAP_SYS_MODULE,CAP_SYS_TIME,CAP_SYS_RAWIO,CAP_MKNOD,CAP_SYS_PTRACE"
DEV_RO_NODES = ("null", "zero", "random", "urandom", "tty")
PIDS_MAX = int(os.environ.get("APPTAINER_EVAL_PIDS_MAX", "4096"))
CNI_SLOTS = max(1, int(os.environ.get("APPTAINER_CNI_SLOTS", "8")))
XFER = "/.harbor_xfer"            # in-sandbox mount point of the per-trial host transfer dir
WRITABLE = os.environ.get("APPTAINER_EVAL_WRITABLE", "copy")   # copy (Docker-like) | overlay (rollout-like)
# Rootless mode, for machines without sudo, CNI or an /etc/subuid range. Off (default) = the root mode above.
# On: no sudo anywhere; `--fakeroot` (root-mapped user namespace + fakeroot command; with the marker file
# <image dir>.no-fakeroot-cmd, `--ignore-fakeroot-command` for images whose glibc cannot load it); network
# APPTAINER_EVAL_NET = slirp (default, see below) | none (own EMPTY network per sandbox, `--net --network none`; the
# verifiers of all TB-2.1 tasks and of many TB-Lite tasks download packages first and then fail) | host (shared host
# network, internet; two copies of one task must then not run at the same time); no CNI lock. Known differences (cannot be done
# without root): no --cpus/--memory ("rootless cgroups is not usable in fakeroot mode"), no
# pids.max, no read-only remount of /dev nodes (mount: permission denied), and only uid 0 exists in the sandbox, so a
# non-root image/exec USER runs as root (as harbor's own singularity backend). sandbox_guard.py replaces the
# process and memory limits at user level.
ROOTLESS = os.environ.get("APPTAINER_EVAL_ROOTLESS", "0") == "1"
NET = os.environ.get("APPTAINER_EVAL_NET", "slirp")
if ROOTLESS and NET not in ("none", "host", "slirp"):
    raise ValueError(f"APPTAINER_EVAL_NET must be none, host or slirp, not {NET!r}")
# NET=slirp (closest to the root mode's own network with NAT): own empty network per sandbox
# (`--net --network none`) + rootless slirp4netns attached to it (NAT to the internet, ports private per sandbox; the
# host's loopback is not reachable). DNS is slirp4netns's own resolver 10.0.2.3 (`--dns`): the host's resolv.conf may
# name a loopback resolver (127.0.0.53 on Ubuntu) that the sandbox cannot reach. Binary: APPTAINER_EVAL_SLIRP (we used
# the static release v1.3.6).
SLIRP = os.environ.get("APPTAINER_EVAL_SLIRP", "slirp4netns")
SLIRP_DNS = "10.0.2.3"
# no_fakeroot_cmd for ALL images (APPTAINER_EVAL_NO_FAKEROOT_CMD=1): root-mapped namespace only, no LD_PRELOAD faking
FORCE_NO_FR = os.environ.get("APPTAINER_EVAL_NO_FAKEROOT_CMD", "0") == "1"


@contextlib.contextmanager
def _cni_lock():
    """Same lock files as TMax env.py's _cni_lock (shared with the RL sandboxes): at most CNI_SLOTS network
    set-ups/tear-downs at once on this machine (many parallel CNI plugin runs fight over the iptables lock)."""
    if ROOTLESS:                            # no CNI in rootless mode: nothing to serialize
        yield
        return
    while True:
        for k in range(CNI_SLOTS):
            path = "/tmp/apptainer_cni.lock" if k == 0 else f"/tmp/apptainer_cni.lock.{k}"
            if not os.path.exists(path):    # never create it as this user: with fs.protected_regular=1 a root RL
                subprocess.run(["sudo", "-n", "touch", path], capture_output=True, timeout=60)  # sandbox service then
            fh = open(path, "r")            # cannot open it ("w") and every RL sandbox start fails. A read fd is enough.
            try:
                fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                fh.close()
                continue
            try:
                yield
            finally:
                fcntl.flock(fh, fcntl.LOCK_UN)
                fh.close()
            return
        time.sleep(0.2)


def _sudo(cmd: list[str], timeout: float = 600) -> subprocess.CompletedProcess:
    return subprocess.run([*([] if ROOTLESS else ["sudo", "-n"]), *cmd], capture_output=True, text=True, timeout=timeout)


def _rm_tree(path: Path, timeout: float = 600):
    """rm -rf of a trial dir. Rootless: files the sandbox's fake root made read-only (e.g. chmod 0 dirs) belong to us,
    so first give ourselves write permission."""
    if ROOTLESS:
        subprocess.run(["chmod", "-R", "u+rwX", str(path)], capture_output=True, timeout=timeout)
    _sudo(["rm", "-rf", str(path)], timeout)


class ApptainerRolloutEnvironment(BaseEnvironment):
    def __init__(self, *args, **kwargs):
        kwargs.pop("mounts_json", None)
        kwargs.pop("keep_containers", None)
        super().__init__(*args, **kwargs)
        self._instance = "aev_" + hashlib.sha1(self.session_id.encode()).hexdigest()[:12]
        self._run_dir = W / "trials" / self._instance
        self._xfer = self._run_dir / "xfer"
        self._overlay = self._run_dir / "overlay"
        self._pid: int | None = None
        self._has_setsid = True
        cfg = json.loads(Path(str(self._sif) + ".json").read_text()) if Path(str(self._sif) + ".json").exists() else {}  # <task>.json
        self._img_workdir = cfg.get("WorkingDir") or "/"
        self._img_user = cfg.get("User") or None
        self._img_env = [e for e in (cfg.get("Env") or []) if "=" in e]

    # ---- harbor interface -------------------------------------------------------------------------------------
    @staticmethod
    def type() -> EnvironmentType:
        return EnvironmentType.SINGULARITY     # closest built-in label; only used in messages

    @property
    def capabilities(self) -> EnvironmentCapabilities:
        return EnvironmentCapabilities(disable_internet=False, windows=False, mounted=True)

    @property
    def _sif(self) -> Path:
        task = self.environment_dir.resolve().parent.name
        suite = "tb21" if "terminal-bench-2-1" in str(self.environment_dir.resolve()) else "tblite"
        return W / "sbx" / suite / task

    def _validate_definition(self):
        if not self._sif.is_dir():
            raise FileNotFoundError(f"no Apptainer sandbox dir for this task: {self._sif} (run build_sandboxes.sh)")

    async def start(self, force_build: bool):
        await asyncio.to_thread(self._start_sync)
        await self.exec(f"chmod 777 {self.env_paths.agent_dir} {self.env_paths.verifier_dir}", user="root")

    def _start_sync(self):
        if self._run_dir.exists():
            _rm_tree(self._run_dir)
        self._xfer.mkdir(parents=True, exist_ok=True)
        if WRITABLE == "copy":
            r = _sudo(["cp", "-a", str(self._sif), str(self._run_dir / "rootfs")], 1800)
            if r.returncode != 0:
                _rm_tree(self._run_dir)
                raise RuntimeError(f"copy of the image dir failed: {(r.stdout + r.stderr)[-300:]}")
            rootfs = self._run_dir / "rootfs"          # no overlay: bind targets must exist in the image dir
            _sudo(["mkdir", "-p", *[str(rootfs) + str(c) for c in (self.env_paths.verifier_dir, self.env_paths.agent_dir,
                                                                  self.env_paths.artifacts_dir, XFER)]], 60)
            image_args = ["--writable", str(rootfs)]
        else:
            (self._overlay / "upper").mkdir(parents=True, exist_ok=True)
            (self._overlay / "work").mkdir(parents=True, exist_ok=True)
            image_args = ["--overlay", str(self._overlay), str(self._sif)]
        tp = self.trial_paths
        binds = [(tp.verifier_dir, self.env_paths.verifier_dir), (tp.agent_dir, self.env_paths.agent_dir),
                 (tp.artifacts_dir, self.env_paths.artifacts_dir), (self._xfer, XFER)]
        for h, _ in binds:
            Path(h).mkdir(parents=True, exist_ok=True)
        cpus = getattr(self.task_env_config, "cpus", None)
        mem = getattr(self.task_env_config, "memory_mb", None)
        if ROOTLESS:                            # see ROOTLESS above: no cgroup limits in fakeroot mode
            # the task's memory limit for sandbox_guard.py (user-level memory guard, kills like the OOM killer)
            (self._run_dir / "limits.json").write_text(json.dumps({"memory_mb": mem, "cpus": cpus}))
            cpus = mem = None
            no_fr = FORCE_NO_FR or Path(str(self._sif) + ".no-fakeroot-cmd").exists()
            self._no_fr = no_fr
            mode = ["--fakeroot", *(["--ignore-fakeroot-command"] if no_fr else []),
                    *(["--net", "--network", "none"] if NET in ("none", "slirp") else []),
                    *(["--dns", SLIRP_DNS] if NET == "slirp" else [])]
            if no_fr and WRITABLE == "copy" and (self._run_dir / "rootfs/etc/apt/apt.conf.d").is_dir():
                # Without the fakeroot command apt cannot drop to its _apt user ("setgroups 65534 failed", every
                # apt-get update/install fails; in the root mode apt drops to _apt and works). A read-only
                # APT::Sandbox::User "root" makes apt download as root: the same result as the root mode.
                # The image stays unchanged (bind over an empty file in the trial's private copy).
                # Dir::Log::Terminal /dev/null: no "W: chown to root:adm of .../term.log failed" line (no adm group).
                conf = W / "99-rootless-sandbox"
                text = 'APT::Sandbox::User "root";\nDir::Log::Terminal "/dev/null";\n'
                if not conf.exists() or conf.read_text() != text:
                    conf.write_text(text)
                tgt = "/etc/apt/apt.conf.d/99-rootless-sandbox"
                (self._run_dir / ("rootfs" + tgt)).touch()
                mode += ["--bind", f"{conf}:{tgt}:ro"]
        else:
            mode = ["--net", "--network", "bridge"]
        cmd = ["apptainer", "instance", "start", "--no-home", "--no-mount", "bind-paths",
               *mode, "--no-mount", "tmp", "--contain", "--no-mount", "cwd",
               "--drop-caps", DROP_CAPS,
               *(["--memory", f"{int(mem)}M", "--memory-swap", f"{int(mem)}M"] if mem else []),
               *(["--cpus", str(cpus)] if cpus else []),
               *sum((["--bind", f"{Path(h).resolve()}:{c}"] for h, c in binds), []),
               "--cleanenv", *image_args[:-1], image_args[-1], self._instance]
        with _cni_lock():
            r = _sudo(cmd)
        if r.returncode != 0:
            _rm_tree(self._run_dir)
            raise RuntimeError(f"apptainer instance start failed for {self.environment_name}: {(r.stdout + r.stderr)[-600:]}")
        try:
            info = json.loads(_sudo(["apptainer", "instance", "list", "--json", self._instance], 60).stdout)
            self._pid = int(info["instances"][0]["pid"])
        except Exception as e:
            self._stop_sync()
            raise RuntimeError(f"cannot read the instance pid: {e!r}")
        if ROOTLESS and NET == "slirp":
            try:
                self._start_slirp()
            except Exception as e:
                self._stop_sync()
                raise RuntimeError(f"slirp4netns failed for {self.environment_name}: {e!r}")
        self._fkey = None
        if ROOTLESS and not self._no_fr:
            try:
                self._start_faked_holder()
            except Exception as e:
                self._stop_sync()
                raise RuntimeError(f"fakeroot holder failed for {self.environment_name}: {e!r}")
        if not ROOTLESS:                        # rootless: neither is possible (see ROOTLESS above)
            if not self._dev_readonly():
                self._stop_sync()
                raise RuntimeError("read-only /dev remount failed (fail closed)")
            cg = Path(f"/sys/fs/cgroup/apptainer/{self._pid}/pids.max")
            _sudo(["bash", "-c", f"[ -f {cg} ] && echo {PIDS_MAX} > {cg}"], 60)
        self._has_setsid = _sudo(["apptainer", "exec", "--drop-caps", DROP_CAPS, f"instance://{self._instance}", "sh", "-c",
                                  "setsid -w true 2>/dev/null"], 60).returncode == 0

    def _start_slirp(self):
        """Attach slirp4netns to the instance's own network namespace: tap0 10.0.2.100/24, gateway 10.0.2.2, DNS
        10.0.2.3 (forwards to the host's resolver). --ready-fd: wait until tap0 is configured; --exit-fd: slirp4netns
        ends when this process ends (no orphan if harbor dies). --enable-seccomp limits slirp4netns itself
        (--enable-sandbox needs root here: 'parent failed')."""
        ready_r, ready_w = os.pipe()
        self._slirp_exit_r, self._slirp_exit_w = os.pipe()
        log = open(self._run_dir / "slirp4netns.log", "wb")
        self._slirp = subprocess.Popen(
            [SLIRP, "--configure", "--mtu=65520", "--disable-host-loopback", "--enable-seccomp",
             "--ready-fd", str(ready_w), "--exit-fd", str(self._slirp_exit_r),
             f"--userns-path=/proc/{self._pid}/ns/user", "--netns-type=path", f"/proc/{self._pid}/ns/net", "tap0"],
            pass_fds=(ready_w, self._slirp_exit_r), stdin=subprocess.DEVNULL, stdout=log, stderr=log,
            start_new_session=True)
        os.close(ready_w)
        os.close(self._slirp_exit_r)
        log.close()
        import select
        ok = select.select([ready_r], [], [], 60)[0]
        got = os.read(ready_r, 1) if ok else b""
        os.close(ready_r)
        if got != b"1":
            raise RuntimeError(f"not ready after 60 s (rc={self._slirp.poll()})")

    def _start_faked_holder(self):
        """ONE fakeroot daemon (faked) for the whole sandbox, as real root keeps ONE file-ownership state: a holder exec
        (`sleep infinity` under the fakeroot command) keeps its faked alive and writes its key to the transfer dir;
        every exec then uses that key (see _exec_argv). Without it each exec had its own faked, which ends with the
        exec: a chown was forgotten by the next command, and a background server (`server &`) lost its faked and
        hung (found with the oracle agent: kv-store-grpc, pypi-server, configure-git-webserver, nginx, mlflow-register)."""
        kf = self._xfer / ".fkey"
        self._faked_holder = subprocess.Popen(
            ["apptainer", "exec", "--cleanenv", "--drop-caps", DROP_CAPS, f"instance://{self._instance}", "bash", "-c",
             f"echo $FAKEROOTKEY > {XFER}/.fkey.tmp && mv {XFER}/.fkey.tmp {XFER}/.fkey; exec sleep infinity"],
            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
        for _ in range(300):
            if kf.exists() and kf.read_text().strip():
                self._fkey = kf.read_text().strip()
                return
            if self._faked_holder.poll() is not None:
                break
            time.sleep(0.1)
        raise RuntimeError(f"no FAKEROOTKEY after 30 s (holder rc={self._faked_holder.poll()})")

    def _current_fkey(self) -> str | None:
        try:
            return (self._xfer / ".fkey").read_text().strip() or None
        except OSError:
            return None

    def _ensure_faked(self):
        """Before every exec: is the sandbox's faked alive? An agent or verifier can kill it (`kill -9 -1`, `pkill`);
        every later exec would then hang until its time-out. Probe: one stat under the current key (libfakeroot asks
        faked; a dead faked = no answer). If it fails, start a new holder (the fake ownership state is lost, as after
        a crash) and log 'faked respawned <instance>' to <APPTAINER_EVAL_DIR>/faked_respawn.log."""
        key = self._current_fkey()
        ok = False
        if key:
            with contextlib.suppress(subprocess.TimeoutExpired):
                ok = subprocess.run(
                    ["apptainer", "exec", "--cleanenv", f"instance://{self._instance}", "sh", "-c",
                     f"export FAKEROOTKEY={key}; stat -c %u / >/dev/null"],
                    stdin=subprocess.DEVNULL, capture_output=True, timeout=5).returncode == 0
        if ok:
            return
        h = getattr(self, "_faked_holder", None)
        if h is not None and h.poll() is None:
            with contextlib.suppress(ProcessLookupError, PermissionError):
                os.killpg(h.pid, 9)
        with contextlib.suppress(OSError):
            (self._xfer / ".fkey").unlink()
        self._start_faked_holder()
        with open(W / "faked_respawn.log", "a") as f:
            f.write(f"{time.strftime('%F %T')} faked respawned {self._instance} {self.environment_name}\n")
        self.logger.warning(f"faked respawned in {self._instance}")

    def _stop_slirp(self):
        p = getattr(self, "_slirp", None)
        if p is None:
            return
        with contextlib.suppress(OSError):
            os.close(self._slirp_exit_w)        # slirp4netns ends on --exit-fd EOF
        try:
            p.wait(timeout=10)
        except subprocess.TimeoutExpired:
            p.kill()
            with contextlib.suppress(subprocess.TimeoutExpired):
                p.wait(timeout=10)
        self._slirp = None

    def _dev_readonly(self) -> bool:
        """env.py _dev_readonly: remount the bind-mounted host /dev nodes read-only inside the sandbox; True only if
        all of them are ro afterwards (checked in /proc/<pid>/mountinfo)."""
        for d in DEV_RO_NODES:
            _sudo(["nsenter", "-t", str(self._pid), "-m", "--", "mount", "-o", "remount,bind,ro", f"/dev/{d}"], 60)
        r = _sudo(["cat", f"/proc/{self._pid}/mountinfo"], 60)
        mi = [l.split() for l in r.stdout.splitlines()]
        return all(any(len(p) > 5 and p[4] == f"/dev/{d}" and "ro" in p[5].split(",") for p in mi) for d in DEV_RO_NODES)

    async def stop(self, delete: bool):
        await self.prepare_logs_for_host()
        await asyncio.to_thread(self._stop_sync)

    def _stop_sync(self):
        with _cni_lock():
            _sudo(["apptainer", "instance", "stop", "-F", self._instance], 300)
        h = getattr(self, "_faked_holder", None)
        if h is not None and h.poll() is None:
            with contextlib.suppress(ProcessLookupError, PermissionError):
                os.killpg(h.pid, 9)
        self._stop_slirp()
        _rm_tree(self._run_dir)

    @staticmethod
    def _host_ids() -> str:
        """uid:gid that maps to our host user inside the sandbox (rootless: the fake root 0:0 IS our user)."""
        return "0:0" if ROOTLESS else f"{os.getuid()}:{os.getgid()}"

    async def prepare_logs_for_host(self) -> None:
        # timeout: after a 120 s command time-out the command keeps running in the sandbox; under the
        # task's CPU limit this chown then starved for ~70 min (train-fasttext) until the watchdog stopped the instance.
        try:
            await self.exec(f"chown -R {self._host_ids()} {self.env_paths.logs_dir}", user="root", timeout_sec=300)
        except Exception as e:  # noqa: BLE001
            self.logger.warning(f"Failed to chown logs directory: {e}")

    # ---- exec ---------------------------------------------------------------------------------------------------
    def _exec_argv(self, command: str, cwd: str | None, env: dict[str, str] | None, user) -> list[str]:
        argv = [*([] if ROOTLESS else ["sudo", "-n"]), "apptainer", "exec", "--cleanenv", "--drop-caps", DROP_CAPS,
                "--pwd", cwd or getattr(self.task_env_config, "workdir", None) or self._img_workdir]
        for e in self._img_env:
            argv += ["--env", e]
        for k, v in (env or {}).items():
            argv += ["--env", f"{k}={v}"]
        argv.append(f"instance://{self._instance}")
        u = user if user is not None else self._img_user
        if ROOTLESS:
            # Each rootless exec runs under its own fakeroot daemon (faked), keyed by FAKEROOTKEY. Vanillux2Agent's
            # persistent bash restores `export -p` of the previous command (/tmp/.vanillux2/env), i.e. the key of a
            # daemon that is gone, and every later command then waits forever (all steps timed out).
            # Read-only keeps the live key; the restore's error is silenced by the agent (2>/dev/null).
            fk = self._current_fkey() if getattr(self, "_fkey", None) else None    # always the live key
            command = (f"export FAKEROOTKEY={fk}; " if fk else "") + "readonly FAKEROOTKEY 2>/dev/null\n" + command
        inner = ["bash", "-c", command]
        if u not in (None, "", "root", 0, "0") and not ROOTLESS:   # rootless: only uid 0 exists (see ROOTLESS above)
            inner = ["su", str(u), "-s", "/bin/bash", "-c", command]
        if self._has_setsid:
            inner = ["setsid", "-w", *inner]
        return argv + inner

    async def exec(self, command: str, cwd: str | None = None, env: dict[str, str] | None = None,
                   timeout_sec: int | None = None, user: str | int | None = None) -> ExecResult:
        """Output goes to a host file (not a pipe), and only the client process is awaited. With a pipe, a background
        process started by the command (e.g. `server &`) keeps the pipe open and the call never returns (trials
        hung for hours); `docker exec` returns when the command itself ends, so this matches Docker.
        The file's mtime is also the trial's progress signal for the driver's watchdog (out_*.log in the run dir)."""
        user = self._resolve_user(user)
        env = self._merge_env(env)
        if ROOTLESS and getattr(self, "_fkey", None):
            await asyncio.to_thread(self._ensure_faked)
        self._exec_n = getattr(self, "_exec_n", 0) + 1
        out_path = self._run_dir / f"out_{self._exec_n:05d}.log"
        out_f = open(out_path, "wb")
        try:
            proc = await asyncio.create_subprocess_exec(*self._exec_argv(command, cwd, env, user),
                                                        stdin=asyncio.subprocess.DEVNULL, stdout=out_f, stderr=out_f)
        finally:
            out_f.close()
        try:
            if timeout_sec:
                await asyncio.wait_for(proc.wait(), timeout=timeout_sec)
            else:
                await proc.wait()
        except asyncio.CancelledError:          # harbor's agent time-out cancels the run: do not leave the client
            with contextlib.suppress(ProcessLookupError, PermissionError):
                proc.kill()
            raise
        except asyncio.TimeoutError:
            with contextlib.suppress(ProcessLookupError, PermissionError):
                proc.terminate()
            try:
                await asyncio.wait_for(proc.wait(), timeout=5)
            except asyncio.TimeoutError:
                with contextlib.suppress(ProcessLookupError, PermissionError):
                    proc.kill()
                with contextlib.suppress(asyncio.TimeoutError):
                    await asyncio.wait_for(proc.wait(), timeout=10)
            raise RuntimeError(f"Command timed out after {timeout_sec} seconds")
        out = out_path.read_bytes() if out_path.exists() else b""
        with contextlib.suppress(OSError):
            out_path.unlink()
        return ExecResult(stdout=out.decode(errors="replace") if out else None, stderr=None,
                          return_code=proc.returncode or 0)

    # ---- files: through the per-trial transfer dir --------------------------------------------------------------
    async def _root(self, command: str, timeout: int = 600) -> ExecResult:
        r = await self.exec(command, user="root", timeout_sec=timeout)
        if r.return_code != 0:
            raise RuntimeError(f"transfer command failed ({r.return_code}): {command[:200]} -> {(r.stdout or '')[-300:]}")
        return r

    async def upload_file(self, source_path: Path | str, target_path: str):
        name = hashlib.sha1(f"{source_path}{time.time()}".encode()).hexdigest()[:10]
        shutil.copy2(source_path, self._xfer / name)
        t = shlex.quote(target_path)
        await self._root(f"mkdir -p $(dirname {t}) && cp -f {XFER}/{name} {t} && rm -f {XFER}/{name}")

    async def upload_dir(self, source_dir: Path | str, target_dir: str):
        name = hashlib.sha1(f"{source_dir}{time.time()}".encode()).hexdigest()[:10]
        shutil.copytree(source_dir, self._xfer / name)
        t = shlex.quote(target_dir)
        await self._root(f"mkdir -p {t} && cp -a {XFER}/{name}/. {t}/ && rm -rf {XFER}/{name}")

    async def download_file(self, source_path: str, target_path: Path | str):
        name = hashlib.sha1(f"{source_path}{time.time()}".encode()).hexdigest()[:10]
        await self._root(f"cp -f {shlex.quote(source_path)} {XFER}/{name} && chown {self._host_ids()} {XFER}/{name}")
        Path(target_path).parent.mkdir(parents=True, exist_ok=True)
        shutil.move(self._xfer / name, target_path)

    async def download_dir(self, source_dir: str, target_dir: Path | str):
        name = hashlib.sha1(f"{source_dir}{time.time()}".encode()).hexdigest()[:10]
        await self._root(f"mkdir -p {XFER}/{name} && cp -a {shlex.quote(source_dir)}/. {XFER}/{name}/ && "
                         f"chown -R {self._host_ids()} {XFER}/{name}")
        Path(target_dir).mkdir(parents=True, exist_ok=True)
        shutil.copytree(self._xfer / name, target_dir, dirs_exist_ok=True)
        shutil.rmtree(self._xfer / name, ignore_errors=True)
