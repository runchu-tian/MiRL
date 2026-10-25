"""Patch TMax's rl_data/generator/env.py so that each sandbox can get its own network (Apptainer bridge network).

Without it, all sandboxes on a machine share the host network, and two sandboxes that start a server on the same port
at the same time collide; a rerun then fails for a reason that has nothing to do with the prefix. With the patch and
APPTAINER_NET_BRIDGE=1 (set by scripts/rerun.sh), each instance starts with `--net --network bridge` (needs root),
instance starts and stops are serialized on the machine (many network set-ups at once fight over the iptables lock),
and instances are stopped with --force. Without the variable, env.py behaves as before. The patch is idempotent.

Usage: python net_bridge.py <TMAX_DIR>
"""
import sys
from pathlib import Path

LOCK = '''def _net_bridge():
    return os.environ.get("APPTAINER_NET_BRIDGE", "0") == "1"


def _cni_lock():
    """With the bridge network, one instance start or stop at a time on this machine; a no-op otherwise."""
    import contextlib
    if not _net_bridge():
        return contextlib.nullcontext()

    @contextlib.contextmanager
    def _lock():
        with open("/tmp/apptainer_cni.lock", "w") as fh:
            fcntl.flock(fh, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(fh, fcntl.LOCK_UN)
    return _lock()


def _fakeroot_flags() -> List[str]:'''

REPLACE = [
    ("def _fakeroot_flags() -> List[str]:", LOCK),
    ('''            subprocess.run(
                ["apptainer", "instance", "stop", self.instance_name],
                capture_output=True
            )''',
     '''            with _cni_lock():
                subprocess.run(
                    ["apptainer", "instance", "stop", *(["--force"] if _net_bridge() else []), self.instance_name],
                    capture_output=True
                )'''),
    ('''            "--no-home",
            "--bind", f"{self.temp_dir}:{self.temp_dir}",''',
     '''            "--no-home",
            *(["--net", "--network", "bridge"] if _net_bridge() else []),
            "--bind", f"{self.temp_dir}:{self.temp_dir}",'''),
    ('''        start_proc = subprocess.run(start_cmd, capture_output=True, text=True)''',
     '''        with _cni_lock():
            start_proc = subprocess.run(start_cmd, capture_output=True, text=True)'''),
]


def main():
    p = Path(sys.argv[1]) / "rl_data" / "generator" / "env.py"
    s = p.read_text()
    if "def _cni_lock" in s:
        print("env.py already has the bridge network patch")
        return
    for old, new in REPLACE:
        if s.count(old) != 1:
            sys.exit("env.py does not look like TMax at 7387d2f9; not patched")
        s = s.replace(old, new)
    p.write_text(s)
    print("env.py patched for the bridge network")


if __name__ == "__main__":
    main()
