"""Sandbox service for MiRL training (paper §3.4): starts task sandboxes, puts each one into the state at the end of its
prefix by rerunning the prefix, and runs the policy's commands and the final tests.

Runs as root when sudo is available (see scripts/train.sh), with one network per sandbox (APPTAINER_NET_BRIDGE=1) and the
sandbox guards of the TMax env.py patch (patches/tmax_env.patch); otherwise as the normal user. The trainer's environment class (prefix_replay_sandbox.py, inside a
Ray actor) talks to it over HTTP on 127.0.0.1.

The shell is the one of error profiling: rl_data.generator.env.InteractiveContainerEnvironment (one persistent bash),
made by run_profiling.make_env, outputs formatted by rollout._fmt_result, command timeout rollout.CMD_TIMEOUT (120 s),
final tests by env.run_final_tests(). Each command of the prefix must give the recorded output, by the rule of the
rerun check (SANDBOX_MATCH=exact, the default, or masked; see mirl/prefixes/rerun.py). A failed start or a mismatch is
retried in a fresh sandbox (MAX_START_RETRIES / MAX_REPLAY_RETRIES); after that, the reset fails and the trainer
drops the sample.

API (JSON POST):
  /reset {task_id, replay: [{command, output}], tag}  -> {ok, sid, tries: [...], replay_s}
  /exec  {sid, command}  -> {output, secs, submitted}
  /tests {sid}           -> {pass, output, secs}
  /close {sid}           -> {ok}
  /close_all {}          -> {closed}   (called by scripts/train.sh when a run ends)
  GET /stats             -> counters
Sandboxes of one task may run at the same time (a GRPO group): orphan cleanup for a task runs only when none of its
sandboxes is live (it would otherwise race with a starting instance).

Environment: TASKS_DIR (TMax task directories, from mirl/profiling/prepare_tasks.sh), SANDBOX_PORT (8650), SANDBOX_LOG,
SANDBOX_MATCH, START_CONCURRENCY, MAX_START_RETRIES, MAX_REPLAY_RETRIES, IDLE_TIMEOUT, and TMAX_DIR.
"""
import collections, json, os, sys, threading, time, traceback, uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "profiling"))
sys.path.insert(0, str(HERE.parent / "prefixes"))
import run_profiling as P                    # make_env, Result, kill_sandbox_orphans
import rollout as R                          # _fmt_result, CMD_TIMEOUT
import rerun                                 # masked_match (the rerun check's masked rule)
from rl_data.generator.sample_solutions import SUBMIT_MARKER

PORT = int(os.environ.get('SANDBOX_PORT', '8650'))
MAX_START_RETRIES = int(os.environ.get('MAX_START_RETRIES', '3'))
MAX_REPLAY_RETRIES = int(os.environ.get('MAX_REPLAY_RETRIES', '3'))
START_CONCURRENCY = int(os.environ.get('START_CONCURRENCY', '16'))
IDLE_TIMEOUT = float(os.environ.get('IDLE_TIMEOUT', '3600'))   # close sessions unused this long (killed trainers)
LOG = os.environ.get('SANDBOX_LOG', 'sandbox_server.jsonl')
MATCH = os.environ.get('SANDBOX_MATCH', 'exact')
if MATCH not in ('exact', 'masked'):
    sys.exit(f'SANDBOX_MATCH must be exact or masked, not {MATCH!r}')
same = (lambda rec, got: rec == got) if MATCH == 'exact' else rerun.masked_match

_start_sem = threading.Semaphore(START_CONCURRENCY)
_sessions = {}                               # sid -> dict(env, task_id, lock, t0)
_live = collections.Counter()                # task_id -> live sandboxes
_task_locks = collections.defaultdict(threading.Lock)
_glock = threading.Lock()
_stats = collections.Counter()


def log(**kw):
    kw['t'] = time.strftime('%F %T')
    with _glock:
        with open(LOG, 'a') as f:
            f.write(json.dumps(kw) + '\n')


def run(env, cmd):
    return R._fmt_result(P.Result(*env.exec(cmd, timeout=R.CMD_TIMEOUT)))


def _cleanup(env):
    try:
        env.cleanup()
    except Exception:
        pass


def _release(task_id):
    with _task_locks[task_id]:
        with _glock:
            _live[task_id] -= 1
            idle = _live[task_id] <= 0
        if idle:                             # no live sandbox of this task: safe to kill its orphans
            try:
                P.kill_sandbox_orphans(task_id, 'train')
            except Exception:
                pass


def reset(task_id, replay, tag=''):
    tries, t0 = [], time.time()
    n_start_fail = n_mismatch = 0
    while True:
        env, t1 = None, time.time()
        with _task_locks[task_id]:
            with _glock:
                _live[task_id] += 1
        try:
            with _start_sem:
                env = P.make_env(task_id)
                started = env.initialize(run_initial_tests=False)
            if not started:
                n_start_fail += 1; tries.append(dict(why='start_failed', secs=round(time.time() - t1, 1)))
                _stats['start_failed'] += 1
                _cleanup(env); _release(task_id)
                if n_start_fail > MAX_START_RETRIES:
                    break
                continue
            bad = None
            for i, e in enumerate(replay):
                got = run(env, e['command'])
                if not same(e['output'], got):
                    bad = dict(why='mismatch', index=i, command=e['command'][:300], recorded=e['output'][:600],
                               replayed=got[:600], secs=round(time.time() - t1, 1))
                    break
            if bad is None:
                sid = uuid.uuid4().hex[:12]
                tries.append(dict(why='ok', secs=round(time.time() - t1, 1)))
                with _glock:
                    _sessions[sid] = dict(env=env, task_id=task_id, lock=threading.Lock(), t0=time.time(), last=time.time())
                _stats['reset_ok'] += 1
                log(ev='reset', ok=True, sid=sid, task_id=task_id, tag=tag, n_cmds=len(replay), tries=tries,
                    secs=round(time.time() - t0, 1))
                return dict(ok=True, sid=sid, tries=tries, replay_s=round(time.time() - t0, 1))
            n_mismatch += 1; tries.append(bad); _stats['mismatch'] += 1
            _cleanup(env); _release(task_id)
            if n_mismatch > MAX_REPLAY_RETRIES:
                break
        except Exception:
            tries.append(dict(why='error', error=traceback.format_exc()[-800:], secs=round(time.time() - t1, 1)))
            _stats['error'] += 1
            if env is not None:
                _cleanup(env)
            _release(task_id)
            if len(tries) > MAX_START_RETRIES + MAX_REPLAY_RETRIES + 1:
                break
    _stats['reset_dropped'] += 1
    log(ev='reset', ok=False, task_id=task_id, tag=tag, n_cmds=len(replay), tries=tries, secs=round(time.time() - t0, 1))
    return dict(ok=False, tries=tries, replay_s=round(time.time() - t0, 1))


def _get(sid):
    with _glock:
        s = _sessions.get(sid)
    if s is None:
        raise KeyError(f'unknown sid {sid}')
    return s


def exec_(sid, command):
    s, t0 = _get(sid), time.time()
    s['last'] = t0
    with s['lock']:
        out = run(s['env'], command)
    _stats['exec'] += 1
    return dict(output=out, secs=round(time.time() - t0, 2), submitted=SUBMIT_MARKER in (command or '') or SUBMIT_MARKER in out)


def tests(sid):
    s, t0 = _get(sid), time.time()
    with s['lock']:
        ok, txt = s['env'].run_final_tests()
    _stats['tests'] += 1; _stats['tests_pass'] += bool(ok)
    return {'pass': bool(ok), 'output': (txt or '')[-2000:], 'secs': round(time.time() - t0, 1)}


def close(sid):
    with _glock:
        s = _sessions.pop(sid, None)
    if s is None:
        return dict(ok=False)
    with s['lock']:
        _cleanup(s['env'])
    _release(s['task_id'])
    _stats['close'] += 1
    return dict(ok=True, life_s=round(time.time() - s['t0'], 1))


def _reaper():
    while True:
        time.sleep(60)
        now = time.time()
        with _glock:
            idle = [sid for sid, s in _sessions.items() if now - s.get('last', s['t0']) > IDLE_TIMEOUT]
        for sid in idle:
            r = close(sid)
            _stats['reaped'] += 1
            log(ev='reaped', sid=sid, life_s=r.get('life_s'))


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, obj):
        b = json.dumps(obj).encode()
        self.send_response(code); self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(b))); self.end_headers(); self.wfile.write(b)

    def do_GET(self):
        if self.path == '/stats':
            with _glock:
                d = dict(_stats); d['live_sessions'] = len(_sessions); d['live_by_task'] = {k: v for k, v in _live.items() if v}
            return self._send(200, d)
        self._send(404, {'error': 'not found'})

    def do_POST(self):
        try:
            body = json.loads(self.rfile.read(int(self.headers.get('Content-Length', 0))) or b'{}')
            if self.path == '/reset':
                r = reset(body['task_id'], body.get('replay') or [], body.get('tag', ''))
            elif self.path == '/exec':
                r = exec_(body['sid'], body.get('command') or '')
            elif self.path == '/tests':
                r = tests(body['sid'])
            elif self.path == '/close':
                r = close(body['sid'])
            elif self.path == '/close_all':          # end of a training run: its prefetched batch is never closed
                with _glock:
                    sids = list(_sessions)
                r = dict(closed=sum(bool(close(x).get('ok')) for x in sids))
            else:
                return self._send(404, {'error': 'not found'})
            self._send(200, r)
        except Exception:
            self._send(500, {'error': traceback.format_exc()[-1500:]})


if __name__ == '__main__':
    P.TASKS = Path(os.environ['TASKS_DIR'])
    P.OUT = Path(LOG).resolve().parent
    P.OUT.mkdir(parents=True, exist_ok=True)
    ThreadingHTTPServer.request_queue_size = 1024   # default 5: a burst of 128 resets got 'connection reset'
    srv = ThreadingHTTPServer(('127.0.0.1', PORT), H)
    srv.daemon_threads = True
    threading.Thread(target=_reaper, daemon=True).start()
    print(f'sandbox server on 127.0.0.1:{PORT} ({MATCH} match, start concurrency {START_CONCURRENCY}, retries start '
          f'{MAX_START_RETRIES} replay {MAX_REPLAY_RETRIES})', flush=True)
    srv.serve_forever()
