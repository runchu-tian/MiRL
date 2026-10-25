"""Build the grpo_fast training rows of MiRL from the prefixes that passed the rerun check.

One row per prefix:
  messages      the prefix messages (system, task, earlier steps and their outputs) + a placeholder assistant turn
                (open-instruct's rlvr_tokenize drops the last message of a row)
  ground_truth  JSON for the reward (prefix_judge.py): the prefix id, its misbehavior (name, definition, examples), the
                number of steps the behavior judge reads (m), and the judge's {task}: the task and the last k steps of
                the prefix, after a "--- ... N former steps not shown ... ---" line when earlier steps are hidden
  dataset       "prefix_judge"
  env_config    the sandbox to continue in (prefix_replay_sandbox.py): task, prefix commands and outputs to rerun, and
                the step limit, 64 steps in total with the prefix included

Usage: python build_rows.py --prefixes <dir>/prefixes_rerun.jsonl --taxonomy <dir> --out <dir>/train.jsonl [--k 5] [--m 5]

Options for the ablations (paper §4.4):
  --start task       one row per environment of the prefixes, starting at the beginning of the task (no prefix, no rerun)
  --reward outcome   the reward is r_o only (no behavior judge)
  --random-envs N    instead of the prefixes, N environments drawn at random from mirl/profiling/environments.json
                     (TMax-189: --random-envs 189 --tasks-dir <dir>/tasks --start task --reward outcome); the task message
                     is built as in error profiling, so this needs TMAX_DIR
"""
import argparse, json, os, random, sys
from pathlib import Path

TOTAL_STEPS = 64
STEP = "--- AGENT STEP ---"


def first_command(m):                                   # teacher._first_command: tool_calls[0] is what the harness ran
    tc = (m.get('tool_calls') or [None])[0]
    if not tc:
        return None
    a = tc['function']['arguments']
    a = json.loads(a) if isinstance(a, str) else a
    return (a or {}).get('command')


def render_context(messages):                           # teacher.render_context
    out = []
    for m in messages:
        if m['role'] == 'user':
            out.append(f"--- TASK ---\n{m.get('content', '')}")
        elif m['role'] == 'assistant':
            out.append(f"{STEP}\n{(m.get('content') or '').strip()}\n$ {first_command(m)}")
        elif m['role'] == 'tool':
            out.append(f"--- OUTPUT ---\n{m.get('content', '')}")
    return "\n\n".join(out)


def judge_task(messages, k):
    """The task and the last k prefix steps, each with its output. When earlier steps are hidden, a marker line says how
    many, so that the judge does not read the first shown step as the first step of the run."""
    saw = render_context(messages)
    a = saw.find(STEP)
    if a < 0:
        return saw.strip()
    task, hist = saw[:a].strip(), [STEP + x for x in saw[a:].split(STEP)[1:]]
    hidden = len(hist) - k
    gap = f"--- ... {hidden} former step{'s' if hidden > 1 else ''} not shown ... ---\n\n" if hidden > 0 else ""
    return (task + '\n\n' + gap + ''.join(hist[-k:])).strip()


def template_ready(messages):
    """tool_call arguments as dicts (vLLM parses the JSON string before rendering the chat template)."""
    out = []
    for m in messages:
        m = {k: v for k, v in m.items() if v is not None and k in ('role', 'content', 'tool_calls', 'tool_call_id', 'name')}
        if m.get('tool_calls'):
            tcs = []
            for tc in m['tool_calls']:
                a = tc['function']['arguments']
                tcs.append({'id': tc.get('id') or 'x', 'type': 'function',
                            'function': {'name': tc['function']['name'], 'arguments': json.loads(a) if isinstance(a, str) else a}})
            m['tool_calls'] = tcs
        out.append(m)
    return out


def row(p, mode, k, m, reward):
    n_prefix_steps = sum(1 for x in p['messages'] if x['role'] == 'assistant')
    new = max(1, TOTAL_STEPS - n_prefix_steps)
    gt = dict(id=p['id'], reward=reward, mode=mode['name'], definition=mode['definition'], examples=mode['examples'],
              m=m, judge_task=judge_task(p['messages'], k))
    return dict(messages=template_ready(p['messages']) + [{'role': 'assistant', 'content': ''}],
                ground_truth=json.dumps(gt), dataset='prefix_judge',
                env_config=dict(max_steps=new, env_configs=[dict(env_name='prefix_replay_sandbox', task_id=p['task'],
                                replay=[{'command': e['command'], 'output': e['output']} for e in p['replay']],
                                max_new_steps=new, prefix_id=p['id'])]))


def task_start_row(task, messages):
    """A rollout from the beginning of the task: system prompt and task message only, 64 steps, reward r_o."""
    gid = task + '#start'
    return dict(messages=messages[:2] + [{'role': 'assistant', 'content': ''}],
                ground_truth=json.dumps(dict(id=gid, reward='outcome')), dataset='prefix_judge',
                env_config=dict(max_steps=TOTAL_STEPS, env_configs=[dict(env_name='prefix_replay_sandbox', task_id=task,
                                replay=[], max_new_steps=TOTAL_STEPS, prefix_id=gid)]))


def random_env_rows(a):
    """N environments drawn at random from the environment list, each starting at the beginning of its task."""
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'profiling'))
    import run_profiling as P
    from rollout import _render_instance, _SYSTEM_TEMPLATE
    P.TASKS = a.tasks_dir
    pool = sorted({e['task'] for e in json.loads(a.envs.read_text())})
    rows = []
    for t in random.Random(a.seed).sample(pool, len(pool))[:a.random_envs]:
        _, instr = P.load_task(t)
        rows.append(task_start_row(t, [{'role': 'system', 'content': _SYSTEM_TEMPLATE},
                                       {'role': 'user', 'content': _render_instance(instr.strip())}]))
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument('--prefixes', type=Path, help='prefixes_rerun.jsonl from mirl/prefixes/rerun.py')
    ap.add_argument('--taxonomy', type=Path, help='output directory of taxonomy.py (misbehaviors.json)')
    ap.add_argument('--out', type=Path, required=True, help='output file (train.jsonl)')
    ap.add_argument('--k', type=int, default=5, help='prefix steps the behavior judge sees before the new steps (default: 5)')
    ap.add_argument('--m', type=int, default=5, help='new steps the behavior judge reads (default: 5)')
    ap.add_argument('--seed', type=int, default=260929, help='seed of the row order and of --random-envs (default: 260929)')
    ap.add_argument('--start', choices=['prefix', 'task'], default='prefix',
                    help='start the rollouts at the prefix (MiRL) or at the beginning of the task (default: prefix)')
    ap.add_argument('--reward', choices=['mirl', 'outcome'], default='mirl',
                    help='r_o AND r_e (MiRL) or r_o only (default: mirl)')
    ap.add_argument('--random-envs', type=int, default=0,
                    help='use this many random environments from --envs instead of the prefixes')
    ap.add_argument('--envs', type=Path, default=Path(__file__).resolve().parent.parent / 'profiling' / 'environments.json',
                    help='environment list for --random-envs (default: the 2,017 environments of the paper)')
    ap.add_argument('--tasks-dir', type=Path, help='TMax task directories, for --random-envs')
    a = ap.parse_args()
    if a.start == 'task' and a.reward == 'mirl':
        ap.error('r_e judges the steps after a prefix: use --reward outcome with --start task')
    if a.random_envs:
        if a.start != 'task' or not a.tasks_dir:
            ap.error('--random-envs needs --start task and --tasks-dir')
        rows = random_env_rows(a)
    else:
        if not (a.prefixes and a.taxonomy):
            ap.error('--prefixes and --taxonomy are required')
        modes = {t['name']: t for t in json.loads((a.taxonomy / 'misbehaviors.json').read_text())}
        prefixes = [json.loads(l) for l in open(a.prefixes)]
        random.Random(a.seed).shuffle(prefixes)          # the prefixes are sorted by misbehavior
        if a.start == 'task':                            # one row per environment, the first of its prefixes in this order
            seen, rows = set(), []
            for p in prefixes:
                if p['task'] not in seen:
                    seen.add(p['task'])
                    rows.append(task_start_row(p['task'], p['messages']))
        else:
            rows = [row(p, modes[p['mode']], a.k, a.m, a.reward) for p in prefixes]
    os.makedirs(a.out.parent, exist_ok=True)
    with open(a.out, 'w') as f:
        for r in rows:
            f.write(json.dumps(r) + '\n')
    envs = {r['env_config']['env_configs'][0]['task_id'] for r in rows}
    print(f"{len(rows)} rows from {len(envs)} environments -> {a.out}")


if __name__ == '__main__':
    main()
