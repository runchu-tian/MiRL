"""One error-profiling rollout (paper §3.2), with full I/O capture.

The policy proposes a step; the error localizer (judge) labels it good or bad; a bad step is replaced by the teacher's
correction, which is judged in the same way, up to K_CAP tries. Only accepted steps execute and enter the context, so a
step judged bad never runs. The judge and the teacher are the same model (Qwen3.8-27B in the paper); their prompts are
in teacher.py.

After K_CAP rejected corrections, the last correction is executed and the rollout continues; such rollouts are removed
later, in the data filtering. The rollout ends when the agent submits, after MAX_STEPS steps, or when the policy's
context exceeds its window.
"""
from __future__ import annotations
import json, os, sys, time
from typing import Any

sys.path.insert(0, os.environ.get("TMAX_DIR", ""))      # TMax checkout at 7387d2f9: rl_data
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import litellm
from rl_data.generator.sample_solutions import SUBMIT_MARKER, TOOL_SCHEMAS, _extract_tool_call
from rl_data.generator.vanillux_solver import _render_instance, _SYSTEM_TEMPLATE, _truncate_observation
import teacher as T

MAX_STEPS = 64          # agent steps per rollout (as in TMax)
CMD_TIMEOUT = 120       # seconds per command
MAX_TOKENS = 16384      # output tokens per policy step
TEACHER_TEMP = 0.7      # judge and teacher
POLICY_TEMP = float(os.environ.get("POLICY_TEMP", "1.0"))    # TMax RL rollout temperature
ROLLOUT_SEED = int(os.environ.get("ROLLOUT_SEED", "0"))       # policy request seed = 100000 * ROLLOUT_SEED + row
K_CAP = 3               # teacher tries per flagged step
JUDGE_MAX_TOKENS = 4096 # answer budget of the judge: its reply is one short JSON verdict
# The correction writes a whole step (thinking + heredocs), like the policy, so it gets the policy's budget.
REWRITE_MAX_TOKENS = int(os.environ.get("REWRITE_MAX_TOKENS", str(MAX_TOKENS)))
TEACHER_CTX = int(os.environ.get("TEACHER_CTX", "65536"))    # context window of the served judge/teacher
# Repetition notice for the judge: when the step repeats the previous command exactly, the judge prompt gets a short
# rubric and the computed facts (teacher.REPEAT_RUBRIC, teacher.repetition_facts).
REPEAT_NOTICE = os.environ.get("REPEAT_NOTICE", "0") == "1"


def _chat(model, messages, api_base, tools=None, max_tokens=MAX_TOKENS, temperature=TEACHER_TEMP,
          response_format=None, thinking_override=True, seed=None):
    # num_retries=3: a single refused connection to a local server must not kill a whole rollout. litellm does not
    # retry 4xx, so a real BadRequest still fails fast; only transport/5xx errors retry.
    kw = dict(model=model, messages=messages, api_base=api_base, max_tokens=max_tokens,
              temperature=temperature, timeout=3600, request_timeout=3600, num_retries=3)
    if thinking_override:                # Qwen3.8-27B; the Qwen3.5 policy uses its stock chat template
        kw["extra_body"] = {"chat_template_kwargs": {"enable_thinking": True, "preserve_thinking": True}}
    if seed is not None:
        kw["seed"] = seed
    if tools:
        kw["tools"] = tools
    if response_format:                  # JSON-schema constrained decoding for the judge
        kw["response_format"] = response_format
    return litellm.completion(**kw)


def _fmt_result(result) -> str:
    out = result.stdout or ""
    if result.stderr:
        out += ("\n" + result.stderr) if out else result.stderr
    out = out.rstrip()
    return f"{_truncate_observation(out) if out else '(no output)'}\n\n(exit_code={result.return_code})"


class Recorder:
    """Every policy, judge and teacher call, and every executed command, in the order they happened."""
    def __init__(self, path):
        self.path = path
        self.events = []
    def add(self, **kw):
        kw["t"] = round(time.time(), 3)
        self.events.append(kw)
    def save(self):
        # recreate the parent if something removed it mid-run: losing the I/O record must never kill a rollout
        # that already did the expensive work
        try:
            os.makedirs(os.path.dirname(self.path), exist_ok=True)
            with open(self.path, "w") as f:
                json.dump(self.events, f, indent=2, default=str)
        except Exception as e:
            print(f"WARN: could not save I/O record {self.path}: {e}", flush=True)


def run_trajectory(task_id, instruction, env_exec, policy_model, teacher_model,
                   policy_base, teacher_base, rec: Recorder, row=None):
    """Returns (messages, stats). env_exec(command, timeout) -> result object."""
    messages = [{"role": "system", "content": _SYSTEM_TEMPLATE},
                {"role": "user", "content": _render_instance(instruction.strip())}]
    stats = dict(task=task_id, steps=0, n_judged=0, n_bad=0, n_rewrites=0,
                 k_list=[], own_rewrite_rejects=0, malformed=0,
                 judge_unparseable=0, ended="cap", notice_fired=0, notice_bad=0)
    executed: list[tuple[str, str]] = []   # (command, output) of every step that ran

    for step_i in range(MAX_STEPS):
        t0 = time.time()
        p_seed = (100000 * ROLLOUT_SEED + int(row)) if (row is not None and ROLLOUT_SEED) else None
        try:
            resp = _chat(policy_model, messages, policy_base, tools=TOOL_SCHEMAS, temperature=POLICY_TEMP,
                         thinking_override=False, seed=p_seed)
        except litellm.exceptions.ContextWindowExceededError:
            stats["ended"] = "context_window"; break
        except litellm.exceptions.BadRequestError as e:
            # vLLM rejected the model's own tool-call JSON (e.g. truncated arguments): count a malformed step
            rec.add(kind="propose_bad_request", step=step_i + 1, actor="policy", error=str(e)[:200])
            stats["malformed"] += 1
            stats["propose_bad_request"] = stats.get("propose_bad_request", 0) + 1
            continue
        msg = resp.choices[0].message.model_dump()
        action = _extract_tool_call(msg)
        rec.add(kind="propose", step=step_i + 1, actor="policy", temperature=POLICY_TEMP, seed=p_seed,
                thinking_override=False, context=T.render_context(messages), output_raw=msg,
                step_rendered=T.render_step(msg), llm_s=round(time.time() - t0, 1))

        # judge the policy's step; on "bad", the teacher corrects it and the correction is judged again
        k = 0
        verdict, jraw = _judge(teacher_model, teacher_base, T.render_context(messages),
                               T.render_step(msg), rec, step_i + 1, stats, notice=_notice(executed, msg))
        stats["n_judged"] += 1
        if action["type"] == "no_tool_call":
            stats["malformed"] += 1
        if verdict == "bad":
            stats["n_bad"] += 1
            while verdict == "bad" and k < K_CAP:
                nm, na = _rewrite(teacher_model, teacher_base, T.render_context(messages), rec, step_i + 1)
                if nm is None:            # the correction call failed (e.g. context overflow): keep the policy step
                    break
                msg, action = nm, na
                k += 1
                stats["n_rewrites"] += 1
                verdict, jraw = _judge(teacher_model, teacher_base, T.render_context(messages),
                                       T.render_step(msg), rec, step_i + 1, stats,
                                       own_rewrite=True, notice=_notice(executed, msg))
                if verdict == "bad":
                    stats["own_rewrite_rejects"] += 1
            stats["k_list"].append(k)
            if verdict == "bad":
                rec.add(kind="k_cap_hit", step=step_i + 1, note=f"accepted last teacher step after k={k}")

        # accepted: it executes and enters the context
        messages.append(msg)
        if action["type"] == "no_tool_call":
            messages.append({"role": "tool", "tool_call_id": action.get("tool_call_id") or "x",
                             "content": "ERROR: no valid bash tool call in your reply."})
            stats["steps"] += 1
            continue

        cmd = action.get("command") or ""
        t1 = time.time()
        result = env_exec(cmd, CMD_TIMEOUT)
        tool_content = _fmt_result(result)
        messages.append({"role": "tool", "tool_call_id": action.get("tool_call_id") or "x",
                         "content": tool_content})
        # author = who wrote the step that actually ran (the policy, or the teacher via a correction)
        author = "teacher_rewrite" if k > 0 else "policy"
        rec.add(kind="execute", step=step_i + 1, actor="policy", author=author, command=cmd,
                output=tool_content, exec_s=round(time.time() - t1, 1), k=k)
        executed.append((cmd, tool_content))
        stats["steps"] += 1

        if action["type"] == "done" or SUBMIT_MARKER in tool_content:
            stats["ended"] = "done"; break

    return messages, stats


def _notice(executed, msg):
    """Repetition facts for the judge; "" when the notice is off or the current command is not identical to the
    immediately previous one (input only; the output is not considered)."""
    if not REPEAT_NOTICE:
        return ""
    return T.repetition_facts(executed, T._first_command(msg))


def _judge(model, base, ctx, act, rec, step, stats, own_rewrite=False, notice=""):
    prompt = T.judge_prompt(ctx, act, notice)
    if notice:
        stats["notice_fired"] += 1
    for attempt in (1, 2):               # an unparseable reply is retried once, then the step counts as good
        t0 = time.time()
        est_in = len(prompt) // 3          # conservative chars -> tokens
        budget = max(1024, min(JUDGE_MAX_TOKENS, TEACHER_CTX - est_in - 1024))
        try:
            r = _chat(model, [{"role": "user", "content": prompt}], base,
                      max_tokens=budget, temperature=TEACHER_TEMP, response_format=T.judge_response_format())
        except litellm.exceptions.BadRequestError as e:
            rec.add(kind="judge_bad_request", step=step, attempt=attempt, error=str(e)[:200])
            stats["judge_bad_request"] = stats.get("judge_bad_request", 0) + 1
            return "good", {}
        except litellm.exceptions.ContextWindowExceededError as e:
            rec.add(kind="judge_context_exceeded", step=step, attempt=attempt, error=str(e)[:200])
            stats["judge_context_exceeded"] = stats.get("judge_context_exceeded", 0) + 1
            return "good", {}          # cannot judge; must not block the rollout
        txt = r.choices[0].message.content or ""
        try:
            verdict, raw = T.parse_verdict(txt)
            rec.add(kind="judge", step=step, own_rewrite=own_rewrite, input=prompt,
                    output_raw=txt, verdict=verdict, fields=raw, notice=notice,
                    response_format=T.judge_response_format(), judge_prompt_sha=T.JUDGE_PROMPT_SHA,
                    llm_s=round(time.time() - t0, 1))
            if notice and verdict == "bad":
                stats["notice_bad"] += 1
            return verdict, raw
        except Exception as e:
            rec.add(kind="judge_unparseable", step=step, attempt=attempt,
                    input=prompt, output_raw=txt, error=str(e),
                    truncated=("</think>" not in txt),
                    finish_reason=getattr(r.choices[0], "finish_reason", None))
    stats["judge_unparseable"] += 1
    return "good", {}


def _rewrite(model, base, ctx, rec, step):
    """The teacher's correction. It sees the context only, never the rejected step."""
    prompt = T.REWRITE_PROMPT.format(ctx=ctx)
    t0 = time.time()
    est_in = len(prompt) // 3          # conservative chars -> tokens, as in the judge call
    budget = max(1024, min(REWRITE_MAX_TOKENS, TEACHER_CTX - est_in - 1024))   # clamped to the served window
    try:
        r = _chat(model, [{"role": "user", "content": prompt}], base, tools=TOOL_SCHEMAS, temperature=TEACHER_TEMP,
                  max_tokens=budget)
    except litellm.exceptions.BadRequestError as e:
        rec.add(kind="rewrite_bad_request", step=step, error=str(e)[:200])
        return None, {"type": "no_tool_call", "command": None, "tool_call_id": None}
    except litellm.exceptions.ContextWindowExceededError as e:
        rec.add(kind="rewrite_context_exceeded", step=step, error=str(e)[:200])
        return None, {"type": "no_tool_call", "command": None, "tool_call_id": None}
    msg = r.choices[0].message.model_dump()
    action = _extract_tool_call(msg)
    rec.add(kind="rewrite", step=step, input=prompt, output_raw=msg,
            step_rendered=T.render_step(msg), llm_s=round(time.time() - t0, 1),
            max_tokens=budget, finish_reason=getattr(r.choices[0], "finish_reason", None),
            note="rewrite sees CONTEXT ONLY, never the rejected step")
    return msg, action
