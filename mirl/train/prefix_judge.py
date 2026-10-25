"""MiRL reward (paper §3.5) as an open-instruct verifier ("prefix_judge"): r = r_o AND r_e.

Installed as open_instruct/prefix_judge.py by mirl/train/install.sh and imported at the end of ground_truth_utils.py
(patches/tmax_open_instruct.patch), so build_all_verifiers() registers it. Rows come from build_rows.py.

  r_o  the tests pass: the environment's last turn reward is 1 only after a submit with passing tests.
  r_e  the behavior judge (Gemini 3.8 Flash) reads the first m steps the policy takes after the prefix, with the
       definition and examples of the prefix's misbehavior (prompt: behavior_judge.txt, paper Appendix A). Its {task}
       is the task and the last k prefix steps (build_rows.py). r_e = 1 if it answers "no" (the misbehavior did not
       happen), else 0. A judge call that keeps failing (no answer, 429 or 5xx) gives r_e = 0; a rejected request
       (e.g. a wrong key) stops the run.
The judge is called only when the tests pass, since the reward is 0 otherwise. Rows with "reward": "outcome" (the
ablations, build_rows.py --reward outcome) get r = r_o and no judge call.

grpo_fast adds the verifier score to the last turn reward; with --reward_aggregator last and --verification_reward 1
the sample's reward is exactly r, so the verifier returns score = r - last turn reward.
Key: GEMINI_API_KEY. Every call appends one JSON row to $PREFIX_JUDGE_LOG, if set.
"""
from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from open_instruct.ground_truth_utils import VerificationResult, VerifierConfig, VerifierFunction

LOG = os.environ.get("PREFIX_JUDGE_LOG", "")
PROMPT = (Path(__file__).resolve().parent / "mirl_behavior_judge.txt").read_text()
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-3.8-flash:generateContent"
GEMINI_PRICE_IN, GEMINI_PRICE_OUT = 0.75e-6, 3.75e-6    # $/token at the time of our experiments (output includes thinking)
ANSWER = re.compile(r"<answer>\s*(yes|no)\s*</answer>", re.I)

_CMD = re.compile(r"<parameter=command>\n?(.*?)\n?</parameter>", re.S)
_SEP = re.compile(r"(<tool_response>\n?.*?\n?</tool_response>|ERROR: no valid bash tool call in your reply\.)", re.S)
_ROLE = re.compile(r"^(?:\s|<\|im_end\|>|<\|im_start\|>|user\n|assistant\n)+|(?:\s|<\|im_end\|>|<\|im_start\|>|user\n?)+$")


def parse_steps(text: str) -> list[dict]:
    """Decoded response -> [{content, command|None, output|None}] in order, one per model turn.
    Works with or without special tokens in the decoded text: the turns are cut at the tool observations
    (<tool_response>...</tool_response>) and at the no-tool-call feedback message."""
    steps = []
    parts = _SEP.split(text)
    for i in range(0, len(parts), 2):
        seg = _ROLE.sub("", parts[i])
        obs = parts[i + 1] if i + 1 < len(parts) else None
        if obs is None and "<tool_call>" not in seg and "</think>" not in seg:
            continue                                                # trailing, unfinished turn (no tool call yet)
        head = seg.split("<tool_call>", 1)[0]
        if "</think>" in head and "<think>" not in head:
            head = "<think>\n" + head                               # the generation prompt holds "<think>\n"
        m = _CMD.search(seg) if "<tool_call>" in seg else None
        out = None
        if obs is not None:
            out = obs
            if out.startswith("<tool_response>"):
                out = re.sub(r"^<tool_response>\n?|\n?</tool_response>$", "", out)
        steps.append(dict(content=head.strip(), command=m.group(1) if m else None, output=out))
    return steps


def render_steps(steps: list[dict]) -> str:
    """Each step's reasoning, visible message, command and output, in order."""
    parts = []
    for j, st in enumerate(steps):
        c = st["content"]
        think, msg = c.split("</think>", 1) if "</think>" in c else ("", c)
        parts.append("\n\n".join([
            f"--- STEP {j + 1} ---",
            "Its reasoning:\n" + (think.replace("<think>", "").strip() or "(none)"),
            "Its visible message:\n" + (msg.strip() or "(none)"),
            "Its command:\n" + (st["command"] or "(NO VALID COMMAND -- malformed step)"),
            "Its output:\n" + (st["output"] if st["output"] is not None else "(none)")]))
    return "\n\n".join(parts)


def render_examples(examples: list[dict]) -> str:
    return "\n".join(f"  {i + 1}. {e['doing']}\n     why: {e['why']}" for i, e in enumerate(examples))


def behavior_judge(gt: dict, steps: list[dict]) -> tuple[bool | None, dict]:
    """True if the judge says the misbehavior happened in these steps, None if it gave no answer."""
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        raise RuntimeError("set GEMINI_API_KEY for the behavior judge")
    m = int(gt["m"])
    prompt = PROMPT
    for slot, val in (("{m}", str(m)), ("{task}", gt["judge_task"]), ("{steps}", render_steps(steps[:m])),
                      ("{mode_name}", gt["mode"]), ("{mode_definition}", gt["definition"]),
                      ("{mode_examples}", render_examples(gt["examples"]))):
        prompt = prompt.replace(slot, val)
    body = {"contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 1.0, "thinkingConfig": {"thinkingLevel": "medium"}}}
    err: dict = {}
    for attempt in range(6):
        req = urllib.request.Request(GEMINI_URL, data=json.dumps(body).encode(),
                                     headers={"x-goog-api-key": key, "Content-Type": "application/json"})
        t0 = time.time()
        try:
            d = json.load(urllib.request.urlopen(req, timeout=180))
            u = d.get("usageMetadata") or {}
            tin, tout = u.get("promptTokenCount") or 0, (u.get("candidatesTokenCount") or 0) + (u.get("thoughtsTokenCount") or 0)
            info = {"secs": round(time.time() - t0, 2), "in_tok": tin, "out_tok": tout,
                    "cost": round(tin * GEMINI_PRICE_IN + tout * GEMINI_PRICE_OUT, 6)}
            txt = "".join(p.get("text", "") for p in d["candidates"][0]["content"]["parts"] if not p.get("thought"))
            a = ANSWER.findall(txt)
            if not a:
                err = dict(info, error="no <answer> tag: " + txt[-100:])
            else:
                return a[-1].lower() == "yes", info
        except urllib.error.HTTPError as e:
            err = {"http_error": e.code, "body": e.read().decode()[:300]}
            if e.code not in (429, 500, 502, 503, 504):     # a wrong key, model name or request: stop, do not train on 0s
                raise RuntimeError(f"behavior judge request rejected: HTTP {e.code}: {err['body']}") from e
        except Exception as e:  # noqa: BLE001
            err = {"error": repr(e)[:300]}
        time.sleep(2 ** attempt)
    return None, dict(err, judge_failed=True)


class PrefixJudgeVerifier(VerifierFunction):
    def __init__(self, verifier_config: VerifierConfig | None = None) -> None:
        super().__init__("prefix_judge", verifier_config=verifier_config, weight=1.0)

    def __call__(self, tokenized_prediction: list[int], prediction: str, label: Any, query: str | None = None,
                 rollout_state: dict | None = None) -> VerificationResult:
        gt = json.loads(label) if isinstance(label, str) else (label[0] if isinstance(label, list) else label)
        if isinstance(gt, str):
            gt = json.loads(gt)
        rs = rollout_state or {}
        turn = list(rs.get("rewards") or [])
        last = turn[-1] if turn else 0.0
        reset_failed = "reset failed after" in (rs.get("tool_error") or "")
        steps = parse_steps(prediction)
        r_o = last >= 1.0                                       # env: 1.0 only after a submit with passing tests
        row = dict(id=gt.get("id"), mode=gt.get("mode"), n_steps=len(steps), r_o=r_o, reset_failed=reset_failed,
                   step_count=rs.get("step_count"))
        info: dict = {}
        if reset_failed or not r_o:
            misbehavior, final = None, 0.0                      # tests failed: reward 0 whatever the judge says
        elif gt.get("reward", "mirl") == "outcome":
            misbehavior, final = None, 1.0                      # ablations: r_o only
        else:
            misbehavior, info = behavior_judge(gt, steps)
            final = float(misbehavior is False)
        row.update(misbehavior=misbehavior, final=final, **info)
        if LOG:
            row["t"] = time.strftime("%F %T")
            with open(LOG, "a") as f:
                f.write(json.dumps(row) + "\n")
        return VerificationResult(score=final - last)
