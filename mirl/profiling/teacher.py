"""Prompts and parsing for the two roles of the teacher model (paper §3.2, appendix prompts):
the error localizer (JUDGE_PROMPT) and the teacher's correction (REWRITE_PROMPT). They are two separate calls.

The judge sees the context and the proposed step, and returns its verdict under a JSON schema (JUDGE_SCHEMA).
The correction sees the context ONLY -- never the rejected step -- and writes its own next step in the agent's format
(a <think> block + one bash tool call).
"""
from __future__ import annotations
import hashlib, json, re
from typing import Any

JUDGE_PROMPT = """You are reviewing one step of an AI agent working on a command-line task.

Below is EXACTLY what the agent saw, and EXACTLY what it did next.
You are seeing the agent's context up to this point and nothing beyond it.
You do not know whether the task eventually succeeded.

Judge this one step.

A step is GOOD if an experienced engineer would plausibly do it here.
A step is BAD if it still works, but raises the chance of failure, and an
experienced engineer would not do it. Examples of bad-but-working steps:
acting on data it has not inspected; writing one large script that does
everything at once instead of building up in checkable pieces; declaring
work finished without verifying the actual output.
A step that cannot work from this state is also bad.

Do not penalise a step for being slow or for taking a different approach
than you would. Only flag it if it makes failure more likely.

A step here is a short <think> reasoning block followed by ONE bash command.

Keep your reasoning SHORT: at most 150 words of thinking before you answer.
Do not re-derive the whole task. Judge the one step.

Reply in this JSON form, nothing else. Fill the fields IN ORDER -- the justification
comes before the verdict, so you must state what the command does before you judge it.

{{
  "command_seen": "<one short line naming what the command does, e.g. 'cat on a binary PDF'>",
  "reason": "<one sentence. REQUIRED. Never null, never empty, on good verdicts too.>",
  "verdict": "good" or "bad"
}}

Before you answer, check "command_seen" against your own reason: if the command you
described is one you would reject, the verdict is "bad".

=== WHAT THE AGENT SAW ===
{ctx}

=== WHAT THE AGENT DID ===
{act}
"""

# Every judge reply is decoded under this schema (vLLM structured output).
# Fields in order, all required, no other fields: the justification is written before the verdict.
JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "command_seen": {"type": "string", "minLength": 1},
        "reason": {"type": "string", "minLength": 1},
        "verdict": {"type": "string", "enum": ["good", "bad"]},
    },
    "required": ["command_seen", "reason", "verdict"],
    "additionalProperties": False,
}


def judge_response_format() -> dict:
    """OpenAI-style response_format that vLLM turns into constrained decoding."""
    return {"type": "json_schema",
            "json_schema": {"name": "step_verdict", "strict": True, "schema": JUDGE_SCHEMA}}


JUDGE_PROMPT_SHA = hashlib.sha256(JUDGE_PROMPT.encode()).hexdigest()[:16]
JUDGE_SCHEMA_SHA = hashlib.sha256(json.dumps(JUDGE_SCHEMA, sort_keys=True).encode()).hexdigest()[:16]

REWRITE_PROMPT = """You are an expert engineer working on a command-line task.

Below is EXACTLY the state so far: the task, and every command already run with
its output. Write the CORRECT next step yourself.

Write a short <think> block explaining your reasoning, then call the `bash` tool
with exactly ONE command. The command may chain shell with && or ; but it is one
command string.

Do not repeat work that the output above already shows is done. Build up in small
checkable pieces rather than one large script. Do not declare the task finished
without verifying the real output.

=== THE STATE SO FAR ===
{ctx}
"""

_FENCE = re.compile(r"^```[a-zA-Z]*\n(.*)\n```$", re.S)


def _strip_fence(s: str) -> str:
    s = s.strip()
    m = _FENCE.match(s)
    return m.group(1).strip() if m else s


def _first_command(msg: dict[str, Any]) -> str:
    """The command of the FIRST tool call, exactly as the harness selects it."""
    for tc in msg.get("tool_calls") or []:
        a = (tc.get("function") or {}).get("arguments")
        if isinstance(a, str):
            try: a = json.loads(a)
            except json.JSONDecodeError: a = {}
        if isinstance(a, dict) and a.get("command"):
            return a["command"]
        return ""        # first call exists but carries no command
    return ""


def render_context(messages: list[dict[str, Any]]) -> str:
    """Flatten the chat history into the plain text the teacher reads.

    Excludes the system prompt (the teacher gets its own instructions).
    """
    out = []
    for m in messages:
        role = m.get("role")
        if role == "system":
            continue
        if role == "user":
            out.append(f"--- TASK ---\n{m.get('content','')}")
        elif role == "assistant":
            think = (m.get("content") or "").strip()
            # FIRST tool call only: that is the one the harness executes
            # (_extract_tool_call takes tool_calls[0]). Keeping the last one would show the
            # judge a different command than the one that ran.
            cmd = _first_command(m)
            out.append(f"--- AGENT STEP ---\n{think}\n$ {cmd}")
        elif role == "tool":
            out.append(f"--- OUTPUT ---\n{m.get('content','')}")
    return "\n\n".join(out)


def render_step(msg: dict[str, Any]) -> str:
    """The step under review, as the teacher sees it."""
    think = (msg.get("content") or "").strip()
    cmd = _first_command(msg)   # the command the harness will run, not the last one listed
    return f"{think}\n$ {cmd}" if cmd else f"{think}\n(NO VALID COMMAND -- malformed step)"


def parse_verdict(text: str) -> tuple[str, dict]:
    """Parse the judge's JSON. Returns (verdict, raw_fields).

    Unparseable -> the caller retries once, then treats the step as 'good'.
    """
    # The teacher emits a <think> block before the JSON, and that reasoning can
    # contain braces. Take only what follows the LAST </think>, then parse.
    s = text
    if "</think>" in s:
        s = s.rsplit("</think>", 1)[1]
    s = _strip_fence(s)
    i, j = s.find("{"), s.rfind("}")
    if i >= 0 and j > i:
        s = s[i:j+1]
    d = json.loads(s)
    v = str(d.get("verdict", "")).strip().lower()
    if v not in ("good", "bad"):
        raise ValueError(f"verdict not good/bad: {v!r}")
    return v, d

REPEAT_RUBRIC = """
Sometimes a step repeats a command that already ran. Repetition is not bad by itself.
Ask one question: could this run return something different from the last one?

A repeat is GOOD when the agent waits for something outside its control, and the repeat
is how it waits: checking `nvidia-smi` until a GPU frees, polling a build or a background
job, waiting for a service to answer. Identical output is normal here. The agent watches
for the moment it changes.

A repeat is GOOD when something in between could change the result: the agent edited a
file, started a process, installed a package, or fixed the command, and now runs it again
to see the effect.

A repeat is BAD when nothing since the last identical run could change the result, and the
agent learned nothing. It re-reads the same state, or re-runs the same failing check,
because it does not know what to do next. Say so. The rewrite will propose a real action.

The REPETITION FACTS below are computed by the harness, not guessed. They tell you that this
step repeats the previous command exactly (input only; the output is not considered). Use them.
"""


def repetition_facts(executed: list[tuple[str, str]], cmd: str) -> str:
    """executed = [(command, output)] so far. Trigger: fire ONLY
    when the current step's command is identical to the immediately previous executed command
    (input == last input). The output is NOT considered. Otherwise return "" and the judge
    prompt is unchanged."""
    if not cmd or not executed:
        return ""
    if executed[-1][0] != cmd:
        return ""
    run = 0                                       # trailing steps in a row with this command
    for k in range(len(executed) - 1, -1, -1):
        if executed[k][0] == cmd: run += 1
        else: break
    total = sum(1 for c, _ in executed if c == cmd)
    return "\n".join([
        "The command in this step is identical to the command the agent ran in the previous step.",
        f"This exact command has now run {total} time{'s' if total != 1 else ''} in this trajectory"
        + (f" ({run} in a row)." if run >= 2 else "."),
        "The agent repeats its previous action with no change in between.",
    ])


def judge_prompt(ctx: str, act: str, notice: str = "") -> str:
    """The judge prompt; with a non-empty notice, the rubric and the facts travel with it."""
    p = JUDGE_PROMPT.format(ctx=ctx, act=act)
    if notice:
        p = p.replace("A step here is a short <think> reasoning block followed by ONE bash command.",
                      "A step here is a short <think> reasoning block followed by ONE bash command.\n" + REPEAT_RUBRIC)
        p = p.replace("=== WHAT THE AGENT DID ===",
                      "=== REPETITION FACTS (computed, not guessed) ===\n" + notice + "\n\n=== WHAT THE AGENT DID ===")
    return p
