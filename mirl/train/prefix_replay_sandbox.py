"""MiRL training environment for open-instruct's grpo_fast: continues a misbehavior prefix in a fresh sandbox that
the sandbox service (sandbox_server.py) has put into the state at the end of the prefix.

Installed into open_instruct/environments/ by mirl/train/install.sh. The sandboxes run as root in sandbox_server.py; this
class (running as the normal user inside a Ray actor) only talks to that service over HTTP, so the shell, output format,
command timeout and final tests are the ones of error profiling.

Per-sample reset kwargs (dataset column env_config -> env_configs[0].kwargs):
  task_id, replay: [{command, output}], max_new_steps (64 - prefix steps), prefix_id
step(bash):  run the command; done at submit (COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT) -> final tests, reward 0/1;
             done at max_new_steps (reward 0).
The reset observation is not used by grpo_fast: the prompt is the prefix messages from the dataset row."""
import asyncio
import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, ClassVar

from openenv.core.env_server.types import State

from open_instruct import logger_utils

from .base import BaseEnvConfig, EnvCall, RLEnvironment, StepResult

logger = logger_utils.setup_logger(__name__)

SUBMIT_MARKER = "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"
NO_TOOL_CALL_MESSAGE = "ERROR: no valid bash tool call in your reply."          # as in mirl/profiling/rollout.py
# rl_data.generator.sample_solutions.TOOL_SCHEMAS (the tool definition of error profiling, verbatim)
TOOL_SCHEMAS = [{"type": "function", "function": {"name": "bash", "description": "Execute a bash command. Each command runs in a "
                 "new subshell.", "parameters": {"type": "object", "properties": {"command": {"type": "string", "description":
                 "The bash command to execute."}}, "required": ["command"]}}}]
EVENT_LOG = os.environ.get("PREFIX_ENV_LOG", "")


def _post(url: str, body: dict, timeout: float) -> dict:
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    for attempt in range(8):  # the service refused/reset the connection before reading it (burst): safe to resend
        try:
            return json.load(urllib.request.urlopen(req, timeout=timeout))
        except (ConnectionResetError, ConnectionRefusedError) as e:
            if attempt == 7:
                raise
            time.sleep(0.5 * 2**min(attempt, 4))
        except urllib.error.URLError as e:
            if not isinstance(e.reason, (ConnectionResetError, ConnectionRefusedError)) or attempt == 7:
                raise
            time.sleep(0.5 * 2**min(attempt, 4))


class PrefixReplaySandboxEnv(RLEnvironment):
    config_name = "prefix_replay_sandbox"

    def __init__(self, server_url: str = "http://127.0.0.1:8650", http_timeout: float = 3600.0, **_: Any):
        self._url = server_url.rstrip("/")
        self._http_timeout = http_timeout
        self._sid: str | None = None
        self._task_id: str | None = None
        self._prefix_id: str | None = None
        self._max_new_steps = 5
        self._step_count = 0
        self._submitted = False
        self._metrics: dict[str, float] = {}

    @classmethod
    def get_tool_definitions(cls) -> list[dict]:
        return TOOL_SCHEMAS

    def get_tool_call_format_error_message(self) -> str | None:
        return NO_TOOL_CALL_MESSAGE

    async def _call(self, path: str, body: dict) -> dict:
        return await asyncio.to_thread(_post, self._url + path, body, self._http_timeout)

    async def _close(self) -> None:
        if self._sid is not None:
            sid, self._sid = self._sid, None
            try:
                await self._call("/close", {"sid": sid})
            except Exception as e:  # noqa: BLE001
                logger.warning("prefix_replay_sandbox close failed: %s", e)

    def _event(self, **kw: Any) -> None:
        if EVENT_LOG:
            kw.update(t=time.strftime("%F %T"), prefix_id=self._prefix_id, task_id=self._task_id)
            with open(EVENT_LOG, "a") as f:
                f.write(json.dumps(kw) + "\n")

    async def reset(self, task_id: str | None = None, **kwargs: Any) -> tuple[StepResult, list[dict]]:
        await self._close()
        self._task_id, self._prefix_id = task_id, kwargs.get("prefix_id")
        self._max_new_steps = int(kwargs.get("max_new_steps", 5))
        self._step_count, self._submitted = 0, False
        t0 = time.time()
        try:
            r = await self._call("/reset", {"task_id": task_id, "replay": kwargs.get("replay") or [], "tag": str(self._prefix_id)})
        except Exception as e:      # a service error (e.g. HTTP 500) must cost one sample, not the run
            self._event(ev="reset", ok=False, retries=0, secs=round(time.time() - t0, 1), why=[f"service error: {type(e).__name__}"])
            raise RuntimeError(f"prefix replay reset failed after a sandbox service error for {self._prefix_id}: {e}") from e
        retries = len(r.get("tries", [])) - (1 if r.get("ok") else 0)
        self._event(ev="reset", ok=bool(r.get("ok")), retries=retries, secs=round(time.time() - t0, 1),
                    why=[t.get("why") for t in r.get("tries", [])])
        if not r.get("ok"):
            # "reset failed after" is a grpo_fast zero-reward marker (SWERL_RESET_FAILURE_ZERO_REWARD=1); with
            # PREFIX_DROP_RESET_FAILED=1 (patches/tmax_open_instruct.patch) the sample then leaves its group and the loss.
            raise RuntimeError(f"prefix replay reset failed after {len(r.get('tries', []))} tries for {self._prefix_id}: "
                               f"{[t.get('why') for t in r.get('tries', [])]}")
        self._sid = r["sid"]
        return StepResult(result="", metadata={"task_id": task_id, "replay_s": r.get("replay_s"), "retries": retries}), TOOL_SCHEMAS

    async def step(self, call: EnvCall) -> StepResult:
        if self._sid is None:
            return StepResult(result="Sandbox is not available.", reward=0.0, done=True, metadata={"error": "no sandbox"})
        if call.name != "bash":
            return StepResult(result=NO_TOOL_CALL_MESSAGE)
        self._step_count += 1
        cmd = (call.args or {}).get("command") or ""
        r = await self._call("/exec", {"sid": self._sid, "command": cmd})
        out = r["output"]
        if r.get("submitted"):
            self._submitted = True
            t = await self._call("/tests", {"sid": self._sid})
            self._event(ev="submit", step=self._step_count, test_pass=t["pass"], test_s=t.get("secs"))
            await self._close()
            return StepResult(result=out, reward=1.0 if t["pass"] else 0.0, done=True,
                              metadata={"test_pass": t["pass"], "submitted": True})
        if self._step_count >= self._max_new_steps:
            self._event(ev="step_limit", step=self._step_count)
            await self._close()
            return StepResult(result=out, reward=0.0, done=True, metadata={"step_limit": True})
        return StepResult(result=out, reward=0.0)

    def state(self) -> State:
        return State(episode_id=self._prefix_id, step_count=self._step_count)

    async def close(self) -> None:
        await self._close()

    async def shutdown(self) -> None:
        await self._close()


@dataclass
class PrefixReplaySandboxEnvConfig(BaseEnvConfig):
    tool_class: ClassVar[type[RLEnvironment]] = PrefixReplaySandboxEnv
    server_url: str = "http://127.0.0.1:8650"
    http_timeout: float = 3600.0

