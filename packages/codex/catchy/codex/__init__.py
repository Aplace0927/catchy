import json
import logging
import os
import shlex
from contextlib import contextmanager
from pathlib import Path
from typing import AsyncIterator

from catchy.core.agents.protocols import Agent
from catchy.core.challenge.models import Challenge
from catchy.core.webhook.models import Webhook
from codex_app_server import (
    AppServerConfig,
    AsyncCodex,
    TextInput,
)
from codex_app_server.generated.v2_all import (
    AgentMessageDeltaNotification,
    CommandExecutionOutputDeltaNotification,
    ErrorNotification,
    FileChangeOutputDeltaNotification,
    ItemCompletedNotification,
    ItemStartedNotification,
    McpToolCallProgressNotification,
    PlanDeltaNotification,
    ReasoningSummaryPartAddedNotification,
    ReasoningSummaryTextDeltaNotification,
    TerminalInteractionNotification,
    TurnCompletedNotification,
    TurnStatus,
)
from codex_app_server.models import JsonObject
from docker import DockerClient

_LOGGER = logging.getLogger(__name__)


class CodexAgent(Agent):
    key: str = "codex"
    _dockerfile = Path(__file__).parent / "Dockerfile"

    def __init__(
        self,
        api_key: str,
        docker_client: DockerClient | None = None,
        model: str = "gpt-5.4",
        config: JsonObject = {},
    ):
        if docker_client is None:
            docker_client = DockerClient.from_env()

        self._docker_client = docker_client
        self._model = model
        self._config = config
        self._api_key = api_key
        self._docker_user = f"{os.getuid()}:{os.getgid()}"
        self._id = f"agent-{self.key}-{model}-{hash(json.dumps(config, sort_keys=True)) % 10000:05d}"

        _LOGGER.info(f"({self._id}) Building Docker image from {self._dockerfile}...")
        self._docker_image, _ = self._docker_client.images.build(
            path=str(self._dockerfile.parent), dockerfile=self._dockerfile.name
        )
        _LOGGER.info(f"({self._id}) Docker image built: {self._docker_image.id}")

    async def stream(
        self, challenge: Challenge, workspace: Path, webhook: Webhook | None = None
    ) -> AsyncIterator[str]:
        if not workspace.exists():
            raise ValueError(f"workspace does not exist: {workspace}")
        if not workspace.is_dir():
            raise ValueError(f"workspace is not a directory: {workspace}")

        with self._docker_container(
            challenge=challenge, workspace=workspace
        ) as container:
            async with AsyncCodex(
                config=AppServerConfig(
                    launch_args_override=(  # pyright: ignore[reportArgumentType]
                        "docker",
                        "exec",
                        "-i",
                        "--user",
                        self._docker_user,
                        "--env",
                        "HOME=/workspace",
                        "--env",
                        "CODEX_HOME=/workspace/.codex",
                        container.id,
                        "codex",
                        "app-server",
                        "--listen",
                        "stdio://",
                    )
                )
            ) as codex:
                threads = (await codex.thread_list()).data

                match threads:
                    case [thread]:
                        _LOGGER.info(
                            f"({self._id})({challenge.id}) Resuming existing thread: {thread.id}"
                        )
                        thread = await codex.thread_resume(thread.id)
                    case []:
                        _LOGGER.info(
                            f"({self._id})({challenge.id}) Starting new thread"
                        )
                        thread = await codex.thread_start(
                            model=self._model, config=self._config
                        )
                    case _:
                        raise RuntimeError(
                            f"Expected at most one thread, but found {len(threads)}"
                        )

                prompt = f"""You are the best at solving CTF challenges.
Solve the challenge in /challenge and explain.

Useful tools are already installed in this container:
- Web challenges: use `curl`, `wget`, `nc`, `socat`, `nmap`, `jq`, Python
  `requests`, `flask`, `PyJWT`, and Playwright. Use `chrome` for Playwright
  Chromium, or `chrome-devtools` to start headless Chromium with the Chrome
  DevTools Protocol on container port 9222.
- Pwn / reversing: use `file`, `strings`, `xxd`, `objdump`, `readelf`,
  `patchelf`, `gdb`, `gdbserver`, `ltrace`, `strace`, `r2`, `pwntools`,
  `angr`, `capstone`, `unicorn`, and `ROPgadget`.
- Crypto / math: use Python with `pycryptodome`, `sympy`, `gmpy2`, `z3`,
  `fpylll`, `numpy`, `scipy`, plus `openssl`, `RsaCtfTool`, `flatter`,
  `cado-nfs`, and `sage` if available.
- Forensics / steg: use `binwalk`, `exiftool`, `steghide`, `stegseek`,
  `zsteg`, `pngcheck`, `imagemagick`, `foremost`, `sleuthkit`, `testdisk`,
  `dcfldd`, `volatility3`, `john`, and archive tools like `zip`, `unzip`,
  `7z`, `xz`, and `zstd`.
- Media / OCR: use `ffmpeg`, `sox`, `tesseract`, `pytesseract`, and `Pillow`.
- Build / runtime helpers: use `gcc`, `g++`, `clang`, `make`, `cmake`,
  `ninja`, `meson`, `node`, `npm`, `ruby`, `java`, `uv`, and Python 3.12 from
  `/opt/ctf-venv`.
- Challenge containers: use `podman` or `buildah` when a challenge provides a
  Dockerfile or service image.

<challenge-description>{challenge.description}</challenge-description>"""

                if webhook is not None:
                    prompt += f"""Frequently share any findings or trial and errors to the webhook at {webhook.url} in {webhook.preferred_language or "English"}."""

                turn = await thread.turn(TextInput(prompt))

                item_message = ""

                async for event in turn.stream():
                    match event.payload:
                        case ItemStartedNotification():
                            item_message = ""
                        case ItemCompletedNotification():
                            if item_message.strip():
                                yield item_message
                            item_message = ""
                        case ErrorNotification() as payload if (
                            payload.turn_id == turn.id
                        ):
                            if payload.will_retry:
                                _LOGGER.warning(
                                    f"({self._id})({challenge.id}) Codex turn error; server will retry: {payload.error.message}"
                                )
                                continue
                            raise RuntimeError(
                                f"Codex reported a non-retryable turn error: {payload.error.message}"
                            )
                        case TurnCompletedNotification() as payload if (
                            payload.turn.id == turn.id
                        ):
                            match payload.turn.status:
                                case TurnStatus.completed:
                                    pass
                                case TurnStatus.failed:
                                    raise RuntimeError(
                                        f"Codex turn failed: {payload.turn.error.message if payload.turn.error else 'unknown error'}"
                                    )
                                case TurnStatus.interrupted:
                                    raise RuntimeError(
                                        f"Codex turn was interrupted: {payload.turn.error.message if payload.turn.error else 'unknown error'}"
                                    )
                                case TurnStatus.in_progress:
                                    raise RuntimeError(
                                        "Codex emitted turn/completed while the turn is still in progress"
                                    )
                        case (
                            AgentMessageDeltaNotification()
                            | PlanDeltaNotification()
                            | ReasoningSummaryTextDeltaNotification()
                            | CommandExecutionOutputDeltaNotification()
                            | FileChangeOutputDeltaNotification()
                        ) as payload:
                            item_message += payload.delta
                        case ReasoningSummaryPartAddedNotification() as payload:
                            _LOGGER.info(f"({self._id})({challenge.id}) {payload}")
                        case TerminalInteractionNotification() as payload:
                            _LOGGER.info(f"({self._id})({challenge.id}) {payload}")
                        case McpToolCallProgressNotification() as payload:
                            _LOGGER.info(f"({self._id})({challenge.id}) {payload}")
                        case _:
                            _LOGGER.debug(
                                f"({self._id})({challenge.id}) Ignoring Codex event: {event.method}"
                            )

    @contextmanager
    def _docker_container(self, challenge: Challenge, workspace: Path):
        assert workspace.is_dir()

        container = self._docker_client.containers.run(
            self._docker_image,
            detach=True,
            stdin_open=True,
            # Codex uses bubblewrap for its Linux sandbox; Docker's default confinement blocks the user/mount namespace setup bwrap needs.
            cap_add=["SYS_ADMIN"],
            security_opt=["seccomp=unconfined", "apparmor=unconfined"],
            environment={
                "HOME": "/workspace",
                "CODEX_HOME": "/workspace/.codex",
                "CHROME_REMOTE_DEBUGGING_PORT": "9222",
            },
            ports={"9222/tcp": None},
            volumes={
                str(challenge.directory): {"bind": "/challenge", "mode": "ro"},
                str(workspace): {"bind": "/workspace", "mode": "rw"},
            },
        )

        _LOGGER.info(f"({self._id}) Started Docker container: {container.id}")
        container.reload()
        chrome_devtools_bindings = container.attrs["NetworkSettings"]["Ports"].get(
            "9222/tcp"
        )
        if chrome_devtools_bindings:
            chrome_devtools_url = (
                f"http://{chrome_devtools_bindings[0]['HostIp']}:"
                f"{chrome_devtools_bindings[0]['HostPort']}"
            )
            _LOGGER.info(
                f"({self._id}) Chrome DevTools Protocol available after running "
                f"`chrome-devtools`: {chrome_devtools_url}"
            )

        result = container.exec_run(
            cmd=[
                "sh",
                "-c",
                " && ".join(
                    [
                        "mkdir -p /workspace/.codex",
                        f"printf %s {shlex.quote(json.dumps({'auth_mode': 'apikey', 'OPENAI_API_KEY': self._api_key}))} > /workspace/.codex/auth.json",
                        f"chown -R {self._docker_user} /workspace",
                    ]
                ),
            ]
        )
        if result.exit_code != 0:
            logs = container.logs().decode()
            container.remove(force=True)
            raise RuntimeError(f"Failed to set up container auth file: {logs}")

        try:
            yield container
        finally:
            _LOGGER.info(
                f"({self._id}) Stopping and removing Docker container: {container.id}"
            )
            container.remove(force=True)
            _LOGGER.info(f"({self._id}) Docker container removed: {container.id}")
