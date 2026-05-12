from __future__ import annotations

import io
import json
import logging
import shlex
import tarfile
import tomllib
from contextlib import contextmanager
from decimal import Decimal
from pathlib import Path
from typing import Any, AsyncGenerator, Literal, cast

import tomli_w
from catchy.core.agents.models import (
    Chunk,
    Event,
    Interrupt,
    ItemCompleted,
    Nop,
    Prompt,
    Steer,
    Stop,
    TurnCompleted,
)
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
from docker import DockerClient
from docker.errors import DockerException
from docker.models.images import Image
from jinja2 import Template
from omegaconf import OmegaConf
from pydantic import (
    BaseModel,
    ConfigDict,
    ValidationInfo,
    field_serializer,
    field_validator,
)

_LOGGER = logging.getLogger(__name__)


class TokenUsage(BaseModel):
    model_config = ConfigDict(frozen=True)

    input_tokens: int = 0
    cached_input_tokens: int = 0
    output_tokens: int = 0

    @field_validator(
        "input_tokens", "cached_input_tokens", "output_tokens", mode="before"
    )
    @classmethod
    def _deserialize_token_count(cls, value: object) -> int:
        return _int_value(value)

    @property
    def billable_input_tokens(self) -> int:
        return max(self.input_tokens - self.cached_input_tokens, 0)

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def __add__(self, other: "TokenUsage") -> "TokenUsage":
        return TokenUsage(
            input_tokens=self.input_tokens + other.input_tokens,
            cached_input_tokens=self.cached_input_tokens + other.cached_input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
        )


class ModelPricing(BaseModel):
    model_config = ConfigDict(frozen=True)

    input_per_million: Decimal
    cached_input_per_million: Decimal
    output_per_million: Decimal


class CostEstimate(BaseModel):
    model_config = ConfigDict(frozen=True)

    model: str
    usage: TokenUsage
    usd: Decimal
    pricing: ModelPricing | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "input_tokens": self.usage.input_tokens,
            "cached_input_tokens": self.usage.cached_input_tokens,
            "output_tokens": self.usage.output_tokens,
            "total_tokens": self.usage.total_tokens,
            "usd": str(self.usd),
            "pricing": None
            if self.pricing is None
            else {
                "input_per_million": str(self.pricing.input_per_million),
                "cached_input_per_million": str(self.pricing.cached_input_per_million),
                "output_per_million": str(self.pricing.output_per_million),
            },
        }


def _pricing(
    input_per_million: str,
    cached_input_per_million: str | None,
    output_per_million: str,
) -> ModelPricing:
    input_price = Decimal(input_per_million)
    return ModelPricing(
        input_per_million=input_price,
        # A missing cached-input price means the model has no cached-input discount.
        cached_input_per_million=Decimal(cached_input_per_million)
        if cached_input_per_million is not None
        else input_price,
        output_per_million=Decimal(output_per_million),
    )


MODEL_PRICING: dict[str, ModelPricing] = {
    "chat-latest": _pricing("5.00", "0.50", "30.00"),
    "gpt-5.5": _pricing("5.00", "0.50", "30.00"),
    "gpt-5.5-pro": _pricing("30.00", None, "180.00"),
    "gpt-5.4": _pricing("2.50", "0.25", "15.00"),
    "gpt-5.4-mini": _pricing("0.75", "0.075", "4.50"),
    "gpt-5.4-nano": _pricing("0.20", "0.02", "1.25"),
    "gpt-5.4-pro": _pricing("30.00", None, "180.00"),
    "gpt-5.3-codex": _pricing("1.75", "0.175", "14.00"),
    "gpt-5.2": _pricing("1.75", "0.175", "14.00"),
    "gpt-5.2-chat-latest": _pricing("1.75", "0.175", "14.00"),
    "gpt-5.2-codex": _pricing("1.75", "0.175", "14.00"),
    "gpt-5.2-pro": _pricing("21.00", None, "168.00"),
    "gpt-5.1": _pricing("1.25", "0.125", "10.00"),
    "gpt-5.1-chat-latest": _pricing("1.25", "0.125", "10.00"),
    "gpt-5.1-codex": _pricing("1.25", "0.125", "10.00"),
    "gpt-5.1-codex-max": _pricing("1.25", "0.125", "10.00"),
    "gpt-5.1-codex-mini": _pricing("0.25", "0.025", "2.00"),
    "gpt-5": _pricing("1.25", "0.125", "10.00"),
    "gpt-5-chat-latest": _pricing("1.25", "0.125", "10.00"),
    "gpt-5-codex": _pricing("1.25", "0.125", "10.00"),
    "gpt-5-mini": _pricing("0.25", "0.025", "2.00"),
    "gpt-5-nano": _pricing("0.05", "0.005", "0.40"),
    "gpt-5-pro": _pricing("15.00", None, "120.00"),
    "gpt-4.1": _pricing("2.00", "0.50", "8.00"),
    "gpt-4.1-mini": _pricing("0.40", "0.10", "1.60"),
    "gpt-4.1-nano": _pricing("0.10", "0.025", "0.40"),
    "gpt-4o": _pricing("2.50", "1.25", "10.00"),
    "gpt-4o-2024-05-13": _pricing("5.00", None, "15.00"),
    "gpt-4o-mini": _pricing("0.15", "0.075", "0.60"),
    "gpt-4o-mini-search-preview": _pricing("0.15", None, "0.60"),
    "gpt-4o-search-preview": _pricing("2.50", None, "10.00"),
    "gpt-4o-mini-realtime-preview": _pricing("0.60", "0.30", "2.40"),
    "gpt-4o-realtime-preview": _pricing("5.00", "2.50", "20.00"),
    "gpt-realtime-2": _pricing("4.00", "0.40", "24.00"),
    "gpt-realtime-1.5": _pricing("4.00", "0.40", "16.00"),
    "gpt-realtime-mini": _pricing("0.60", "0.06", "2.40"),
    "gpt-audio": _pricing("2.50", None, "10.00"),
    "gpt-audio-mini": _pricing("0.60", None, "2.40"),
    "gpt-4o-audio-preview": _pricing("2.50", None, "10.00"),
    "gpt-4o-mini-audio-preview": _pricing("0.15", None, "0.60"),
    "gpt-5-search-api": _pricing("1.25", "0.125", "10.00"),
    "codex-mini-latest": _pricing("1.50", "0.375", "6.00"),
    "computer-use-preview": _pricing("3.00", None, "12.00"),
    "o1": _pricing("15.00", "7.50", "60.00"),
    "o1-mini": _pricing("1.10", "0.55", "4.40"),
    "o1-preview": _pricing("15.00", "7.50", "60.00"),
    "o1-pro": _pricing("150.00", None, "600.00"),
    "o3": _pricing("2.00", "0.50", "8.00"),
    "o3-deep-research": _pricing("10.00", "2.50", "40.00"),
    "o3-mini": _pricing("1.10", "0.55", "4.40"),
    "o3-pro": _pricing("20.00", None, "80.00"),
    "o4-mini": _pricing("1.10", "0.275", "4.40"),
    "o4-mini-deep-research": _pricing("2.00", "0.50", "8.00"),
}
MODEL_PRICING.update(
    {
        "gpt-5.5-2026-04-23": MODEL_PRICING["gpt-5.5"],
        "gpt-5.5-pro-2026-04-23": MODEL_PRICING["gpt-5.5-pro"],
        "gpt-5.4-2026-03-05": MODEL_PRICING["gpt-5.4"],
        "gpt-5.4-mini-2026-03-17": MODEL_PRICING["gpt-5.4-mini"],
        "gpt-5.4-nano-2026-03-17": MODEL_PRICING["gpt-5.4-nano"],
        "gpt-5.4-pro-2026-03-05": MODEL_PRICING["gpt-5.4-pro"],
        "gpt-5.2-2025-12-11": MODEL_PRICING["gpt-5.2"],
        "gpt-5.2-pro-2025-12-11": MODEL_PRICING["gpt-5.2-pro"],
        "gpt-5.1-2025-11-13": MODEL_PRICING["gpt-5.1"],
        "gpt-5-2025-08-07": MODEL_PRICING["gpt-5"],
        "gpt-5-mini-2025-08-07": MODEL_PRICING["gpt-5-mini"],
        "gpt-5-nano-2025-08-07": MODEL_PRICING["gpt-5-nano"],
        "gpt-5-pro-2025-10-06": MODEL_PRICING["gpt-5-pro"],
        "gpt-4.1-2025-04-14": MODEL_PRICING["gpt-4.1"],
        "gpt-4.1-mini-2025-04-14": MODEL_PRICING["gpt-4.1-mini"],
        "gpt-4.1-nano-2025-04-14": MODEL_PRICING["gpt-4.1-nano"],
        "gpt-4o-2024-08-06": MODEL_PRICING["gpt-4o"],
        "gpt-4o-2024-11-20": MODEL_PRICING["gpt-4o"],
        "gpt-4o-mini-2024-07-18": MODEL_PRICING["gpt-4o-mini"],
        "computer-use-preview-2025-03-11": MODEL_PRICING["computer-use-preview"],
        "o1-2024-12-17": MODEL_PRICING["o1"],
        "o1-mini-2024-09-12": MODEL_PRICING["o1-mini"],
        "o1-preview-2024-09-12": MODEL_PRICING["o1-preview"],
        "o1-pro-2025-03-19": MODEL_PRICING["o1-pro"],
        "o3-2025-04-16": MODEL_PRICING["o3"],
        "o3-deep-research-2025-06-26": MODEL_PRICING["o3-deep-research"],
        "o3-mini-2025-01-31": MODEL_PRICING["o3-mini"],
        "o3-pro-2025-06-10": MODEL_PRICING["o3-pro"],
        "o4-mini-2025-04-16": MODEL_PRICING["o4-mini"],
        "o4-mini-deep-research-2025-06-26": MODEL_PRICING["o4-mini-deep-research"],
    }
)


def estimate_cost(model: str, usage: TokenUsage) -> CostEstimate:
    pricing = MODEL_PRICING.get(model)
    if pricing is None:
        return CostEstimate(model=model, usage=usage, usd=Decimal("0"), pricing=None)

    usd = (
        Decimal(usage.billable_input_tokens) * pricing.input_per_million
        + Decimal(usage.cached_input_tokens) * pricing.cached_input_per_million
        + Decimal(usage.output_tokens) * pricing.output_per_million
    ) / Decimal(1_000_000)

    return CostEstimate(
        model=model, usage=usage, usd=usd.quantize(Decimal("0.000001")), pricing=pricing
    )


def usage_from_codex_session_event(event: dict[str, Any]) -> TokenUsage:
    payload = event.get("payload")
    if not isinstance(payload, dict):
        return TokenUsage()
    payload = cast(dict[str, Any], payload)

    info = payload.get("info")
    if not isinstance(info, dict):
        return TokenUsage()
    info = cast(dict[str, Any], info)

    raw_usage = info.get("last_token_usage") or info.get("total_token_usage")
    if not isinstance(raw_usage, dict):
        return TokenUsage()
    return TokenUsage.model_validate(raw_usage)


def estimate_codex_session_jsonl_cost(path: Path, *, model: str) -> CostEstimate:
    usage = TokenUsage()
    if not path.exists():
        return estimate_cost(model, usage)

    with path.open() as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(event, dict):
                typed_event = cast(dict[str, Any], event)
                if typed_event.get("type") == "event_msg":
                    usage += usage_from_codex_session_event(typed_event)

    return estimate_cost(model, usage)


def _int_value(value: object) -> int:
    if isinstance(value, bool):
        return 0
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str) and value.isdecimal():
        return int(value)
    return 0


def _deep_merge(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    merged = dict(left)
    for key, value in right.items():
        existing = merged.get(key)
        if isinstance(existing, dict) and isinstance(value, dict):
            merged[key] = _deep_merge(
                cast(dict[str, Any], existing),
                cast(dict[str, Any], value),
            )
        else:
            merged[key] = value
    return merged


class _Model(BaseModel):
    provider: Literal["openai"] = "openai"
    name: str = "gpt-5.5"
    api_key: str
    base_url: str | None = None
    organization_id: str | None = None


class _Directory(BaseModel):
    challenge: str = "/challenge"
    workspace: str = "/workspace"
    metadata: str = "/metadata"


class _Container(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    provider: Literal["docker"] = "docker"
    socket: str = "/var/run/docker.sock"
    image: Image

    @field_validator("image", mode="before")
    @classmethod
    def _deserialize_image(cls, value: Image | str, info: ValidationInfo) -> Image:
        if isinstance(value, Image):
            return value

        socket = info.data.get("socket", "/var/run/docker.sock")
        client: DockerClient | None = None
        try:
            client = DockerClient(base_url=f"unix://{socket}")
            try:
                return client.images.get(value)
            except DockerException:
                _LOGGER.info("Pulling Docker image: %s", value)
                return client.images.pull(value)
        except DockerException as exc:
            raise ValueError(
                f"Failed to resolve Docker image {value!r}: {exc}"
            ) from exc
        finally:
            if client is not None:
                client.close()

    @field_serializer("image")
    def _serialize_image(self, value: Image) -> str:
        return value.tags[0] if value.tags else value.id or value.short_id or ""


class _PromptTemplate(BaseModel):
    user: str


class Configuration(BaseModel):
    id: str
    model: _Model
    directory: _Directory
    container: _Container
    prompt: _PromptTemplate


class CodexAgent(Agent):
    key: str = "codex"

    @staticmethod
    def from_configuration(configuration: Configuration) -> CodexAgent:
        return CodexAgent(
            id=configuration.id,
            model_name=configuration.model.name,
            model_api_key=configuration.model.api_key,
            model_base_url=configuration.model.base_url,
            model_organization_id=configuration.model.organization_id,
            container_challenge_directory=configuration.directory.challenge,
            container_workspace_directory=configuration.directory.workspace,
            container_metadata_directory=configuration.directory.metadata,
            docker_image=configuration.container.image,
            docker_client=DockerClient(
                base_url=f"unix://{configuration.container.socket}"
            ),
            user_prompt_template=configuration.prompt.user,
            docker_socket=configuration.container.socket,
        )

    def __init__(
        self,
        id: str,
        model_name: str,
        model_api_key: str,
        container_challenge_directory: str,
        container_workspace_directory: str,
        container_metadata_directory: str,
        docker_image: Image,
        docker_client: DockerClient,
        user_prompt_template: str,
        model_base_url: str | None = None,
        model_organization_id: str | None = None,
        docker_socket: str = "/var/run/docker.sock",
        # model_config: JsonObject = {},
    ):
        self._id = id
        self._model_name = model_name
        self._model_api_key = model_api_key
        self._model_base_url = model_base_url
        self._model_organization_id = model_organization_id
        self._container_challenge_directory = container_challenge_directory
        self._container_workspace_directory = container_workspace_directory
        self._container_metadata_directory = container_metadata_directory
        self._docker_image = docker_image
        self._docker_client = docker_client
        self._docker_socket = docker_socket
        self._user_prompt_template = user_prompt_template

        image_name = docker_image.tags[0] if docker_image.tags else docker_image.id
        try:
            container = self._docker_client.containers.run(
                self._docker_image,
                command=["sh", "-c", "command -v codex >/dev/null 2>&1"],
                detach=True,
                remove=True,
            )
            result = container.wait()
            exit_code = result.get("StatusCode", 1)
            if exit_code != 0:
                logs = container.logs().decode()
                raise RuntimeError(
                    f"Codex executable was not found in Docker image {image_name}: "
                    f"{logs}"
                )
        except DockerException as error:
            raise RuntimeError(
                f"Failed to check Codex executable in Docker image {image_name}: {error}"
            ) from error

    @property
    def id(self) -> str:
        return self._id

    @property
    def configuration(self) -> Configuration:
        return Configuration(
            id=self._id,
            model=_Model(
                name=self._model_name,
                api_key=self._model_api_key,
                base_url=self._model_base_url,
                organization_id=self._model_organization_id,
            ),
            directory=_Directory(
                challenge=self._container_challenge_directory,
                workspace=self._container_workspace_directory,
                metadata=self._container_metadata_directory,
            ),
            container=_Container(
                socket=self._docker_socket,
                image=self._docker_image,
            ),
            prompt=_PromptTemplate(
                user=self._user_prompt_template,
            ),
        )

    async def stream(
        self,
        challenge: Challenge,
        workspace: Path,
        metadata_directory: Path,
        webhook: Webhook | None = None,
        prompt: str | None = None,
    ) -> AsyncGenerator[Event, Interrupt]:
        if not workspace.exists():
            raise ValueError(f"workspace does not exist: {workspace}")
        if not workspace.is_dir():
            raise ValueError(f"workspace is not a directory: {workspace}")
        if not metadata_directory.exists():
            raise ValueError(f"metadata directory does not exist: {metadata_directory}")
        if not metadata_directory.is_dir():
            raise ValueError(
                f"metadata directory is not a directory: {metadata_directory}"
            )

        OmegaConf.save(
            config=OmegaConf.create(self.configuration.model_dump(mode="json")),
            f=metadata_directory / "configuration.yaml",
        )

        with self._docker_container(
            challenge=challenge,
            workspace=workspace,
            metadata_directory=metadata_directory,
        ) as container:
            async with AsyncCodex(
                config=AppServerConfig(
                    launch_args_override=(  # pyright: ignore[reportArgumentType]
                        "docker",
                        "exec",
                        "-i",
                        "--env",
                        f"HOME={self._container_workspace_directory}",
                        "--env",
                        f"CODEX_HOME={self._container_metadata_directory}/.codex",
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
                            f"({self.id})({challenge.id}) Resuming existing thread: {thread.id}"
                        )
                        thread = await codex.thread_resume(thread.id)
                    case []:
                        _LOGGER.info(f"({self.id})({challenge.id}) Starting new thread")
                        thread = await codex.thread_start(
                            model=self._model_name,
                            config={},  # TODO: support custom model config
                        )
                    case _:
                        raise RuntimeError(
                            f"Expected at most one thread, but found {len(threads)}"
                        )

                default_prompt = Template(self._user_prompt_template).render(
                    challenge=challenge,
                    webhook=webhook,
                )
                next_prompt: str | None = prompt or default_prompt

                while next_prompt is not None:
                    turn = await thread.turn(TextInput(next_prompt))
                    next_prompt = None

                    async for codex_event in turn.stream():
                        match codex_event.payload:
                            case ItemStartedNotification():
                                event = Nop()
                            case ItemCompletedNotification():
                                event = ItemCompleted()
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
                                        event = TurnCompleted()
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
                                | FileChangeOutputDeltaNotification()
                            ) as payload:
                                event = Chunk(tag="action", text=payload.delta)
                            case CommandExecutionOutputDeltaNotification() as payload:
                                event = Chunk(tag="observation", text=payload.delta)
                            case ReasoningSummaryPartAddedNotification() as payload:
                                _LOGGER.info(f"({self._id})({challenge.id}) {payload}")
                                event = Nop()
                            case TerminalInteractionNotification() as payload:
                                _LOGGER.info(f"({self._id})({challenge.id}) {payload}")
                                event = Nop()
                            case McpToolCallProgressNotification() as payload:
                                _LOGGER.info(f"({self._id})({challenge.id}) {payload}")
                                event = Nop()
                            case _:
                                _LOGGER.debug(
                                    f"({self._id})({challenge.id}) Ignoring Codex event: {codex_event.method}"
                                )
                                event = Nop()

                        interrupt = yield event

                        match interrupt:
                            case Steer() as steer:
                                await turn.steer(TextInput(steer.text))
                            case Prompt() as prompt_interrupt:
                                next_prompt = prompt_interrupt.text
                                break
                            case Stop():
                                await turn.interrupt()
                                return
                            case Nop():
                                ...

    @contextmanager
    def _docker_container(
        self, challenge: Challenge, workspace: Path, metadata_directory: Path
    ):
        assert workspace.is_dir()
        assert metadata_directory.is_dir()

        codex_home = f"{self._container_metadata_directory}/.codex"

        container = self._docker_client.containers.run(
            self._docker_image,
            detach=True,
            stdin_open=True,
            # Codex uses bubblewrap for its Linux sandbox; Docker's default confinement blocks the user/mount namespace setup bwrap needs.
            cap_add=["SYS_ADMIN"],
            security_opt=["seccomp=unconfined", "apparmor=unconfined"],
            environment={
                "HOME": self._container_workspace_directory,
                "CODEX_HOME": codex_home,
                "CHROME_REMOTE_DEBUGGING_PORT": "9222",
            },
            ports={"9222/tcp": None},
            volumes={
                str(challenge.directory): {
                    "bind": self._container_challenge_directory,
                    "mode": "ro",
                },
                str(workspace): {
                    "bind": self._container_workspace_directory,
                    "mode": "rw",
                },
                str(metadata_directory): {
                    "bind": self._container_metadata_directory,
                    "mode": "rw",
                },
            },
        )

        try:
            _LOGGER.info(f"({self._id}) Started Docker container: {container.id}")
            self._configure_codex_home(container, codex_home)
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

            yield container
        finally:
            _LOGGER.info(
                f"({self._id}) Stopping and removing Docker container: {container.id}"
            )
            container.remove(force=True)
            _LOGGER.info(f"({self._id}) Docker container removed: {container.id}")

    def _configure_codex_home(self, container: Any, codex_home: str) -> None:
        runtime_config = self._read_container_toml(
            container, f"{codex_home}/config.toml"
        )
        auth_payload = {"auth_mode": "apikey", "OPENAI_API_KEY": self._model_api_key}
        if self._model_organization_id:
            auth_payload["OPENAI_ORGANIZATION"] = self._model_organization_id
            auth_payload["OPENAI_ORG_ID"] = self._model_organization_id

        self._put_container_files(
            container,
            codex_home,
            {
                "auth.json": json.dumps(auth_payload),
                "config.toml": tomli_w.dumps(self._build_codex_config(runtime_config)),
            },
        )

    def _build_codex_config(
        self, runtime_config: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        config = _deep_merge(
            self._load_container_codex_config(),
            runtime_config or {},
        )
        config["model"] = self._model_name
        if self._model_base_url:
            config["openai_base_url"] = self._model_base_url
        return config

    def _read_container_toml(self, container: Any, path: str) -> dict[str, Any]:
        output = self._run_container_command(
            container,
            ["sh", "-c", f"cat {shlex.quote(path)} 2>/dev/null || true"],
        )
        return tomllib.loads(output.decode())

    def _put_container_files(
        self, container: Any, directory: str, files: dict[str, str]
    ) -> None:
        self._run_container_command(container, ["mkdir", "-p", directory])
        archive_buffer = io.BytesIO()
        with tarfile.open(fileobj=archive_buffer, mode="w") as archive:
            for name, content in files.items():
                encoded = content.encode()
                info = tarfile.TarInfo(name=name)
                info.size = len(encoded)
                archive.addfile(info, io.BytesIO(encoded))
        archive_buffer.seek(0)
        if not container.put_archive(directory, archive_buffer.getvalue()):
            raise RuntimeError(f"failed to copy Codex configuration into {directory}")

    def _run_container_command(self, container: Any, command: list[str]) -> bytes:
        result = container.exec_run(command)
        exit_code = int(getattr(result, "exit_code", 1))
        output = getattr(result, "output", b"")
        if exit_code != 0:
            text = output.decode() if isinstance(output, bytes) else str(output)
            raise RuntimeError(f"container command failed: {command!r}: {text}")
        return output if isinstance(output, bytes) else str(output).encode()

    def _load_container_codex_config(self) -> dict[str, Any]:
        paths = [
            f"{self._container_metadata_directory}/.codex/config.toml",
            "/metadata/.codex/config.toml",
        ]
        for path in dict.fromkeys(paths):
            try:
                output = self._docker_client.containers.run(
                    self._docker_image,
                    command=["cat", path],
                    remove=True,
                )
            except DockerException:
                continue
            text = output.decode()
            return tomllib.loads(text)
        return {}
