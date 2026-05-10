from __future__ import annotations

import json
import logging
import os
import shlex
from contextlib import contextmanager
from pathlib import Path
from typing import AsyncGenerator, Literal

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
from docker.errors import ContainerError, DockerException
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


class _Model(BaseModel):
    provider: Literal["openai"] = "openai"
    name: str = "gpt-5.5"
    api_key: str


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
        docker_socket: str = "/var/run/docker.sock",
        # model_config: JsonObject = {},
    ):
        self._id = id
        self._model_name = model_name
        self._model_api_key = model_api_key
        self._container_challenge_directory = container_challenge_directory
        self._container_workspace_directory = container_workspace_directory
        self._container_metadata_directory = container_metadata_directory
        self._docker_image = docker_image
        self._docker_client = docker_client
        self._docker_socket = docker_socket
        self._user_prompt_template = user_prompt_template

        image_name = docker_image.tags[0] if docker_image.tags else docker_image.id
        try:
            self._docker_client.containers.run(
                self._docker_image,
                command=["sh", "-c", "command -v codex >/dev/null 2>&1"],
                remove=True,
            )
        except ContainerError as error:
            raise RuntimeError(
                f"Codex executable was not found in Docker image {image_name}"
            ) from error
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
    ) -> AsyncGenerator[str, str | None]:
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

                prompt = Template(self._user_prompt_template).render(
                    challenge=challenge,
                    webhook=webhook,
                )

                turn = await thread.turn(TextInput(prompt))

                item_message = ""

                async for event in turn.stream():
                    match event.payload:
                        case ItemStartedNotification():
                            item_message = ""
                        case ItemCompletedNotification():
                            if item_message.strip():
                                steering_message = yield item_message

                                if steering_message is not None:
                                    await turn.steer(TextInput(steering_message))

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
                        f"mkdir -p {shlex.quote(codex_home)}",
                        f"printf %s {shlex.quote(json.dumps({'auth_mode': 'apikey', 'OPENAI_API_KEY': self._model_api_key}))} > {shlex.quote(f'{codex_home}/auth.json')}",
                        f"chown -R {os.getuid()}:{os.getgid()} {shlex.quote(self._container_workspace_directory)} {shlex.quote(self._container_metadata_directory)}",
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
