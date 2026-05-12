from __future__ import annotations

import io
import json
import logging
import os
import shlex
import tarfile
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, AsyncGenerator, Literal, cast

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
from claude_agent_sdk import (
    ClaudeAgentOptions,
    ClaudeSDKClient,
    ResultMessage,
    StreamEvent,
    ToolResultBlock,
    UserMessage,
)
from claude_agent_sdk.types import SystemPromptPreset, ToolsPreset
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
    provider: Literal["anthropic"] = "anthropic"
    name: str = "claude-sonnet-4-5"
    api_key: str
    base_url: str | None = None


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


class ClaudeCodeAgent(Agent):
    key: str = "claude-code"

    @staticmethod
    def from_configuration(configuration: Configuration) -> ClaudeCodeAgent:
        return ClaudeCodeAgent(
            id=configuration.id,
            model_name=configuration.model.name,
            model_api_key=configuration.model.api_key,
            model_base_url=configuration.model.base_url,
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
        docker_socket: str = "/var/run/docker.sock",
    ):
        self._id = id
        self._model_name = model_name
        self._model_api_key = model_api_key
        self._model_base_url = model_base_url
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
                command=["sh", "-c", "command -v claude >/dev/null 2>&1"],
                detach=True,
                remove=True,
            )
            result = container.wait()
            exit_code = result.get("StatusCode", 1)
            if exit_code != 0:
                logs = container.logs().decode()
                raise RuntimeError(
                    f"Claude Code executable was not found in Docker image "
                    f"{image_name}: {logs}"
                )
        except DockerException as error:
            raise RuntimeError(
                f"Failed to check Claude Code executable in Docker image "
                f"{image_name}: {error}"
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

        default_prompt = Template(self._user_prompt_template).render(
            challenge=challenge,
            webhook=webhook,
        )
        next_prompt: str | None = prompt or default_prompt

        with self._docker_container(
            challenge=challenge,
            workspace=workspace,
            metadata_directory=metadata_directory,
        ) as container:
            claude_config_dir = f"{self._container_metadata_directory}/.claude"
            wrapper_path = self._write_cli_wrapper(
                metadata_directory=metadata_directory,
                container_id=container.id,
                claude_config_dir=claude_config_dir,
                container_user=self._container_exec_user(),
            )
            env = self._claude_env()
            system_prompt: SystemPromptPreset = {
                "type": "preset",
                "preset": "claude_code",
            }
            tools: ToolsPreset = {"type": "preset", "preset": "claude_code"}
            options = ClaudeAgentOptions(
                cli_path=wrapper_path,
                cwd=workspace,
                env=env,
                model=self._model_name,
                system_prompt=system_prompt,
                tools=tools,
                permission_mode="bypassPermissions",
                include_partial_messages=True,
                continue_conversation=self._has_claude_sessions(metadata_directory),
                stderr=lambda line: _LOGGER.debug(
                    "(%s)(%s) Claude Code stderr: %s",
                    self._id,
                    challenge.id,
                    line,
                ),
            )

            async with ClaudeSDKClient(options=options) as client:
                while next_prompt is not None:
                    await client.query(next_prompt)
                    next_prompt = None

                    async for message in client.receive_response():
                        event = self._event_from_message(message)

                        interrupt = yield event

                        match interrupt:
                            case Steer() as steer:
                                await client.query(steer.text)
                            case Prompt() as prompt_interrupt:
                                await client.interrupt()
                                next_prompt = prompt_interrupt.text
                                break
                            case Stop():
                                await client.interrupt()
                                return
                            case Nop():
                                ...

    @contextmanager
    def _docker_container(
        self, challenge: Challenge, workspace: Path, metadata_directory: Path
    ) -> Generator[Any]:
        assert workspace.is_dir()
        assert metadata_directory.is_dir()

        claude_config_dir = f"{self._container_metadata_directory}/.claude"

        container = self._docker_client.containers.run(
            self._docker_image,
            detach=True,
            stdin_open=True,
            environment={
                "HOME": self._container_workspace_directory,
                "CLAUDE_CONFIG_DIR": claude_config_dir,
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
            _LOGGER.info("(%s) Started Docker container: %s", self._id, container.id)
            self._configure_claude_home(container, claude_config_dir)
            self._prepare_claude_runtime(container, claude_config_dir)
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
                    "(%s) Chrome DevTools Protocol available after running "
                    "`chrome-devtools`: %s",
                    self._id,
                    chrome_devtools_url,
                )

            yield container
        finally:
            _LOGGER.info(
                "(%s) Stopping and removing Docker container: %s",
                self._id,
                container.id,
            )
            container.remove(force=True)
            _LOGGER.info("(%s) Docker container removed: %s", self._id, container.id)

    def _configure_claude_home(self, container: Any, claude_config_dir: str) -> None:
        runtime_settings = self._read_container_json(
            container, f"{claude_config_dir}/settings.json"
        )
        settings = _deep_merge(
            self._load_container_claude_settings(),
            runtime_settings,
        )
        self._put_container_files(
            container,
            claude_config_dir,
            {"settings.json": json.dumps(settings, indent=2) + "\n"},
        )

    def _prepare_claude_runtime(self, container: Any, claude_config_dir: str) -> None:
        uid = self._container_uid()
        gid = self._container_gid()
        home = shlex.quote(self._container_workspace_directory)
        config_dir = shlex.quote(claude_config_dir)
        script = """
set -eu
uid=__CATCHY_UID__
gid=__CATCHY_GID__
home=__CATCHY_HOME__
config_dir=__CATCHY_CONFIG_DIR__

test ! -d /root || chmod 755 /root

if ! command -v sudo >/dev/null 2>&1; then
  echo "sudo is not installed in the Claude Code image" >&2
  exit 1
fi

group_name="$(getent group "$gid" | cut -d: -f1 || true)"
if [ -z "$group_name" ]; then
  group_name="catchy_$gid"
  groupadd -g "$gid" "$group_name"
fi

user_name="$(getent passwd "$uid" | cut -d: -f1 || true)"
if [ -z "$user_name" ]; then
  user_name="catchy_$uid"
  useradd -u "$uid" -g "$gid" -M -d "$home" -s /bin/bash "$user_name"
fi

mkdir -p /etc/sudoers.d
printf '%s ALL=(ALL) NOPASSWD:ALL\\n' "$user_name" > /etc/sudoers.d/catchy-claude-code
chmod 0440 /etc/sudoers.d/catchy-claude-code

chown -R "$uid:$gid" "$config_dir"
"""
        script = (
            script.replace("__CATCHY_UID__", str(uid))
            .replace("__CATCHY_GID__", str(gid))
            .replace("__CATCHY_HOME__", home)
            .replace("__CATCHY_CONFIG_DIR__", config_dir)
        )
        self._run_container_command(container, ["sh", "-c", script])

    def _read_container_json(self, container: Any, path: str) -> dict[str, Any]:
        output = self._run_container_command(
            container,
            ["sh", "-c", f"cat {shlex.quote(path)} 2>/dev/null || true"],
        )
        text = output.decode().strip()
        if not text:
            return {}
        data = json.loads(text)
        if not isinstance(data, dict):
            raise ValueError(f"container JSON file must contain an object: {path}")
        return cast(dict[str, Any], data)

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
            raise RuntimeError(
                f"failed to copy Claude Code configuration into {directory}"
            )

    def _run_container_command(self, container: Any, command: list[str]) -> bytes:
        result = container.exec_run(command)
        exit_code = int(getattr(result, "exit_code", 1))
        output = getattr(result, "output", b"")
        if exit_code != 0:
            text = output.decode() if isinstance(output, bytes) else str(output)
            raise RuntimeError(f"container command failed: {command!r}: {text}")
        return output if isinstance(output, bytes) else str(output).encode()

    def _load_container_claude_settings(self) -> dict[str, Any]:
        paths = [
            f"{self._container_metadata_directory}/.claude/settings.json",
            "/metadata/.claude/settings.json",
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
            text = output.decode().strip()
            if not text:
                return {}
            data = json.loads(text)
            if not isinstance(data, dict):
                raise ValueError(
                    f"image Claude Code settings must contain an object: {path}"
                )
            return cast(dict[str, Any], data)
        return {}

    def _write_cli_wrapper(
        self,
        *,
        metadata_directory: Path,
        container_id: str,
        claude_config_dir: str,
        container_user: str,
    ) -> Path:
        wrapper_path = metadata_directory / "claude-docker-cli"
        env_keys = sorted(self._claude_env().keys())
        env_args = [
            ("--env", f"HOME={self._container_workspace_directory}"),
            ("--env", f"CLAUDE_CONFIG_DIR={claude_config_dir}"),
            (
                "--env",
                f"CHROME_REMOTE_DEBUGGING_PORT={os.environ.get('CHROME_REMOTE_DEBUGGING_PORT', '9222')}",
            ),
        ]
        env_args.extend(("--env", key) for key in env_keys)

        lines = ["#!/bin/sh", "set -eu", "exec docker exec -i \\"]
        lines.append(f"  --user {shlex.quote(container_user)} \\")
        lines.append(
            f"  --workdir {shlex.quote(self._container_workspace_directory)} \\"
        )
        for flag, value in env_args:
            lines.append(f"  {flag} {shlex.quote(value)} \\")
        lines.append(f'  {shlex.quote(container_id)} claude "$@"')
        wrapper_path.write_text("\n".join(lines) + "\n")
        wrapper_path.chmod(0o700)
        return wrapper_path

    def _container_exec_user(self) -> str:
        return f"{self._container_uid()}:{self._container_gid()}"

    def _container_uid(self) -> int:
        return os.getuid() or 1000

    def _container_gid(self) -> int:
        return os.getgid() or 1000

    def _claude_env(self) -> dict[str, str]:
        env = {
            "ANTHROPIC_API_KEY": self._model_api_key,
        }
        if self._model_base_url:
            env["ANTHROPIC_BASE_URL"] = self._model_base_url
        return env

    def _has_claude_sessions(self, metadata_directory: Path) -> bool:
        projects_directory = metadata_directory / ".claude" / "projects"
        return projects_directory.exists() and any(projects_directory.rglob("*.jsonl"))

    def _event_from_message(self, message: object) -> Event:
        if isinstance(message, StreamEvent):
            return self._event_from_stream_event(message)
        if isinstance(message, UserMessage):
            return self._event_from_user_message(message)
        if isinstance(message, ResultMessage):
            if message.is_error:
                details = message.result or ", ".join(message.errors or [])
                raise RuntimeError(
                    f"Claude Code turn failed: {details or message.subtype}"
                )
            return TurnCompleted()
        return Nop()

    def _event_from_stream_event(self, message: StreamEvent) -> Event:
        event = cast(dict[str, object], message.event)
        event_type = event.get("type")

        if event_type == "content_block_start":
            raw_content_block = event.get("content_block")
            if isinstance(raw_content_block, dict):
                content_block = cast(dict[str, object], raw_content_block)
            else:
                content_block = {}
            if content_block.get("type") == "tool_use":
                raw_tool_name = content_block.get("name")
                tool_name = raw_tool_name if isinstance(raw_tool_name, str) else "tool"
                return Chunk(
                    tag="tool_use",
                    text=self._tool_use_text(content_block, fallback_name=tool_name),
                )

        if event_type == "content_block_delta":
            raw_delta = event.get("delta")
            if isinstance(raw_delta, dict):
                delta = cast(dict[str, object], raw_delta)
                if delta.get("type") == "text_delta":
                    raw_text = delta.get("text")
                    text = raw_text if isinstance(raw_text, str) else ""
                    return Chunk(tag="action", text=text)
                if delta.get("type") == "input_json_delta":
                    raw_partial_json = delta.get("partial_json")
                    partial_json = (
                        raw_partial_json if isinstance(raw_partial_json, str) else ""
                    )
                    return Chunk(tag="tool_input", text=partial_json)
                if delta.get("type") == "thinking_delta":
                    raw_thinking = delta.get("thinking")
                    thinking = raw_thinking if isinstance(raw_thinking, str) else ""
                    return Chunk(tag="thinking", text=thinking)

        if event_type == "content_block_stop":
            return ItemCompleted()

        return Nop()

    def _tool_use_text(
        self, content_block: dict[str, object], *, fallback_name: str
    ) -> str:
        payload: dict[str, object] = {"name": fallback_name}
        raw_id = content_block.get("id")
        if isinstance(raw_id, str):
            payload["id"] = raw_id
        raw_input = content_block.get("input")
        if isinstance(raw_input, dict):
            payload["input"] = raw_input
        return json.dumps(payload, ensure_ascii=False, sort_keys=True)

    def _event_from_user_message(self, message: UserMessage) -> Event:
        if not isinstance(message.content, list):
            return Nop()

        chunks: list[str] = []
        for block in message.content:
            if isinstance(block, ToolResultBlock):
                text = self._tool_result_text(block.content)
                if text:
                    chunks.append(text)

        if chunks:
            return Chunk(tag="observation", text="\n".join(chunks))
        return Nop()

    def _tool_result_text(self, content: object) -> str:
        if content is None:
            return ""
        if isinstance(content, str):
            return content
        return json.dumps(content, ensure_ascii=False)
