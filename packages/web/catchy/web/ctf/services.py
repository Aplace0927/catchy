from __future__ import annotations

import asyncio
import importlib
import json
import tarfile
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, cast

from asgiref.sync import sync_to_async
from django.conf import settings
from django.core.exceptions import PermissionDenied
from django.db import (
    IntegrityError,
    OperationalError,
    close_old_connections,
    transaction,
)
from django.db.models import Max
from django.utils import timezone

from catchy.codex import estimate_codex_session_jsonl_cost
from catchy.core.agents.protocols import Agent
from catchy.core.challenge.models import Challenge as CoreChallenge
from catchy.core.webhook.models import Webhook

from .models import (
    AgentConfiguration,
    Credential,
    ModelConfiguration,
    SteeringMessage,
    StreamEvent,
    Thread,
    ThreadCostSnapshot,
)

_STREAM_EVENT_WRITE_ATTEMPTS = 6
_STREAM_EVENT_WRITE_RETRY_DELAY_SECONDS = 0.05


def start_thread(thread: Thread) -> Any:
    worker = threading.Thread(
        target=_run_thread_in_local_worker,
        args=(thread.pk,),
        daemon=True,
        name=f"catchy-thread-{thread.pk}",
    )
    worker.start()
    thread.task_result_id = f"local-thread:{worker.name}"
    thread.save(update_fields=["task_result_id", "updated_at"])
    return worker


def _run_thread_in_local_worker(thread_id: int) -> None:
    close_old_connections()
    try:
        run_thread_sync(thread_id)
    finally:
        close_old_connections()


def run_thread_sync(thread_id: int) -> None:
    thread = (
        Thread.objects.select_related(
            "challenge",
            "challenge__ctf",
            "agent",
            "model",
            "credential",
        )
        .select_related("created_by")
        .prefetch_related(
            "agent__use_groups", "model__use_groups", "credential__allowed_groups"
        )
        .get(pk=thread_id)
    )

    thread_root = _thread_root(thread)
    source_directory = thread_root / "source"
    workspace = thread_root / "workspace"
    metadata = thread_root / "metadata"
    source_directory.mkdir(parents=True, exist_ok=True)
    workspace.mkdir(parents=True, exist_ok=True)
    metadata.mkdir(parents=True, exist_ok=True)
    _safe_extract_tar_gz(Path(thread.challenge.source_archive.path), source_directory)

    thread.thread_root = str(thread_root)
    thread.workspace_path = str(workspace)
    thread.metadata_path = str(metadata)
    thread.status = Thread.Status.RUNNING
    thread.error = ""
    thread.save(
        update_fields=[
            "thread_root",
            "workspace_path",
            "metadata_path",
            "status",
            "error",
            "updated_at",
        ]
    )
    _record_event(thread, source="system", kind="thread.started", text="Thread started")

    try:
        agent = load_agent(
            thread.agent,
            model_configuration=thread.model,
            credential=thread.credential,
            user=thread.created_by,
        )
        core_challenge = CoreChallenge(
            id=thread.challenge.challenge_id,
            description=thread.challenge.description,
            directory=source_directory,
        )
        webhook_data = thread.challenge.webhook_mapping()
        webhook = Webhook(**webhook_data) if webhook_data else None
        asyncio.run(
            _run_agent_stream(
                thread_id=thread.pk,
                agent=agent,
                challenge=core_challenge,
                workspace=workspace,
                metadata=metadata,
                webhook=webhook,
                model_name=_thread_model_name(thread),
            )
        )
    except Exception as exc:
        Thread.objects.filter(pk=thread.pk).update(
            status=Thread.Status.FAILED,
            error=str(exc),
            updated_at=timezone.now(),
        )
        _record_event(thread, source="system", kind="thread.failed", text=str(exc))
        raise

    Thread.objects.filter(pk=thread.pk).update(
        status=Thread.Status.COMPLETED,
        updated_at=timezone.now(),
    )
    _record_event(
        thread, source="system", kind="thread.completed", text="Thread completed"
    )


def load_agent(
    agent_configuration: AgentConfiguration,
    *,
    model_configuration: ModelConfiguration | None = None,
    credential: Credential | None = None,
    user: Any | None = None,
) -> Agent:
    data = build_agent_configuration(
        agent_configuration,
        model_configuration=model_configuration,
        credential=credential,
        user=user,
    )
    class_path = _agent_class_path(data)
    agent_class = _import_agent_class(class_path)
    configuration_class = getattr(
        importlib.import_module(agent_class.__module__),
        "Configuration",
        None,
    )
    if not hasattr(configuration_class, "model_validate"):
        raise TypeError(
            f"agent module must expose Configuration: {agent_class.__module__}"
        )

    from_configuration = getattr(agent_class, "from_configuration", None)
    if not callable(from_configuration):
        raise TypeError(f"agent class must expose from_configuration: {class_path}")

    agent = from_configuration(cast(Any, configuration_class).model_validate(data))
    if not isinstance(agent, Agent):
        raise TypeError(f"from_configuration did not return an Agent: {class_path}")
    return agent


def build_agent_configuration(
    agent_configuration: AgentConfiguration,
    *,
    model_configuration: ModelConfiguration | None = None,
    credential: Credential | None = None,
    user: Any | None = None,
) -> dict[str, Any]:
    if user is not None and not agent_configuration.can_use(user):
        raise PermissionDenied("agent configuration is not accessible")

    data = dict(agent_configuration.resolved_mapping(user=user))
    if model_configuration is None and credential is None:
        return data

    if model_configuration is None:
        raise ValueError("model configuration is required")
    if credential is None:
        raise ValueError("credential is required")
    if user is not None and not model_configuration.can_use(user):
        raise PermissionDenied("model configuration is not accessible")
    if user is not None and not credential.can_view(user):
        raise PermissionDenied("credential is not accessible")
    if credential.kind != Credential.Kind.OPENAI:
        raise ValueError(f"unsupported credential kind: {credential.kind}")

    existing_model = data.get("model", {})
    model_data = dict(existing_model) if isinstance(existing_model, dict) else {}
    model_data.update(
        {
            "provider": "openai",
            "name": model_configuration.name,
            "api_key": credential.api_key,
            "base_url": credential.base_url,
        }
    )
    if credential.organization_id:
        model_data["organization_id"] = credential.organization_id
    else:
        model_data.pop("organization_id", None)
    data["model"] = model_data
    return data


def ingest_codex_sessions(thread: Thread, *, model_name: str | None = None) -> None:
    metadata = thread.metadata_directory
    if metadata is None:
        return

    sessions_root = metadata / ".codex" / "sessions"
    if not sessions_root.exists():
        return

    session_paths = sorted(sessions_root.glob("**/*.jsonl"))
    for path in session_paths:
        _ingest_session_file(thread, path)

    if model_name and session_paths:
        estimate = estimate_codex_session_jsonl_cost(
            session_paths[-1], model=model_name
        )
        thread.latest_cost_usd = estimate.usd
        thread.latest_cost = estimate.as_dict()
        thread.save(update_fields=["latest_cost_usd", "latest_cost", "updated_at"])
        ThreadCostSnapshot.objects.create(
            thread=thread,
            usd=estimate.usd,
            usage=estimate.as_dict(),
        )


async def _run_agent_stream(
    *,
    thread_id: int,
    agent: Agent,
    challenge: CoreChallenge,
    workspace: Path,
    metadata: Path,
    webhook: Webhook | None,
    model_name: str,
) -> None:
    stream = agent.stream(
        challenge=challenge,
        workspace=workspace,
        metadata_directory=metadata,
        webhook=webhook,
    )
    steering_message: str | None = None
    while True:
        try:
            if steering_message is None:
                delta = await stream.__anext__()
            else:
                delta = await stream.asend(steering_message)
                steering_message = None
        except StopAsyncIteration:
            return

        await sync_to_async(_record_stream_delta, thread_sensitive=True)(
            thread_id,
            delta,
            model_name,
        )
        steering_message = await sync_to_async(
            _pop_next_steering_message,
            thread_sensitive=True,
        )(thread_id)


def _record_stream_delta(thread_id: int, delta: str, model_name: str) -> None:
    thread = Thread.objects.get(pk=thread_id)
    _record_event(
        thread,
        source="agent_stream",
        kind="delta",
        text=delta,
    )
    ingest_codex_sessions(thread, model_name=model_name)


def _pop_next_steering_message(thread_id: int) -> str | None:
    message = (
        SteeringMessage.objects.filter(thread_id=thread_id, delivered_at__isnull=True)
        .order_by("created_at")
        .first()
    )
    if message is None:
        return None

    message.delivered_at = timezone.now()
    message.save(update_fields=["delivered_at", "updated_at"])
    _record_event(
        message.thread,
        source="user",
        kind="steer",
        text=message.text,
        raw={"steering_message_id": message.pk},
    )
    return message.text


def _ingest_session_file(thread: Thread, path: Path) -> None:
    try:
        relative_path = str(path.relative_to(Path(thread.metadata_path)))
    except ValueError:
        relative_path = str(path)

    with path.open() as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue
            dedupe_key = f"jsonl:{relative_path}:{line_number}"
            if StreamEvent.objects.filter(
                thread=thread, dedupe_key=dedupe_key
            ).exists():
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError:
                raw = {"line": line}
            kind, text = _summarize_codex_event(raw)
            if not text:
                continue
            _record_event(
                thread,
                source="codex_jsonl",
                kind=kind,
                text=text,
                raw=raw if isinstance(raw, dict) else {"value": raw},
                dedupe_key=dedupe_key,
            )


def _summarize_codex_event(raw: Any) -> tuple[str, str]:
    if not isinstance(raw, dict):
        return "raw", str(raw)

    event_type = str(raw.get("type", "event"))
    payload = raw.get("payload")
    if not isinstance(payload, dict):
        return event_type, ""

    if event_type == "response_item":
        item_type = str(payload.get("type", "response_item"))
        if item_type == "message":
            parts = payload.get("content")
            if isinstance(parts, list):
                texts = [
                    str(part.get("text") or part.get("output_text"))
                    for part in parts
                    if isinstance(part, dict)
                    and (part.get("text") or part.get("output_text"))
                ]
                return item_type, "\n".join(texts)
        if item_type in {"function_call", "function_call_output"}:
            return item_type, json.dumps(payload, ensure_ascii=False)
        return item_type, ""

    if event_type == "event_msg":
        payload_type = str(payload.get("type", "event_msg"))
        message = payload.get("message")
        if isinstance(message, str):
            return payload_type, message
        if payload_type == "token_count":
            return payload_type, json.dumps(payload.get("info", {}), ensure_ascii=False)
        return payload_type, ""

    return event_type, ""


def _record_event(
    thread: Thread,
    *,
    source: str,
    kind: str,
    text: str,
    raw: dict[str, Any] | None = None,
    dedupe_key: str | None = None,
) -> StreamEvent:
    delay = _STREAM_EVENT_WRITE_RETRY_DELAY_SECONDS
    for attempt in range(_STREAM_EVENT_WRITE_ATTEMPTS):
        try:
            return _record_event_once(
                thread,
                source=source,
                kind=kind,
                text=text,
                raw=raw,
                dedupe_key=dedupe_key,
            )
        except OperationalError as exc:
            if (
                not _is_database_locked(exc)
                or attempt == _STREAM_EVENT_WRITE_ATTEMPTS - 1
            ):
                raise
            close_old_connections()
            time.sleep(delay)
            delay *= 2
        except IntegrityError:
            if dedupe_key is not None:
                event = StreamEvent.objects.filter(
                    thread=thread,
                    dedupe_key=dedupe_key,
                ).first()
                if event is not None:
                    return event
            if attempt == _STREAM_EVENT_WRITE_ATTEMPTS - 1:
                raise
            time.sleep(delay)
            delay *= 2
    raise RuntimeError("stream event write retry loop exited unexpectedly")


def _record_event_once(
    thread: Thread,
    *,
    source: str,
    kind: str,
    text: str,
    raw: dict[str, Any] | None = None,
    dedupe_key: str | None = None,
) -> StreamEvent:
    with transaction.atomic():
        sequence = (
            StreamEvent.objects.filter(thread=thread).aggregate(Max("sequence"))[
                "sequence__max"
            ]
            or 0
        ) + 1
        if dedupe_key is None:
            dedupe_key = f"{source}:{sequence}"
        event, _created = StreamEvent.objects.get_or_create(
            thread=thread,
            dedupe_key=dedupe_key,
            defaults={
                "sequence": sequence,
                "source": source,
                "kind": kind,
                "text": text,
                "raw": raw or {},
            },
        )
        return event


def _is_database_locked(exc: OperationalError) -> bool:
    message = str(exc).lower()
    return "database is locked" in message or "database table is locked" in message


def _agent_class_path(data: dict[str, Any]) -> str:
    class_path = data.get("class", "catchy.codex.CodexAgent")
    if class_path == "CodexAgent":
        return "catchy.codex.CodexAgent"
    if not isinstance(class_path, str) or not class_path:
        raise ValueError("agent configuration has an invalid class")
    return class_path


def _import_agent_class(class_path: str) -> type[Any]:
    module_name, separator, attribute_name = class_path.rpartition(".")
    if not separator or not module_name or not attribute_name:
        raise ValueError(
            f"agent class must be a fully qualified import path: {class_path!r}"
        )
    module = importlib.import_module(module_name)
    agent_class = getattr(module, attribute_name, None)
    if not isinstance(agent_class, type):
        raise TypeError(f"agent class is not a class: {class_path!r}")
    return agent_class


def _thread_model_name(thread: Thread) -> str:
    if thread.model is not None:
        return thread.model.name
    model = thread.agent.resolved_mapping(user=thread.created_by).get("model", {})
    if isinstance(model, dict) and isinstance(model.get("name"), str):
        return str(model["name"])
    return "unknown"


def _thread_root(thread: Thread) -> Path:
    if thread.thread_root:
        return Path(thread.thread_root)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    return Path(settings.MEDIA_ROOT) / "threads" / f"thread-{thread.pk}-{timestamp}"


def _safe_extract_tar_gz(archive_path: Path, destination: Path) -> None:
    with tarfile.open(archive_path, mode="r:gz") as archive:
        destination_root = destination.resolve()
        for member in archive.getmembers():
            target = (destination / member.name).resolve()
            try:
                target.relative_to(destination_root)
            except ValueError as exc:
                raise ValueError(
                    f"archive member escapes destination: {member.name}"
                ) from exc
        archive.extractall(destination)
