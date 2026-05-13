from __future__ import annotations

import json
from typing import Any, cast

from django.db import migrations


def normalize_token_usage_records(apps: Any, schema_editor: Any) -> None:
    stream_event_model = apps.get_model("ctf", "StreamEvent")
    thread_model = apps.get_model("ctf", "Thread")
    snapshot_model = apps.get_model("ctf", "ThreadCostSnapshot")

    for event in (
        stream_event_model.objects.filter(kind="token_count")
        .select_related("thread", "thread__model")
        .iterator()
    ):
        event_raw: object = getattr(event, "raw", None)
        raw = cast(dict[str, Any], event_raw) if isinstance(event_raw, dict) else {}
        model_name = _model_name(event.thread, raw)
        normalized = _normalized_event_raw(
            raw, model_name=model_name, source=event.source
        )
        if normalized is None:
            continue
        stream_event_model.objects.filter(pk=event.pk).update(
            raw=normalized,
            text=json.dumps(normalized["usage"], separators=(",", ":")),
        )

    for thread in thread_model.objects.select_related("model").iterator():
        thread_latest_cost: object = getattr(thread, "latest_cost", None)
        latest_cost = (
            cast(dict[str, Any], thread_latest_cost)
            if isinstance(thread_latest_cost, dict)
            else {}
        )
        if not latest_cost:
            continue
        snapshot = _token_usage_snapshot(
            latest_cost,
            model_name=_model_name(thread, latest_cost),
        )
        if snapshot is None:
            continue
        thread_model.objects.filter(pk=thread.pk).update(
            latest_cost=snapshot,
        )

    for snapshot_record in snapshot_model.objects.select_related(
        "thread", "thread__model"
    ).iterator():
        snapshot_usage: object = getattr(snapshot_record, "usage", None)
        usage = (
            cast(dict[str, Any], snapshot_usage)
            if isinstance(snapshot_usage, dict)
            else {}
        )
        if not usage:
            continue
        snapshot = _token_usage_snapshot(
            usage,
            model_name=_model_name(snapshot_record.thread, usage),
        )
        if snapshot is None:
            continue
        snapshot_model.objects.filter(pk=snapshot_record.pk).update(
            usage=snapshot,
        )


def _normalized_event_raw(
    raw: dict[str, Any],
    *,
    model_name: str,
    source: str,
) -> dict[str, Any] | None:
    snapshot = _token_usage_snapshot(raw, model_name=model_name)
    if snapshot is None:
        return None
    original_raw = raw.get("raw") if _is_normalized_event_raw(raw) else raw
    normalized: dict[str, Any] = {
        "provider": snapshot["provider"],
        "model": snapshot["model"],
        "source": _string_value(raw.get("source")) or source,
        "usage": _event_usage_from_snapshot(snapshot),
    }
    if isinstance(original_raw, dict) and original_raw:
        normalized["raw"] = original_raw
    return normalized


def _token_usage_snapshot(
    raw: dict[str, Any],
    *,
    model_name: str,
) -> dict[str, Any] | None:
    usage = _token_usage_from_raw(raw)
    if usage is None:
        return None

    input_tokens = _int_value(usage.get("input_tokens") or usage.get("inputTokens"))
    cached_input_tokens = _int_value(
        usage.get("cached_input_tokens") or usage.get("cachedInputTokens")
    )
    cache_creation_input_tokens = _int_value(
        usage.get("cache_creation_input_tokens")
        or usage.get("cacheCreationInputTokens")
    )
    cache_read_input_tokens = _int_value(
        usage.get("cache_read_input_tokens") or usage.get("cacheReadInputTokens")
    )
    output_tokens = _int_value(usage.get("output_tokens") or usage.get("outputTokens"))
    reasoning_output_tokens = _int_value(
        usage.get("reasoning_output_tokens") or usage.get("reasoningOutputTokens")
    )
    total_tokens = _int_value(usage.get("total_tokens") or usage.get("totalTokens"))
    if not total_tokens:
        total_tokens = (
            input_tokens
            + cache_creation_input_tokens
            + cache_read_input_tokens
            + output_tokens
        )

    return {
        "provider": _provider(raw),
        "model": _string_value(raw.get("model")) or model_name,
        "input_tokens": input_tokens,
        "cached_input_tokens": cached_input_tokens,
        "cache_creation_input_tokens": cache_creation_input_tokens,
        "cache_read_input_tokens": cache_read_input_tokens,
        "output_tokens": output_tokens,
        "reasoning_output_tokens": reasoning_output_tokens,
        "total_tokens": total_tokens,
    }


def _token_usage_from_raw(raw: dict[str, Any]) -> dict[str, Any] | None:
    payload = _first_dict(raw.get("payload"), raw.get("message"), raw)
    info = _first_dict(
        payload.get("info"),
        payload.get("usage"),
        payload.get("tokenUsage"),
        raw.get("info"),
        raw.get("usage"),
        raw.get("tokenUsage"),
        payload,
    )
    usage = _first_dict(
        info.get("total_token_usage"),
        info.get("last_token_usage"),
        info.get("total"),
        info.get("last"),
        info,
    )
    if not usage:
        return None
    return {str(key): value for key, value in usage.items()}


def _event_usage_from_snapshot(snapshot: dict[str, Any]) -> dict[str, int]:
    usage = {
        "input_tokens": _int_value(snapshot.get("input_tokens")),
        "output_tokens": _int_value(snapshot.get("output_tokens")),
        "total_tokens": _int_value(snapshot.get("total_tokens")),
    }
    for key in (
        "cached_input_tokens",
        "cache_creation_input_tokens",
        "cache_read_input_tokens",
        "reasoning_output_tokens",
    ):
        value = _int_value(snapshot.get(key))
        if value:
            usage[key] = value
    return usage


def _model_name(thread: Any, raw: dict[str, Any]) -> str:
    model = _string_value(raw.get("model"))
    if model:
        return model
    latest_cost = getattr(thread, "latest_cost", {})
    if isinstance(latest_cost, dict):
        typed_latest_cost = cast(dict[str, Any], latest_cost)
        model = _string_value(typed_latest_cost.get("model"))
        if model:
            return model
    thread_model = getattr(thread, "model", None)
    model = _string_value(getattr(thread_model, "name", None))
    return model or "unknown"


def _provider(raw: dict[str, Any]) -> str:
    provider = _string_value(raw.get("provider"))
    if provider:
        return provider
    nested = raw.get("raw")
    if isinstance(nested, dict):
        typed_nested = cast(dict[str, Any], nested)
        provider = _string_value(typed_nested.get("provider"))
        if provider:
            return provider
    return "openai"


def _is_normalized_event_raw(raw: dict[str, Any]) -> bool:
    return isinstance(raw.get("usage"), dict) and (
        isinstance(raw.get("provider"), str)
        or isinstance(raw.get("model"), str)
        or isinstance(raw.get("raw"), dict)
    )


def _first_dict(*values: object) -> dict[str, Any]:
    for value in values:
        if isinstance(value, dict):
            return cast(dict[str, Any], value)
    return {}


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


def _string_value(value: object) -> str:
    return value if isinstance(value, str) else ""


class Migration(migrations.Migration):
    dependencies = [
        ("ctf", "0011_add_auth_json_and_oauth_credentials"),
    ]

    operations = [
        migrations.RunPython(normalize_token_usage_records, migrations.RunPython.noop),
    ]
