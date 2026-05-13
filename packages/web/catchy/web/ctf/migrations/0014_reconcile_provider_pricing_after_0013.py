from __future__ import annotations

from decimal import Decimal
from typing import Any, cast

from django.db import migrations


PROVIDER_NAMES = {
    "openai": "OpenAI",
    "anthropic": "Anthropic",
}

CREDENTIAL_KIND_PROVIDERS = {
    "openai": "openai",
    "codex_auth_json": "openai",
    "anthropic": "anthropic",
    "claude_oauth_token": "anthropic",
}

PRICING_PRESETS = (
    ("openai", "gpt-5.5", Decimal("5.00"), Decimal("0.50"), Decimal("30.00")),
    ("openai", "gpt-5.4-mini", Decimal("0.75"), Decimal("0.075"), Decimal("4.50")),
    ("openai", "gpt-5.4-nano", Decimal("0.20"), Decimal("0.02"), Decimal("1.25")),
    ("openai", "gpt-5.4", Decimal("2.50"), Decimal("0.25"), Decimal("15.00")),
    ("openai", "gpt-5.3-codex", Decimal("1.75"), Decimal("0.175"), Decimal("14.00")),
    ("openai", "gpt-5.2", Decimal("1.75"), Decimal("0.175"), Decimal("14.00")),
    ("openai", "gpt-5-mini", Decimal("0.25"), Decimal("0.025"), Decimal("2.00")),
    ("openai", "gpt-5-nano", Decimal("0.05"), Decimal("0.005"), Decimal("0.40")),
    ("openai", "gpt-5", Decimal("1.25"), Decimal("0.125"), Decimal("10.00")),
    ("openai", "gpt-4.1", Decimal("2.00"), Decimal("0.50"), Decimal("8.00")),
    ("openai", "gpt-4o", Decimal("2.50"), Decimal("1.25"), Decimal("10.00")),
    ("openai", "o4-mini", Decimal("1.10"), Decimal("0.275"), Decimal("4.40")),
    ("openai", "o3", Decimal("2.00"), Decimal("0.50"), Decimal("8.00")),
    (
        "anthropic",
        "claude-opus-4.7",
        Decimal("5.00"),
        Decimal("0.50"),
        Decimal("25.00"),
    ),
    (
        "anthropic",
        "claude-opus-4.6",
        Decimal("5.00"),
        Decimal("0.50"),
        Decimal("25.00"),
    ),
    (
        "anthropic",
        "claude-opus-4.5",
        Decimal("5.00"),
        Decimal("0.50"),
        Decimal("25.00"),
    ),
    (
        "anthropic",
        "claude-sonnet-4.6",
        Decimal("3.00"),
        Decimal("0.30"),
        Decimal("15.00"),
    ),
    (
        "anthropic",
        "claude-sonnet-4.5",
        Decimal("3.00"),
        Decimal("0.30"),
        Decimal("15.00"),
    ),
    (
        "anthropic",
        "claude-sonnet-4",
        Decimal("3.00"),
        Decimal("0.30"),
        Decimal("15.00"),
    ),
    (
        "anthropic",
        "claude-haiku-4.5",
        Decimal("1.00"),
        Decimal("0.10"),
        Decimal("5.00"),
    ),
    (
        "anthropic",
        "claude-haiku-3.5",
        Decimal("0.80"),
        Decimal("0.08"),
        Decimal("4.00"),
    ),
)

PRICING_PRESETS_BY_MODEL = sorted(
    PRICING_PRESETS,
    key=lambda preset: len(preset[1]),
    reverse=True,
)

GPT_5_FALLBACK_RATES = (Decimal("1.25"), Decimal("0.125"), Decimal("10.00"))
CORRECTED_GPT5_VARIANTS = {"gpt-5-mini", "gpt-5-nano"}


def reconcile_provider_pricing_and_cost_storage(
    apps: Any,
    schema_editor: Any,
) -> None:
    provider_model = apps.get_model("ctf", "Provider")
    credential_model = apps.get_model("ctf", "Credential")
    model_configuration_model = apps.get_model("ctf", "ModelConfiguration")
    model_pricing_model = apps.get_model("ctf", "ModelPricing")
    thread_model = apps.get_model("ctf", "Thread")
    snapshot_model = apps.get_model("ctf", "ThreadCostSnapshot")

    providers = {
        slug: provider_model.objects.get_or_create(
            slug=slug,
            defaults={"name": name},
        )[0]
        for slug, name in PROVIDER_NAMES.items()
    }

    for kind, provider_slug in CREDENTIAL_KIND_PROVIDERS.items():
        credential_model.objects.filter(kind=kind, provider__isnull=True).update(
            provider=providers[provider_slug]
        )

    for model_configuration in model_configuration_model.objects.iterator():
        preset = _preset_for_model_name(model_configuration.name)
        if preset is None:
            continue
        provider_slug, preset_model, input_rate, cached_rate, output_rate = preset
        pricing, created = model_pricing_model.objects.get_or_create(
            model=model_configuration,
            provider=providers[provider_slug],
            defaults={
                "input_per_million": input_rate,
                "cached_input_per_million": cached_rate,
                "output_per_million": output_rate,
            },
        )
        if created:
            continue
        if (
            preset_model in CORRECTED_GPT5_VARIANTS
            and _pricing_rates(pricing) == GPT_5_FALLBACK_RATES
        ):
            pricing.input_per_million = input_rate
            pricing.cached_input_per_million = cached_rate
            pricing.output_per_million = output_rate
            pricing.save(
                update_fields=[
                    "input_per_million",
                    "cached_input_per_million",
                    "output_per_million",
                    "updated_at",
                ]
            )

    for thread in thread_model.objects.iterator():
        latest_cost = _dict_value(getattr(thread, "latest_cost", None))
        scrubbed = _without_saved_cost(latest_cost)
        if scrubbed != latest_cost:
            thread_model.objects.filter(pk=thread.pk).update(latest_cost=scrubbed)

    for snapshot in snapshot_model.objects.iterator():
        usage = _dict_value(getattr(snapshot, "usage", None))
        scrubbed = _without_saved_cost(usage)
        if scrubbed != usage:
            snapshot_model.objects.filter(pk=snapshot.pk).update(usage=scrubbed)

    _drop_column_if_exists(schema_editor, "ctf_thread", "latest_cost_usd")
    _drop_column_if_exists(schema_editor, "ctf_threadcostsnapshot", "usd")


def _drop_column_if_exists(
    schema_editor: Any, table_name: str, column_name: str
) -> None:
    connection = schema_editor.connection
    with connection.cursor() as cursor:
        columns = {
            column.name
            for column in connection.introspection.get_table_description(
                cursor,
                table_name,
            )
        }
    if column_name not in columns:
        return
    quoted_table = schema_editor.quote_name(table_name)
    quoted_column = schema_editor.quote_name(column_name)
    schema_editor.execute(f"ALTER TABLE {quoted_table} DROP COLUMN {quoted_column}")


def _preset_for_model_name(
    model_name: object,
) -> tuple[str, str, Decimal, Decimal, Decimal] | None:
    normalized = _string_value(model_name).strip().lower()
    if not normalized:
        return None
    for preset in PRICING_PRESETS_BY_MODEL:
        _provider_slug, preset_model, _input_rate, _cached_rate, _output_rate = preset
        if normalized == preset_model or normalized.startswith(f"{preset_model}-"):
            return preset
    if normalized.startswith("claude-opus"):
        return _preset_by_provider_and_model("anthropic", "claude-opus-4.5")
    if normalized.startswith("claude-sonnet"):
        return _preset_by_provider_and_model("anthropic", "claude-sonnet-4")
    if normalized.startswith("claude-haiku"):
        return _preset_by_provider_and_model("anthropic", "claude-haiku-3.5")
    if normalized.startswith("gpt-4o"):
        return _preset_by_provider_and_model("openai", "gpt-4o")
    if normalized.startswith("gpt-4.1"):
        return _preset_by_provider_and_model("openai", "gpt-4.1")
    if normalized.startswith("gpt-5-mini"):
        return _preset_by_provider_and_model("openai", "gpt-5-mini")
    if normalized.startswith("gpt-5-nano"):
        return _preset_by_provider_and_model("openai", "gpt-5-nano")
    if normalized.startswith("gpt-5"):
        return _preset_by_provider_and_model("openai", "gpt-5")
    if normalized.startswith("o3"):
        return _preset_by_provider_and_model("openai", "o3")
    if normalized.startswith("o4-mini"):
        return _preset_by_provider_and_model("openai", "o4-mini")
    return None


def _preset_by_provider_and_model(
    provider_slug: str,
    model_name: str,
) -> tuple[str, str, Decimal, Decimal, Decimal] | None:
    for preset in PRICING_PRESETS:
        if preset[0] == provider_slug and preset[1] == model_name:
            return preset
    return None


def _pricing_rates(pricing: Any) -> tuple[Decimal, Decimal, Decimal]:
    return (
        Decimal(str(pricing.input_per_million)),
        Decimal(str(pricing.cached_input_per_million)),
        Decimal(str(pricing.output_per_million)),
    )


def _string_value(value: object) -> str:
    return value if isinstance(value, str) else ""


def _dict_value(value: object) -> dict[str, Any]:
    return cast(dict[str, Any], value) if isinstance(value, dict) else {}


def _without_saved_cost(value: dict[str, Any]) -> dict[str, Any]:
    if not value:
        return value
    scrubbed = dict(value)
    scrubbed.pop("usd", None)
    scrubbed.pop("pricing", None)
    return scrubbed


class Migration(migrations.Migration):
    dependencies = [
        ("ctf", "0013_provider_credential_provider_modelpricing"),
    ]

    operations = [
        migrations.RunPython(
            reconcile_provider_pricing_and_cost_storage,
            migrations.RunPython.noop,
        ),
    ]
