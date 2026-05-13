from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal


@dataclass(frozen=True)
class PricingPreset:
    key: str
    label: str
    provider_slug: str
    input_per_million: Decimal
    cached_input_per_million: Decimal
    output_per_million: Decimal

    @property
    def model_name(self) -> str:
        _provider, _separator, model_name = self.key.partition(":")
        return model_name

    def as_pricing(self) -> dict[str, Decimal]:
        return {
            "input_per_million": self.input_per_million,
            "cached_input_per_million": self.cached_input_per_million,
            "output_per_million": self.output_per_million,
        }


PRICING_PRESETS: tuple[PricingPreset, ...] = (
    PricingPreset(
        "openai:gpt-5.5",
        "OpenAI GPT-5.5",
        "openai",
        Decimal("5.00"),
        Decimal("0.50"),
        Decimal("30.00"),
    ),
    PricingPreset(
        "openai:gpt-5.4",
        "OpenAI GPT-5.4",
        "openai",
        Decimal("2.50"),
        Decimal("0.25"),
        Decimal("15.00"),
    ),
    PricingPreset(
        "openai:gpt-5.4-mini",
        "OpenAI GPT-5.4 mini",
        "openai",
        Decimal("0.75"),
        Decimal("0.075"),
        Decimal("4.50"),
    ),
    PricingPreset(
        "openai:gpt-5.4-nano",
        "OpenAI GPT-5.4 nano",
        "openai",
        Decimal("0.20"),
        Decimal("0.02"),
        Decimal("1.25"),
    ),
    PricingPreset(
        "openai:gpt-5.3-codex",
        "OpenAI GPT-5.3 Codex",
        "openai",
        Decimal("1.75"),
        Decimal("0.175"),
        Decimal("14.00"),
    ),
    PricingPreset(
        "openai:gpt-5.2",
        "OpenAI GPT-5.2",
        "openai",
        Decimal("1.75"),
        Decimal("0.175"),
        Decimal("14.00"),
    ),
    PricingPreset(
        "openai:gpt-5-mini",
        "OpenAI GPT-5 mini",
        "openai",
        Decimal("0.25"),
        Decimal("0.025"),
        Decimal("2.00"),
    ),
    PricingPreset(
        "openai:gpt-5-nano",
        "OpenAI GPT-5 nano",
        "openai",
        Decimal("0.05"),
        Decimal("0.005"),
        Decimal("0.40"),
    ),
    PricingPreset(
        "openai:gpt-5",
        "OpenAI GPT-5",
        "openai",
        Decimal("1.25"),
        Decimal("0.125"),
        Decimal("10.00"),
    ),
    PricingPreset(
        "openai:gpt-4.1",
        "OpenAI GPT-4.1",
        "openai",
        Decimal("2.00"),
        Decimal("0.50"),
        Decimal("8.00"),
    ),
    PricingPreset(
        "openai:gpt-4o",
        "OpenAI GPT-4o",
        "openai",
        Decimal("2.50"),
        Decimal("1.25"),
        Decimal("10.00"),
    ),
    PricingPreset(
        "openai:o3",
        "OpenAI o3",
        "openai",
        Decimal("2.00"),
        Decimal("0.50"),
        Decimal("8.00"),
    ),
    PricingPreset(
        "openai:o4-mini",
        "OpenAI o4-mini",
        "openai",
        Decimal("1.10"),
        Decimal("0.275"),
        Decimal("4.40"),
    ),
    PricingPreset(
        "anthropic:claude-opus-4.7",
        "Anthropic Claude Opus 4.7",
        "anthropic",
        Decimal("5.00"),
        Decimal("0.50"),
        Decimal("25.00"),
    ),
    PricingPreset(
        "anthropic:claude-opus-4.6",
        "Anthropic Claude Opus 4.6",
        "anthropic",
        Decimal("5.00"),
        Decimal("0.50"),
        Decimal("25.00"),
    ),
    PricingPreset(
        "anthropic:claude-opus-4.5",
        "Anthropic Claude Opus 4.5",
        "anthropic",
        Decimal("5.00"),
        Decimal("0.50"),
        Decimal("25.00"),
    ),
    PricingPreset(
        "anthropic:claude-sonnet-4.6",
        "Anthropic Claude Sonnet 4.6",
        "anthropic",
        Decimal("3.00"),
        Decimal("0.30"),
        Decimal("15.00"),
    ),
    PricingPreset(
        "anthropic:claude-sonnet-4.5",
        "Anthropic Claude Sonnet 4.5",
        "anthropic",
        Decimal("3.00"),
        Decimal("0.30"),
        Decimal("15.00"),
    ),
    PricingPreset(
        "anthropic:claude-sonnet-4",
        "Anthropic Claude Sonnet 4",
        "anthropic",
        Decimal("3.00"),
        Decimal("0.30"),
        Decimal("15.00"),
    ),
    PricingPreset(
        "anthropic:claude-haiku-4.5",
        "Anthropic Claude Haiku 4.5",
        "anthropic",
        Decimal("1.00"),
        Decimal("0.10"),
        Decimal("5.00"),
    ),
    PricingPreset(
        "anthropic:claude-haiku-3.5",
        "Anthropic Claude Haiku 3.5",
        "anthropic",
        Decimal("0.80"),
        Decimal("0.08"),
        Decimal("4.00"),
    ),
)

PRICING_PRESET_BY_KEY = {preset.key: preset for preset in PRICING_PRESETS}
