from __future__ import annotations

import tarfile
from typing import Any

from django import forms
from django.utils.text import slugify
from omegaconf import OmegaConf

from .models import AgentConfiguration, Challenge, Credential, Ctf, ModelConfiguration


class CredentialForm(forms.ModelForm):
    class Meta:
        model = Credential
        fields = [
            "name",
            "slug",
            "kind",
            "api_key",
            "base_url",
            "organization_id",
            "allowed_groups",
        ]
        widgets = {"api_key": forms.PasswordInput(render_value=True)}


class ModelConfigurationForm(forms.ModelForm):
    class Meta:
        model = ModelConfiguration
        fields = ["name", "slug", "view_groups", "use_groups"]


class AgentConfigurationForm(forms.ModelForm):
    class Meta:
        model = AgentConfiguration
        fields = ["name", "slug", "yaml", "view_groups", "use_groups"]
        widgets = {"yaml": forms.Textarea(attrs={"rows": 22, "cols": 100})}

    def __init__(self, *args: Any, user=None, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.user = user

    def clean_yaml(self) -> str:
        yaml = self.cleaned_data["yaml"]
        try:
            AgentConfiguration(yaml=yaml).resolved_mapping(user=self.user)
        except Exception as exc:
            raise forms.ValidationError(f"invalid agent YAML: {exc}") from exc
        return yaml


class CtfForm(forms.ModelForm):
    settings_yaml = forms.CharField(
        required=False,
        widget=forms.Textarea(attrs={"rows": 8, "cols": 80}),
        help_text="Optional YAML mapping for future CTF-level settings.",
    )

    class Meta:
        model = Ctf
        fields = ["title", "slug", "description", "view_groups", "init_groups"]

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        if self.instance.pk:
            self.fields["settings_yaml"].initial = self.instance.settings

    def clean_settings_yaml(self) -> str:
        value = str(self.cleaned_data.get("settings_yaml", ""))
        _clean_yaml_mapping(value)
        return value

    def save(self, commit: bool = True) -> Ctf:
        instance = super().save(commit=False)
        instance.settings = self.cleaned_data["settings_yaml"]
        if commit:
            instance.save()
            self.save_m2m()
        return instance


class ChallengeForm(forms.ModelForm):
    webhook_url = forms.URLField(
        required=False,
        label="Webhook URL",
        help_text="Endpoint the agent can POST to during the run. Leave blank to disable.",
        widget=forms.URLInput(attrs={"placeholder": "https://example.com/hook"}),
    )
    webhook_preferred_language = forms.CharField(
        required=False,
        label="Preferred language",
        help_text="Spoken language the agent should respond in (e.g. English, 한국어). Optional.",
        widget=forms.TextInput(attrs={"placeholder": "English"}),
    )
    clear_webhook = forms.BooleanField(
        required=False,
        label="Remove webhook",
        help_text="Check to remove the existing webhook from this challenge.",
    )
    config_yaml = forms.CharField(
        required=False,
        label="Config (YAML)",
        help_text="Free-form YAML mapping forwarded to the challenge runner.",
        widget=forms.Textarea(attrs={"rows": 8, "cols": 80}),
    )

    fieldsets = [
        ("Basics", ["challenge_id", "description", "source_archive"]),
        ("Webhook", ["webhook_url", "webhook_preferred_language", "clear_webhook"]),
        ("Advanced", ["config_yaml"]),
    ]

    class Meta:
        model = Challenge
        fields = ["challenge_id", "description", "source_archive"]
        help_texts = {
            "description": "Markdown is supported.",
        }

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        if self.instance.pk:
            webhook_data = _safe_yaml_mapping(self.instance.webhook)
            existing_webhook_url = str(webhook_data.get("url") or "")
            self.fields["webhook_preferred_language"].initial = webhook_data.get(
                "preferred_language", ""
            )
            if existing_webhook_url:
                self.fields[
                    "webhook_url"
                ].help_text = "A webhook URL is set. Leave blank to keep it, or enter a new URL to replace."
                self.fields["webhook_url"].widget.attrs["placeholder"] = (
                    "(URL set — leave blank to keep)"
                )
            else:
                self.fields["clear_webhook"].widget = forms.HiddenInput()
            self.fields["config_yaml"].initial = self.instance.config
            self.fields["source_archive"].required = False
            self.fields[
                "source_archive"
            ].help_text = "Leave blank to keep the existing archive."
        else:
            self.fields["clear_webhook"].widget = forms.HiddenInput()

    def clean_source_archive(self):
        archive = self.cleaned_data.get("source_archive")
        if not archive:
            if self.instance.pk and self.instance.source_archive:
                return self.instance.source_archive
            raise forms.ValidationError("This field is required.")
        if not hasattr(archive, "name"):
            return archive
        name = archive.name.lower()
        if not (name.endswith(".tar.gz") or name.endswith(".tgz")):
            raise forms.ValidationError("source must be a .tar.gz or .tgz archive")
        try:
            archive.file.seek(0)
            with tarfile.open(fileobj=archive.file, mode="r:gz"):
                pass
        except tarfile.TarError as exc:
            raise forms.ValidationError("source archive is not a valid tar.gz") from exc
        finally:
            archive.file.seek(0)
        return archive

    def clean(self) -> dict[str, Any]:
        cleaned = super().clean()
        url = (cleaned.get("webhook_url") or "").strip()
        lang = (cleaned.get("webhook_preferred_language") or "").strip()
        clear = bool(cleaned.get("clear_webhook"))
        existing_url = ""
        if self.instance.pk:
            existing_url = str(
                _safe_yaml_mapping(self.instance.webhook).get("url") or ""
            )
        effective_url = "" if clear else (url or existing_url)
        if lang and not effective_url:
            self.add_error(
                "webhook_url",
                "Webhook URL is required when a preferred language is set.",
            )
        return cleaned

    def clean_config_yaml(self) -> str:
        value = str(self.cleaned_data.get("config_yaml", ""))
        _clean_yaml_mapping(value)
        return value

    def save(self, commit: bool = True) -> Challenge:
        instance = super().save(commit=False)
        instance.webhook = self._serialize_webhook()
        instance.config = self.cleaned_data["config_yaml"]
        if commit:
            instance.save()
        return instance

    def _serialize_webhook(self) -> str:
        if self.cleaned_data.get("clear_webhook"):
            return ""
        url = (self.cleaned_data.get("webhook_url") or "").strip()
        lang = (self.cleaned_data.get("webhook_preferred_language") or "").strip()
        if not url and self.instance.pk:
            url = str(_safe_yaml_mapping(self.instance.webhook).get("url") or "")
        if not url:
            return ""
        payload: dict[str, Any] = {"url": url}
        if lang:
            payload["preferred_language"] = lang
        return OmegaConf.to_yaml(OmegaConf.create(payload))


class ThreadCreateForm(forms.Form):
    name = forms.CharField(
        max_length=80,
        required=False,
        widget=forms.TextInput(attrs={"placeholder": "bright-cipher-0427"}),
    )
    agent = forms.ModelChoiceField(queryset=AgentConfiguration.objects.none())
    model = forms.ModelChoiceField(queryset=ModelConfiguration.objects.none())
    credential = forms.ModelChoiceField(queryset=Credential.objects.none())

    def __init__(
        self,
        *args: Any,
        user,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        agent_ids = [
            agent.pk
            for agent in AgentConfiguration.objects.prefetch_related("use_groups")
            if agent.can_use(user)
        ]
        self.fields["agent"].queryset = AgentConfiguration.objects.filter(
            pk__in=agent_ids
        )
        model_ids = [
            model.pk
            for model in ModelConfiguration.objects.prefetch_related("use_groups")
            if model.can_use(user)
        ]
        self.fields["model"].queryset = ModelConfiguration.objects.filter(
            pk__in=model_ids
        )
        credential_ids = [
            credential.pk
            for credential in Credential.objects.prefetch_related("allowed_groups")
            if credential.can_view(user)
        ]
        self.fields["credential"].queryset = Credential.objects.filter(
            pk__in=credential_ids
        )

    def clean_name(self) -> str:
        value = self.cleaned_data.get("name", "")
        if not value:
            return ""
        name = slugify(value)[:80]
        if not name:
            raise forms.ValidationError("Enter a name with letters or numbers.")
        return name


def _clean_yaml_mapping(value: str) -> dict[str, Any]:
    if not value.strip():
        return {}
    try:
        data = OmegaConf.to_container(OmegaConf.create(value), resolve=True)
    except Exception as exc:
        raise forms.ValidationError(f"invalid YAML: {exc}") from exc
    if not isinstance(data, dict):
        raise forms.ValidationError("YAML value must be a mapping")
    return {str(key): item for key, item in data.items()}


def _safe_yaml_mapping(value: str) -> dict[str, Any]:
    if not value or not value.strip():
        return {}
    try:
        data = OmegaConf.to_container(OmegaConf.create(value), resolve=True)
    except Exception:
        return {}
    if not isinstance(data, dict):
        return {}
    return {str(key): item for key, item in data.items()}
