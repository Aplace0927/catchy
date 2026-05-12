from django.contrib import admin

from .models import (
    AgentConfiguration,
    Challenge,
    Credential,
    Ctf,
    ModelConfiguration,
    SteeringMessage,
    StreamEvent,
    Thread,
    ThreadCostSnapshot,
)


@admin.register(Credential)
class CredentialAdmin(admin.ModelAdmin):
    list_display = ["name", "slug", "kind", "base_url", "created_at"]
    search_fields = ["name", "slug", "base_url", "organization_id"]
    filter_horizontal = ["allowed_groups"]


@admin.register(ModelConfiguration)
class ModelConfigurationAdmin(admin.ModelAdmin):
    list_display = ["name", "slug", "created_at"]
    search_fields = ["name", "slug"]
    filter_horizontal = ["view_groups", "use_groups"]


@admin.register(AgentConfiguration)
class AgentConfigurationAdmin(admin.ModelAdmin):
    list_display = ["name", "slug", "created_at"]
    search_fields = ["name", "slug"]
    filter_horizontal = ["view_groups", "use_groups"]


@admin.register(Ctf)
class CtfAdmin(admin.ModelAdmin):
    list_display = ["title", "slug", "created_at"]
    search_fields = ["title", "slug"]
    filter_horizontal = ["view_groups", "init_groups"]


@admin.register(Challenge)
class ChallengeAdmin(admin.ModelAdmin):
    list_display = ["challenge_id", "ctf", "created_at"]
    search_fields = ["challenge_id", "description"]


@admin.register(Thread)
class ThreadAdmin(admin.ModelAdmin):
    list_display = [
        "id",
        "name",
        "ctf",
        "challenge",
        "agent",
        "model",
        "credential",
        "status",
        "is_public",
        "latest_cost_usd",
    ]
    list_filter = ["status", "is_public", "ctf", "agent", "model", "credential"]
    search_fields = ["name", "challenge__challenge_id", "ctf__title"]


@admin.register(StreamEvent)
class StreamEventAdmin(admin.ModelAdmin):
    list_display = ["thread", "sequence", "source", "kind", "created_at"]
    list_filter = ["source", "kind"]


@admin.register(SteeringMessage)
class SteeringMessageAdmin(admin.ModelAdmin):
    list_display = ["thread", "created_by", "delivered_at", "created_at"]
    list_filter = ["delivered_at"]


@admin.register(ThreadCostSnapshot)
class ThreadCostSnapshotAdmin(admin.ModelAdmin):
    list_display = ["thread", "usd", "created_at"]
