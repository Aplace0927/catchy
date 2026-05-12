from __future__ import annotations

import json
import time
from collections.abc import Iterator
from typing import Any, TypedDict

from catchy.codex import TokenUsage, estimate_cost
from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.contrib.auth.views import redirect_to_login
from django.core.exceptions import PermissionDenied
from django.db.models import Q, QuerySet
from django.http import HttpRequest, HttpResponse, StreamingHttpResponse
from django.shortcuts import get_object_or_404, redirect, render
from django.utils import timezone
from django.views.decorators.http import require_POST

from .forms import (
    AgentConfigurationForm,
    ChallengeForm,
    CredentialForm,
    CtfForm,
    ModelConfigurationForm,
    ThreadCreateForm,
)
from .models import (
    AgentConfiguration,
    Challenge,
    Credential,
    Ctf,
    ModelConfiguration,
    SteeringMessage,
    StreamEvent,
    Thread,
)
from .services import build_agent_configuration, start_thread


class _ChallengeGroup(TypedDict):
    challenge: Challenge
    threads: list[Thread]


class _ThreadGroup(TypedDict):
    ctf: Ctf
    challenges: list[_ChallengeGroup]
    thread_count: int


def index(request: HttpRequest) -> HttpResponse:
    ctfs = [
        ctf
        for ctf in Ctf.objects.prefetch_related("view_groups")
        if ctf.can_view(request.user)
    ]
    ctf_ids = [ctf.pk for ctf in ctfs]
    thread_filter = Q(is_public=True)
    if request.user.is_authenticated:
        thread_filter |= Q(ctf_id__in=ctf_ids)
    threads = (
        Thread.objects.select_related(
            "ctf", "challenge", "agent", "model", "credential"
        )
        .filter(thread_filter)
        .distinct()[:20]
    )
    public_thread_groups = _group_threads_by_ctf_and_challenge(
        Thread.objects.select_related(
            "ctf", "challenge", "agent", "model", "credential"
        ).filter(is_public=True)[:40]
    )
    public_thread_count = sum(group["thread_count"] for group in public_thread_groups)
    return render(
        request,
        "ctf/index.html",
        {
            "ctfs": ctfs,
            "threads": threads,
            "public_thread_groups": public_thread_groups,
            "public_thread_count": public_thread_count,
        },
    )


@login_required
def credential_list(request: HttpRequest) -> HttpResponse:
    credentials = [
        credential
        for credential in Credential.objects.prefetch_related("allowed_groups")
        if credential.can_view(request.user)
    ]
    return render(
        request,
        "ctf/credential_list.html",
        {"credentials": credentials},
    )


@login_required
def credential_create(request: HttpRequest) -> HttpResponse:
    form = CredentialForm(request.POST or None)
    if request.method == "POST" and form.is_valid():
        credential = form.save(commit=False)
        credential.created_by = request.user
        credential.save()
        form.save_m2m()
        messages.success(request, "Credential saved.")
        return redirect("ctf:credential_list")
    return render(
        request,
        "ctf/form.html",
        {"form": form, "title": "New credential"},
    )


@login_required
def model_list(request: HttpRequest) -> HttpResponse:
    models = [
        model
        for model in ModelConfiguration.objects.prefetch_related(
            "view_groups", "use_groups"
        )
        if model.can_view(request.user)
    ]
    return render(request, "ctf/model_list.html", {"models": models})


@login_required
def model_create(request: HttpRequest) -> HttpResponse:
    form = ModelConfigurationForm(request.POST or None)
    if request.method == "POST" and form.is_valid():
        model = form.save(commit=False)
        model.created_by = request.user
        model.save()
        form.save_m2m()
        messages.success(request, "Model saved.")
        return redirect("ctf:model_list")
    return render(request, "ctf/form.html", {"form": form, "title": "New model"})


@login_required
def model_update(request: HttpRequest, slug: str) -> HttpResponse:
    model = get_object_or_404(
        ModelConfiguration.objects.prefetch_related("view_groups", "use_groups"),
        slug=slug,
    )
    if not model.can_view(request.user):
        raise PermissionDenied

    form = ModelConfigurationForm(request.POST or None, instance=model)
    if request.method == "POST" and form.is_valid():
        form.save()
        messages.success(request, "Model updated.")
        return redirect("ctf:model_list")
    return render(
        request,
        "ctf/form.html",
        {"form": form, "title": f"Edit model: {model.name}"},
    )


@login_required
def agent_list(request: HttpRequest) -> HttpResponse:
    agents = [
        agent
        for agent in AgentConfiguration.objects.prefetch_related(
            "view_groups", "use_groups"
        )
        if agent.can_view(request.user)
    ]
    return render(request, "ctf/agent_list.html", {"agents": agents})


@login_required
def agent_create(request: HttpRequest) -> HttpResponse:
    form = AgentConfigurationForm(request.POST or None, user=request.user)
    if request.method == "POST" and form.is_valid():
        agent = form.save(commit=False)
        agent.created_by = request.user
        agent.save()
        form.save_m2m()
        messages.success(request, "Agent configuration saved.")
        return redirect(agent)
    return render(request, "ctf/form.html", {"form": form, "title": "New agent"})


@login_required
def agent_update(request: HttpRequest, slug: str) -> HttpResponse:
    agent = get_object_or_404(AgentConfiguration, slug=slug)
    if not agent.can_view(request.user):
        raise PermissionDenied

    form = AgentConfigurationForm(
        request.POST or None, instance=agent, user=request.user
    )
    if request.method == "POST" and form.is_valid():
        agent = form.save()
        messages.success(request, "Agent configuration updated.")
        return redirect(agent)
    return render(
        request,
        "ctf/form.html",
        {"form": form, "title": f"Edit agent: {agent.name}"},
    )


@login_required
def agent_detail(request: HttpRequest, slug: str) -> HttpResponse:
    agent = get_object_or_404(AgentConfiguration, slug=slug)
    if not agent.can_view(request.user):
        raise PermissionDenied
    resolves = False
    try:
        agent.resolved_mapping(user=request.user)
        resolves = True
    except Exception as exc:
        messages.error(request, f"Could not resolve YAML: {exc}")
    return render(
        request,
        "ctf/agent_detail.html",
        {"agent": agent, "resolves": resolves},
    )


@login_required
def ctf_create(request: HttpRequest) -> HttpResponse:
    form = CtfForm(request.POST or None)
    if request.method == "POST" and form.is_valid():
        ctf = form.save(commit=False)
        ctf.created_by = request.user
        ctf.save()
        form.save_m2m()
        messages.success(request, "CTF saved.")
        return redirect(ctf)
    return render(request, "ctf/form.html", {"form": form, "title": "New CTF"})


@login_required
def ctf_update(request: HttpRequest, slug: str) -> HttpResponse:
    ctf = get_object_or_404(
        Ctf.objects.prefetch_related("view_groups", "init_groups"), slug=slug
    )
    if not ctf.can_init_thread(request.user):
        raise PermissionDenied
    form = CtfForm(request.POST or None, instance=ctf)
    if request.method == "POST" and form.is_valid():
        ctf = form.save()
        messages.success(request, "CTF updated.")
        return redirect(ctf)
    return render(
        request,
        "ctf/form.html",
        {"form": form, "title": f"Edit CTF: {ctf.title}"},
    )


@login_required
def ctf_detail(request: HttpRequest, slug: str) -> HttpResponse:
    ctf = get_object_or_404(
        Ctf.objects.prefetch_related("view_groups", "init_groups"), slug=slug
    )
    if not ctf.can_view(request.user):
        raise PermissionDenied

    return render(
        request,
        "ctf/ctf_detail.html",
        {
            "ctf": ctf,
            "challenges": ctf.challenges.all(),
            "can_init": ctf.can_init_thread(request.user),
        },
    )


@login_required
def challenge_create(request: HttpRequest, ctf_slug: str) -> HttpResponse:
    ctf = get_object_or_404(Ctf, slug=ctf_slug)
    if not ctf.can_init_thread(request.user):
        raise PermissionDenied
    form = ChallengeForm(request.POST or None, request.FILES or None)
    if request.method == "POST" and form.is_valid():
        challenge = form.save(commit=False)
        challenge.ctf = ctf
        challenge.created_by = request.user
        challenge.save()
        messages.success(request, "Challenge saved.")
        return redirect(ctf)
    return render(
        request,
        "ctf/form.html",
        {"form": form, "title": f"New challenge for {ctf.title}"},
    )


@login_required
def challenge_update(
    request: HttpRequest, ctf_slug: str, challenge_id: str
) -> HttpResponse:
    challenge = get_object_or_404(
        Challenge.objects.select_related("ctf").prefetch_related("ctf__init_groups"),
        ctf__slug=ctf_slug,
        challenge_id=challenge_id,
    )
    if not challenge.ctf.can_init_thread(request.user):
        raise PermissionDenied
    form = ChallengeForm(
        request.POST or None,
        request.FILES or None,
        instance=challenge,
    )
    if request.method == "POST" and form.is_valid():
        challenge = form.save()
        messages.success(request, "Challenge updated.")
        return redirect(challenge)
    return render(
        request,
        "ctf/form.html",
        {"form": form, "title": f"Edit challenge: {challenge.challenge_id}"},
    )


@login_required
def challenge_detail(
    request: HttpRequest, ctf_slug: str, challenge_id: str
) -> HttpResponse:
    challenge = get_object_or_404(
        Challenge.objects.select_related("ctf").prefetch_related(
            "ctf__view_groups", "ctf__init_groups"
        ),
        ctf__slug=ctf_slug,
        challenge_id=challenge_id,
    )
    ctf = challenge.ctf
    if not ctf.can_view(request.user):
        raise PermissionDenied

    thread_form = ThreadCreateForm(user=request.user)
    return render(
        request,
        "ctf/challenge_detail.html",
        {
            "ctf": ctf,
            "challenge": challenge,
            "threads": challenge.threads.select_related("agent", "model", "credential"),
            "thread_form": thread_form,
            "can_init": ctf.can_init_thread(request.user),
        },
    )


@login_required
@require_POST
def thread_create(
    request: HttpRequest, ctf_slug: str, challenge_id: str
) -> HttpResponse:
    ctf = get_object_or_404(Ctf.objects.prefetch_related("init_groups"), slug=ctf_slug)
    if not ctf.can_init_thread(request.user):
        raise PermissionDenied
    challenge = get_object_or_404(Challenge, ctf=ctf, challenge_id=challenge_id)

    form = ThreadCreateForm(request.POST, user=request.user)
    if not form.is_valid():
        messages.error(request, "Could not start thread.")
        return redirect(challenge)

    agent = form.cleaned_data["agent"]
    if not agent.can_use(request.user):
        raise PermissionDenied
    model = form.cleaned_data["model"]
    if not model.can_use(request.user):
        raise PermissionDenied
    credential = form.cleaned_data["credential"]
    if not credential.can_view(request.user):
        raise PermissionDenied

    try:
        build_agent_configuration(
            agent,
            model_configuration=model,
            credential=credential,
            user=request.user,
        )
    except PermissionDenied:
        raise
    except Exception as exc:
        messages.error(request, f"Could not resolve agent configuration: {exc}")
        return redirect(challenge)

    thread = Thread.objects.create(
        ctf=ctf,
        challenge=challenge,
        agent=agent,
        model=model,
        credential=credential,
        created_by=request.user,
        name=form.cleaned_data["name"],
    )
    start_thread(thread)
    messages.success(request, "Thread queued.")
    return redirect(thread)


def thread_detail(request: HttpRequest, pk: int) -> HttpResponse:
    thread = get_object_or_404(
        Thread.objects.select_related(
            "ctf", "challenge", "agent", "model", "credential"
        ),
        pk=pk,
    )
    if not thread.can_view(request.user):
        if not request.user.is_authenticated:
            return redirect_to_login(request.get_full_path())
        raise PermissionDenied
    can_manage_thread = thread.ctf.can_view(request.user)
    events = list(thread.events.all()[:2000])
    return render(
        request,
        "ctf/thread_detail.html",
        {
            "thread": thread,
            "events": events,
            "events_json": [
                _event_payload(event, cost_model=_thread_cost_model(thread))
                for event in events
            ],
            "can_manage_thread": can_manage_thread,
        },
    )


@login_required
@require_POST
def thread_publish(request: HttpRequest, pk: int) -> HttpResponse:
    thread = get_object_or_404(Thread.objects.select_related("ctf"), pk=pk)
    if not thread.can_publish(request.user):
        raise PermissionDenied

    thread.is_public = request.POST.get("is_public") == "1"
    thread.save(update_fields=["is_public", "updated_at"])
    messages.success(
        request,
        "Thread published." if thread.is_public else "Thread unpublished.",
    )
    return redirect(thread)


@login_required
@require_POST
def thread_steer(request: HttpRequest, pk: int) -> HttpResponse:
    thread = get_object_or_404(Thread.objects.select_related("ctf"), pk=pk)
    if not thread.ctf.can_view(request.user):
        raise PermissionDenied

    text = request.POST.get("text", "").strip()
    if not text:
        messages.error(request, "Steer message cannot be empty.")
        return redirect(thread)
    active_statuses = {Thread.Status.QUEUED, Thread.Status.RUNNING}
    stopped_statuses = {Thread.Status.COMPLETED, Thread.Status.FAILED}
    should_resume = thread.status in stopped_statuses
    if thread.status not in active_statuses | stopped_statuses:
        messages.error(request, "This thread cannot receive steer messages.")
        return redirect(thread)

    SteeringMessage.objects.create(
        thread=thread,
        created_by=request.user,
        text=text,
    )
    if should_resume:
        Thread.objects.filter(pk=thread.pk, status__in=stopped_statuses).update(
            status=Thread.Status.QUEUED,
            error="",
            updated_at=timezone.now(),
        )
        thread.status = Thread.Status.QUEUED
        thread.error = ""
        start_thread(thread)
        messages.success(request, "Steer message queued; thread is resuming.")
    else:
        messages.success(request, "Steer message queued.")
    return redirect(thread)


def thread_stream(request: HttpRequest, pk: int) -> HttpResponse:
    thread = get_object_or_404(Thread.objects.select_related("ctf"), pk=pk)
    if not thread.can_view(request.user):
        if not request.user.is_authenticated:
            return redirect_to_login(request.get_full_path())
        raise PermissionDenied
    last_sequence = max(
        _nonnegative_int(request.GET.get("after")),
        _nonnegative_int(request.headers.get("Last-Event-ID")),
    )
    response = StreamingHttpResponse(
        _event_stream(thread.pk, last_sequence),
        content_type="text/event-stream",
    )
    response["Cache-Control"] = "no-cache"
    response["X-Accel-Buffering"] = "no"
    return response


def _event_stream(thread_id: int, last_sequence: int = 0) -> Iterator[str]:
    while True:
        thread = Thread.objects.get(pk=thread_id)
        for event in _events_after(thread_id, last_sequence):
            last_sequence = event.sequence
            yield f"id: {event.sequence}\n"
            yield "event: stream\n"
            yield f"data: {json.dumps(_event_payload(event, cost_model=_thread_cost_model(thread)), ensure_ascii=False)}\n\n"

        yield "event: cost\n"
        yield f"data: {json.dumps({'cost': thread.latest_cost}, ensure_ascii=False)}\n\n"

        if thread.status in {Thread.Status.COMPLETED, Thread.Status.FAILED}:
            yield "event: status\n"
            yield f"data: {json.dumps({'status': thread.status, 'error': thread.error}, ensure_ascii=False)}\n\n"
            return
        time.sleep(1)


def _group_threads_by_ctf_and_challenge(
    threads: QuerySet[Thread],
) -> list[_ThreadGroup]:
    groups: list[_ThreadGroup] = []
    group_by_ctf_id: dict[int, _ThreadGroup] = {}
    challenge_index: dict[tuple[int, int], _ChallengeGroup] = {}
    for thread in threads:
        ctf_group = group_by_ctf_id.get(thread.ctf_id)
        if ctf_group is None:
            ctf_group = {"ctf": thread.ctf, "challenges": [], "thread_count": 0}
            group_by_ctf_id[thread.ctf_id] = ctf_group
            groups.append(ctf_group)
        ch_key = (thread.ctf_id, thread.challenge_id)
        challenge_group = challenge_index.get(ch_key)
        if challenge_group is None:
            challenge_group = {"challenge": thread.challenge, "threads": []}
            challenge_index[ch_key] = challenge_group
            ctf_group["challenges"].append(challenge_group)
        challenge_group["threads"].append(thread)
        ctf_group["thread_count"] += 1
    return groups


def _events_after(thread_id: int, sequence: int) -> QuerySet[StreamEvent]:
    return StreamEvent.objects.filter(
        thread_id=thread_id, sequence__gt=sequence
    ).order_by("sequence")


def _event_payload(
    event: StreamEvent, *, cost_model: str | None = None
) -> dict[str, object]:
    cost_usd = _event_cost_usd(event, cost_model=cost_model)
    return {
        "sequence": event.sequence,
        "source": event.source,
        "kind": event.kind,
        "text": event.text,
        "raw": event.raw,
        "cost_usd": str(cost_usd) if cost_usd is not None else None,
        "created_at": event.created_at.isoformat(),
    }


def _event_cost_usd(event: StreamEvent, *, cost_model: str | None) -> object | None:
    if not cost_model or event.source != "codex_jsonl" or event.kind != "token_count":
        return None
    usage = _token_count_total_usage(event.raw)
    if usage is None:
        return None
    return estimate_cost(cost_model, usage).usd


def _token_count_total_usage(raw: dict[str, Any]) -> TokenUsage | None:
    payload = raw.get("payload")
    if not isinstance(payload, dict):
        return None
    info = payload.get("info")
    if not isinstance(info, dict):
        return None
    usage = info.get("total_token_usage") or info.get("last_token_usage")
    if not isinstance(usage, dict):
        return None
    return TokenUsage(
        input_tokens=_int_value(usage.get("input_tokens")),
        cached_input_tokens=_int_value(usage.get("cached_input_tokens")),
        output_tokens=_int_value(usage.get("output_tokens")),
    )


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


def _thread_cost_model(thread: Thread) -> str | None:
    latest_cost = thread.latest_cost
    if isinstance(latest_cost, dict):
        model = latest_cost.get("model")
        if isinstance(model, str) and model:
            return model
    return None


def _nonnegative_int(value: str | None) -> int:
    if value is None:
        return 0
    try:
        return max(int(value), 0)
    except ValueError:
        return 0
