from __future__ import annotations

from typing import Any
from unittest.mock import patch

from django.contrib.auth.models import Group, User
from django.core.exceptions import PermissionDenied
from django.db import OperationalError
from django.test import TestCase
from django.urls import reverse

from catchy.web.ctf import services
from catchy.web.ctf.models import (
    AgentConfiguration,
    Challenge,
    Ctf,
    Secret,
    StreamEvent,
    Thread,
)


class SecretAgentPermissionTests(TestCase):
    def setUp(self) -> None:
        self.allowed_group = Group.objects.create(name="secret-users")
        self.allowed_user = User.objects.create_user(
            username="allowed",
            password="password",
        )
        self.allowed_user.groups.add(self.allowed_group)
        self.denied_user = User.objects.create_user(
            username="denied",
            password="password",
        )
        self.secret = Secret.objects.create(name="api-token", value="top-secret")
        self.secret.allowed_groups.add(self.allowed_group)

    def test_agent_resolves_secret_for_allowed_user(self) -> None:
        agent = AgentConfiguration(
            name="Codex",
            slug="codex",
            yaml="model:\n  name: ${secret:api-token}\n",
        )

        self.assertEqual(
            agent.resolved_mapping(user=self.allowed_user),
            {"model": {"name": "top-secret"}},
        )

    def test_agent_resolution_rejects_disallowed_user(self) -> None:
        agent = AgentConfiguration(
            name="Codex",
            slug="codex",
            yaml="model:\n  name: ${secret:api-token}\n",
        )

        with self.assertRaises(PermissionDenied):
            agent.resolved_mapping(user=self.denied_user)

    def test_agent_create_rejects_secret_user_cannot_view(self) -> None:
        self.client.force_login(self.denied_user)

        response = self.client.post(
            reverse("ctf:agent_create"),
            {
                "name": "Codex",
                "slug": "codex",
                "yaml": "model:\n  name: ${secret:api-token}\n",
            },
        )

        self.assertEqual(response.status_code, 200)
        self.assertFalse(AgentConfiguration.objects.filter(slug="codex").exists())

    def test_thread_create_rejects_agent_secret_user_cannot_view(self) -> None:
        ctf = Ctf.objects.create(title="Study", slug="study")
        challenge = Challenge.objects.create(
            ctf=ctf,
            challenge_id="canary",
            source_archive="ctfs/study/challenges/canary/source.tgz",
        )
        agent = AgentConfiguration.objects.create(
            name="Codex",
            slug="codex",
            yaml="model:\n  name: ${secret:api-token}\n",
        )
        self.client.force_login(self.denied_user)

        response = self.client.post(
            reverse(
                "ctf:thread_create",
                kwargs={"ctf_slug": ctf.slug, "challenge_id": challenge.challenge_id},
            ),
            {"agent": str(agent.pk)},
        )

        self.assertEqual(response.status_code, 403)
        self.assertFalse(Thread.objects.exists())


class ThreadCreateNameTests(TestCase):
    def setUp(self) -> None:
        self.user = User.objects.create_user(username="member", password="password")
        self.ctf = Ctf.objects.create(title="Study", slug="study")
        self.challenge = Challenge.objects.create(
            ctf=self.ctf,
            challenge_id="canary",
            source_archive="ctfs/study/challenges/canary/source.tgz",
        )
        self.agent = AgentConfiguration.objects.create(
            name="Codex",
            slug="codex",
            yaml="{}",
        )
        self.client.force_login(self.user)

    def test_thread_create_accepts_optional_name(self) -> None:
        with patch("catchy.web.ctf.views.start_thread") as start_thread:
            response = self.client.post(
                self._thread_create_url(),
                {"agent": str(self.agent.pk), "name": "My First Run"},
            )

        thread = Thread.objects.get()
        self.assertRedirects(response, thread.get_absolute_url())
        self.assertEqual(thread.name, "my-first-run")
        self.assertEqual(str(thread), "my-first-run")
        self.assertEqual(start_thread.call_args.args[0].pk, thread.pk)

    def test_thread_create_generates_kebab_name_when_blank(self) -> None:
        with patch("catchy.web.ctf.views.start_thread"):
            response = self.client.post(
                self._thread_create_url(),
                {"agent": str(self.agent.pk), "name": ""},
            )

        thread = Thread.objects.get()
        self.assertRedirects(response, thread.get_absolute_url())
        self.assertRegex(thread.name, r"^[a-z]+-[a-z]+-\d{4}$")
        self.assertEqual(str(thread), thread.name)

    def _thread_create_url(self) -> str:
        return reverse(
            "ctf:thread_create",
            kwargs={
                "ctf_slug": self.ctf.slug,
                "challenge_id": self.challenge.challenge_id,
            },
        )


class StreamEventRecordingTests(TestCase):
    def setUp(self) -> None:
        self.ctf = Ctf.objects.create(title="Study", slug="study")
        self.challenge = Challenge.objects.create(
            ctf=self.ctf,
            challenge_id="canary",
            source_archive="ctfs/study/challenges/canary/source.tgz",
        )
        self.agent = AgentConfiguration.objects.create(
            name="Codex",
            slug="codex",
            yaml="{}",
        )
        self.thread = Thread.objects.create(
            ctf=self.ctf,
            challenge=self.challenge,
            agent=self.agent,
        )

    def test_record_event_retries_transient_sqlite_lock(self) -> None:
        calls = 0
        record_event_once = services._record_event_once

        def flaky_record_event_once(*args: Any, **kwargs: Any) -> StreamEvent:
            nonlocal calls
            calls += 1
            if calls == 1:
                raise OperationalError("database is locked")
            return record_event_once(*args, **kwargs)

        with (
            patch(
                "catchy.web.ctf.services._record_event_once",
                side_effect=flaky_record_event_once,
            ),
            patch("catchy.web.ctf.services.time.sleep"),
        ):
            event = services._record_event(
                self.thread,
                source="agent_stream",
                kind="delta",
                text="hello",
            )

        self.assertEqual(calls, 2)
        self.assertEqual(event.sequence, 1)
        self.assertEqual(event.text, "hello")


class PublicThreadAccessTests(TestCase):
    def setUp(self) -> None:
        self.user = User.objects.create_user(username="member", password="password")
        self.ctf = Ctf.objects.create(title="Study", slug="study")
        self.agent = AgentConfiguration.objects.create(
            name="Codex",
            slug="codex",
            yaml="{}",
        )

    def test_anonymous_dashboard_groups_only_public_threads_by_ctf(self) -> None:
        public_thread = self._create_thread("public", is_public=True)
        second_public_thread = self._create_thread("second-public", is_public=True)
        private_thread = self._create_thread("private", is_public=False)
        other_ctf = Ctf.objects.create(title="Other", slug="other")
        other_public_thread = self._create_thread(
            "other-public",
            is_public=True,
            ctf=other_ctf,
        )

        response = self.client.get(reverse("ctf:index"))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, str(public_thread))
        self.assertContains(response, str(second_public_thread))
        self.assertContains(response, str(other_public_thread))
        self.assertContains(response, "Public threads")
        self.assertNotContains(response, str(private_thread))
        self.assertEqual(response.context["public_thread_count"], 3)

        groups = response.context["public_thread_groups"]
        grouped_threads = {
            group["ctf"].slug: {
                thread.pk
                for challenge_group in group["challenges"]
                for thread in challenge_group["threads"]
            }
            for group in groups
        }
        self.assertEqual(
            grouped_threads,
            {
                "study": {public_thread.pk, second_public_thread.pk},
                "other": {other_public_thread.pk},
            },
        )

    def test_anonymous_can_view_public_thread(self) -> None:
        thread = self._create_thread("public-detail", is_public=True)

        response = self.client.get(thread.get_absolute_url())

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, str(thread))
        self.assertContains(response, "public")
        self.assertNotContains(response, "Publish")
        self.assertNotContains(response, "Steer</button>")

    def test_anonymous_private_thread_redirects_to_login(self) -> None:
        thread = self._create_thread("private-detail", is_public=False)

        response = self.client.get(thread.get_absolute_url())

        self.assertEqual(response.status_code, 302)
        self.assertIn("/accounts/login/", response["Location"])

    def test_published_thread_detail_shows_unpublish_button(self) -> None:
        thread = self._create_thread("published", is_public=True)
        self.client.force_login(self.user)

        response = self.client.get(thread.get_absolute_url())

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Unpublish")
        self.assertNotContains(response, ">Publish</button>")

    def test_authenticated_user_can_publish_and_unpublish_thread(self) -> None:
        thread = self._create_thread("publishable", is_public=False)
        self.client.force_login(self.user)

        response = self.client.post(
            reverse("ctf:thread_publish", kwargs={"pk": thread.pk}),
            {"is_public": "1"},
        )
        thread.refresh_from_db()
        self.assertRedirects(response, thread.get_absolute_url())
        self.assertTrue(thread.is_public)

        response = self.client.post(
            reverse("ctf:thread_publish", kwargs={"pk": thread.pk}),
            {"is_public": "0"},
        )
        thread.refresh_from_db()
        self.assertRedirects(response, thread.get_absolute_url())
        self.assertFalse(thread.is_public)

    def test_thread_detail_includes_cumulative_token_count_cost(self) -> None:
        thread = self._create_thread("costed", is_public=True)
        thread.latest_cost = {"model": "gpt-5.5"}
        thread.save(update_fields=["latest_cost", "updated_at"])
        StreamEvent.objects.create(
            thread=thread,
            sequence=1,
            dedupe_key="token-count",
            source="codex_jsonl",
            kind="token_count",
            text="{}",
            raw={
                "type": "event_msg",
                "payload": {
                    "type": "token_count",
                    "info": {
                        "total_token_usage": {
                            "input_tokens": 14705,
                            "cached_input_tokens": 0,
                            "output_tokens": 165,
                        }
                    },
                },
            },
        )

        response = self.client.get(thread.get_absolute_url())

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.context["events_json"][0]["cost_usd"], "0.078475")

    def test_thread_stream_starts_after_requested_sequence(self) -> None:
        thread = self._create_thread("stream-after", is_public=True)
        thread.latest_cost = {"model": "gpt-5.5", "usd": "0.000000"}
        thread.save(update_fields=["latest_cost", "updated_at"])
        StreamEvent.objects.create(
            thread=thread,
            sequence=1,
            dedupe_key="one",
            source="system",
            kind="thread.started",
            text="one",
        )
        StreamEvent.objects.create(
            thread=thread,
            sequence=2,
            dedupe_key="two",
            source="system",
            kind="thread.completed",
            text="two",
        )

        response = self.client.get(
            reverse("ctf:thread_stream", kwargs={"pk": thread.pk}),
            {"after": "1"},
        )
        body = b"".join(response.streaming_content).decode()

        self.assertNotIn('"sequence": 1', body)
        self.assertIn("id: 2", body)
        self.assertIn('"sequence": 2', body)
        self.assertEqual(response.headers["X-Accel-Buffering"], "no")

    def test_thread_stream_resumes_after_last_event_id(self) -> None:
        thread = self._create_thread("stream-last-event-id", is_public=True)
        for sequence in range(1, 4):
            StreamEvent.objects.create(
                thread=thread,
                sequence=sequence,
                dedupe_key=str(sequence),
                source="system",
                kind="thread.event",
                text=str(sequence),
            )

        response = self.client.get(
            reverse("ctf:thread_stream", kwargs={"pk": thread.pk}),
            {"after": "1"},
            headers={"Last-Event-ID": "2"},
        )
        body = b"".join(response.streaming_content).decode()

        self.assertNotIn('"sequence": 1', body)
        self.assertNotIn('"sequence": 2', body)
        self.assertIn("id: 3", body)
        self.assertIn('"sequence": 3', body)

    def _create_thread(
        self,
        challenge_id: str,
        *,
        is_public: bool,
        ctf: Ctf | None = None,
    ) -> Thread:
        ctf = ctf or self.ctf
        challenge = Challenge.objects.create(
            ctf=ctf,
            challenge_id=challenge_id,
            source_archive=f"ctfs/{ctf.slug}/challenges/{challenge_id}/source.tgz",
        )
        return Thread.objects.create(
            ctf=ctf,
            challenge=challenge,
            agent=self.agent,
            status=Thread.Status.COMPLETED,
            is_public=is_public,
        )
