# Catchy Web

Django 6 web UI for managing CTF challenges, agent YAML, model names,
credentials, threads, and stream archives.

```bash
uv sync
uv run python -m catchy.web.manage migrate
uv run python -m catchy.web.manage createsuperuser
uv run python -m catchy.web.manage runserver
```

The default task backend is Django's immediate backend. Configure `TASKS` with a
real backend before running long agent streams from production web requests.
