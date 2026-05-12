from __future__ import annotations

from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("ctf", "0010_add_anthropic_credentials"),
    ]

    operations = [
        migrations.AlterField(
            model_name="credential",
            name="base_url",
            field=models.URLField(blank=True, default=""),
        ),
        migrations.AlterField(
            model_name="credential",
            name="kind",
            field=models.CharField(
                choices=[
                    ("anthropic", "Anthropic API key"),
                    ("claude_oauth_token", "Claude Code OAuth token"),
                    ("codex_auth_json", "Codex auth.json"),
                    ("openai", "OpenAI API key"),
                ],
                default="openai",
                max_length=30,
            ),
        ),
    ]
