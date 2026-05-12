from __future__ import annotations

from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("ctf", "0009_steeringmessage_kind_alter_steeringmessage_text_and_more"),
    ]

    operations = [
        migrations.AlterField(
            model_name="credential",
            name="kind",
            field=models.CharField(
                choices=[("anthropic", "Anthropic"), ("openai", "OpenAI")],
                default="openai",
                max_length=30,
            ),
        ),
        migrations.AlterField(
            model_name="credential",
            name="base_url",
            field=models.URLField(blank=True, default="https://api.openai.com/v1"),
        ),
    ]
