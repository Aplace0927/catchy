from __future__ import annotations

import django.db.models.deletion
import django.utils.timezone
from django.conf import settings
from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("auth", "0012_alter_user_first_name_max_length"),
        ("ctf", "0006_thread_name"),
        migrations.swappable_dependency(settings.AUTH_USER_MODEL),
    ]

    operations = [
        migrations.RenameModel(
            old_name="Secret",
            new_name="Credential",
        ),
        migrations.RenameField(
            model_name="credential",
            old_name="value",
            new_name="api_key",
        ),
        migrations.AddField(
            model_name="credential",
            name="kind",
            field=models.CharField(
                choices=[("openai", "OpenAI")],
                default="openai",
                max_length=30,
            ),
        ),
        migrations.AddField(
            model_name="credential",
            name="base_url",
            field=models.URLField(default="https://api.openai.com/v1"),
        ),
        migrations.AddField(
            model_name="credential",
            name="organization_id",
            field=models.CharField(blank=True, max_length=200),
        ),
        migrations.AlterField(
            model_name="credential",
            name="allowed_groups",
            field=models.ManyToManyField(
                blank=True,
                related_name="credentials",
                to="auth.group",
            ),
        ),
        migrations.AlterField(
            model_name="credential",
            name="created_by",
            field=models.ForeignKey(
                blank=True,
                null=True,
                on_delete=django.db.models.deletion.SET_NULL,
                related_name="created_credentials",
                to=settings.AUTH_USER_MODEL,
            ),
        ),
        migrations.CreateModel(
            name="ModelConfiguration",
            fields=[
                (
                    "id",
                    models.BigAutoField(
                        auto_created=True,
                        primary_key=True,
                        serialize=False,
                        verbose_name="ID",
                    ),
                ),
                ("created_at", models.DateTimeField(default=django.utils.timezone.now)),
                ("updated_at", models.DateTimeField(auto_now=True)),
                ("name", models.CharField(max_length=200, unique=True)),
                ("label", models.CharField(blank=True, max_length=200)),
                (
                    "created_by",
                    models.ForeignKey(
                        blank=True,
                        null=True,
                        on_delete=django.db.models.deletion.SET_NULL,
                        related_name="created_model_configurations",
                        to=settings.AUTH_USER_MODEL,
                    ),
                ),
                (
                    "use_groups",
                    models.ManyToManyField(
                        blank=True,
                        related_name="usable_model_configurations",
                        to="auth.group",
                    ),
                ),
                (
                    "view_groups",
                    models.ManyToManyField(
                        blank=True,
                        related_name="viewable_model_configurations",
                        to="auth.group",
                    ),
                ),
            ],
            options={
                "ordering": ["name"],
            },
        ),
        migrations.AddField(
            model_name="thread",
            name="credential",
            field=models.ForeignKey(
                blank=True,
                null=True,
                on_delete=django.db.models.deletion.PROTECT,
                related_name="threads",
                to="ctf.credential",
            ),
        ),
        migrations.AddField(
            model_name="thread",
            name="model",
            field=models.ForeignKey(
                blank=True,
                null=True,
                on_delete=django.db.models.deletion.PROTECT,
                related_name="threads",
                to="ctf.modelconfiguration",
            ),
        ),
    ]
