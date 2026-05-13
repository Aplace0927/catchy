from __future__ import annotations

import uuid

from django.db import migrations, models


def backfill_thread_uuids(apps, schema_editor):
    thread_model = apps.get_model("ctf", "Thread")
    for thread in thread_model.objects.only("pk", "uuid").iterator():
        if thread.uuid:
            continue
        thread_model.objects.filter(pk=thread.pk).update(uuid=uuid.uuid4())


def clear_thread_uuids(apps, schema_editor):
    apps.get_model("ctf", "Thread").objects.update(uuid=None)


class Migration(migrations.Migration):
    dependencies = [
        ("ctf", "0014_reconcile_provider_pricing_after_0013"),
    ]

    operations = [
        migrations.AddField(
            model_name="thread",
            name="uuid",
            field=models.UUIDField(null=True, editable=False),
        ),
        migrations.RunPython(backfill_thread_uuids, clear_thread_uuids),
        migrations.AlterField(
            model_name="thread",
            name="uuid",
            field=models.UUIDField(
                default=uuid.uuid4,
                editable=False,
                unique=True,
            ),
        ),
    ]
