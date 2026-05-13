from __future__ import annotations

from django.conf import settings
from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        migrations.swappable_dependency(settings.AUTH_USER_MODEL),
        ("ctf", "0015_thread_uuid"),
    ]

    operations = [
        migrations.AddField(
            model_name="credential",
            name="allowed_users",
            field=models.ManyToManyField(
                blank=True,
                related_name="credentials",
                to=settings.AUTH_USER_MODEL,
            ),
        ),
    ]
