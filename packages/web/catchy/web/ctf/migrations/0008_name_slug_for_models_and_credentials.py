from django.db import migrations, models
from django.utils.text import slugify


def populate_names_and_slugs(apps, schema_editor):
    credential_model = apps.get_model("ctf", "Credential")
    for credential in credential_model.objects.all():
        name = credential.label or credential.slug
        credential_model.objects.filter(pk=credential.pk).update(name=name)

    model_configuration_model = apps.get_model("ctf", "ModelConfiguration")
    seen_slugs = set()
    for model_configuration in model_configuration_model.objects.order_by("pk"):
        base_slug = slugify(model_configuration.name)[:200] or (
            f"model-{model_configuration.pk}"
        )
        slug = base_slug
        suffix = 2
        while slug in seen_slugs or model_configuration_model.objects.filter(
            slug=slug
        ).exclude(pk=model_configuration.pk).exists():
            suffix_text = f"-{suffix}"
            slug = f"{base_slug[: 200 - len(suffix_text)]}{suffix_text}"
            suffix += 1
        seen_slugs.add(slug)
        model_configuration_model.objects.filter(pk=model_configuration.pk).update(
            slug=slug
        )


class Migration(migrations.Migration):
    dependencies = [
        ("ctf", "0007_credentials_models_thread_runtime"),
    ]

    operations = [
        migrations.RenameField(
            model_name="credential",
            old_name="name",
            new_name="slug",
        ),
        migrations.AddField(
            model_name="credential",
            name="name",
            field=models.CharField(default="", max_length=200),
            preserve_default=False,
        ),
        migrations.AlterField(
            model_name="credential",
            name="slug",
            field=models.SlugField(max_length=200, unique=True),
        ),
        migrations.AddField(
            model_name="modelconfiguration",
            name="slug",
            field=models.SlugField(blank=True, default="", max_length=200),
            preserve_default=False,
        ),
        migrations.RunPython(
            populate_names_and_slugs,
            reverse_code=migrations.RunPython.noop,
        ),
        migrations.RemoveField(
            model_name="credential",
            name="label",
        ),
        migrations.RemoveField(
            model_name="modelconfiguration",
            name="label",
        ),
        migrations.AlterField(
            model_name="modelconfiguration",
            name="name",
            field=models.CharField(max_length=200),
        ),
        migrations.AlterField(
            model_name="modelconfiguration",
            name="slug",
            field=models.SlugField(max_length=200, unique=True),
        ),
    ]
