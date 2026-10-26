import django.db.models.deletion
from django.db import migrations, models
import uuid


class Migration(migrations.Migration):
    initial = True
    dependencies = []
    operations = [
        migrations.CreateModel(
            name="AnalysisJob",
            fields=[
                ("id", models.UUIDField(default=uuid.uuid4, editable=False, primary_key=True, serialize=False)),
                ("owner_session", models.CharField(db_index=True, max_length=64)),
                ("status", models.CharField(db_index=True, default="queued", max_length=16)),
                ("progress", models.PositiveSmallIntegerField(default=0)),
                ("input_name", models.CharField(max_length=255)),
                ("kind", models.CharField(max_length=16)),
                ("model_id", models.CharField(blank=True, max_length=80)),
                ("parameters", models.JSONField(default=dict)),
                ("result", models.JSONField(default=dict)),
                ("error", models.TextField(blank=True)),
                ("cancel_requested", models.BooleanField(default=False)),
                ("created_at", models.DateTimeField(auto_now_add=True)),
                ("updated_at", models.DateTimeField(auto_now=True)),
            ],
            options={"ordering": ["-created_at"]},
        ),
        migrations.CreateModel(
            name="AssistantRecord",
            fields=[
                ("id", models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")),
                ("owner_session", models.CharField(db_index=True, max_length=64)),
                ("mode", models.CharField(default="chat", max_length=20)),
                ("question", models.TextField(blank=True)),
                ("answer", models.TextField(blank=True)),
                ("source", models.CharField(default="llm", max_length=16)),
                ("model", models.CharField(blank=True, max_length=100)),
                ("usage", models.JSONField(default=dict)),
                ("status", models.CharField(default="completed", max_length=16)),
                ("error", models.CharField(blank=True, max_length=300)),
                ("created_at", models.DateTimeField(auto_now_add=True)),
                ("job", models.ForeignKey(blank=True, null=True, on_delete=django.db.models.deletion.CASCADE, to="football_portal.analysisjob")),
            ],
            options={"ordering": ["-created_at"]},
        ),
    ]
