import uuid

from django.db import models


class AnalysisJob(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    owner_session = models.CharField(max_length=64, db_index=True)
    status = models.CharField(max_length=16, default="queued", db_index=True)
    progress = models.PositiveSmallIntegerField(default=0)
    input_name = models.CharField(max_length=255)
    kind = models.CharField(max_length=16)
    model_id = models.CharField(max_length=80, blank=True)
    parameters = models.JSONField(default=dict)
    result = models.JSONField(default=dict)
    error = models.TextField(blank=True)
    cancel_requested = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["-created_at"]


class AssistantRecord(models.Model):
    owner_session = models.CharField(max_length=64, db_index=True)
    job = models.ForeignKey(AnalysisJob, null=True, blank=True, on_delete=models.CASCADE)
    mode = models.CharField(max_length=20, default="chat")
    question = models.TextField(blank=True)
    answer = models.TextField(blank=True)
    source = models.CharField(max_length=16, default="llm")
    model = models.CharField(max_length=100, blank=True)
    usage = models.JSONField(default=dict)
    status = models.CharField(max_length=16, default="completed")
    error = models.CharField(max_length=300, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at"]


# Register the additive manual-review model without changing original app models.
from .var_models import VarReviewCase  # noqa: E402,F401
