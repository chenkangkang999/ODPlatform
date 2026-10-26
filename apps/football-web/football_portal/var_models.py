"""Manual VAR-review records, separate from detector output and decisions."""
import uuid

from django.db import models


class VarReviewCase(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    owner_session = models.CharField(max_length=64, db_index=True)
    source_job = models.ForeignKey("football_portal.AnalysisJob", null=True, blank=True,
                                   on_delete=models.SET_NULL, related_name="var_reviews")
    title = models.CharField(max_length=160)
    event_type = models.CharField(max_length=40, default="offside")
    status = models.CharField(max_length=20, default="pending", db_index=True)
    input_name = models.CharField(max_length=255)
    media_kind = models.CharField(max_length=10)
    source_filename = models.CharField(max_length=40)
    duration_seconds = models.FloatField(null=True, blank=True)
    observations = models.TextField(blank=True)
    verdict = models.TextField(blank=True)
    keyframes = models.JSONField(default=list)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["-created_at"]

