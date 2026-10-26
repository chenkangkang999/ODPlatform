from django.urls import path

from . import views

app_name = "football_portal"
urlpatterns = [
    path("", views.shell, name="home"),
    path("api/status/", views.status_view, name="status"),
    path("api/models/", views.models_view, name="models"),
    path("api/jobs/", views.jobs_view, name="jobs"),
    path("api/demo/", views.demo_view, name="demo"),
    path("api/jobs/<uuid:job_id>/", views.job_detail, name="job"),
    path("api/jobs/<uuid:job_id>/cancel/", views.cancel_job, name="cancel"),
    path("api/jobs/<uuid:job_id>/artifact/<str:key>/", views.artifact, name="artifact"),
    path("api/import/", views.import_view, name="import"),
    path("api/assistant/", views.assistant, name="assistant"),
    path("api/local-summary/", views.local_summary_view, name="local-summary"),
    path("api/assistant/history/", views.assistant_history, name="assistant-history"),
    path("api/usage/", views.usage, name="usage"),
]
