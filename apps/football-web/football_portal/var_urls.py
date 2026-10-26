from django.urls import path

from . import var_views

app_name = "football_var"
urlpatterns = [
    path("cases/", var_views.cases, name="cases"),
    path("cases/<uuid:case_id>/", var_views.case_detail, name="case"),
    path("cases/<uuid:case_id>/keyframes/", var_views.keyframes, name="keyframes"),
    path("cases/<uuid:case_id>/keyframes/<uuid:frame_id>/", var_views.delete_keyframe, name="delete-keyframe"),
    path("cases/<uuid:case_id>/artifact/<str:key>/", var_views.artifact, name="artifact"),
    path("cases/<uuid:case_id>/report.json", var_views.report_json, name="report-json"),
    path("cases/<uuid:case_id>/report.md", var_views.report_md, name="report-md"),
]
