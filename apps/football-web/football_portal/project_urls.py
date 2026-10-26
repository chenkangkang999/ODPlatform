from django.urls import include, path
from django.views.generic import RedirectView
from .resource_views import var_resources
from . import monitor_views

urlpatterns = [
    path("", RedirectView.as_view(url="/football/", permanent=False)),
    path("football/api/monitor/", monitor_views.monitor_view, name="football-monitor"),
    path("football/api/events/", monitor_views.events_view, name="football-events"),
    path("football/api/datasets/", monitor_views.datasets_view, name="football-datasets"),
    path("football/api/var/resources/", var_resources, name="var-resources"),
    path("football/api/var/", include("football_portal.var_urls")),
    path("football/", include("football_portal.urls")),
    path("", include("defect_manage.urls")),
]
