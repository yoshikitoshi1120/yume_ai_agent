from django.urls import path
from.views import ai_agent_interaction

urlpatterns = [
    path('ai_agent', ai_agent_interaction),
]