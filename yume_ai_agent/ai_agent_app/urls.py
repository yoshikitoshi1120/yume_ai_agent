from django.urls import path
from . import views

urlpatterns = [
    path('start-ai-response/', views.start_ai_response, name='start_ai_response'),
    path('get-task-result/<str:task_id>/', views.get_task_result, name='get_task_result'),
]