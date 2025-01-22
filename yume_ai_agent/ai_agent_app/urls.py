from django.urls import path
from . import views

urlpatterns = [
    path('start-ai-response/', views.start_ai_response, name='start_ai_response'),
    path('generate-bot-twitter/', views.generate_bot_twitter, name='generate_bot_twitter'),
]
