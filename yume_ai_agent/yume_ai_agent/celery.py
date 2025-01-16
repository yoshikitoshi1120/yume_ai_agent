from __future__ import absolute_import, unicode_literals
import os
from celery import Celery

# set Django settings module
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'yume_ai_agent.settings')

app = Celery('yume_ai_agent')

# read config from Django settings.py
app.config_from_object('django.conf:settings', namespace='CELERY')

# auto discover task
app.autodiscover_tasks()