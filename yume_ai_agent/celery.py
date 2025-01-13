from __future__ import absolute_import, unicode_literals
import os
from celery import Celery

# 设置Django settings模块
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'your_project_name.settings')

app = Celery('your_project_name')

# 从Django settings.py中读取配置
app.config_from_object('django.conf:settings', namespace='CELERY')

# 自动发现任务
app.autodiscover_tasks()