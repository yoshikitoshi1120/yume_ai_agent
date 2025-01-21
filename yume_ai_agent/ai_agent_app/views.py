import logging

from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from celery.result import AsyncResult
from .agent import YUMEAgent

agent = YUMEAgent()


@require_http_methods(["POST"])
@csrf_exempt
def start_ai_response(request):
    uuid = request.POST.get('uuid', None)
    if not uuid:
        return JsonResponse({"error": "UUID is missing."}, status=400)
    user_message = request.POST.get('message', '')
    ai_message = agent.generate_ai_response(uuid, user_message)
    return JsonResponse({"ai_message": ai_message})


logger = logging.getLogger(__name__)


@require_http_methods(["POST"])
@csrf_exempt
def generate_bot_twitter(request):
    response = agent.publish_tweet()
    return JsonResponse(response)
