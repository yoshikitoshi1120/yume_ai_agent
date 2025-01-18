import logging

from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from celery.result import AsyncResult
from .tasks import generate_ai_response

@require_http_methods(["POST"])
@csrf_exempt
def start_ai_response(request):
    uuid = request.POST.get('uuid', None)
    if not uuid:
        return JsonResponse({"error": "UUID is missing."}, status=400)
    user_message = request.POST.get('message', '')
    task = generate_ai_response.delay(uuid, user_message)
    return JsonResponse({"task_id": task.id})


logger = logging.getLogger(__name__)


@require_http_methods(["GET"])
@csrf_exempt
def get_task_result(request, task_id):
    try:
        task = AsyncResult(task_id)
    except Exception as e:
        logger.error(f'Error retrieving task {task_id}: {e}')
        return JsonResponse({"status": "error", "message": str(e)}, status=404)

    if task.ready():
        if task.failed():
            logger.error(f'Task {task_id} failed: {task.result}')
            return JsonResponse({"status": "failed", "error": str(task.result)}, status=200)
        else:
            logger.info(f'Task {task_id} succeeded: {task.result}')
            return JsonResponse({"status": "success", "result": task.result}, status=200)
    else:
        logger.info(f'Task {task_id} is pending.')
        return JsonResponse({"status": "pending"}, status=202)