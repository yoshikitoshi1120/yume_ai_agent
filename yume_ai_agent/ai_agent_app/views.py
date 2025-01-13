from django.http import JsonResponse
from django.views.decorators.http import require_http_methods
from celery.result import AsyncResult
from .tasks import generate_ai_response

@require_http_methods(["POST"])
def start_ai_response(request):
    uuid = request.POST.get('uuid', None)
    if not uuid:
        return JsonResponse({"error": "UUID is missing."}, status=400)
    user_message = request.POST.get('message', '')
    task = generate_ai_response.delay(uuid, user_message)
    return JsonResponse({"task_id": task.id})

@require_http_methods(["GET"])
def get_task_result(request, task_id):
    task = AsyncResult(task_id)
    if task.ready():
        if task.failed():
            return JsonResponse({"status": "failed", "error": str(task.result)})
        else:
            result = task.get()
            return JsonResponse({"status": "success", "result": result})
    else:
        return JsonResponse({"status": "pending"})