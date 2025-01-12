from django.http import JsonResponse
import json
from utils import handle_user_input
from django.views.decorators.csrf import csrf_exempt

@csrf_exempt
def ai_agent_interaction(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            user_input = data.get('input', '')

            response_text = handle_user_input(user_input)

            return JsonResponse({"response": response_text})
        except Exception as e:
            return JsonResponse({"error": str(e)}, status=500)
    return JsonResponse({"error": "Only POST requests are accepted"}, status=400)