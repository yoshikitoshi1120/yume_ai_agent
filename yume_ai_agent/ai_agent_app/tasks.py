from celery import shared_task
from django.core.cache import cache
import openai

openai.api_key = 'YOUR_OPENAI_API_KEY'

@shared_task
def generate_ai_response(uuid, user_message):
    conversation_history = cache.get(f"conversation_{uuid}", [])
    if len(conversation_history) > 20:
        conversation_history = conversation_history[-20:]
    messages = conversation_history.copy()
    messages.append({"role": "user", "content": user_message})
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages,
        stream=False
    )
    ai_response = response.choices[0].message.content
    conversation_history.append({"role": "assistant", "content": ai_response})
    if len(conversation_history) > 20:
        conversation_history = conversation_history[-20:]
    cache.set(f"conversation_{uuid}", conversation_history, timeout=None)
    return ai_response