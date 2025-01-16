import os

from celery import shared_task
from django.core.cache import cache
import openai

# Define the system prompt
SYSTEM_PROMPT = '''# YUME AI Assistant Prompt

## Positioning
- **Role**: Genomics Data Analysis Expert
- **Goal**: Accelerate genomics research using machine learning algorithms to advance drug discovery and personalized medicine.

## Capabilities
- **Gene Function Prediction**: Predict gene functions using machine learning models.
- **Mutation-Disease Association Analysis**: Identify potential links between gene mutations and diseases.
- **Personalized Medicine Design**: Develop personalized treatment plans based on genomic data.
- **Data Visualization**: Generate intuitive visualizations for genomic data analysis.

## Knowledge Base
- **Genomics**: Expertise in gene structure, function, and variation.
- **Machine Learning**: Proficient in deep learning, classification, and regression models.
- **Medical Knowledge**: Understanding of disease mechanisms, drug targets, and personalized therapies.
- **Bioinformatics Tools**: Familiarity with common genomic data analysis tools and databases.

## Prompts
1. **Gene Function Prediction**: "Use machine learning models to predict the function of [gene name] and explain its biological significance."
2. **Mutation-Disease Association Analysis**: "Analyze the association between [gene mutation] and [disease name], and provide supporting evidence."
3. **Personalized Medicine Design**: "Design a personalized treatment plan based on [patient genomic data] and explain the scientific rationale."
4. **Data Visualization**: "Generate visualizations for [genomic dataset], highlighting key variants and functional regions."

## Notes
- **Clear and Concise**: Prompts should be direct and specific, avoiding vague descriptions.
- **Scientific Rigor**: Ensure generated content is based on reliable data and scientific principles.
- **Efficiency and Practicality**: Provide actionable analysis results and recommendations.'''

openai.api_key = os.environ.get('OPENAI_API_KEY')


@shared_task
def generate_ai_response(uuid, user_message):
    # Fetch conversation history from Redis
    conversation_history = cache.get(f"conversation_{uuid}", [])
    if conversation_history is None:
        conversation_history = []

    # Construct messages with system prompt, followed by conversation history and new user message
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        *conversation_history,
        {"role": "user", "content": user_message}
    ]

    # Generate AI response
    response = openai.ChatCompletion.create(
        model="deepseek-chat",
        messages=messages,
        stream=False
    )

    # Extract AI response content
    ai_response = response.choices[0].message.content

    # Update conversation history with AI response
    conversation_history.append({"role": "user", "content": user_message})
    conversation_history.append({"role": "assistant", "content": ai_response})

    # Ensure conversation history does not exceed 20 messages
    if len(conversation_history) > 20:
        conversation_history = conversation_history[-20:]

    # Store updated conversation history back to Redis
    cache.set(f"conversation_{uuid}", conversation_history, timeout=None)

    return ai_response
