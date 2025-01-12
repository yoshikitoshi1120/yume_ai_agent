import spacy
import openai
import os
import requests
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

nlp = spacy.load("en_core_web_sm")

openai.api_key = os.environ.get('OPENAI_API_KEY')


def query_sol_price():
    url = "https://api.coingecko.com/api/v3/coins/solana/market_chart?vs_currency=usd&days=7"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    else:
        raise Exception("Failed to fetch sol price. Status code: " + str(response.status_code))


def analyze_user_intent(user_input):
    doc = nlp(user_input)
    intent = ""
    verbs = []
    nouns = []
    for token in doc:
        if token.pos_ == "VERB":
            verbs.append(token.text)
        elif token.pos_ == "NOUN":
            nouns.append(token.text)
    if "continue" in verbs:
        intent = "text_continuation"
    elif "check" in verbs and "grammar" in nouns:
        intent = "grammar_check"
    elif "analyze" in verbs and "sentiment" in nouns:
        intent = "sentiment_analysis"
    elif "generate" in verbs and "summary" in nouns:
        intent = "text_summary"
    elif "ask" in verbs and "question" in nouns:
        intent = "knowledge_question"
    elif "write" in verbs or "create" in verbs:
        intent = "text_creation"
    else:
        intent = identify_investment_intent(user_input)
    return intent


def identify_investment_intent(text):
    lemmatizer = WordNetLemmatizer()
    tokens = word_tokenize(text)
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
    keywords_price_trend = ["price", "trend", "movement", "history", "fluctuation"]
    keywords_investment_advice = ["investment", "advice", "buy", "sell", "hold", "invest", "strategy"]
    keywords_sol_coin = ["sol", "solana", "coin"]

    has_price_trend_keyword = any(lemmatized_token in keywords_price_trend for lemmatized_token in lemmatized_tokens)
    has_investment_advice_keyword = any(
        lemmatized_token in keywords_investment_advice for lemmatized_token in lemmatized_tokens)
    has_sol_coin_keyword = any(lemmatized_token in keywords_sol_coin for lemmatized_token in lemmatized_tokens)

    if has_price_trend_keyword and has_investment_advice_keyword and has_sol_coin_keyword:
        return "query_and_advice_sol"
    elif has_price_trend_keyword and has_sol_coin_keyword:
        return "query_sol"
    elif has_investment_advice_keyword and has_sol_coin_keyword:
        return "advice_sol"
    else:
        return "no_related_intent"


def handle_user_input(user_input):
    intent = analyze_user_intent(user_input)
    if intent == "text_continuation":
        return text_continuation(user_input)
    elif intent == "grammar_check":
        return grammar_check(user_input)
    elif intent == "sentiment_analysis":
        return sentiment_analysis(user_input)
    elif intent == "text_summary":
        return text_summary(user_input)
    elif intent == "knowledge_question":
        return knowledge_question_answer(user_input)
    elif intent == "text_creation":
        return text_creation(user_input)
    elif intent == "query_and_advice_sol":
        return query_and_advice_sol(user_input)
    elif intent == "query_sol":
        return query_sol(user_input)
    elif intent == "advice_sol":
        return advice_sol(user_input)
    else:
        return yume_medical_advice(user_input)


def yume_medical_advice(input_text):
    try:
        prompt = f"As a machine learning algorithm expert, design a personalized medical solution for the user based on their input, {input_text},leveraging YUME's expertise in genomic data analysis, which utilizes machine learning algorithms to predict gene function, identify gene mutation-disease associations, and aid in drug discovery and genetic disease research. Ensure the solution is comprehensive, taking into account relevant genomic data, potential genetic factors, and individual health conditions. Provide a step-by-step plan with explanations of how the proposed solution addresses the user's specific needs, and present the response in the same language as the user's input." + input_text
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=prompt,
            max_tokens=2000,
            temperature=0.2
        )
        return response.choices[0].text.strip()
    except openai.error.OpenAIError as e:
        return f"call ai LLM error:{str(e)}"


def advice_sol(input_text):
    try:
        prompt = f"Please provide reasonable investment advice for SOL coin based on its market situation, taking into account the user's input: {input_text}. Consider various factors that may affect the value of SOL coin, such as market trends, competitor analysis, technological developments, and regulatory factors. Explain your advice clearly and provide supporting evidence. The advice should be presented in the same language as the input text."
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=prompt,
            max_tokens=2000,
            temperature=0.2
        )
        return response.choices[0].text.strip()
    except openai.error.OpenAIError as e:
        return f"call ai LLM error:{str(e)}"


def query_sol(input_text):
    try:
        prompt = f"Please query the price trend of SOL coin based on the user's input: {input_text}. Present the price trend of SOL coin in a clear and understandable manner, which may include using graphs, tables, or detailed textual descriptions. Ensure that the information is accurate and up-to-date, and provide the data in the same language as the input text."
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=prompt,
            max_tokens=200,
            temperature=0.2
        )
        return response.choices[0].text.strip()
    except openai.error.OpenAIError as e:
        return f"call ai LLM error:{str(e)}"


def query_and_advice_sol(input_text):
    try:
        prompt = f"Please query the price trend of SOL coin based on the user's input: {input_text}, and then provide reasonable investment advice. You should utilize reliable data sources and financial analysis methods to ensure the accuracy and rationality of the advice. Consider factors such as the historical price performance of SOL coin, market trends, and any relevant economic indicators. Additionally, explain the rationale behind your investment advice clearly. Provide the information in the same language as the input text"
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=prompt,
            max_tokens=200,
            temperature=0.2
        )
        return response.choices[0].text.strip()
    except openai.error.OpenAIError as e:
        return f"call ai LLM error:{str(e)}"


def text_continuation(input_text):
    try:
        prompt = f"Please continue writing a reasonable piece of content based on the following text, while maintaining a consistent style. The original text is: {input_text}. Ensure that the continuation flows smoothly and makes sense in the context of the original. Return the continuation in the same language as the input text."
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=prompt,
            max_tokens=200,
            temperature=0.7
        )
        return response.choices[0].text.strip()
    except openai.error.OpenAIError as e:
        return f"call ai LLM error:{str(e)}"


def grammar_check(input_text):
    try:
        prompt = f"Please check the following text for grammatical errors. If there are any errors, identify them and provide correct suggestions for modification. The text is as follows: {input_text}. Please provide your analysis and suggestions in the same language as the input text."
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=prompt,
            max_tokens=150,
            temperature=0.2
        )
        return response.choices[0].text.strip()
    except openai.error.OpenAIError as e:
        return f"call ai LLM error:{str(e)}"


def sentiment_analysis(input_text):
    try:
        prompt = f"Please analyze the emotional tendency of the following text: {input_text}. Determine whether the sentiment of the text is positive, negative, or neutral, and provide reasons for your judgment. Consider the overall tone, word choice, and context of the text. Return your analysis in the same language as the input text."
        response = openai.Completion.create(
            engine="text-davinci-03",
            prompt=prompt,
            max_tokens=50,
            temperature=0.3
        )
        return response.choices[0].text.strip()
    except openai.error.OpenAIError as e:
        return f"call ai LLM error:{str(e)}"


def knowledge_question_answer(input_text):
    try:
        prompt = f"Please answer the following question in a clear, accurate, and organized manner. Provide reliable evidence for your answer. If professional knowledge is involved, please refer to authoritative sources. {input_text}. Ensure that your response is in the same language as the input."
        response = openai.Completion.create(
            engine="text-davinci-03",
            prompt=prompt,
            max_tokens=1500,
            temperature=0.7
        )
        return response.choices[0].text.strip()
    except openai.error.OpenAIError as e:
        return f"call ai LLM error:{str(e)}"


def text_creation(input_text):
    try:
        prompt = f"Write an engaging and easy-to-understand article about {input_text}. The article should have an introduction to draw the reader's attention, a main body with detailed information, and a conclusion to summarize the key points. The length of the article should be between 500 and 800 words. Make sure the language of the article matches the language of the input."
        response = openai.Completion.create(
            engine="text-davinci-03",
            prompt=prompt,
            max_tokens=3000,
            temperature=0.8
        )
        return response.choices[0].text.strip()
    except openai.error.OpenAIError as e:
        return f"call ai LLM error:{str(e)}"


def text_summary(input_text):
    try:
        prompt = f"Please generate a summary of the following text: {input_text}. Ensure the summary captures the main points, key information, and essence of the original text, while being concise and coherent. Return the summary in the same language as the input text."
        response = openai.Completion.create(
            engine="text-davinci-03",
            prompt=prompt,
            max_tokens=1000,
            temperature=0.5
        )
        return response.choices[0].text.strip()
    except openai.error.OpenAIError as e:
        return f"call ai LLM error:{str(e)}"


import chardet


def handle_encoding(user_input):
    detected_encoding = chardet.detect(user_input)['encoding']
    if detected_encoding:
        user_input = user_input.decode(detected_encoding).encode('utf-8')
    return user_input
