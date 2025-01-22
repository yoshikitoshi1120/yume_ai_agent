from typing import List, Dict

from django.core.cache import cache
from django.conf import settings
from openai import OpenAI
import tweepy


def _validate_tweet(tweet: str) -> bool:
    """Validate tweet content before publishing"""
    if len(tweet) > 280:
        raise ValueError("Tweet exceeds 280 character limit")

    return True


def _init_twitter_client() -> tweepy.API:
    """Initialize and verify Twitter API credentials"""
    try:
        auth = tweepy.OAuth1UserHandler(
            consumer_key=settings.TWITTER_CONSUMER_KEY,
            consumer_secret=settings.TWITTER_CONSUMER_SECRET,
            access_token=settings.TWITTER_ACCESS_TOKEN,
            access_token_secret=settings.TWITTER_ACCESS_SECRET,
        )
        api = tweepy.API(auth)
        api.verify_credentials()  # Test credentials
        return api
    except tweepy.TweepyException as e:
        raise RuntimeError(f"Twitter authentication failed: {str(e)}")


def _manage_conversation_history(uuid: str, new_messages: List[Dict]) -> List[Dict]:
    """
    Manage conversation history in Redis cache

    Args:
        uuid: Unique user session identifier
        new_messages: New messages to append

    Returns:
        Updated conversation history
    """
    # Retrieve existing history
    history = cache.get(f"conversation_{uuid}", [])

    # Update and trim history
    updated_history = (history + new_messages)[-20:]

    # Persist to cache
    cache.set(f"conversation_{uuid}", updated_history, timeout=None)

    return updated_history


class YUMEAgent:
    """
       AI agent specializing in genomics data analysis and social engagement

       Key Features:
       1. Medical research conversation system
       2. Automated research tweet generation
       3. Intelligent @mention optimization for domain experts
       4. Conversation history management
       """
    # Define the chat system prompt
    CHAT_SYSTEM_PROMPT = '''# YUME AI Assistant Prompt

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

    TWEET_SYSTEM_PROMPT = '''
    [Role]
    Expert technical writer crafting sub-280char genomic tweets

    [Hard Constraints]
    - STRICT CHARACTER LIMIT: 275 chars (including spaces/emojis)
    - REQUIRED ELEMENTS: Hook, Technical Body, Engagement, Hashtags

    [Content Requirements]
    Core components (order and formatting can vary):

    1. 🧬 HOOK (1 line)
       - Must start with DNA emoji 🧬
       - Include: [Breakthrough Verb] + [Key Innovation]
       - Example: "🧬 AI maps tumor evolution pathways!"

    2. TECHNICAL CORE (3 elements)
       - Method: [Technical approach] (e.g., "Multiome sequencing")
       - Data: [Quantifiable result] (e.g., "1M cells analyzed")
       - Impact: [Medical application] (e.g., "Metastasis tracking")

    3. ENGAGEMENT (choose one)
       - @Mention: Reference expert work
       - Question: Open technical challenge
       - Collab: Partnership proposal

    4. HASHTAGS (2-3 focused tags)
       - Combine field + focus (e.g., #SpatialOncology)

    [Formatting Freedom]
    Choose ONE of these display styles:

    Style A: Classic Technical (Recommended)
    🧬 Hook!
    │ Method: Abbreviated approach
    │ Data: Key metric (N=XX)
    └ Impact: Disease application
    @Expert Validate [specific aspect]? 
    #Tag1 #Tag2

    Style B: Compact Flow
    🧬 Hook ➔ Method | Data ➔ Impact
    @Expert Contextual question?
    #CombinedTags

    Style C: Visual Priority
    🧬 HOOK
    ◈ Tech: Method
    ◈ Proof: Data
    ◈ Goal: Impact
    ↗️ Engagement action
    #FocusedTag

    [Optimization Rules]
    - Use vertical bars/arrows/emojis as separators
    - Vary bullet symbols between tweets (│, ➔, ◈, ▸)
    - Hashtag position: End or after engagement
    - Allow 2-line hook when using Style C

    [Examples]

    Example A (Classic):
    🧬 CRISPR edits reach new precision!
    │ Method: Prime editing 2.0
    │ Data: 0.01% off-target (50k sites)
    └ Impact: Sickle cell cure
    @GeneEditReview Replicate in primary cells?
    #CRISPRopt #GeneTherapy (269 chars)

    Example B (Compact):
    🧬 Spatial omics decodes TME ➔ scRNA+protein | 200k cells ➔ IO resistance
    @SingleCellAI Clinical validation plan? 
    #SpatialIO #Multiome (271 chars)

    Example C (Visual):
    🧬 AI predicts protein-drug interactions
    ◈ Tech: Geometric deep learning
    ◈ Proof: 92% accuracy (150k pairs)
    ◈ Goal: Rare disease targets
    ↗️ Collaborate on wet-lab tests
    #AIDrugDiscovery (263 chars)
    '''

    def __init__(self):
        """Initialize AI agent components"""
        # Initialize LLM client
        self.llm_client = OpenAI(
            api_key=settings.OPENAI_API_KEY,
            base_url="https://api.deepseek.com"
        )

        self.twitter_api = _init_twitter_client()

    def generate_ai_response(self, uuid: str, user_message: str) -> str:
        """
        Generate research-focused AI response

        Args:
            uuid: User session ID
            user_message: User input text

        Returns:
            AI-generated response content
        """
        # Build message chain
        messages = [
            {"role": "system", "content": self.CHAT_SYSTEM_PROMPT},
            *cache.get(f"conversation_{uuid}", []),
            {"role": "user", "content": user_message}
        ]

        # Generate response
        response = self.llm_client.chat.completions.create(
            model="deepseek-chat",
            messages=messages,
            temperature=0.7,
            max_tokens=500
        )

        # Extract and store response
        ai_response = response.choices[0].message.content
        _manage_conversation_history(uuid, [
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": ai_response}
        ])

        return ai_response

    def _generate_tweet(self) -> str:
        """
        Generate technical research tweet

        Returns:
            Formatted tweet content adhering to Twitter guidelines
        """
        # Generate content
        response = self.llm_client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "system", "content": self.TWEET_SYSTEM_PROMPT},
                      {"role": "user", "content": "Generate a technical tweet,the length of the tweet should less than 280 characters"},],
            temperature=0.7,
            max_tokens=280
        )

        return response.choices[0].message.content.strip()

    def publish_tweet(self) -> dict:
        """
        Execute complete tweet publication workflow

        Returns:
            Dictionary containing:
            - tweet_content: Published tweet content
            - tweet_id: Published tweet ID
            - success: Boolean status
            - error: Error message (if any)
        """
        result = {
            "tweet_content": None,
            "tweet_id": None,
            "success": False,
            "error": None
        }

        try:
            # Generate base content
            tweet = self._generate_tweet()
            result["tweet_content"] = tweet

            # Validate content
            _validate_tweet(tweet)

            # Publish tweet
            response = self.twitter_api.update_status(tweet)
            result["tweet_id"] = response.id_str
            result["success"] = True

        except tweepy.TweepyException as e:
            result["error"] = f"Twitter API error: {str(e)}"
        except ValueError as e:
            result["error"] = f"Validation error: {str(e)}"
        except Exception as e:
            result["error"] = f"Unexpected error: {str(e)}"

        return result
