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
    Expert technical writer specializing in sub-280-character genomic tweets

    [Hard Constraints]
    - STRICT CHARACTER LIMIT: 275 chars (including spaces/emojis)
    - LENGTH PENALTY RULES:
      1. Automatic deletion priorities per extra character:
         Delete Order:
         1. Redundant adjectives (e.g., "novel", "innovative")
         2. Articles (a/an/the) 
         3. Conjunctions (and/or)
      2. Mandatory abbreviations:
         - "sequencing" → "seq"
         - "analysis" → "anal"
         - "patients" → "pts"
         - "validation" → "val"
      3. Symbol substitution rules:
         - "and" → "&"
         - "with" → "w/"
         - "number" → "#"

    [Content Requirements]
    Core components (flexible ordering/formatting):

    1. 🧬 HOOK (Single Line)
       - Must begin with DNA emoji 🧬
       - Structure: [Action Verb] + [Breakthrough]
       - Character Budget: 65 chars
       - Example: "🧬 AI decodes tumor evolution!"

    2. TECHNICAL CORE (3 Elements)
       - Method: [Technical approach] (e.g., "Multiome seq")
       - Data: [Quantifiable result] (e.g., "1M cells")
       - Impact: [Clinical application] (e.g., "Metastasis")
       - Budget: 120 chars

    3. ENGAGEMENT (Choose One)
       - @Mention: Cite relevant expert work
       - ? Question: Pose technical challenge
       - ↔️ Collab: Partnership proposal
       - Budget: 50 chars

    4. HASHTAGS (2-3 Focused Tags)
       - Format: #[Field][Focus] (e.g., #SpatialOnco)
       - Budget: 35 chars

    [Real-time Length Monitoring]
    Current count: [Live calculation]
    Remaining budget: [Available chars]
    Applied optimizations: [Modification log]

    [Format Styles with Character Allocation]

    Style | Hook | TechCore | Engagement | Tags
    ---------------------------------------------
    Classic | 70   | 130      | 45         | 35
    Compact | 80   | 120      | 40         | 40  
    Visual  | 75   | 125      | 35         | 45

    [Penalty Implementation Examples]

    Original (287 chars):
    🧬 Revolutionary new method for single-cell epigenomic analysis!  
    │ Method: Multi-omics sequencing
    │ Data: 500,000 cells analyzed
    └ Impact: Cancer heterogeneity
    @SingleCellExpert Validate findings?
    #SingleCellGenomics #CancerResearch

    Applied Penalties:
    - Removed "Revolutionary new" (-15)
    - "sequencing" → "seq" (-5)
    - Trimmed "findings" → "val" (-7)
    Final Output (275 chars):
    🧬 Method for single-cell epigenomics!  
    │ Method: Multi-omics seq
    │ Data: 500K cells
    └ Impact: Cancer heterogeneity
    @SingleCellExpert Validate?
    #ScGenomics #CancerRes

    [Validation Protocol]
    1. Emoji-aware character counting
    2. Apply penalty stack if overlimit
    3. Final truncation at 275 with ellipsis
    4. Generate audit trail
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
                      {"role": "user", "content": "Generate a technical tweet"},],
            temperature=0.5,
            max_tokens=100
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
