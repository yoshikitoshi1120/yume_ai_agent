import random
from typing import List, Dict

from django.core.cache import cache
from django.conf import settings
from openai import OpenAI
from sentence_transformers import SentenceTransformer
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
            access_token_secret=settings.TWITTER_ACCESS_TOKEN_SECRET,
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

    def __init__(self):
        """Initialize AI agent components"""
        # Initialize LLM client
        self.llm_client = OpenAI(
            api_key=settings.OPENAI_API_KEY,
            base_url="https://api.deepseek.com"
        )

        # Initialize semantic similarity model
        self.semantic_model = SentenceTransformer('all-MiniLM-L6-v2')

        self.twitter_api = _init_twitter_client()

        # # Domain expert database
        # self.experts = [
        #     {
        #         "handle": "@DrGenomics",
        #         "research_focus": "Cancer genomics and CRISPR-based therapies",
        #         "embeddings": None  # Lazy initialization
        #     },
        #     # Additional experts...
        # ]
        # self._init_expert_embeddings()

        # Tweet topic repository
        self.tweet_topics = [
            # Genome Editing Advances
            ("High-precision CRISPR-Cas9 editing systems",
             "Base editing efficiency optimization",
             "Off-target rate <0.1%"),

            # Single-Cell Technologies
            ("Single-cell multi-omics integration",
             "Spatial transcriptomics + epigenetics co-modeling",
             "CellBender denoising algorithm"),

            # Neuroscience Applications
            ("Alzheimer's disease risk gene mapping",
             "GWAS meta-analysis breakthroughs",
             "APOE ε4 allele carrier risk"),

            # Cancer Genomics
            ("AI-driven tumor heterogeneity analysis",
             "Deep learning clonal evolution tracking",
             "PyClone-VI algorithm"),

            # Rare Disease Diagnostics
            ("Whole-exome sequencing for rare disorders",
             "Phenotype-driven variant prioritization",
             "ACMG guidelines v3.0 compliance"),

            # Drug Discovery
            ("AI-accelerated anticancer drug development",
             "Molecular docking & drug-likeness prediction",
             "Diffusion generative models"),

            # Gene Therapy
            ("AAV vector optimization strategies",
             "Tissue-specific promoter engineering",
             "3x hepatic targeting efficiency"),

            # Microbiome Research
            ("Gut microbiome-host gene interactions",
             "Metagenomic-metabolomic integration",
             "MAGs reconstruction rate >90%"),

            # Epigenetics
            ("DNA methylation clock calibration",
             "Deep neural network age prediction",
             "Horvath clock ±1.2yr error"),

            # Population Genomics
            ("1000 Genomes Project discoveries",
             "Population-specific variant spectra",
             "gnomAD v4 dataset"),

            # Immunogenomics
            ("T-cell receptor repertoire deep decoding",
             "VDJ recombination pattern recognition",
             "TRUST4 algorithm"),

            # Synthetic Biology
            ("Genome-scale metabolic modeling",
             "Constraint-based phenotype prediction",
             "COBRApy toolkit"),

            # Prenatal Testing
            ("Non-invasive prenatal testing (NIPT) innovations",
             "cfDNA fragmentation pattern analysis",
             "Z-score 99.2% sensitivity"),

            # Aging Research
            ("AI telomere length prediction models",
             "Genome-wide SNP association analysis",
             "TelSeq R²=0.81"),

            # Precision Nutrition
            ("Genotype-guided dietary interventions",
             "Polygenic risk score optimization",
             "NutrigenomicsDB integration")
        ]

    # def _init_expert_embeddings(self):
    #     """Precompute embedding vectors for expert research focuses"""
    #     for expert in self.experts:
    #         expert["embeddings"] = self.semantic_model.encode(
    #             expert["research_focus"],
    #             convert_to_tensor=True
    #         )

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
        # Select random topic
        topic, aspect, keyword = random.choice(self.tweet_topics)
        prompt = f'''
    [Role]
You are YUME, an AI agent specializing in genomic data analysis. Your goal is to create engaging technical tweets that:
1. Highlight recent research breakthroughs
2. Appeal to both experts and informed laypeople
3. Drive engagement (retweets, clicks, discussions)

[Content Requirements]
Format each tweet with:
1. Hook: Start with eye-catching elements in this order:
   - 🧬 Emoji relating to genetics
   - Strong adjective (e.g., "Groundbreaking", "Surprising")
   - Core finding/technology
2. Body: Include exactly:
   - 1 technical method (e.g., "single-cell RNA sequencing")
   - 1 data point (e.g., "73% accuracy in cross-validation")
   - 1 disease/medical application
3. Call-to-action: End with either:
   - Thought-provoking question
   - Collaboration opportunity
   - Resource sharing
4. Hashtags: Use 3-4 from this rotating list:
   - Technical: #CRISPR #GWAS #DeepGenomics
   - Medical: #PrecisionMedicine #CancerResearch
   - Ethics: #AIinHealthcare #GenomicEthics
   - General: #ScienceNews #MedTech

[Style Guide]
- Length: 240-260 characters (mobile-friendly)
- Tone: Professional enthusiasm (avoid hype)
- Technical level: Assume PhD-level audience but avoid jargon
- Use: 
   • Bullet points/symbols → ✓
   • Line breaks → 🧬--- 
   • Strategic emojis → 🧪💡

[Example Output]
🧬 Novel framework achieves 89% accuracy predicting drug-gene interactions!
✓ Combines GNNs & multi-omics data
✓ Validated on 15 cancer types
Could this accelerate personalized chemo regimens? 
#DeepGenomics #CancerResearch #AIforHealth

[Current Task]
Generate a tweet about {topic} focusing on {aspect}. Use active voice and include:{keyword}
'''
        # Generate content
        response = self.llm_client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.8,
            max_tokens=280
        )

        return response.choices[0].message.content.strip()

    def _optimize_mentions(self, tweet: str) -> str:
        """
        Enhance tweet with relevant expert mentions

        Args:
            tweet: Original tweet content

        Returns:
            Tweet with optimized @mentions
        """
        # if random.random() < 0.25:  # 25% mention probability
        #     # Calculate semantic similarity
        #     tweet_embedding = self.semantic_model.encode(tweet, convert_to_tensor=True)
        #     similarities = [
        #         (expert["handle"], util.pytorch_cos_sim(tweet_embedding, expert["embeddings"]).item())
        #         for expert in self.experts
        #     ]
        #
        #     # Filter and select mentions
        #     relevant_mentions = [item for item in similarities if item[1] > 0.4]
        #     sorted_mentions = sorted(relevant_mentions, key=lambda x: x[1], reverse=True)[:2]
        #
        #     if sorted_mentions:
        #         return f"{tweet} {' '.join(m[0] for m in sorted_mentions)}"

        return tweet

    def publish_tweet(self) -> dict:
        """
        Execute complete tweet publication workflow

        Returns:
            Dictionary containing:
            - original: Raw tweet content
            - optimized: Final tweet with mentions
            - tweet_id: Published tweet ID
            - success: Boolean status
            - error: Error message (if any)
        """
        result = {
            "original": None,
            "optimized": None,
            "tweet_id": None,
            "success": False,
            "error": None
        }

        try:
            # Generate base content
            raw_tweet = self._generate_tweet()
            result["original"] = raw_tweet

            # Add intelligent mentions
            optimized_tweet = self._optimize_mentions(raw_tweet)
            result["optimized"] = optimized_tweet

            # Validate content
            _validate_tweet(optimized_tweet)

            # Publish tweet
            response = self.twitter_api.update_status(optimized_tweet)
            result["tweet_id"] = response.id_str
            result["success"] = True

        except tweepy.TweepyException as e:
            result["error"] = f"Twitter API error: {str(e)}"
        except ValueError as e:
            result["error"] = f"Validation error: {str(e)}"
        except Exception as e:
            result["error"] = f"Unexpected error: {str(e)}"

        return result
