#!/usr/bin/env python3
"""
Twitter Automation Bot with Flask Server and Telegram Integration
Single file containing Flask server, Telegram bot, Twitter integration, and AI features
"""

import os
import json
import asyncio
import threading
import time
from datetime import datetime, timedelta
from typing import Optional, Dict, Any

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

# Flask imports
from flask import Flask, render_template_string, request, redirect, url_for, session, jsonify

# Telegram bot imports
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes, CallbackQueryHandler

# Twitter API imports
import tweepy

# AI imports
from openai import OpenAI
import google.generativeai as genai
from groq import Groq

# Scheduler imports
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.interval import IntervalTrigger

# Utility imports
import requests
import logging
import re

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# SECURITY: Suppress HTTP request logging to prevent token exposure
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("telegram").setLevel(logging.WARNING)
logging.getLogger("telegram.ext").setLevel(logging.WARNING)
logging.getLogger("telegram.request").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

# Global variables
app = Flask(__name__)
app.secret_key = os.environ.get('SESSION_SECRET')
if not app.secret_key:
    raise ValueError("SESSION_SECRET environment variable is required")
scheduler = BackgroundScheduler()
telegram_app = None
admin_chat_id = None
telegram_loop = None

# Configuration from environment variables
TELEGRAM_BOT_TOKEN = os.environ.get('TELEGRAM_BOT_TOKEN')
ADMIN_PASSWORD = os.environ.get('ADMIN_PASSWORD')
if not ADMIN_PASSWORD:
    raise ValueError("ADMIN_PASSWORD environment variable is required")
POSTING_INTERVAL_HOURS = int(os.environ.get('POSTING_INTERVAL_HOURS', '2'))
DEVELOPER_TG = os.environ.get('DEVELOPER_TG', '@Alcboss112')

# Twitter API configuration
TWITTER_BEARER_TOKEN = os.environ.get('TWITTER_BEARER_TOKEN')
TWITTER_API_KEY = os.environ.get('TWITTER_API_KEY')
TWITTER_API_SECRET = os.environ.get('TWITTER_API_SECRET')
TWITTER_ACCESS_TOKEN = os.environ.get('TWITTER_ACCESS_TOKEN')
TWITTER_ACCESS_SECRET = os.environ.get('TWITTER_ACCESS_SECRET')

# AI configuration
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY')
GROQ_API_KEY = os.environ.get('GROQ_API_KEY')
DEEPSEEK_API_KEY = os.environ.get('DEEPSEEK_API_KEY')

# Initialize AI clients
openai_client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None
groq_client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None
deepseek_client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com") if DEEPSEEK_API_KEY else None

# Initialize Gemini
gemini_model = None
if GEMINI_API_KEY:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        gemini_model = genai.GenerativeModel('gemini-1.5-flash')
    except Exception as e:
        logger.warning(f"Failed to initialize Gemini: {e}")
        gemini_model = None

# Initialize Twitter API
twitter_api = None
if all([TWITTER_API_KEY, TWITTER_API_SECRET, TWITTER_ACCESS_TOKEN, TWITTER_ACCESS_SECRET]):
    twitter_api = tweepy.Client(
        bearer_token=TWITTER_BEARER_TOKEN,
        consumer_key=TWITTER_API_KEY,
        consumer_secret=TWITTER_API_SECRET,
        access_token=TWITTER_ACCESS_TOKEN,
        access_token_secret=TWITTER_ACCESS_SECRET,
        wait_on_rate_limit=False  # Don't sleep on rate limits - handle manually
    )

# Data storage (in-memory for simplicity)
app_data = {
    'base_script': '',
    'last_post_time': None,
    'next_post_time': None,
    'post_history': [],
    'posting_enabled': True,
    'current_countdown': 0,
    'initial_post_done': False,
    'last_heartbeat': None,
    'server_start_time': datetime.now().isoformat(),
    'awaiting_input': None,  # 'base' or 'instant' when waiting for user input
    'temp_message_id': None  # Store message ID for editing
}

# ============================================================================
# AI CONTENT GENERATION FUNCTIONS
# ============================================================================

def generate_content_with_openai(base_script: str, research_data: str = "") -> Dict[str, Any]:
    """Generate Twitter content using OpenAI"""
    if not openai_client:
        return {"success": False, "content": "", "error": "OpenAI client not available"}

    try:
        # Get character limit from environment
        char_limit = int(os.environ.get('CHARACTER_LIMIT', 280))
        
        prompt = f"""
        Create a Twitter post that is EXACTLY {char_limit} characters including everything.

        REQUIRED CONTENT TO INCLUDE LITERALLY:
        Base script: "{base_script}"
        Signature: "{DEVELOPER_TG}"
        
        RESEARCH TO ADD: {research_data}

        CRITICAL REQUIREMENTS:
        - Post must be EXACTLY {char_limit} characters total (count every character)
        - MUST include the COMPLETE base script word-for-word: "{base_script}"
        - MUST include signature: "{DEVELOPER_TG}"
        - Add relevant content based on research to reach exactly {char_limit} characters
        - Use trending hashtags and mentions from research if available
        - Make added content natural and engaging
        - Count characters carefully to hit exactly {char_limit}
        
        Format: [Additional engaging content] + [Complete base script] + [Signature]
        VERIFY: Total character count = {char_limit}
        """

        # Try multiple models in order of preference
        models_to_try = ["gpt-4o-mini", "gpt-3.5-turbo"]

        for model in models_to_try:
            try:
                response = openai_client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=150,
                    temperature=0.8
                )

                content = response.choices[0].message.content
                if content:
                    return {"success": True, "content": optimize_character_count(content.strip()), "error": None}

            except Exception as model_error:
                logger.warning(f"Model {model} failed: {model_error}")
                    # If rate limited or quota exceeded, don't try more OpenAI models
                if any(x in str(model_error).lower() for x in ["quota", "429", "rate limit", "too many requests", "rate_limit_exceeded", "insufficient_quota"]):
                    logger.warning(f"OpenAI rate limited/quota exceeded: {model_error}")
                    break
                continue

        return {"success": False, "content": "", "error": "All OpenAI models failed"}

    except Exception as e:
        logger.error(f"OpenAI generation error: {e}")
        return {"success": False, "content": "", "error": str(e)[:50]}

def generate_content_with_groq(base_script: str, research_data: str = "") -> Dict[str, Any]:
    """Generate Twitter content using Groq"""
    if not groq_client:
        return {"success": False, "content": "", "error": "Groq client not available"}

    try:
        # Get character limit from environment
        char_limit = int(os.environ.get('CHARACTER_LIMIT', 280))
        
        prompt = f"""
        Create a Twitter post that is EXACTLY {char_limit} characters including everything.

        REQUIRED CONTENT TO INCLUDE LITERALLY:
        Base script: "{base_script}"
        Signature: "{DEVELOPER_TG}"
        
        RESEARCH TO ADD: {research_data}

        CRITICAL REQUIREMENTS:
        - Post must be EXACTLY {char_limit} characters total (count every character)
        - MUST include the COMPLETE base script word-for-word: "{base_script}"
        - MUST include signature: "{DEVELOPER_TG}"
        - Add relevant content based on research to reach exactly {char_limit} characters
        - Use trending hashtags and mentions from research if available
        - Make added content natural and engaging
        - Count characters carefully to hit exactly {char_limit}
        
        Format: [Additional engaging content] + [Complete base script] + [Signature]
        VERIFY: Total character count = {char_limit}
        """

        models_to_try = ["llama-3.3-70b-versatile", "llama-3.1-70b-versatile", "mixtral-8x7b-32768"]

        for model in models_to_try:
            try:
                response = groq_client.chat.completions.create(
                    messages=[{"role": "user", "content": prompt}],
                    model=model,
                    max_tokens=150,
                    temperature=0.8
                )

                content = response.choices[0].message.content
                if content:
                    return {"success": True, "content": optimize_character_count(content.strip()), "error": None}

            except Exception as model_error:
                logger.warning(f"Groq model {model} failed: {model_error}")
                # Check for rate limits or quota issues
                if any(x in str(model_error).lower() for x in ["rate limit", "429", "too many requests", "quota", "rate_limit_exceeded"]):
                    logger.warning("Groq rate limited, skipping remaining models")
                    break
                continue

        return {"success": False, "content": "", "error": "All Groq models failed"}

    except Exception as e:
        logger.error(f"Groq generation error: {e}")
        return {"success": False, "content": "", "error": str(e)[:50]}

def generate_content_with_deepseek(base_script: str, research_data: str = "") -> Dict[str, Any]:
    """Generate Twitter content using DeepSeek"""
    if not deepseek_client:
        return {"success": False, "content": "", "error": "DeepSeek client not available"}

    try:
        # Get character limit from environment
        char_limit = int(os.environ.get('CHARACTER_LIMIT', 280))
        
        prompt = f"""
        Create a Twitter post that is EXACTLY {char_limit} characters including everything.

        REQUIRED CONTENT TO INCLUDE LITERALLY:
        Base script: "{base_script}"
        Signature: "{DEVELOPER_TG}"
        
        RESEARCH TO ADD: {research_data}

        CRITICAL REQUIREMENTS:
        - Post must be EXACTLY {char_limit} characters total (count every character)
        - MUST include the COMPLETE base script word-for-word: "{base_script}"
        - MUST include signature: "{DEVELOPER_TG}"
        - Add relevant content based on research to reach exactly {char_limit} characters
        - Use trending hashtags and mentions from research if available
        - Make added content natural and engaging
        - Count characters carefully to hit exactly {char_limit}
        
        Format: [Additional engaging content] + [Complete base script] + [Signature]
        VERIFY: Total character count = {char_limit}
        """

        response = deepseek_client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=150,
            temperature=0.8
        )

        content = response.choices[0].message.content
        if content:
            return {"success": True, "content": optimize_character_count(content.strip()), "error": None}

        return {"success": False, "content": "", "error": "No content generated"}

    except Exception as e:
        error_msg = str(e).lower()
        if any(x in error_msg for x in ["rate limit", "429", "too many requests", "quota", "rate_limit_exceeded", "insufficient_quota"]):
            logger.warning(f"DeepSeek rate limited: {e}")
            return {"success": False, "content": "", "error": "Rate limited"}
        logger.error(f"DeepSeek generation error: {e}")
        return {"success": False, "content": "", "error": str(e)[:50]}

def generate_content_with_gemini(base_script: str, research_data: str = "") -> Dict[str, Any]:
    """Generate Twitter content using Gemini AI"""
    if not gemini_model:
        return {"success": False, "content": "", "error": "Gemini client not available"}

    try:
        # Get character limit from environment
        char_limit = int(os.environ.get('CHARACTER_LIMIT', 280))
        
        prompt = f"""
        Create a Twitter post that is EXACTLY {char_limit} characters including everything.

        REQUIRED CONTENT TO INCLUDE LITERALLY:
        Base script: "{base_script}"
        Signature: "{DEVELOPER_TG}"
        
        RESEARCH TO ADD: {research_data}

        CRITICAL REQUIREMENTS:
        - Post must be EXACTLY {char_limit} characters total (count every character)
        - MUST include the COMPLETE base script word-for-word: "{base_script}"
        - MUST include signature: "{DEVELOPER_TG}"
        - Add relevant content based on research to reach exactly {char_limit} characters
        - Use trending hashtags and mentions from research if available
        - Make added content natural and engaging
        - Count characters carefully to hit exactly {char_limit}
        
        Format: [Additional engaging content] + [Complete base script] + [Signature]
        VERIFY: Total character count = {char_limit}
        """

        # Try Gemini generation
        try:
            response = gemini_model.generate_content(prompt)

            if response.text:
                return {"success": True, "content": optimize_character_count(response.text.strip()), "error": None}

        except Exception as model_error:
            error_msg = str(model_error).lower()
            if any(x in error_msg for x in ["rate limit", "429", "quota", "too many requests", "rate_limit_exceeded", "resource_exhausted"]):
                logger.warning(f"Gemini rate limited: {model_error}")
                return {"success": False, "content": "", "error": "Rate limited"}

            logger.warning(f"Gemini generation failed: {model_error}")
            # Try fallback model if available
            try:
                fallback_model = genai.GenerativeModel('gemini-1.0-pro')
                response = fallback_model.generate_content(prompt)
                if response.text:
                    return {"success": True, "content": optimize_character_count(response.text.strip()), "error": None}
            except Exception as fallback_error:
                logger.warning(f"Gemini fallback model also failed: {fallback_error}")
                pass

        return {"success": False, "content": "", "error": "Gemini generation failed"}

    except Exception as e:
        logger.error(f"Gemini generation error: {e}")
        return {"success": False, "content": "", "error": str(e)[:50]}

def research_trending_topics(base_script: str) -> str:
    """Research trending topics, hashtags, and @ mentions related to the base script"""
    research_results = []
    
    try:
        # Step 1: Extract keywords from base script for search
        keywords = extract_keywords_from_base_script(base_script)
        logger.info(f"Extracted keywords for research: {keywords[:3]}...")
        
        # Step 2: Search for trending hashtags on Twitter/Google
        trending_hashtags = search_trending_hashtags(keywords)
        if trending_hashtags:
            research_results.append(f"Trending hashtags: {trending_hashtags}")
            logger.info(f"Found trending hashtags: {trending_hashtags}")
        
        # Step 3: Search for relevant @ mentions and influencers
        relevant_mentions = search_relevant_mentions(keywords)
        if relevant_mentions:
            research_results.append(f"Key voices: {relevant_mentions}")
            logger.info(f"Found relevant mentions: {relevant_mentions}")
        
        # Step 4: Analyze why these hashtags/mentions are trending
        trend_analysis = analyze_trend_context(base_script, trending_hashtags, relevant_mentions)
        if trend_analysis:
            research_results.append(trend_analysis)
            logger.info(f"Trend analysis: {trend_analysis[:100]}...")
        
        # Step 5: Search for current events related to the topic
        current_events = search_current_events(keywords)
        if current_events:
            research_results.append(current_events)
            logger.info(f"Current events: {current_events[:100]}...")
        
        # Combine all research results
        if research_results:
            combined_research = " | ".join(research_results)
            return combined_research[:500]  # Limit research length
        
    except Exception as e:
        logger.warning(f"Advanced research error: {e}")
        
        # Fallback to basic AI research if advanced search fails
        try:
            if openai_client:
                research_prompt = f"""
                Based on this topic: "{base_script}"
                
                Generate current, trending information including:
                - Popular hashtags currently being used for this topic
                - Key influencers or accounts discussing this topic
                - Recent developments and current events
                - Why these trends are important right now
                
                Provide 2-3 sentences of relevant context.
                """
                
                response = openai_client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": research_prompt}],
                    max_tokens=150,
                    temperature=0.7,
                    timeout=10
                )
                
                content = response.choices[0].message.content
                return content.strip() if content else ""
        except Exception as ai_error:
            logger.warning(f"AI research fallback also failed: {ai_error}")
    
    return "Current trends and insights incorporated."

def extract_keywords_from_base_script(base_script: str) -> list:
    """Extract relevant keywords from base script for research"""
    # Remove common words and extract meaningful keywords
    import re
    common_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'will', 'would', 'could', 'should', 'may', 'might', 'can', 'about', 'just', 'more', 'also', 'now', 'new', 'this', 'that', 'these', 'those'}
    
    # Clean and split the base script
    words = re.findall(r'\b[a-zA-Z]{3,}\b', base_script.lower())
    keywords = [word for word in words if word not in common_words]
    
    # Return top 5 most relevant keywords
    return keywords[:5]

def search_trending_hashtags(keywords: list) -> str:
    """Search for trending hashtags related to keywords"""
    try:
        # Try to search Twitter API first for real hashtags
        if twitter_api and keywords:
            trending_hashtags = []
            
            for keyword in keywords[:3]:  # Limit to avoid rate limits
                try:
                    # Search for recent tweets with the keyword to find hashtags
                    search_query = f"{keyword} -is:retweet"
                    tweets = twitter_api.search_recent_tweets(
                        query=search_query,
                        max_results=10,
                        tweet_fields=['public_metrics', 'created_at']
                    )
                    
                    if tweets and hasattr(tweets, 'data') and tweets.data:
                        for tweet in tweets.data:
                            # Extract hashtags from tweet text
                            if hasattr(tweet, 'text'):
                                hashtags = re.findall(r'#\w+', tweet.text)
                                trending_hashtags.extend(hashtags[:2])  # Limit hashtags per tweet
                            
                except Exception as twitter_error:
                    logger.warning(f"Twitter search failed for {keyword}: {twitter_error}")
                    continue
            
            if trending_hashtags:
                # Remove duplicates and return top hashtags
                unique_hashtags = list(dict.fromkeys(trending_hashtags))[:4]
                return " ".join(unique_hashtags)
                
    except Exception as e:
        logger.warning(f"Hashtag search error: {e}")
    
    # Fallback: Generate relevant hashtags based on keywords
    if keywords:
        hashtag_suggestions = []
        for keyword in keywords[:3]:
            hashtag_suggestions.append(f"#{keyword.capitalize()}")
        return " ".join(hashtag_suggestions)
    
    return ""

def search_relevant_mentions(keywords: list) -> str:
    """Search for relevant @ mentions and influencers"""
    try:
        if not keywords:
            return ""
            
        # Try to search Twitter for actual mentions if API available
        if twitter_api and keywords:
            real_mentions = []
            
            for keyword in keywords[:2]:  # Limit to avoid rate limits
                try:
                    # Search for tweets mentioning the keyword to find real accounts
                    search_query = f"{keyword} -is:retweet"
                    tweets = twitter_api.search_recent_tweets(
                        query=search_query,
                        max_results=10,
                        tweet_fields=['author_id', 'public_metrics']
                    )
                    
                    if tweets and hasattr(tweets, 'data') and tweets.data:
                        for tweet in tweets.data:
                            # Extract @ mentions from tweet text
                            if hasattr(tweet, 'text'):
                                mentions = re.findall(r'@\w+', tweet.text)
                                # Filter out common mentions and add relevant ones
                                relevant = [m for m in mentions if len(m) > 3 and 
                                          not m.lower().endswith('bot') and 
                                          not m.lower().endswith('spam')]
                                real_mentions.extend(relevant[:1])  # Limit mentions per tweet
                                
                except Exception as twitter_error:
                    logger.warning(f"Twitter mention search failed for {keyword}: {twitter_error}")
                    continue
            
            if real_mentions:
                # Remove duplicates and return top mentions
                unique_mentions = list(dict.fromkeys(real_mentions))[:2]
                return " ".join(unique_mentions)
        
        # Fallback: Generate more realistic mention patterns based on keywords
        if keywords:
            # More realistic patterns for established projects/topics
            mention_suggestions = []
            for keyword in keywords[:2]:
                keyword_clean = keyword.lower().replace('#', '').replace('$', '')
                if len(keyword_clean) > 2:
                    # Common patterns for crypto/tech projects
                    if any(crypto_term in keyword_clean for crypto_term in ['token', 'coin', 'crypto', 'defi', 'nft']):
                        mention_suggestions.append(f"@{keyword_clean}official")
                    else:
                        mention_suggestions.append(f"@{keyword_clean}")
            
            return " ".join(mention_suggestions[:2])
                
    except Exception as e:
        logger.warning(f"Mention search error: {e}")
    
    return ""

def analyze_trend_context(base_script: str, hashtags: str, mentions: str) -> str:
    """Analyze why hashtags and mentions are trending"""
    try:
        if not any([hashtags, mentions]):
            return ""
            
        # Use AI to analyze trend context
        context_prompt = f"""
        Analyze why these social media elements are currently relevant:
        Topic: {base_script}
        Hashtags: {hashtags}
        Mentions: {mentions}
        
        Explain in 1-2 sentences why these are trending and how they relate to the topic.
        """
        
        if openai_client:
            response = openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": context_prompt}],
                max_tokens=80,
                temperature=0.6,
                timeout=8
            )
            
            content = response.choices[0].message.content
            return content.strip() if content else ""
            
    except Exception as e:
        logger.warning(f"Trend analysis error: {e}")
    
    return ""

def search_current_events(keywords: list) -> str:
    """Search for current events related to keywords"""
    try:
        if not keywords:
            return ""
            
        # Use AI to generate current events context
        events_prompt = f"""
        Based on these keywords: {', '.join(keywords[:3])}
        
        What are the most recent and relevant developments, news, or events happening right now?
        Focus on very recent (last 7 days) information that would make content more timely and engaging.
        Respond in 1-2 sentences.
        """
        
        if openai_client:
            response = openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": events_prompt}],
                max_tokens=80,
                temperature=0.7,
                timeout=8
            )
            
            content = response.choices[0].message.content
            return content.strip() if content else ""
            
    except Exception as e:
        logger.warning(f"Current events search error: {e}")
    
    return ""

def optimize_character_count(content: str) -> str:
    """Optimize content to fit CHARACTER_LIMIT from .env"""
    # Get character limit from environment variable, default to 280
    char_limit = int(os.environ.get('CHARACTER_LIMIT', 280))
    target_min = max(char_limit - 10, char_limit * 0.9)  # Target at least 90% of limit
    
    # Remove extra whitespace
    content = re.sub(r'\s+', ' ', content.strip())

    # If content is too long, try to trim it intelligently
    if len(content) > char_limit:
        # Try to cut at sentence boundaries
        sentences = content.split('.')
        if len(sentences) > 1:
            content = sentences[0] + '.'
            if len(content) <= char_limit:
                return content

        # If still too long, trim to (limit-3) chars and add "..."
        content = content[:char_limit-3] + "..."

    # If content is too short, ensure it's close to the character limit by adding relevant hashtags
    if len(content) < target_min:
        hashtags = [" #tech", " #innovation", " #trending", " #AI", " #update", " #news", " #viral", " #social"]
        for hashtag in hashtags:
            if len(content + hashtag) <= char_limit:
                content += hashtag
            if len(content) >= target_min:
                break

    return content

# ============================================================================
# TWITTER FUNCTIONS
# ============================================================================

def post_to_twitter(content: str) -> Dict[str, Any]:
    """Post content to Twitter and return result with link"""
    if not twitter_api:
        return {'success': False, 'error': 'Twitter API not configured', 'link': None}

    try:
        # Post the tweet
        response = twitter_api.create_tweet(text=content)

        if response and hasattr(response, 'data') and response.data:
            tweet_data = response.data
            tweet_id = str(tweet_data.get('id', 'unknown')) if tweet_data else 'unknown'
            # Get user info to construct proper URL
            try:
                user = twitter_api.get_me()
                username = user.data.username if user and hasattr(user, 'data') and user.data else "unknown"
            except Exception as e:
                logger.warning(f"Could not get username: {e}")
                username = "unknown"

            tweet_url = f"https://twitter.com/{username}/status/{tweet_id}"

            return {
                'success': True,
                'tweet_id': tweet_id,
                'link': tweet_url,
                'content': content
            }
        else:
            return {'success': False, 'error': 'No response data', 'link': None}

    except Exception as e:
        error_str = str(e)
        logger.error(f"Twitter posting error: {error_str}")

        # Check if it's a rate limit error
        if '429' in error_str or 'rate limit' in error_str.lower() or 'too many requests' in error_str.lower():
            logger.warning("🚦 Twitter rate limit hit - will schedule retry in 15 minutes")
            return {'success': False, 'error': 'RATE_LIMITED', 'link': None, 'retry_needed': True}

        return {'success': False, 'error': error_str, 'link': None}

def check_content_uniqueness(new_content: str, base_script: str) -> bool:
    """Check if new content is sufficiently unique compared to recent posts"""
    if not app_data['post_history']:
        return True  # No previous posts to compare against
    
    # Get recent posts with same base script
    recent_posts = [post for post in app_data['post_history'][:10] 
                   if post.get('base_script') == base_script]
    
    if not recent_posts:
        return True  # No posts with same base script
    
    # Simple similarity check - compare key phrases and words
    new_words = set(new_content.lower().split())
    
    for post in recent_posts:
        old_words = set(post['content'].lower().split())
        # Calculate similarity (intersection over union)
        intersection = len(new_words.intersection(old_words))
        union = len(new_words.union(old_words))
        similarity = intersection / union if union > 0 else 0
        
        # If more than 60% similar, consider it too repetitive
        if similarity > 0.6:
            logger.warning(f"Content too similar to recent post (similarity: {similarity:.2f})")
            return False
    
    return True

def generate_unique_content_with_retries(base_script: str, research_data: str, max_retries: int = 3) -> Dict[str, Any]:
    """Generate unique content with retry logic to avoid repetition"""
    
    for attempt in range(max_retries):
        logger.info(f"Content generation attempt {attempt + 1}/{max_retries}")
        
        # Add uniqueness instruction to research data for subsequent attempts
        if attempt > 0:
            uniqueness_note = f" | IMPORTANT: Make this post VERY different from previous attempts. This is attempt {attempt + 1}. Use different angles, tone, or focus. Previous posts for this topic may have been too similar."
            enhanced_research = research_data + uniqueness_note
        else:
            enhanced_research = research_data
        
        # Try multiple AI services in order: OpenAI → Gemini → Groq → DeepSeek
        content = None
        ai_used = "none"
        
        # Try OpenAI first
        if openai_client:
            openai_result = generate_content_with_openai(base_script, enhanced_research)
            if openai_result["success"]:
                content = openai_result["content"]
                ai_used = "openai"
        
        # Try Gemini if OpenAI failed (correct order)
        if not content and gemini_model:
            gemini_result = generate_content_with_gemini(base_script, enhanced_research)
            if gemini_result["success"]:
                content = gemini_result["content"]
                ai_used = "gemini"
        
        # Try Groq if Gemini failed
        if not content and groq_client:
            groq_result = generate_content_with_groq(base_script, enhanced_research)
            if groq_result["success"]:
                content = groq_result["content"]
                ai_used = "groq"
        
        # Try DeepSeek if all others failed
        if not content and deepseek_client:
            deepseek_result = generate_content_with_deepseek(base_script, enhanced_research)
            if deepseek_result["success"]:
                content = deepseek_result["content"]
                ai_used = "deepseek"
        
        if not content:
            return {'success': False, 'error': 'All AI services failed', 'link': None, 'ai_used': 'none'}
        
        # Check if content is unique
        if check_content_uniqueness(content, base_script):
            logger.info(f"✅ Unique content generated on attempt {attempt + 1} using {ai_used}")
            return {'success': True, 'content': content, 'ai_used': ai_used}
        else:
            logger.warning(f"❌ Content not unique enough, retrying... (attempt {attempt + 1})")
    
    # If all retries failed, return the last content with a warning
    logger.warning(f"⚠️ Could not generate unique content after {max_retries} attempts, using last attempt")
    return {'success': True, 'content': content, 'ai_used': ai_used, 'warning': 'Content may be similar to recent posts'}

def create_and_post_content(base_script: str, instant: bool = False) -> Dict[str, Any]:
    """Create AI-generated content and post to Twitter"""
    if not base_script:
        return {'success': False, 'error': 'No base script provided', 'link': None}

    logger.info(f"Creating content with base script: {base_script}")

    # Research trending topics with timeout
    research_data = research_trending_topics(base_script)

    # Generate unique content with retry logic to avoid repetition
    generation_result = generate_unique_content_with_retries(base_script, research_data)
    
    if not generation_result['success']:
        # Fallback if all AI services fail
        script_terms = base_script.replace('#', '').replace('@', '').replace('$', '').split()
        key_terms = [term for term in script_terms if len(term) > 3][:3]

        if key_terms:
            content = f"🚀 Exciting developments in {' '.join(key_terms[:2])}! Stay updated with the latest trends and insights. {DEVELOPER_TG} #trending #updates"
        else:
            content = f"🚀 {base_script[:100]} - Stay updated with the latest trends and developments! {DEVELOPER_TG} #trending #updates"

        content = optimize_character_count(content)
        ai_used = "fallback"
        logger.warning("All AI services failed, using fallback content")
    else:
        content = generation_result['content']
        ai_used = generation_result['ai_used']
        if 'warning' in generation_result:
            logger.warning(f"Content generation warning: {generation_result['warning']}")

    logger.info(f"Content generated using: {ai_used}")

    # Post to Twitter
    result = post_to_twitter(content)
    
    # Ensure ai_used is always included in the result
    result['ai_used'] = ai_used

    # Update app data
    now = datetime.now()
    post_entry = {
        'timestamp': now.isoformat(),
        'content': content,
        'base_script': base_script,
        'success': result['success'],
        'link': result.get('link'),
        'error': result.get('error'),
        'type': 'instant' if instant else 'scheduled',
        'ai_used': ai_used
    }

    app_data['post_history'].insert(0, post_entry)

    # Keep only last 50 posts
    if len(app_data['post_history']) > 50:
        app_data['post_history'] = app_data['post_history'][:50]

    if not instant:
        app_data['last_post_time'] = now
        app_data['next_post_time'] = now + timedelta(hours=POSTING_INTERVAL_HOURS)

    return result

# ============================================================================
# TELEGRAM BOT FUNCTIONS
# ============================================================================

def get_main_keyboard():
    """Create main menu keyboard with buttons"""
    keyboard = [
        [InlineKeyboardButton("📝 Set Base Script", callback_data="base_menu")],
        [InlineKeyboardButton("⚡ Instant Post", callback_data="instant_post")],
        [InlineKeyboardButton("📊 Status", callback_data="status")],
        [InlineKeyboardButton("📈 History", callback_data="history")]
    ]
    return InlineKeyboardMarkup(keyboard)

def get_base_menu_keyboard():
    """Create base script menu keyboard"""
    keyboard = []
    if app_data['base_script']:
        keyboard.append([InlineKeyboardButton("📝 Update Script", callback_data="set_base")])
        keyboard.append([InlineKeyboardButton("🗑️ Remove Script", callback_data="remove_base")])
    else:
        keyboard.append([InlineKeyboardButton("📝 Set Script", callback_data="set_base")])

    keyboard.append([InlineKeyboardButton("◀️ Back", callback_data="main_menu")])
    return InlineKeyboardMarkup(keyboard)

async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /start command"""
    global admin_chat_id

    if update.message:
        # Store admin chat ID for notifications
        admin_chat_id = update.message.chat_id
        logger.info(f"Admin chat ID stored: {admin_chat_id}")

        welcome_msg = f"""
🤖 *Twitter Automation Bot* 

Welcome! This bot automatically posts to Twitter every {POSTING_INTERVAL_HOURS} hours using AI-generated content.

*Current base script:* {app_data['base_script'] or 'Not set'}

Choose an option below:

Developer: {DEVELOPER_TG}
        """
        await update.message.reply_text(welcome_msg, parse_mode='Markdown', reply_markup=get_main_keyboard())

async def button_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle button callbacks"""
    query = update.callback_query
    if query:
        await query.answer()

    if query and query.data == "main_menu":
        welcome_msg = f"""
🤖 *Twitter Automation Bot* 

Welcome! This bot automatically posts to Twitter every {POSTING_INTERVAL_HOURS} hours using AI-generated content.

*Current base script:* {app_data['base_script'] or 'Not set'}

Choose an option below:

Developer: {DEVELOPER_TG}
        """
        await query.edit_message_text(welcome_msg, parse_mode='Markdown', reply_markup=get_main_keyboard())
        app_data['awaiting_input'] = None

    elif query and query.data == "base_menu":
        base_msg = f"""
📝 *Base Script Management*

Current base script: {app_data['base_script'] or 'Not set'}

Choose an action:
        """
        await query.edit_message_text(base_msg, parse_mode='Markdown', reply_markup=get_base_menu_keyboard())

    elif query and query.data == "set_base":
        app_data['awaiting_input'] = 'base'
        if query and query.message:
            app_data['temp_message_id'] = query.message.message_id
            await query.edit_message_text(
                "📝 *Set Base Script*\n\nPlease send me the base script for automated posts:",
                parse_mode='Markdown',
                reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("❌ Cancel", callback_data="base_menu")]])
            )

    elif query and query.data == "remove_base":
        app_data['base_script'] = ''
        if query:
            await query.edit_message_text(
                "✅ *Base script removed*\n\nAutomated posting is now disabled.",
                parse_mode='Markdown',
                reply_markup=get_base_menu_keyboard()
            )

    elif query and query.data == "instant_post":
        app_data['awaiting_input'] = 'instant'
        if query and query.message:
            app_data['temp_message_id'] = query.message.message_id
            await query.edit_message_text(
                "⚡ *Instant Post*\n\nPlease send me the content for immediate posting:",
                parse_mode='Markdown',
                reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("❌ Cancel", callback_data="main_menu")]])
            )

    elif query and query.data == "status":
        status_msg = f"""
📊 *Bot Status*

*Base Script:* {app_data['base_script'] or 'Not set'}
*Posting Enabled:* {'Yes' if app_data['posting_enabled'] and app_data['base_script'] else 'No'}
*Next Post:* {get_next_post_countdown()}
*Last Post:* {app_data['last_post_time'].strftime('%Y-%m-%d %H:%M') if app_data['last_post_time'] else 'Never'}
*Total Posts:* {len(app_data['post_history'])}

*AI Status:*
• OpenAI: {'✅' if openai_client else '❌'}
• Groq: {'✅' if groq_client else '❌'}
• DeepSeek: {'✅' if deepseek_client else '❌'}
• Gemini: {'✅' if GEMINI_API_KEY else '❌'}
• Twitter: {'✅' if twitter_api else '❌'}

Developer: {DEVELOPER_TG}
        """
        await query.edit_message_text(status_msg, parse_mode='Markdown', 
                                    reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("◀️ Back", callback_data="main_menu")]]))

    elif query and query.data == "history":
        if not app_data['post_history']:
            await query.edit_message_text("📈 *Recent Posts*\n\nNo posts yet.", parse_mode='Markdown',
                                        reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("◀️ Back", callback_data="main_menu")]]))
            return

        history_msg = "📈 *Recent Posts*\n\n"

        for i, post in enumerate(app_data['post_history'][:5]):
            timestamp = datetime.fromisoformat(post['timestamp']).strftime('%m/%d %H:%M')
            status = "✅" if post['success'] else "❌"
            post_type = "⚡" if post['type'] == 'instant' else "⏰"
            ai_used = post.get('ai_used', 'unknown')

            history_msg += f"{post_type} {status} {timestamp} ({ai_used})\n"
            history_msg += f"Content: {post['content'][:80]}...\n"
            if post['link']:
                history_msg += f"Link: {post['link']}\n"
            history_msg += "\n"

        if query:
            await query.edit_message_text(history_msg, parse_mode='Markdown',
                                        reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("◀️ Back", callback_data="main_menu")]]))

async def handle_text_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle text messages based on current state"""
    if not update.message or not app_data['awaiting_input']:
        return

    text = update.message.text

    if app_data['awaiting_input'] == 'base':
        # Setting base script
        app_data['base_script'] = text
        app_data['awaiting_input'] = None

        # Try to edit the message if possible
        try:
            await context.bot.edit_message_text(
                chat_id=update.message.chat_id,
                message_id=app_data['temp_message_id'],
                text=f"✅ *Base script updated*\n\nNew script: {text}\n\nNext automated post in: {get_next_post_countdown()}",
                parse_mode='Markdown',
                reply_markup=get_base_menu_keyboard()
            )
        except:
            # If editing fails, send new message
            await update.message.reply_text(
                f"✅ *Base script updated*\n\nNew script: {text}\n\nNext automated post in: {get_next_post_countdown()}",
                parse_mode='Markdown',
                reply_markup=get_main_keyboard()
            )

        logger.info(f"Base script updated: {text}")

    elif app_data['awaiting_input'] == 'instant':
        # Instant posting
        app_data['awaiting_input'] = None

        # Try to edit the message
        try:
            await context.bot.edit_message_text(
                chat_id=update.message.chat_id,
                message_id=app_data['temp_message_id'],
                text="🔄 Creating and posting content...",
                parse_mode='Markdown'
            )
        except:
            pass

        result = create_and_post_content(text, instant=True)

        if result['success']:
            success_msg = f"""
✅ *Posted successfully!*

Content: {result['content'][:100]}...
Link: {result['link']}
AI Used: {result.get('ai_used', 'unknown')}

Developer: {DEVELOPER_TG}
            """
            try:
                await context.bot.edit_message_text(
                    chat_id=update.message.chat_id,
                    message_id=app_data['temp_message_id'],
                    text=success_msg,
                    parse_mode='Markdown',
                    reply_markup=get_main_keyboard()
                )
            except:
                await update.message.reply_text(success_msg, parse_mode='Markdown', reply_markup=get_main_keyboard())
        else:
            error_msg = f"""
❌ *Posting failed*

Error: {result['error']}

Content was: {result.get('content', text)[:100]}...
            """
            try:
                await context.bot.edit_message_text(
                    chat_id=update.message.chat_id,
                    message_id=app_data['temp_message_id'],
                    text=error_msg,
                    parse_mode='Markdown',
                    reply_markup=get_main_keyboard()
                )
            except:
                await update.message.reply_text(error_msg, parse_mode='Markdown', reply_markup=get_main_keyboard())

def send_notification_to_telegram_sync(message: str):
    """Send notification message to admin via Telegram using HTTP API (synchronous)"""
    try:
        # Use admin_chat_id from environment or stored value
        chat_id = admin_chat_id or os.environ.get('ADMIN_CHAT_ID')

        if TELEGRAM_BOT_TOKEN and chat_id:
            # Send message via HTTP Bot API
            url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
            data = {
                'chat_id': chat_id,
                'text': message,
                'parse_mode': 'Markdown'
            }

            response = requests.post(url, data=data, timeout=10)
            if response.status_code == 200:
                logger.info(f"Telegram notification sent: {message[:50]}...")
            else:
                logger.error(f"Telegram API error: {response.status_code} - {response.text}")
        else:
            logger.info(f"Telegram notification (no chat_id): {message}")
    except Exception as e:
        logger.error(f"Telegram notification error: {e}")

# ============================================================================
# SCHEDULER FUNCTIONS
# ============================================================================

def scheduled_post_job():
    """Scheduled job for automated posting"""
    # Determine if this is the initial post or regular post
    post_type = "Initial" if not app_data['initial_post_done'] else "Regular"

    logger.info(f"🎯 SCHEDULED POST JOB TRIGGERED - {post_type} post starting")

    if not app_data['posting_enabled'] or not app_data['base_script']:
        logger.info(f"🚫 {post_type} scheduled posting skipped - {'disabled' if not app_data['posting_enabled'] else 'no base script'}")
        if not app_data['initial_post_done']:
            app_data['initial_post_done'] = True
        app_data['next_post_time'] = datetime.now() + timedelta(hours=POSTING_INTERVAL_HOURS)
        return

    logger.info(f"🚀 Running {post_type.lower()} scheduled post job with base script: '{app_data['base_script'][:50]}{'...' if len(app_data['base_script']) > 50 else ''}'")

    base_script = app_data.get('base_script') or ''
    if base_script:
        result = create_and_post_content(base_script)
    else:
        logger.warning("No base script available for scheduled posting")
        return

    # Mark initial post as done
    if not app_data['initial_post_done']:
        app_data['initial_post_done'] = True

    # Update next post time
    app_data['next_post_time'] = datetime.now() + timedelta(hours=POSTING_INTERVAL_HOURS)

    # Send notification
    if result['success']:
        notification = f"✅ {post_type} post successful!\nLink: {result['link']}\nAI Used: {result.get('ai_used', 'unknown')}\nContent: {result['content'][:100]}..."
        logger.info(f"✅ {post_type} post successful: {result['link']}")
    else:
        notification = f"❌ {post_type} post failed!\nError: {result['error']}"
        logger.error(f"❌ {post_type} post failed: {result['error']}")

    # Send notification synchronously from scheduler thread
    send_notification_to_telegram_sync(notification)

def get_next_post_countdown() -> str:
    """Get human-readable countdown to next post"""
    if not app_data['next_post_time']:
        return "Not scheduled"

    now = datetime.now()
    if now >= app_data['next_post_time']:
        return "Due now"

    diff = app_data['next_post_time'] - now
    hours = int(diff.total_seconds() // 3600)
    minutes = int((diff.total_seconds() % 3600) // 60)

    return f"{hours}h {minutes}m"

def update_countdown():
    """Update countdown timer"""
    app_data['current_countdown'] = get_next_post_countdown()

def keep_alive_heartbeat():
    """Keep-alive heartbeat to maintain server activity"""
    try:
        # Ping self or external URL to keep alive
        ping_url = os.environ.get('PING_URL', 'http://127.0.0.1:5000/api/status')
        response = requests.get(ping_url, timeout=5)
        app_data['last_heartbeat'] = datetime.now().isoformat()

        logger.info(f"💓 Heartbeat successful - Server alive. Ping: {ping_url} ({response.status_code}). Next post: {get_next_post_countdown()}")
    except Exception as e:
        logger.warning(f"💓 Heartbeat failed: {e}")
        app_data['last_heartbeat'] = datetime.now().isoformat()

# ============================================================================
# FLASK WEB SERVER
# ============================================================================

DASHBOARD_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Twitter Automation Bot Dashboard</title>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }
        .container { max-width: 1200px; margin: 0 auto; background: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        .header { text-align: center; color: #1da1f2; margin-bottom: 30px; }
        .status-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; margin-bottom: 30px; }
        .status-card { background: #f8f9fa; padding: 20px; border-radius: 8px; border-left: 4px solid #1da1f2; }
        .status-card h3 { margin: 0 0 15px 0; color: #333; }
        .status-card p { margin: 5px 0; }
        .form-section { background: #f8f9fa; padding: 20px; border-radius: 8px; margin-bottom: 20px; }
        .form-group { margin-bottom: 15px; }
        .form-group label { display: block; margin-bottom: 5px; font-weight: bold; }
        .form-group input, .form-group textarea { width: 100%; padding: 8px; border: 1px solid #ddd; border-radius: 4px; }
        .btn { background: #1da1f2; color: white; padding: 10px 20px; border: none; border-radius: 4px; cursor: pointer; }
        .btn:hover { background: #0d8bd9; }
        .btn-danger { background: #dc3545; }
        .btn-danger:hover { background: #c82333; }
        .posts-history { background: #f8f9fa; padding: 20px; border-radius: 8px; }
        .post-item { background: white; padding: 15px; border-radius: 6px; margin-bottom: 10px; border-left: 3px solid #28a745; }
        .post-item.failed { border-left-color: #dc3545; }
        .post-meta { font-size: 0.9em; color: #666; margin-bottom: 8px; }
        .post-content { margin-bottom: 8px; }
        .post-link { color: #1da1f2; text-decoration: none; }
        .developer-credit { text-align: center; margin-top: 30px; color: #666; font-style: italic; }
        .ai-status { display: flex; flex-wrap: wrap; gap: 10px; }
        .ai-service { background: #e9ecef; padding: 8px 12px; border-radius: 15px; font-size: 0.9em; }
        .ai-service.available { background: #d4edda; color: #155724; }
        .ai-service.unavailable { background: #f8d7da; color: #721c24; }
    </style>
    <script>
        function refreshPage() { location.reload(); }
        setInterval(refreshPage, 30000); // Refresh every 30 seconds
    </script>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🤖 Twitter Automation Bot Dashboard</h1>
            <p>Automated Twitter posting with AI content generation</p>
        </div>

        <div class="status-grid">
            <div class="status-card">
                <h3>📊 System Status</h3>
                <p><strong>Posting Enabled:</strong> {{ 'Yes' if data.posting_enabled and data.base_script else 'No' }}</p>
                <p><strong>Next Post:</strong> {{ countdown }}</p>
                <p><strong>Last Post:</strong> {{ data.last_post_time.strftime('%Y-%m-%d %H:%M') if data.last_post_time else 'Never' }}</p>
                <p><strong>Total Posts:</strong> {{ data.post_history|length }}</p>
                <p><strong>Server Started:</strong> {{ data.server_start_time[:19] if data.server_start_time else 'Unknown' }}</p>
            </div>

            <div class="status-card">
                <h3>🤖 AI Services Status</h3>
                <div class="ai-status">
                    <span class="ai-service {{ 'available' if data.openai_available else 'unavailable' }}">
                        OpenAI {{ '✅' if data.openai_available else '❌' }}
                    </span>
                    <span class="ai-service {{ 'available' if data.groq_available else 'unavailable' }}">
                        Groq {{ '✅' if data.groq_available else '❌' }}
                    </span>
                    <span class="ai-service {{ 'available' if data.deepseek_available else 'unavailable' }}">
                        DeepSeek {{ '✅' if data.deepseek_available else '❌' }}
                    </span>
                    <span class="ai-service {{ 'available' if data.gemini_available else 'unavailable' }}">
                        Gemini {{ '✅' if data.gemini_available else '❌' }}
                    </span>
                    <span class="ai-service {{ 'available' if data.twitter_available else 'unavailable' }}">
                        Twitter {{ '✅' if data.twitter_available else '❌' }}
                    </span>
                </div>
            </div>

            <div class="status-card">
                <h3>📝 Current Base Script</h3>
                <p>{{ data.base_script if data.base_script else 'Not set - automated posting disabled' }}</p>
            </div>
        </div>

        <div class="form-section">
            <h3>📝 Update Base Script</h3>
            <form method="POST" action="/update_base">
                <div class="form-group">
                    <label for="base_script">Base Script for AI Content Generation:</label>
                    <textarea id="base_script" name="base_script" rows="3" placeholder="Enter the base topic/script for automated AI content generation...">{{ data.base_script }}</textarea>
                </div>
                <button type="submit" class="btn">Update Base Script</button>
                {% if data.base_script %}
                <button type="submit" name="action" value="clear" class="btn btn-danger">Clear Script</button>
                {% endif %}
            </form>
        </div>

        <div class="form-section">
            <h3>⚡ Instant Post</h3>
            <form method="POST" action="/instant_post">
                <div class="form-group">
                    <label for="instant_content">Content for Immediate AI Processing and Posting:</label>
                    <textarea id="instant_content" name="instant_content" rows="3" placeholder="Enter content for immediate AI-enhanced posting..."></textarea>
                </div>
                <button type="submit" class="btn">Post Now</button>
            </form>
        </div>

        <div class="posts-history">
            <h3>📈 Recent Posts ({{ data.post_history|length }})</h3>
            {% if data.post_history %}
                {% for post in data.post_history[:10] %}
                <div class="post-item {{ 'failed' if not post.success else '' }}">
                    <div class="post-meta">
                        {{ post.timestamp[:19] }} | 
                        {{ '⚡ Instant' if post.type == 'instant' else '⏰ Scheduled' }} | 
                        {{ '✅ Success' if post.success else '❌ Failed' }} |
                        AI: {{ post.ai_used|title if post.ai_used else 'Unknown' }}
                    </div>
                    <div class="post-content">{{ post.content }}</div>
                    {% if post.link %}
                    <a href="{{ post.link }}" target="_blank" class="post-link">View Tweet</a>
                    {% endif %}
                    {% if post.error %}
                    <div style="color: #dc3545; font-size: 0.9em;">Error: {{ post.error }}</div>
                    {% endif %}
                </div>
                {% endfor %}
            {% else %}
                <p>No posts yet. Set a base script to enable automated posting.</p>
            {% endif %}
        </div>

        <div class="developer-credit">
            <p>Developed by {{ developer }} | Auto-refresh every 30 seconds</p>
        </div>
    </div>
</body>
</html>
"""

@app.route('/')
def index():
    """Main dashboard"""
    if not session.get('authenticated'):
        return redirect(url_for('login'))

    # Add AI service availability status
    enhanced_data = dict(app_data)
    enhanced_data.update({
        'openai_available': openai_client is not None,
        'groq_available': groq_client is not None,
        'deepseek_available': deepseek_client is not None,
        'gemini_available': GEMINI_API_KEY is not None,
        'twitter_available': twitter_api is not None
    })

    return render_template_string(DASHBOARD_TEMPLATE, 
                                data=enhanced_data, 
                                countdown=get_next_post_countdown(),
                                developer=DEVELOPER_TG)

@app.route('/login', methods=['GET', 'POST'])
def login():
    """Admin login"""
    if request.method == 'POST':
        password = request.form.get('password')
        if password == ADMIN_PASSWORD:
            session['authenticated'] = True
            return redirect(url_for('index'))
        else:
            return render_template_string(LOGIN_TEMPLATE, error="Invalid password")

    return render_template_string(LOGIN_TEMPLATE)

LOGIN_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Twitter Bot Admin Login</title>
    <style>
        body { font-family: Arial, sans-serif; background: #f5f5f5; display: flex; justify-content: center; align-items: center; height: 100vh; margin: 0; }
        .login-container { background: white; padding: 40px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); width: 300px; text-align: center; }
        .login-container h2 { color: #1da1f2; margin-bottom: 30px; }
        .form-group { margin-bottom: 20px; text-align: left; }
        .form-group label { display: block; margin-bottom: 5px; font-weight: bold; }
        .form-group input { width: 100%; padding: 10px; border: 1px solid #ddd; border-radius: 4px; box-sizing: border-box; }
        .btn { background: #1da1f2; color: white; padding: 12px 24px; border: none; border-radius: 4px; cursor: pointer; width: 100%; }
        .btn:hover { background: #0d8bd9; }
        .error { color: #dc3545; margin-top: 10px; }
        .developer-credit { margin-top: 20px; color: #666; font-size: 0.9em; }
    </style>
</head>
<body>
    <div class="login-container">
        <h2>🤖 Twitter Bot Admin</h2>
        <form method="POST">
            <div class="form-group">
                <label for="password">Admin Password:</label>
                <input type="password" id="password" name="password" required>
            </div>
            <button type="submit" class="btn">Login</button>
        </form>
        {% if error %}
        <div class="error">{{ error }}</div>
        {% endif %}
        <div class="developer-credit">
            <p>Developed by {{ DEVELOPER_TG }}</p>
        </div>
    </div>
</body>
</html>
"""

@app.route('/update_base', methods=['POST'])
def update_base():
    """Update base script"""
    if not session.get('authenticated'):
        return redirect(url_for('login'))

    action = request.form.get('action')
    if action == 'clear':
        app_data['base_script'] = ''
    else:
        app_data['base_script'] = request.form.get('base_script', '').strip()

    return redirect(url_for('index'))

@app.route('/instant_post', methods=['POST'])
def instant_post():
    """Handle instant post"""
    if not session.get('authenticated'):
        return redirect(url_for('login'))

    content = request.form.get('instant_content', '').strip()
    if content:
        result = create_and_post_content(content, instant=True)
        # Result will be visible in post history on main page

    return redirect(url_for('index'))

@app.route('/test_post', methods=['POST'])
def test_post():
    """Test the posting system manually"""
    if not session.get('authenticated'):
        return redirect(url_for('login'))

    logger.info("🧪 Manual test post triggered")

    # Use base script for test
    base_script = app_data.get('base_script', '')
    if not base_script:
        logger.error("❌ No base script available for test")
        return redirect(url_for('index'))

    logger.info(f"🧪 Testing with base script: {base_script[:50]}...")
    result = create_and_post_content(base_script, instant=True)
    logger.info(f"🧪 Test result: {result}")

    return redirect(url_for('index'))

@app.route('/logout')
def logout():
    """Logout"""
    session.pop('authenticated', None)
    return redirect(url_for('login'))

@app.route('/api/status')
def api_status():
    """API status endpoint for keep-alive"""
    return jsonify({
        'status': 'alive',
        'next_post': get_next_post_countdown(),
        'posts_count': len(app_data['post_history']),
        'base_script_set': bool(app_data['base_script']),
        'timestamp': datetime.now().isoformat()
    })

# ============================================================================
# MAIN FUNCTION
# ============================================================================

def initialize_telegram_bot():
    """Initialize Telegram bot"""
    global telegram_app

    if not TELEGRAM_BOT_TOKEN:
        logger.warning("No Telegram bot token provided")
        return None

    try:
        telegram_app = Application.builder().token(TELEGRAM_BOT_TOKEN).build()

        # Add handlers
        telegram_app.add_handler(CommandHandler("start", start_command))
        telegram_app.add_handler(CallbackQueryHandler(button_callback))
        telegram_app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text_message))

        logger.info("Telegram bot initialized successfully")
        return telegram_app
    except Exception as e:
        logger.error(f"Failed to initialize Telegram bot: {e}")
        return None

def setup_scheduler():
    """Set up the posting scheduler"""
    try:
        # Clear any existing jobs
        scheduler.remove_all_jobs()

        # Calculate initial delay (2 minutes from now)
        start_time = datetime.now() + timedelta(minutes=2)
        app_data['next_post_time'] = start_time

        # Add scheduled posting job - runs once after 2 minutes, then every 2 hours
        # First add a one-time job for the initial post
        scheduler.add_job(
            func=scheduled_post_job,
            trigger='date',
            run_date=start_time,
            id='initial_posting',
            name='Initial Twitter Posting',
            max_instances=1,
            replace_existing=True
        )

        # Then add the recurring job that starts after first post
        recurring_start = start_time + timedelta(hours=POSTING_INTERVAL_HOURS)
        scheduler.add_job(
            func=scheduled_post_job,
            trigger=IntervalTrigger(hours=POSTING_INTERVAL_HOURS),
            start_date=recurring_start,
            id='scheduled_posting',
            name='Scheduled Twitter Posting',
            max_instances=1,
            replace_existing=True
        )

        # Add countdown update job every minute
        scheduler.add_job(
            func=update_countdown,
            trigger=IntervalTrigger(minutes=1),
            id='countdown_update',
            name='Countdown Update'
        )

        # Add keep-alive heartbeat every 10 minutes
        scheduler.add_job(
            func=keep_alive_heartbeat,
            trigger=IntervalTrigger(minutes=10),
            id='keep_alive',
            name='Keep Alive Heartbeat'
        )

        scheduler.start()
        logger.info(f"✅ Scheduler initialized - Initial post in 2 minutes, then every {POSTING_INTERVAL_HOURS} hours")
        logger.info(f"📅 Initial post scheduled for: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"📅 Recurring posts start at: {recurring_start.strftime('%Y-%m-%d %H:%M:%S')}")

    except Exception as e:
        logger.error(f"Failed to setup scheduler: {e}")

def main():
    """Main function"""
    logger.info("🚀 Starting Twitter Automation Bot")

    # Initialize components
    telegram_app = initialize_telegram_bot()
    setup_scheduler()

    # Start Telegram bot polling
    if telegram_app:
        try:
            # Start polling in a separate thread
            import threading
            def run_telegram_bot():
                try:
                    asyncio.new_event_loop().run_until_complete(
                        telegram_app.run_polling(drop_pending_updates=True, stop_signals=None)
                    )
                except Exception as e:
                    logger.error(f"Telegram bot polling error: {e}")

            telegram_thread = threading.Thread(target=run_telegram_bot, daemon=True)
            telegram_thread.start()

            logger.info("📱 Telegram bot polling started successfully")
            logger.info("📱 Bot ready to receive /start and other commands")
            logger.info("📱 Web admin interface also available for management")
        except Exception as e:
            logger.error(f"Telegram bot setup error: {e}")
            logger.info("📱 Continuing with web interface only")
    else:
        logger.info("📱 Continuing with web interface - Telegram bot optional")

    # Configure Flask for production
    app.config['DEBUG'] = False
    app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

    # Set cache control headers
    @app.after_request
    def after_request(response):
        response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
        response.headers['Pragma'] = 'no-cache'
        response.headers['Expires'] = '0'
        return response

    # Dynamic port selection with 5000 first for Replit webview compatibility  
    env_port = int(os.environ.get('PORT', 5000))
    logger.info(f"Starting Flask server (env port: {env_port})")

    # Try ports in order: 5000 -> ENV_PORT -> 8080 (5000 required for webview)
    ports_to_try = [5000]
    if env_port != 5000:
        ports_to_try.append(env_port)
    if 8080 not in ports_to_try:
        ports_to_try.append(8080)

    for try_port in ports_to_try:
        try:
            logger.info(f"Attempting to start Flask server on port {try_port}")
            app.run(host='0.0.0.0', port=try_port, debug=False, threaded=True, use_reloader=False)
            break
        except OSError as e:
            logger.warning(f"Port {try_port} unavailable: {e}")
            if try_port == ports_to_try[-1]:
                logger.error("All ports failed - server cannot start")
                raise

if __name__ == '__main__':
    main()
