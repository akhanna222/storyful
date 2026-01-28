"""
Reputation Risk Intelligence - Utility Functions

This module provides:
- Twitter/X API integration
- LLM (OpenAI/Anthropic) wrapper functions
- Reputation risk scoring formulas
- Credibility and amplification assessment helpers
"""

import os
import json
import asyncio
import hashlib
from datetime import datetime, timezone
from typing import Any, Optional

import httpx
import tweepy
from tweepy import asynchronous as tweepy_async

# LLM clients - initialized lazily
_openai_client = None
_anthropic_client = None


# =============================================================================
# CONFIGURATION
# =============================================================================

def get_env(key: str, default: str = "") -> str:
    """Get environment variable with optional default."""
    return os.getenv(key, default)


def get_llm_provider() -> str:
    """Get configured LLM provider."""
    return get_env("LLM_PROVIDER", "openai").lower()


def get_llm_model() -> str:
    """Get configured LLM model."""
    provider = get_llm_provider()
    default = "gpt-4-turbo-preview" if provider == "openai" else "claude-3-opus-20240229"
    return get_env("LLM_MODEL", default)


# =============================================================================
# TWITTER/X API INTEGRATION
# =============================================================================

def get_twitter_client() -> tweepy.Client:
    """Get authenticated Twitter API v2 client."""
    bearer_token = get_env("TWITTER_BEARER_TOKEN")
    if not bearer_token:
        raise ValueError("TWITTER_BEARER_TOKEN not configured")

    return tweepy.Client(
        bearer_token=bearer_token,
        consumer_key=get_env("TWITTER_API_KEY"),
        consumer_secret=get_env("TWITTER_API_SECRET"),
        access_token=get_env("TWITTER_ACCESS_TOKEN"),
        access_token_secret=get_env("TWITTER_ACCESS_TOKEN_SECRET"),
        wait_on_rate_limit=True
    )


async def search_tweets(
    query: str,
    max_results: int = 20,
    include_author_info: bool = True
) -> list[dict]:
    """
    Search recent tweets matching query.

    Args:
        query: Twitter search query (e.g., "Tesla -is:retweet")
        max_results: Maximum tweets to return (10-100)
        include_author_info: Whether to fetch author details

    Returns:
        List of tweet dictionaries with full metadata
    """
    client = get_twitter_client()

    # Build tweet fields for rich metadata
    tweet_fields = [
        "id", "text", "created_at", "author_id",
        "public_metrics", "entities", "attachments",
        "context_annotations", "conversation_id"
    ]

    user_fields = [
        "id", "name", "username", "verified", "description",
        "public_metrics", "created_at", "profile_image_url"
    ]

    expansions = ["author_id", "attachments.media_keys"]
    media_fields = ["type", "url", "preview_image_url"]

    try:
        # Execute search
        response = client.search_recent_tweets(
            query=query,
            max_results=min(max_results, 100),
            tweet_fields=tweet_fields,
            user_fields=user_fields,
            expansions=expansions,
            media_fields=media_fields
        )

        if not response.data:
            return []

        # Build user lookup
        users = {}
        if response.includes and "users" in response.includes:
            for user in response.includes["users"]:
                users[user.id] = user

        # Build media lookup
        media = {}
        if response.includes and "media" in response.includes:
            for m in response.includes["media"]:
                media[m.media_key] = m

        # Format tweets with full metadata
        tweets = []
        for tweet in response.data:
            author = users.get(tweet.author_id)

            tweet_data = {
                "id": str(tweet.id),
                "text": tweet.text,
                "created_at": tweet.created_at.isoformat() if tweet.created_at else None,
                "url": f"https://twitter.com/i/web/status/{tweet.id}",
                "author": {
                    "id": str(tweet.author_id),
                    "name": author.name if author else "Unknown",
                    "username": author.username if author else "unknown",
                    "verified": getattr(author, "verified", False) if author else False,
                    "description": author.description if author else "",
                    "followers_count": author.public_metrics.get("followers_count", 0) if author and author.public_metrics else 0,
                    "following_count": author.public_metrics.get("following_count", 0) if author and author.public_metrics else 0,
                    "tweet_count": author.public_metrics.get("tweet_count", 0) if author and author.public_metrics else 0,
                    "profile_image_url": getattr(author, "profile_image_url", None) if author else None
                },
                "metrics": {
                    "likes": tweet.public_metrics.get("like_count", 0) if tweet.public_metrics else 0,
                    "retweets": tweet.public_metrics.get("retweet_count", 0) if tweet.public_metrics else 0,
                    "replies": tweet.public_metrics.get("reply_count", 0) if tweet.public_metrics else 0,
                    "quotes": tweet.public_metrics.get("quote_count", 0) if tweet.public_metrics else 0
                },
                "has_media": bool(tweet.attachments and tweet.attachments.get("media_keys")),
                "entities": tweet.entities if tweet.entities else {},
                "context_annotations": tweet.context_annotations if tweet.context_annotations else []
            }

            tweets.append(tweet_data)

        return tweets

    except tweepy.TweepyException as e:
        print(f"Twitter API error: {e}")
        return []


def build_search_query(
    company_name: str,
    keywords: list[str] = None,
    exclude_retweets: bool = True,
    language: str = "en"
) -> str:
    """
    Build Twitter search query for company monitoring.

    Args:
        company_name: Primary company name
        keywords: Additional keywords/handles to include
        exclude_retweets: Whether to exclude retweets
        language: Tweet language filter

    Returns:
        Formatted Twitter search query string
    """
    terms = [company_name]
    if keywords:
        terms.extend(keywords)

    # Build OR query for terms
    query_parts = [f'"{term}"' if " " in term else term for term in terms]
    query = f"({' OR '.join(query_parts)})"

    # Add filters
    if exclude_retweets:
        query += " -is:retweet"
    if language:
        query += f" lang:{language}"

    return query


# =============================================================================
# LLM INTEGRATION
# =============================================================================

async def call_llm(
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.3,
    max_tokens: int = 2000
) -> dict:
    """
    Call LLM (OpenAI or Anthropic) with prompts.

    Args:
        system_prompt: System/context prompt
        user_prompt: User message/query
        temperature: Response randomness (0-1)
        max_tokens: Maximum response tokens

    Returns:
        Parsed JSON response from LLM
    """
    provider = get_llm_provider()
    model = get_llm_model()

    if provider == "openai":
        return await _call_openai(system_prompt, user_prompt, model, temperature, max_tokens)
    elif provider == "anthropic":
        return await _call_anthropic(system_prompt, user_prompt, model, temperature, max_tokens)
    else:
        raise ValueError(f"Unsupported LLM provider: {provider}")


async def _call_openai(
    system_prompt: str,
    user_prompt: str,
    model: str,
    temperature: float,
    max_tokens: int
) -> dict:
    """Call OpenAI API."""
    global _openai_client

    if _openai_client is None:
        from openai import AsyncOpenAI
        _openai_client = AsyncOpenAI(api_key=get_env("OPENAI_API_KEY"))

    try:
        response = await _openai_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=temperature,
            max_tokens=max_tokens,
            response_format={"type": "json_object"}
        )

        content = response.choices[0].message.content
        return json.loads(content)

    except json.JSONDecodeError as e:
        print(f"Failed to parse LLM response as JSON: {e}")
        return {"error": "Invalid JSON response", "raw": content}
    except Exception as e:
        print(f"OpenAI API error: {e}")
        return {"error": str(e)}


async def _call_anthropic(
    system_prompt: str,
    user_prompt: str,
    model: str,
    temperature: float,
    max_tokens: int
) -> dict:
    """Call Anthropic Claude API."""
    global _anthropic_client

    if _anthropic_client is None:
        from anthropic import AsyncAnthropic
        _anthropic_client = AsyncAnthropic(api_key=get_env("ANTHROPIC_API_KEY"))

    try:
        response = await _anthropic_client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=system_prompt,
            messages=[
                {"role": "user", "content": user_prompt}
            ]
        )

        content = response.content[0].text

        # Extract JSON from response (Claude may include explanation)
        json_start = content.find("{")
        json_end = content.rfind("}") + 1
        if json_start != -1 and json_end > json_start:
            json_str = content[json_start:json_end]
            return json.loads(json_str)

        return json.loads(content)

    except json.JSONDecodeError as e:
        print(f"Failed to parse LLM response as JSON: {e}")
        return {"error": "Invalid JSON response", "raw": content}
    except Exception as e:
        print(f"Anthropic API error: {e}")
        return {"error": str(e)}


# =============================================================================
# REPUTATION RISK SCORING
# =============================================================================

# Threat type severity weights (max 30 points)
THREAT_SEVERITY_WEIGHTS = {
    "product_safety": 25,
    "data_privacy": 25,
    "executive_misconduct": 23,
    "financial_misconduct": 23,
    "regulatory": 22,
    "environmental": 20,
    "labor_practices": 20,
    "customer_harm": 18,
    "competitive_attack": 12
}

# Stakeholder action risk scores
STAKEHOLDER_ACTION_SCORES = {
    "boycott_potential": 8,
    "investigation_trigger": 10,
    "lawsuit_risk": 9,
    "stock_sell_off": 9,
    "employee_exodus": 7,
    "partner_termination": 6,
    "negative_reviews": 4,
    "media_campaign": 7
}


def assess_threat_severity(
    threat_type: str,
    has_evidence: bool,
    credibility: str,
    severity_indicators: list[str] = None
) -> float:
    """
    Calculate threat severity score (0-30 points).

    Args:
        threat_type: Category of threat
        has_evidence: Whether evidence is attached
        credibility: Source credibility level (high/medium/low)
        severity_indicators: Additional severity signals

    Returns:
        Threat severity score (0-30)
    """
    base = THREAT_SEVERITY_WEIGHTS.get(threat_type, 15)

    # Evidence multiplier
    if has_evidence:
        base *= 1.3

    # Credibility multiplier
    if credibility == "high":
        base *= 1.2
    elif credibility == "low":
        base *= 0.7

    # Severity indicator bonuses
    if severity_indicators:
        if "verified_account" in severity_indicators:
            base += 2
        if "firsthand_claim" in severity_indicators:
            base += 3
        if "multiple_reports" in severity_indicators:
            base += 2
        if "media_engaged" in severity_indicators:
            base += 3

    return min(base, 30)  # Cap at 30 points


def assess_stakeholder_impact(
    affected_groups: list[str],
    impact_levels: dict[str, str],
    action_risks: dict[str, list[str]],
    financial_materiality: str
) -> float:
    """
    Calculate stakeholder impact score (0-35 points).

    Args:
        affected_groups: List of affected stakeholder groups
        impact_levels: Impact level per group (none/low/medium/high/critical)
        action_risks: Potential actions per group
        financial_materiality: Overall financial impact level

    Returns:
        Stakeholder impact score (0-35)
    """
    score = 0

    # Impact level scoring
    impact_values = {"none": 0, "low": 1, "medium": 2, "high": 4, "critical": 6}
    for group in affected_groups:
        level = impact_levels.get(group, "low")
        score += impact_values.get(level, 1)

    # Action risk scoring
    for group, actions in action_risks.items():
        for action in actions:
            score += STAKEHOLDER_ACTION_SCORES.get(action, 2)

    # Financial materiality multiplier
    materiality_multipliers = {"low": 0.8, "medium": 1.0, "high": 1.3}
    score *= materiality_multipliers.get(financial_materiality, 1.0)

    return min(score, 35)  # Cap at 35 points


def assess_amplification_risk(
    virality_score: float,
    media_pickup_probability: float,
    current_stage: str,
    time_to_peak: str,
    tipping_point_factors: list[str] = None
) -> float:
    """
    Calculate amplification risk score (0-35 points).

    Args:
        virality_score: Current virality (0-100)
        media_pickup_probability: Likelihood of mainstream media coverage (0-1)
        current_stage: Crisis stage (emerging/developing/escalating/peak/declining)
        time_to_peak: Time window (hours/days/weeks)
        tipping_point_factors: Factors that could accelerate

    Returns:
        Amplification risk score (0-35)
    """
    score = 0

    # Base virality contribution (0-10 points)
    score += (virality_score / 100) * 10

    # Media pickup probability (0-12 points)
    score += media_pickup_probability * 12

    # Stage scoring (0-8 points)
    stage_scores = {
        "emerging": 2,
        "developing": 4,
        "escalating": 6,
        "peak": 8,
        "declining": 3
    }
    score += stage_scores.get(current_stage, 2)

    # Time urgency (0-5 points)
    time_scores = {"hours": 5, "days": 3, "weeks": 1}
    score += time_scores.get(time_to_peak, 2)

    # Tipping point factors
    if tipping_point_factors:
        score += min(len(tipping_point_factors) * 1.5, 5)

    return min(score, 35)  # Cap at 35 points


def calculate_reputation_risk_score(
    threat_data: dict,
    stakeholder_data: dict,
    amplification_data: dict
) -> dict:
    """
    Calculate composite reputation risk score (0-100).

    Args:
        threat_data: Output from threat detection agent
        stakeholder_data: Output from stakeholder impact agent
        amplification_data: Output from amplification risk agent

    Returns:
        Complete risk score breakdown
    """
    # Calculate component scores
    threat_score = assess_threat_severity(
        threat_type=threat_data.get("threat_type", "unknown"),
        has_evidence="evidence" in threat_data.get("severity_indicators", []) or
                     "evidence_attached" in threat_data.get("severity_indicators", []),
        credibility=threat_data.get("credibility", "medium"),
        severity_indicators=threat_data.get("severity_indicators", [])
    )

    stakeholder_score = assess_stakeholder_impact(
        affected_groups=stakeholder_data.get("primary_stakeholders", []),
        impact_levels={
            group: info.get("impact_level", "low")
            for group, info in stakeholder_data.get("impact_by_stakeholder", {}).items()
        },
        action_risks={
            group: [info.get("action_risk", "")] if isinstance(info.get("action_risk"), str)
                   else info.get("action_risk", [])
            for group, info in stakeholder_data.get("impact_by_stakeholder", {}).items()
        },
        financial_materiality=stakeholder_data.get("financial_materiality", "medium")
    )

    amplification_score = assess_amplification_risk(
        virality_score=amplification_data.get("amplification_score", 0),
        media_pickup_probability=amplification_data.get("media_pickup_probability", 0),
        current_stage=amplification_data.get("current_stage", "emerging"),
        time_to_peak=amplification_data.get("time_to_peak", "days"),
        tipping_point_factors=amplification_data.get("tipping_point_factors", [])
    )

    total = threat_score + stakeholder_score + amplification_score

    return {
        "total_score": round(total, 1),
        "breakdown": {
            "threat": round(threat_score, 1),
            "stakeholder": round(stakeholder_score, 1),
            "amplification": round(amplification_score, 1)
        },
        "risk_level": categorize_risk_level(total),
        "urgency": calculate_urgency(amplification_data.get("time_to_peak", "days"))
    }


def categorize_risk_level(score: float) -> str:
    """Categorize risk score into level."""
    if score >= 86:
        return "crisis"
    elif score >= 71:
        return "alert"
    elif score >= 51:
        return "investigate"
    elif score >= 26:
        return "monitor"
    else:
        return "noise"


def calculate_urgency(time_to_peak: str) -> str:
    """Calculate response urgency based on time to peak."""
    urgency_map = {
        "hours": "immediate",
        "days": "same_day",
        "weeks": "this_week"
    }
    return urgency_map.get(time_to_peak, "monitor")


# =============================================================================
# CREDIBILITY ASSESSMENT
# =============================================================================

def extract_credibility_signals(tweet: dict) -> dict:
    """
    Extract credibility signals from tweet and author.

    Args:
        tweet: Tweet data dictionary

    Returns:
        Credibility assessment with signals
    """
    author = tweet.get("author", {})
    credibility_score = 0
    signals = []

    # Verified account
    if author.get("verified"):
        credibility_score += 30
        signals.append("verified")

    # Follower count
    followers = author.get("followers_count", 0)
    if followers > 100000:
        credibility_score += 25
        signals.append("major_influencer")
    elif followers > 10000:
        credibility_score += 20
        signals.append("influencer")
    elif followers > 1000:
        credibility_score += 10
        signals.append("established_account")

    # Bio indicators for journalists/experts
    bio = (author.get("description") or "").lower()
    if any(word in bio for word in ["journalist", "reporter", "editor", "correspondent"]):
        credibility_score += 40
        signals.append("media")
    if any(word in bio for word in ["analyst", "researcher", "professor", "expert"]):
        credibility_score += 30
        signals.append("expert")
    if any(word in bio for word in ["employee", "former", "insider", "worked at"]):
        credibility_score += 25
        signals.append("insider")

    # Evidence indicators
    if tweet.get("has_media"):
        credibility_score += 15
        signals.append("evidence_attached")

    # Engagement ratio (high engagement = more credible/viral)
    metrics = tweet.get("metrics", {})
    total_engagement = metrics.get("likes", 0) + metrics.get("retweets", 0) + metrics.get("replies", 0)
    if total_engagement > 10000:
        credibility_score += 20
        signals.append("viral")
    elif total_engagement > 1000:
        credibility_score += 10
        signals.append("high_engagement")

    # Determine credibility level
    if credibility_score > 60:
        level = "high"
    elif credibility_score > 30:
        level = "medium"
    else:
        level = "low"

    return {
        "credibility": level,
        "signals": signals,
        "score": credibility_score
    }


def detect_coordination_signals(
    tweets: list[dict],
    time_window_minutes: int = 60
) -> dict:
    """
    Detect potential coordination in a set of tweets.

    Args:
        tweets: List of tweet data dictionaries
        time_window_minutes: Time window for clustering

    Returns:
        Coordination analysis
    """
    if len(tweets) < 3:
        return {"is_coordinated": False, "signals": [], "confidence": 0}

    signals = []
    confidence = 0

    # Check for similar hashtags
    all_hashtags = []
    for tweet in tweets:
        entities = tweet.get("entities", {})
        hashtags = [h.get("tag", "").lower() for h in entities.get("hashtags", [])]
        all_hashtags.extend(hashtags)

    if all_hashtags:
        hashtag_counts = {}
        for tag in all_hashtags:
            hashtag_counts[tag] = hashtag_counts.get(tag, 0) + 1

        # If same hashtag appears in >50% of tweets
        for tag, count in hashtag_counts.items():
            if count > len(tweets) * 0.5:
                signals.append(f"common_hashtag:#{tag}")
                confidence += 20

    # Check for similar text (copy-paste)
    texts = [tweet.get("text", "")[:100].lower() for tweet in tweets]
    for i, text1 in enumerate(texts):
        for text2 in texts[i+1:]:
            similarity = _text_similarity(text1, text2)
            if similarity > 0.8:
                signals.append("copy_paste_detected")
                confidence += 30
                break

    # Check timing clustering
    timestamps = []
    for tweet in tweets:
        created = tweet.get("created_at")
        if created:
            try:
                ts = datetime.fromisoformat(created.replace("Z", "+00:00"))
                timestamps.append(ts)
            except:
                pass

    if len(timestamps) >= 3:
        timestamps.sort()
        time_diffs = [(timestamps[i+1] - timestamps[i]).total_seconds() / 60
                      for i in range(len(timestamps)-1)]
        avg_diff = sum(time_diffs) / len(time_diffs)

        if avg_diff < 5:  # Average 5 minutes between posts
            signals.append("timing_cluster")
            confidence += 25

    return {
        "is_coordinated": confidence > 40,
        "signals": signals,
        "confidence": min(confidence, 100)
    }


def _text_similarity(text1: str, text2: str) -> float:
    """Simple text similarity using character overlap."""
    if not text1 or not text2:
        return 0.0

    set1 = set(text1.split())
    set2 = set(text2.split())

    if not set1 or not set2:
        return 0.0

    intersection = len(set1 & set2)
    union = len(set1 | set2)

    return intersection / union if union > 0 else 0.0


# =============================================================================
# UTILITY HELPERS
# =============================================================================

def generate_threat_id(tweet: dict) -> str:
    """Generate unique ID for a threat based on tweet data."""
    unique_str = f"{tweet.get('id', '')}-{tweet.get('text', '')[:50]}"
    return hashlib.sha256(unique_str.encode()).hexdigest()[:16]


def format_timestamp(iso_string: str) -> str:
    """Format ISO timestamp for display."""
    if not iso_string:
        return "Unknown"
    try:
        dt = datetime.fromisoformat(iso_string.replace("Z", "+00:00"))
        now = datetime.now(timezone.utc)
        diff = now - dt

        if diff.total_seconds() < 60:
            return "Just now"
        elif diff.total_seconds() < 3600:
            mins = int(diff.total_seconds() / 60)
            return f"{mins}m ago"
        elif diff.total_seconds() < 86400:
            hours = int(diff.total_seconds() / 3600)
            return f"{hours}h ago"
        else:
            days = int(diff.total_seconds() / 86400)
            return f"{days}d ago"
    except:
        return iso_string


def format_number(num: int) -> str:
    """Format large numbers for display (e.g., 1.5K, 2.3M)."""
    if num >= 1_000_000:
        return f"{num/1_000_000:.1f}M"
    elif num >= 1_000:
        return f"{num/1_000:.1f}K"
    else:
        return str(num)
