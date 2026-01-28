"""
Reputation Risk Intelligence MVP - FastAPI Backend

This module provides:
- FastAPI REST API endpoints
- Three-agent pipeline for reputation risk analysis
- In-memory storage for MVP
- Agent orchestration and scoring
"""

import os
import asyncio
from datetime import datetime, timezone
from typing import Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from utils import (
    search_tweets,
    build_search_query,
    call_llm,
    calculate_reputation_risk_score,
    extract_credibility_signals,
    detect_coordination_signals,
    generate_threat_id,
    format_timestamp,
    format_number
)

# Load environment variables
load_dotenv()

# =============================================================================
# IN-MEMORY STORAGE (MVP Only)
# =============================================================================

# Active monitoring sessions
monitoring_sessions: dict = {}

# Detected threats
threats_db: dict = {}

# Escalated threats for human review
escalated_threats: dict = {}


# =============================================================================
# PYDANTIC MODELS
# =============================================================================

class MonitorRequest(BaseModel):
    company_name: str = Field(..., description="Company name to monitor")
    keywords: list[str] = Field(default=[], description="Additional keywords/handles")
    tweet_count: int = Field(default=20, ge=10, le=100, description="Number of tweets to fetch")
    focus_areas: list[str] = Field(default=[], description="Optional: filter by threat types")


class EscalateRequest(BaseModel):
    notes: str = Field(default="", description="Notes for escalation")


class ThreatResponse(BaseModel):
    id: str
    timestamp: str
    tweet_data: dict
    threat_analysis: dict
    stakeholder_analysis: dict
    amplification_analysis: dict
    risk_score: dict
    requires_human_review: bool
    escalated: bool = False
    escalation_notes: str = ""


# =============================================================================
# AGENT PROMPTS
# =============================================================================

THREAT_DETECTION_PROMPT = """You are a corporate reputation risk analyst specializing in identifying threats to company reputation.

Your task: Analyze this tweet and determine if it represents a REPUTATIONAL THREAT to the company.

REPUTATIONAL THREAT DEFINITION:
An allegation, claim, or narrative that could damage:
- Brand trust and customer loyalty
- Stakeholder confidence (investors, partners, employees)
- Regulatory standing or legal position
- Corporate legitimacy or social license to operate

THREAT TYPES TO IDENTIFY:
1. product_safety: Defects, injuries, safety failures, recalls
2. data_privacy: Breaches, surveillance, misuse of customer data
3. executive_misconduct: Leadership wrongdoing, ethical failures
4. financial_misconduct: Fraud, accounting issues, misleading investors
5. labor_practices: Unsafe conditions, discrimination, unfair treatment
6. environmental: Pollution, climate impact, greenwashing
7. customer_harm: Predatory practices, deceptive marketing, poor service
8. regulatory: Investigation, violation, non-compliance
9. competitive_attack: Rival-amplified negative narrative

NOT THREATS (ignore these):
- General complaints without serious allegations
- Positive or neutral mentions
- Competitor comparisons (unless alleging harm)
- Price complaints alone
- Personal opinions without factual claims

CREDIBILITY SIGNALS (strengthen threat assessment):
- Verified account or journalist
- Firsthand account (employee, customer, witness)
- Evidence attached (photos, documents, data)
- Multiple similar reports
- Expert or institutional source

OUTPUT FORMAT (JSON only):
{
  "is_threat": boolean,
  "threat_type": "one_of_the_types_above or null if not a threat",
  "severity_indicators": ["verified_account", "evidence", "multiple_reports", etc.],
  "entities_at_risk": ["CEO_name", "Product_X", "Division_Y"],
  "claim_summary": "brief factual summary of allegation",
  "credibility": "high|medium|low",
  "reasoning": "why this is/isn't a reputation threat"
}

TWEET DATA:
Author: {author_name} (@{author_username}), {followers} followers, Verified: {verified}
Bio: {author_bio}
Content: {tweet_text}
Engagement: {likes} likes, {retweets} retweets, {replies} replies
Has Media: {has_media}
Timestamp: {timestamp}

Company being monitored: {company_name}

Analyze and respond with JSON only."""


STAKEHOLDER_IMPACT_PROMPT = """You are analyzing the STAKEHOLDER IMPACT of a confirmed reputational threat.

STAKEHOLDER CATEGORIES:
1. CUSTOMERS/USERS - Direct product/service users
   Impact factors: Purchase decisions, brand loyalty, word-of-mouth

2. INVESTORS/SHAREHOLDERS - Financial stakeholders
   Impact factors: Stock price, valuation, investment decisions

3. REGULATORS/GOVERNMENT - Oversight bodies
   Impact factors: Investigations, fines, license risk

4. EMPLOYEES - Current and prospective workforce
   Impact factors: Morale, retention, talent acquisition

5. MEDIA/PUBLIC - General reputation
   Impact factors: News coverage, social perception, viral risk

6. PARTNERS/SUPPLIERS - B2B relationships
   Impact factors: Contract risk, partnership strain

For each AFFECTED stakeholder group, assess:
- IMPACT LEVEL: none|low|medium|high|critical
- ACTION RISK: What might they do? (boycott, sell_shares, investigate, quit, terminate_contract, negative_reviews, etc.)
- REASONING: Why this stakeholder cares

THREAT CONTEXT:
{threat_analysis}

ORIGINAL TWEET:
Author: {author_name} (@{author_username}), {followers} followers
Content: {tweet_text}
Engagement: {likes} likes, {retweets} retweets

Company: {company_name}

OUTPUT FORMAT (JSON only):
{
  "primary_stakeholders": ["customers", "regulators", etc.],
  "impact_by_stakeholder": {
    "customers": {
      "impact_level": "high",
      "action_risk": "boycott_potential",
      "reasoning": "Product safety directly affects purchase decisions"
    },
    "regulators": {
      "impact_level": "medium",
      "action_risk": "investigation_trigger",
      "reasoning": "Safety allegations may prompt agency review"
    }
  },
  "overall_stakeholder_reach": "widespread|targeted|niche",
  "financial_materiality": "high|medium|low"
}

Respond with JSON only."""


AMPLIFICATION_RISK_PROMPT = """You are predicting whether this reputational threat will ESCALATE from social media to a full-blown crisis.

AMPLIFICATION FACTORS TO ASSESS:

1. VIRALITY INDICATORS
   - Engagement velocity (is it growing fast?)
   - Shareability (emotional, shocking, visual?)
   - Current reach vs potential reach

2. MEDIA PICKUP LIKELIHOOD
   - Journalist engagement already?
   - Newsworthy elements (scale, harm, hypocrisy, public figure)?
   - Slow news day or competing stories?

3. COORDINATION SIGNALS
   - Organic complaint or organized campaign?
   - Activist involvement?
   - Competitor amplification?

4. TIMING FACTORS
   - Company events (earnings, product launch, shareholder meeting)?
   - Existing negative narrative this feeds into?
   - Day of week, time of day?

5. NARRATIVE STICKINESS
   - Simple, memorable story?
   - Fits existing criticism of company?
   - Confirms stakeholder fears?

ESCALATION PATHWAYS:
- Social → Mainstream Media → Regulatory Action
- Employee → Industry Press → Talent Flight
- Customer → Forums/Reddit → Organized Movement
- Activist → Shareholder Pressure → Board Scrutiny

CRISIS STAGES:
- emerging: Early signals, limited awareness
- developing: Growing attention, no mainstream coverage yet
- escalating: Media pickup, official response pressure building
- peak: Maximum visibility, crisis management active
- declining: Attention fading, post-crisis phase

THREAT & STAKEHOLDER CONTEXT:
{threat_analysis}
{stakeholder_analysis}

ORIGINAL TWEET:
Author: {author_name} (@{author_username}), {followers} followers, Verified: {verified}
Content: {tweet_text}
Engagement: {likes} likes, {retweets} retweets, {replies} replies
Posted: {timestamp}

Company: {company_name}

OUTPUT FORMAT (JSON only):
{
  "amplification_score": 0-100,
  "escalation_pathway": "social_to_media|grassroots_to_organized|insider_to_investigation",
  "time_to_peak": "hours|days|weeks",
  "media_pickup_probability": 0.0-1.0,
  "current_stage": "emerging|developing|escalating|peak|declining",
  "tipping_point_factors": ["verified_journalist_engaged", "competitor_amplifying", etc.],
  "recommended_response_window": "immediate|same_day|24_hours|this_week|monitor",
  "reasoning": "why this will/won't blow up"
}

Respond with JSON only."""


# =============================================================================
# AGENT FUNCTIONS
# =============================================================================

async def threat_detection_agent(tweet: dict, company_name: str) -> dict:
    """
    Agent 1: Detect if tweet represents a reputational threat.
    """
    author = tweet.get("author", {})
    metrics = tweet.get("metrics", {})

    prompt = THREAT_DETECTION_PROMPT.format(
        author_name=author.get("name", "Unknown"),
        author_username=author.get("username", "unknown"),
        followers=format_number(author.get("followers_count", 0)),
        verified=author.get("verified", False),
        author_bio=author.get("description", "No bio"),
        tweet_text=tweet.get("text", ""),
        likes=format_number(metrics.get("likes", 0)),
        retweets=format_number(metrics.get("retweets", 0)),
        replies=format_number(metrics.get("replies", 0)),
        has_media=tweet.get("has_media", False),
        timestamp=format_timestamp(tweet.get("created_at", "")),
        company_name=company_name
    )

    result = await call_llm(
        system_prompt="You are a corporate reputation risk analyst. Respond with valid JSON only.",
        user_prompt=prompt,
        temperature=0.2
    )

    # Add credibility signals from utility function
    credibility = extract_credibility_signals(tweet)
    if "credibility" not in result:
        result["credibility"] = credibility["credibility"]
    if "severity_indicators" not in result:
        result["severity_indicators"] = []
    result["severity_indicators"].extend(credibility["signals"])

    return result


async def stakeholder_impact_agent(
    tweet: dict,
    threat_analysis: dict,
    company_name: str
) -> dict:
    """
    Agent 2: Assess stakeholder impact of the threat.
    """
    author = tweet.get("author", {})
    metrics = tweet.get("metrics", {})

    prompt = STAKEHOLDER_IMPACT_PROMPT.format(
        threat_analysis=str(threat_analysis),
        author_name=author.get("name", "Unknown"),
        author_username=author.get("username", "unknown"),
        followers=format_number(author.get("followers_count", 0)),
        tweet_text=tweet.get("text", ""),
        likes=format_number(metrics.get("likes", 0)),
        retweets=format_number(metrics.get("retweets", 0)),
        company_name=company_name
    )

    result = await call_llm(
        system_prompt="You are a stakeholder impact analyst. Respond with valid JSON only.",
        user_prompt=prompt,
        temperature=0.2
    )

    return result


async def amplification_risk_agent(
    tweet: dict,
    threat_analysis: dict,
    stakeholder_analysis: dict,
    company_name: str
) -> dict:
    """
    Agent 3: Predict amplification and escalation risk.
    """
    author = tweet.get("author", {})
    metrics = tweet.get("metrics", {})

    prompt = AMPLIFICATION_RISK_PROMPT.format(
        threat_analysis=str(threat_analysis),
        stakeholder_analysis=str(stakeholder_analysis),
        author_name=author.get("name", "Unknown"),
        author_username=author.get("username", "unknown"),
        followers=format_number(author.get("followers_count", 0)),
        verified=author.get("verified", False),
        tweet_text=tweet.get("text", ""),
        likes=format_number(metrics.get("likes", 0)),
        retweets=format_number(metrics.get("retweets", 0)),
        replies=format_number(metrics.get("replies", 0)),
        timestamp=format_timestamp(tweet.get("created_at", "")),
        company_name=company_name
    )

    result = await call_llm(
        system_prompt="You are an amplification risk analyst. Respond with valid JSON only.",
        user_prompt=prompt,
        temperature=0.3
    )

    return result


# =============================================================================
# AGENT ORCHESTRATION
# =============================================================================

async def analyze_for_reputation_risk(
    tweet: dict,
    company_name: str,
    focus_areas: list[str] = None
) -> Optional[dict]:
    """
    Three-agent pipeline for reputation risk assessment.

    Args:
        tweet: Tweet data dictionary
        company_name: Company being monitored
        focus_areas: Optional filter for threat types

    Returns:
        Reputation event dictionary or None if not a threat
    """
    # Stage 1: Is this a threat?
    threat_analysis = await threat_detection_agent(tweet, company_name)

    if not threat_analysis.get("is_threat", False):
        return None  # Not a reputational risk, skip

    # Filter by focus areas if specified
    if focus_areas and threat_analysis.get("threat_type"):
        if threat_analysis["threat_type"] not in focus_areas:
            return None

    # Stage 2: Who cares and how much?
    stakeholder_analysis = await stakeholder_impact_agent(
        tweet, threat_analysis, company_name
    )

    # Stage 3: Will this blow up?
    amplification_analysis = await amplification_risk_agent(
        tweet, threat_analysis, stakeholder_analysis, company_name
    )

    # Synthesize final risk score
    risk_score = calculate_reputation_risk_score(
        threat_analysis, stakeholder_analysis, amplification_analysis
    )

    # Build reputation event
    threat_id = generate_threat_id(tweet)
    reputation_event = {
        "id": threat_id,
        "timestamp": tweet.get("created_at", datetime.now(timezone.utc).isoformat()),
        "tweet_data": tweet,
        "threat_analysis": threat_analysis,
        "stakeholder_analysis": stakeholder_analysis,
        "amplification_analysis": amplification_analysis,
        "risk_score": risk_score,
        "requires_human_review": risk_score["total_score"] > 70,
        "escalated": False,
        "escalation_notes": "",
        "company_name": company_name
    }

    # Only store if actual threat (score > 25)
    if risk_score["total_score"] > 25:
        threats_db[threat_id] = reputation_event

    return reputation_event


async def run_monitoring_pipeline(
    session_id: str,
    company_name: str,
    keywords: list[str],
    tweet_count: int,
    focus_areas: list[str]
):
    """
    Background task to run monitoring pipeline.
    """
    try:
        # Update session status
        if session_id in monitoring_sessions:
            monitoring_sessions[session_id]["status"] = "fetching"

        # Build search query
        query = build_search_query(company_name, keywords)

        # Fetch tweets
        tweets = await search_tweets(query, max_results=tweet_count)

        if session_id in monitoring_sessions:
            monitoring_sessions[session_id]["tweets_fetched"] = len(tweets)
            monitoring_sessions[session_id]["status"] = "analyzing"

        # Analyze each tweet
        threats_found = 0
        for i, tweet in enumerate(tweets):
            result = await analyze_for_reputation_risk(tweet, company_name, focus_areas)

            if result and result["risk_score"]["total_score"] > 25:
                threats_found += 1

            if session_id in monitoring_sessions:
                monitoring_sessions[session_id]["tweets_analyzed"] = i + 1

        # Update final status
        if session_id in monitoring_sessions:
            monitoring_sessions[session_id]["status"] = "completed"
            monitoring_sessions[session_id]["threats_found"] = threats_found
            monitoring_sessions[session_id]["completed_at"] = datetime.now(timezone.utc).isoformat()

    except Exception as e:
        if session_id in monitoring_sessions:
            monitoring_sessions[session_id]["status"] = "error"
            monitoring_sessions[session_id]["error"] = str(e)


# =============================================================================
# FASTAPI APPLICATION
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    print("Starting Reputation Risk Intelligence MVP...")
    yield
    print("Shutting down...")


app = FastAPI(
    title="Reputation Risk Intelligence MVP",
    description="Agentic AI system for monitoring company reputational risks on Twitter/X",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# API ENDPOINTS
# =============================================================================

@app.get("/")
async def serve_frontend():
    """Serve the frontend HTML."""
    return FileResponse("index.html")


@app.post("/api/monitor/start")
async def start_monitoring(request: MonitorRequest, background_tasks: BackgroundTasks):
    """
    Start monitoring company mentions for reputation risks.

    Initiates a background task to:
    1. Fetch recent tweets mentioning the company
    2. Analyze each tweet through the 3-agent pipeline
    3. Store detected threats in memory
    """
    session_id = f"session_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"

    monitoring_sessions[session_id] = {
        "id": session_id,
        "company_name": request.company_name,
        "keywords": request.keywords,
        "focus_areas": request.focus_areas,
        "tweet_count": request.tweet_count,
        "status": "starting",
        "tweets_fetched": 0,
        "tweets_analyzed": 0,
        "threats_found": 0,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "completed_at": None,
        "error": None
    }

    background_tasks.add_task(
        run_monitoring_pipeline,
        session_id,
        request.company_name,
        request.keywords,
        request.tweet_count,
        request.focus_areas
    )

    return {
        "session_id": session_id,
        "message": f"Started monitoring for {request.company_name}",
        "status": "starting"
    }


@app.get("/api/monitor/status/{session_id}")
async def get_monitoring_status(session_id: str):
    """Get status of a monitoring session."""
    if session_id not in monitoring_sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    return monitoring_sessions[session_id]


@app.get("/api/monitor/sessions")
async def list_sessions():
    """List all monitoring sessions."""
    return list(monitoring_sessions.values())


@app.get("/api/threats/active")
async def get_active_threats(
    min_score: int = 26,
    risk_level: Optional[str] = None,
    company: Optional[str] = None
):
    """
    Get all active threats with optional filtering.

    Args:
        min_score: Minimum risk score (default 26 to exclude noise)
        risk_level: Filter by risk level (noise/monitor/investigate/alert/crisis)
        company: Filter by company name
    """
    threats = []

    for threat in threats_db.values():
        score = threat["risk_score"]["total_score"]

        # Apply filters
        if score < min_score:
            continue
        if risk_level and threat["risk_score"]["risk_level"] != risk_level:
            continue
        if company and threat.get("company_name", "").lower() != company.lower():
            continue

        threats.append(threat)

    # Sort by risk score (highest first)
    threats.sort(key=lambda x: x["risk_score"]["total_score"], reverse=True)

    return {
        "count": len(threats),
        "threats": threats
    }


@app.get("/api/threats/{threat_id}")
async def get_threat_detail(threat_id: str):
    """Get full details for a specific threat."""
    if threat_id not in threats_db:
        raise HTTPException(status_code=404, detail="Threat not found")

    return threats_db[threat_id]


@app.post("/api/threats/{threat_id}/escalate")
async def escalate_threat(threat_id: str, request: EscalateRequest):
    """Mark a threat for human review with notes."""
    if threat_id not in threats_db:
        raise HTTPException(status_code=404, detail="Threat not found")

    threats_db[threat_id]["escalated"] = True
    threats_db[threat_id]["escalation_notes"] = request.notes
    threats_db[threat_id]["escalated_at"] = datetime.now(timezone.utc).isoformat()

    # Also store in escalated threats for quick access
    escalated_threats[threat_id] = threats_db[threat_id]

    return {
        "message": "Threat escalated successfully",
        "threat_id": threat_id
    }


@app.get("/api/threats/escalated")
async def get_escalated_threats():
    """Get all threats marked for human review."""
    return {
        "count": len(escalated_threats),
        "threats": list(escalated_threats.values())
    }


@app.post("/api/threats/{threat_id}/false-positive")
async def mark_false_positive(threat_id: str):
    """Mark a threat as false positive and remove from active list."""
    if threat_id not in threats_db:
        raise HTTPException(status_code=404, detail="Threat not found")

    # Remove from active threats
    del threats_db[threat_id]

    # Remove from escalated if present
    if threat_id in escalated_threats:
        del escalated_threats[threat_id]

    return {"message": "Marked as false positive", "threat_id": threat_id}


@app.get("/api/stats")
async def get_stats():
    """Get overall statistics."""
    total_threats = len(threats_db)

    risk_levels = {"crisis": 0, "alert": 0, "investigate": 0, "monitor": 0, "noise": 0}
    threat_types = {}

    for threat in threats_db.values():
        level = threat["risk_score"]["risk_level"]
        risk_levels[level] = risk_levels.get(level, 0) + 1

        threat_type = threat["threat_analysis"].get("threat_type", "unknown")
        threat_types[threat_type] = threat_types.get(threat_type, 0) + 1

    return {
        "total_threats": total_threats,
        "escalated_count": len(escalated_threats),
        "by_risk_level": risk_levels,
        "by_threat_type": threat_types,
        "active_sessions": len([s for s in monitoring_sessions.values() if s["status"] not in ["completed", "error"]])
    }


# =============================================================================
# DEMO DATA ENDPOINT (for testing without Twitter API)
# =============================================================================

@app.post("/api/demo/load")
async def load_demo_data():
    """Load demo threats for testing the UI without Twitter API credentials."""

    demo_tweets = [
        {
            "id": "demo_1",
            "text": "Breaking: FDA investigating Tesla after 5 brake failure deaths reported. Internal documents show company knew about defect for months. Thread with evidence...",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "url": "https://twitter.com/demo/status/1",
            "author": {
                "id": "123",
                "name": "TechInvestigator",
                "username": "techinvestigator",
                "verified": True,
                "description": "Investigative journalist covering automotive safety",
                "followers_count": 89000,
                "following_count": 1200,
                "tweet_count": 15000,
                "profile_image_url": None
            },
            "metrics": {"likes": 15200, "retweets": 8900, "replies": 2100, "quotes": 450},
            "has_media": True,
            "entities": {},
            "context_annotations": []
        },
        {
            "id": "demo_2",
            "text": "I'm a former Tesla engineer. We knowingly shipped cars with battery defects that could cause fires. Management ignored all safety reports. I have the internal memos to prove it.",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "url": "https://twitter.com/demo/status/2",
            "author": {
                "id": "456",
                "name": "Anonymous Whistleblower",
                "username": "safetyfirst_anon",
                "verified": False,
                "description": "Former automotive engineer. Truth matters.",
                "followers_count": 2500,
                "following_count": 180,
                "tweet_count": 340,
                "profile_image_url": None
            },
            "metrics": {"likes": 4500, "retweets": 3200, "replies": 890, "quotes": 120},
            "has_media": True,
            "entities": {},
            "context_annotations": []
        },
        {
            "id": "demo_3",
            "text": "Tesla laid off 10% of workforce with zero notice. Friends got emails at 2am saying they're terminated effective immediately. No severance. This is how @elonmusk treats people.",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "url": "https://twitter.com/demo/status/3",
            "author": {
                "id": "789",
                "name": "Sarah Chen",
                "username": "sarahc_tech",
                "verified": False,
                "description": "Tech recruiter | Helping people find great jobs",
                "followers_count": 12000,
                "following_count": 890,
                "tweet_count": 8900,
                "profile_image_url": None
            },
            "metrics": {"likes": 2800, "retweets": 1500, "replies": 420, "quotes": 85},
            "has_media": False,
            "entities": {},
            "context_annotations": []
        },
        {
            "id": "demo_4",
            "text": "Tesla customer service is absolutely terrible. Been waiting 3 weeks for a callback about my car issue.",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "url": "https://twitter.com/demo/status/4",
            "author": {
                "id": "101",
                "name": "Frustrated Customer",
                "username": "carbuyer2024",
                "verified": False,
                "description": "",
                "followers_count": 150,
                "following_count": 340,
                "tweet_count": 89,
                "profile_image_url": None
            },
            "metrics": {"likes": 45, "retweets": 8, "replies": 12, "quotes": 0},
            "has_media": False,
            "entities": {},
            "context_annotations": []
        },
        {
            "id": "demo_5",
            "text": "SEC investigating Tesla for potential securities fraud. Sources say it's related to Musk's tweets about production numbers. Stock down 8% in after-hours.",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "url": "https://twitter.com/demo/status/5",
            "author": {
                "id": "202",
                "name": "Market Watch",
                "username": "marketwatch_live",
                "verified": True,
                "description": "Breaking financial news | Market analysis | @WSJ affiliate",
                "followers_count": 450000,
                "following_count": 890,
                "tweet_count": 125000,
                "profile_image_url": None
            },
            "metrics": {"likes": 8900, "retweets": 5600, "replies": 1200, "quotes": 890},
            "has_media": False,
            "entities": {},
            "context_annotations": []
        }
    ]

    # Analyze each demo tweet
    company_name = "Tesla"
    for tweet in demo_tweets:
        await analyze_for_reputation_risk(tweet, company_name)

    return {
        "message": "Demo data loaded",
        "tweets_analyzed": len(demo_tweets),
        "threats_detected": len(threats_db)
    }


@app.delete("/api/demo/clear")
async def clear_demo_data():
    """Clear all demo data."""
    threats_db.clear()
    escalated_threats.clear()
    monitoring_sessions.clear()
    return {"message": "All data cleared"}


# =============================================================================
# RUN SERVER
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
