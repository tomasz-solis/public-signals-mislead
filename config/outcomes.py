"""
Feature decision context for the public-signals-mislead project.

`company_action` is the observable part of the story: did the company keep
supporting the feature, pull back from it, or leave too little public evidence
to tell?

`business_outcome` is stricter. It stays `UNKNOWN` unless public evidence
supports a real positive or negative business result.
"""

SUPPORTED_ACTIONS = {"SUPPORTED"}
PULLED_BACK_ACTIONS = {"PULLED_BACK"}
KNOWN_BUSINESS_OUTCOMES = {"POSITIVE", "NEGATIVE"}


FEATURE_DECISION_CONTEXT = {
    "Password Sharing Crackdown": {
        "company_action": "SUPPORTED",
        "business_outcome": "POSITIVE",
        "evidence_summary": "9.3M paid net additions (Q1 2024)",
        "source": "Netflix Q1 2024 shareholder letter",
        "evidence_tier": "TIER1",
        "url": "https://ir.netflix.net/financials/quarterly-earnings/default.aspx",
    },
    "Extra Member": {
        "company_action": "SUPPORTED",
        "business_outcome": "POSITIVE",
        "evidence_summary": "Contributed to 9.3M growth",
        "source": "Netflix Q1 2024 shareholder letter",
        "evidence_tier": "TIER1",
        "url": "https://ir.netflix.net/financials/quarterly-earnings/default.aspx",
    },
    "Ad-Supported Tier": {
        "company_action": "SUPPORTED",
        "business_outcome": "POSITIVE",
        "evidence_summary": "23M monthly active users (Jan 2024)",
        "source": "Netflix press release",
        "evidence_tier": "TIER1",
        "url": "https://about.netflix.com/en/news/netflix-ads-plan-grows",
    },
    "AI DJ": {
        "company_action": "SUPPORTED",
        "business_outcome": "POSITIVE",
        "evidence_summary": "Billions of streams, top engagement driver",
        "source": "Spotify Q4 2023 shareholder letter",
        "evidence_tier": "TIER1",
        "url": "https://investors.spotify.com",
    },
    "Premium Price Increase": {
        "company_action": "SUPPORTED",
        "business_outcome": "POSITIVE",
        "evidence_summary": "10% ARPU increase YoY",
        "source": "Spotify Q3 2023 earnings call",
        "evidence_tier": "TIER1",
        "url": "https://investors.spotify.com/financials/press-releases/default.aspx",
    },
    "Multiview": {
        "company_action": "SUPPORTED",
        "business_outcome": "POSITIVE",
        "evidence_summary": "Millions of viewers during NFL",
        "source": "YouTube Blog",
        "evidence_tier": "TIER1",
        "url": "https://blog.youtube/news-and-events/nfl-sunday-ticket-youtube-tv/",
    },
    "Strength Training": {
        "company_action": "SUPPORTED",
        "business_outcome": "POSITIVE",
        "evidence_summary": "32% YoY growth in minutes consumed",
        "source": "Peloton shareholder letter",
        "evidence_tier": "TIER1",
        "url": "https://investor.onepeloton.com",
    },
    "Audiobooks": {
        "company_action": "SUPPORTED",
        "business_outcome": "UNKNOWN",
        "evidence_summary": "Spotify kept audiobooks inside the Premium offer and described them as helpful to retention and listening hours.",
        "source": "Spotify Q3 2023 earnings",
        "evidence_tier": "TIER2",
        "url": "https://investors.spotify.com",
        "caveat": "Public framing supports continued company support, but not a hard business outcome.",
    },
    "Star Content Hub": {
        "company_action": "SUPPORTED",
        "business_outcome": "UNKNOWN",
        "evidence_summary": "Disney highlighted Star as an engagement lever in international markets.",
        "source": "Disney Investor Day",
        "evidence_tier": "TIER2",
        "url": "https://thewaltdisneycompany.com/disney-investor-day-2020/",
        "caveat": "Directionally useful, but not a public revenue or retention result.",
    },
    "IMAX Enhanced": {
        "company_action": "SUPPORTED",
        "business_outcome": "UNKNOWN",
        "evidence_summary": "Disney+ expanded IMAX Enhanced beyond the initial launch set instead of quietly dropping it.",
        "source": "Disney+ press release",
        "evidence_tier": "TIER2",
        "url": "https://press.disneyplus.com/news/imax-enhanced-on-disney-plus-launches-november-12-disney-plus-day",
        "caveat": "This shows continued support, not a hard business result.",
    },
    "Unlimited DVR": {
        "company_action": "SUPPORTED",
        "business_outcome": "UNKNOWN",
        "evidence_summary": "Unlimited DVR remained a named YouTube TV base-plan benefit in official help docs.",
        "source": "YouTube TV Help",
        "evidence_tier": "TIER2",
        "url": "https://support.google.com/youtubetv/answer/7069119?hl=en",
        "caveat": "Official docs confirm ongoing support, but no public adoption metric was found.",
    },
    "Background Play": {
        "company_action": "SUPPORTED",
        "business_outcome": "UNKNOWN",
        "evidence_summary": "Background Play stayed listed as a core YouTube Premium benefit on supported devices.",
        "source": "YouTube Help",
        "evidence_tier": "TIER2",
        "url": "https://support.google.com/youtube/answer/6308116?hl=en",
        "caveat": "The public record shows maintenance, not outcome certainty.",
    },
    "Offline Downloads": {
        "company_action": "SUPPORTED",
        "business_outcome": "UNKNOWN",
        "evidence_summary": "Offline downloads remained part of the official YouTube Premium benefit set.",
        "source": "YouTube Help",
        "evidence_tier": "TIER2",
        "url": "https://support.google.com/youtube/answer/6308116?hl=en",
        "caveat": "The public record shows maintenance, not outcome certainty.",
    },
    "Classical App": {
        "company_action": "SUPPORTED",
        "business_outcome": "UNKNOWN",
        "evidence_summary": "Apple kept the dedicated Classical app as a live product surface for the segment.",
        "source": "Apple newsroom",
        "evidence_tier": "TIER2",
        "url": "https://www.apple.com/newsroom/",
        "caveat": "Continued presence is observable; user value and business impact are not.",
    },
    "Live TV Cloud DVR": {
        "company_action": "SUPPORTED",
        "business_outcome": "UNKNOWN",
        "evidence_summary": "Hulu rolled unlimited DVR into all Hulu + Live TV plans at no extra cost.",
        "source": "Hulu press release",
        "evidence_tier": "TIER2",
        "url": "https://press.hulu.com/pressrelease/unlimited-dvr-coming-to-all-hulu-live-tv-subscribers-at-no-additional-cost-on-april-13/",
        "caveat": "This shows product support and rollout breadth, not a hard revenue or retention result.",
    },
    "Running Content": {
        "company_action": "SUPPORTED",
        "business_outcome": "UNKNOWN",
        "evidence_summary": "Peloton kept a dedicated running catalog across app and tread experiences.",
        "source": "Peloton product pages",
        "evidence_tier": "TIER2",
        "url": "https://www.onepeloton.com/app",
        "caveat": "The catalog is observable, but the segment's business value is not public.",
    },
    "Games": {
        "company_action": "PULLED_BACK",
        "business_outcome": "NEGATIVE",
        "evidence_summary": "<1% daily usage (0.5% of subs)",
        "source": "CNBC via Apptopia",
        "evidence_tier": "TIER1",
        "url": "https://www.cnbc.com/2023/08/07/netflix-games-have-low-engagement.html",
        "caveat": "The hard outcome is negative; the exact internal follow-through is still only partially visible publicly.",
    },
    "GroupWatch": {
        "company_action": "PULLED_BACK",
        "business_outcome": "UNKNOWN",
        "evidence_summary": "Removed from Disney+ in September 2023, based on a help-center notice reported publicly.",
        "source": "Disney+ help-center notice via ComicBook",
        "evidence_tier": "TIER2",
        "url": "https://comicbook.com/irl/news/disney-plus-groupwatch-feature-no-longer-available/",
        "caveat": "Removal is observable. Whether the feature still had value for a niche audience is not.",
    },
    "App-Only Membership": {
        "company_action": "PULLED_BACK",
        "business_outcome": "NEGATIVE",
        "evidence_summary": "Below expectations",
        "source": "Peloton Q4 2023 earnings",
        "evidence_tier": "TIER1",
        "url": "https://investor.onepeloton.com",
    },
    "Watch Party": {
        "company_action": "UNKNOWN",
        "business_outcome": "UNKNOWN",
        "evidence_summary": "Hulu highlighted Watch Party in marketing copy, but the public record is too soft to classify either the company action or the business outcome.",
        "source": "Hulu year-end report",
        "evidence_tier": "TIER2",
        "url": "https://press.hulu.com/pressrelease/year-with-hulu-2021-its-in-your-dna/",
        "caveat": "Kept in the repo as a cautionary example: public commentary exists, but it should not be treated as outcome proof.",
    },
}

# Backward-compatible alias for older scripts or notebooks.
KNOWN_OUTCOMES = FEATURE_DECISION_CONTEXT


FEATURE_TYPES = {
    "Password Sharing Crackdown": "MONETIZATION",
    "Extra Member": "MONETIZATION",
    "Ad-Supported Tier": "MONETIZATION",
    "Premium Price Increase": "MONETIZATION",
    "Price Increase": "MONETIZATION",
    "Premium Plus Tier": "MONETIZATION",
    "AI DJ": "AI",
    "AI Playlist": "AI",
    "Daylist": "AI",
    "Wrapped AI Podcast": "AI",
    "Grok AI": "AI",
    "Star Content Hub": "CONTENT",
    "Classical App": "CONTENT",
    "Audiobooks": "CONTENT",
    "ESPN Integration": "CONTENT",
    "Strength Training": "CONTENT",
    "Rowing Classes": "CONTENT",
    "Running Content": "CONTENT",
    "Background Play": "UTILITY",
    "Offline Downloads": "UTILITY",
    "Downloads Offline": "UTILITY",
    "Download to Watch Offline": "UTILITY",
    "Downloads Feature Update": "UTILITY",
    "Queue Management": "UTILITY",
    "Profile Transfer": "UTILITY",
    "Parental Controls Update": "UTILITY",
    "GroupWatch": "SOCIAL",
    "Watch Party": "SOCIAL",
    "Multiview": "SOCIAL",
    "IMAX Enhanced": "TECH",
    "Spatial Audio": "TECH",
    "Lossless Audio": "TECH",
    "Dolby Atmos": "TECH",
    "Sing Feature": "TECH",
    "Live Sports": "LIVE",
    "Unlimited DVR": "LIVE",
    "Live TV Cloud DVR": "LIVE",
    "Games": "GAMES",
    "App-Only Membership": "OTHER",
}


def get_feature_context(feature_name: str) -> dict:
    """Return the decision context for a feature or an empty dict when missing."""
    return FEATURE_DECISION_CONTEXT.get(feature_name, {})


def get_feature_type(feature_name: str) -> str:
    """Return the repo's coarse feature type or ``UNKNOWN`` when missing."""
    return FEATURE_TYPES.get(feature_name, "UNKNOWN")


def get_all_contextualized_features() -> list:
    """Return every feature that has public decision context in the config."""
    return list(FEATURE_DECISION_CONTEXT.keys())


def get_action_counts() -> dict:
    """Count configured features by observed company action."""
    counts = {}
    for context in FEATURE_DECISION_CONTEXT.values():
        action = context["company_action"]
        counts[action] = counts.get(action, 0) + 1
    return counts


def get_business_outcome_counts() -> dict:
    """Count configured features by known business outcome coverage."""
    counts = {}
    for context in FEATURE_DECISION_CONTEXT.values():
        outcome = context["business_outcome"]
        counts[outcome] = counts.get(outcome, 0) + 1
    return counts
