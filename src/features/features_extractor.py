"""Feature extraction module implementing TELS paper features.

Includes Commonness (CMNS), overlap metrics (LE/CE), popularity features (vE, rE),
and context-based metrics (absorption, purity, DIST).
"""

import numpy as np
from typing import Set
from src.core import Candidate, EntityContext


def tokenize(text: str) -> Set[str]:
    """Tokenize text into lowercase word set."""
    return set(text.lower().split()) if text else set()


def compute_commonness(anchor_score: int, total_score: int) -> float:
    """CMNS: how often the mention refers to this entity vs others."""
    if total_score == 0:
        return 0.0
    return anchor_score / total_score


def compute_le_overlap(tweet_tokens: Set[str], anchors: dict) -> float:
    """LE Overlap: overlap between tweet and entity reference anchors."""
    if not tweet_tokens or not anchors:
        return 0.0
    anchor_tokens = set()
    for anchor_text in anchors.keys():
        anchor_tokens.update(tokenize(anchor_text))
    if not anchor_tokens:
        return 0.0
    intersection = tweet_tokens & anchor_tokens
    union = tweet_tokens | anchor_tokens
    return len(intersection) / len(union) if union else 0.0


def compute_ce_overlap(tweet_tokens: Set[str], categories: dict) -> float:
    """CE Overlap: overlap between tweet and entity categories."""
    if not tweet_tokens or not categories:
        return 0.0
    category_tokens = set()
    for cat_text in categories.keys():
        category_tokens.update(tokenize(cat_text))
    if not category_tokens:
        return 0.0
    intersection = tweet_tokens & category_tokens
    union = tweet_tokens | category_tokens
    return len(intersection) / len(union) if union else 0.0


def compute_in_category(ngram: str, categories: dict) -> float:
    """Check if n-gram appears in any category name."""
    ngram_lower = ngram.lower()
    for cat in categories.keys():
        if ngram_lower in cat.lower():
            return 1.0
    return 0.0


def compute_absorption(ngram: str, anchors: dict) -> float:
    """Absorption: proportion of anchors that contain the n-gram."""
    if not anchors:
        return 0.0
    ngram_lower = ngram.lower()
    contained_count = sum(1 for anchor in anchors.keys() if ngram_lower in anchor.lower())
    return contained_count / len(anchors)


def compute_purity(ngram: str, anchors: dict) -> float:
    """Purity: ratio of exact n-gram occurrences in references."""
    if not anchors:
        return 0.0
    ngram_lower = ngram.lower()
    exact_count = sum(1 for anchor in anchors.keys() if anchor.lower() == ngram_lower)
    return exact_count / len(anchors)


def compute_tfidf_approx(ngram: str, anchors: dict, org_anchors: dict) -> float:
    """Approximate TF-IDF based on anchor frequency in entity content."""
    ngram_lower = ngram.lower()
    tf = 0
    for anchor, count in anchors.items():
        if ngram_lower in anchor.lower():
            tf += count
    for anchor, count in org_anchors.items():
        if ngram_lower in anchor.lower():
            tf += count
    return np.log1p(tf)


def compute_distinct_refs(anchors: dict, org_anchors: dict) -> float:
    """DIST: count of distinct references linking to entity."""
    all_refs = set(anchors.keys()) | set(org_anchors.keys())
    return float(len(all_refs))


def string_similarity(s1: str, s2: str) -> float:
    """Jaccard similarity between two strings (word-level)."""
    s1, s2 = s1.lower(), s2.lower()
    if not s1 or not s2:
        return 0.0
    set1, set2 = set(s1.split()), set(s2.split())
    if not set1 or not set2:
        return 0.0
    return len(set1 & set2) / len(set1 | set2)


def extract_features(
    candidate: Candidate, 
    total_anchor_score: int,
    tweet_text: str = ""
) -> np.ndarray:
    """Extract all features for a candidate mention-entity pair.
    
    Args:
        candidate: The candidate containing n-gram and entity info
        total_anchor_score: Sum of anchor scores for all candidates from same n-gram
        tweet_text: Original tweet text for context-based features
    
    Returns:
        Feature vector with all 10+ features
    """
    # Get entity context if available
    ctx = candidate.entity_context
    tweet_tokens = tokenize(tweet_text or candidate.tweet_text)
    
    # 1. CMNS (Commonness)
    commonness = compute_commonness(candidate.anchor_score, total_anchor_score)
    
    # 2. String similarity (n-gram to page title)
    title_sim = string_similarity(candidate.ngram, candidate.page_title)
    
    # 3. N-gram length
    ngram_len = float(len(candidate.ngram.split()))
    
    # 4-6. T_OR: Redirect/Anchor type indicators
    is_redirect = 1.0 if candidate.anchor_type == 0 else 0.0
    is_anchor = 1.0 if candidate.anchor_type == 1 else 0.0
    is_both = 1.0 if candidate.anchor_type == 2 else 0.0
    
    # 7. rE: Page rank (logged)
    log_rank = np.log1p(candidate.page_rank * 1e9)
    
    # 8. vE: Page views (logged)
    log_views = np.log1p(candidate.page_views)
    
    # 9. Number of categories (logged)
    log_cats = np.log1p(candidate.num_categories)
    
    # 10. Number of anchors (logged)
    log_anchors = np.log1p(candidate.num_anchors)
    
    # Advanced features requiring EntityContext
    if ctx:
        # 11. LE Overlap
        le_overlap = compute_le_overlap(tweet_tokens, ctx.anchors)
        
        # 12. CE Overlap
        ce_overlap = compute_ce_overlap(tweet_tokens, ctx.categories)
        
        # 13. In Category
        in_category = compute_in_category(candidate.ngram, ctx.categories)
        
        # 14. Absorption
        absorption = compute_absorption(candidate.ngram, ctx.anchors)
        
        # 15. Purity
        purity = compute_purity(candidate.ngram, ctx.anchors)
        
        # 16. TF-IDF approximation
        tfidf = compute_tfidf_approx(candidate.ngram, ctx.anchors, ctx.org_anchors)
        
        # 17. DIST: Distinct references
        dist = compute_distinct_refs(ctx.anchors, ctx.org_anchors)
    else:
        le_overlap = 0.0
        ce_overlap = 0.0
        in_category = 0.0
        absorption = 0.0
        purity = 0.0
        tfidf = 0.0
        dist = 0.0

    return np.array([
        commonness,      # 0: CMNS
        title_sim,       # 1: Title similarity
        ngram_len,       # 2: N-gram length
        is_redirect,     # 3: T_OR redirect flag
        is_anchor,       # 4: T_OR anchor flag
        is_both,         # 5: T_OR both flag
        log_rank,        # 6: rE (PageRank)
        log_views,       # 7: vE (PageViews)
        log_cats,        # 8: Categories count
        log_anchors,     # 9: Anchors count
        le_overlap,      # 10: LE Overlap
        ce_overlap,      # 11: CE Overlap
        in_category,     # 12: In Category
        absorption,      # 13: Absorption
        purity,          # 14: Purity
        tfidf,           # 15: TF-IDF
        dist,            # 16: DIST
    ], dtype=np.float32)


FEATURE_NAMES = [
    'commonness', 'title_similarity', 'ngram_length',
    'is_redirect', 'is_anchor', 'is_both',
    'log_pagerank', 'log_pageviews', 'log_categories', 'log_anchors',
    'le_overlap', 'ce_overlap', 'in_category',
    'absorption', 'purity', 'tfidf', 'distinct_refs'
]
