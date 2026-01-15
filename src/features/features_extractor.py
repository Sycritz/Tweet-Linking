import numpy as np
from src.core import Candidate
from src.utils.utils import compute_commonness, string_similarity

# Basic features for Now, TODO: implement all features listed in the paper
def extract_features(candidate: Candidate, total_anchor_score: int) -> np.ndarray:
    commonness = compute_commonness(candidate.anchor_score, total_anchor_score)
    sim = string_similarity(candidate.ngram, candidate.page_title)
    ngram_len = len(candidate.ngram.split())
    is_redirect = 1.0 if candidate.anchor_type == 0 else 0.0
    is_anchor = 1.0 if candidate.anchor_type == 1 else 0.0
    is_both = 1.0 if candidate.anchor_type == 2 else 0.0
    log_rank = np.log1p(candidate.page_rank * 1e9)
    log_views = np.log1p(candidate.page_views)
    log_cats = np.log1p(candidate.num_categories)
    log_anchors = np.log1p(candidate.num_anchors)

    return np.array([
        commonness,
        sim,
        ngram_len,
        is_redirect,
        is_anchor,
        is_both,
        log_rank,
        log_views,
        log_cats,
        log_anchors
    ], dtype=np.float32)

