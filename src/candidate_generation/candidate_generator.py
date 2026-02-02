"""Candidate generation module for entity linking.

Generates candidate mention-entity pairs from tweets using inverted index lookup.
"""

from src.core import InvertedIndex, PageContext, Candidate


def generate_candidates(
    tweet: str, 
    index: InvertedIndex, 
    context: PageContext, 
    top_k: int = 5,
    use_full_context: bool = True
) -> list[Candidate]:
    """Generate candidate mention-entity pairs for a tweet.
    
    Args:
        tweet: Raw tweet text
        index: Inverted index for anchor lookup
        context: Page context database
        top_k: Maximum candidates per n-gram
        use_full_context: If True, fetch full EntityContext for each candidate
    
    Returns:
        List of Candidate objects with entity information
    """
    from src.preprocessing.preprocessing import clean_tweet, get_ngrams
    
    cleaned = clean_tweet(tweet)
    ngrams = get_ngrams(cleaned)
    candidates = []

    for ngram in ngrams:
        postings = index.get(ngram)
        for page_id, score, typ in postings[:top_k]:
            if use_full_context:
                entity_ctx = context.get_full_context(page_id)
                if entity_ctx:
                    candidates.append(
                        Candidate(
                            ngram=ngram,
                            page_id=page_id,
                            anchor_score=score,
                            anchor_type=typ,
                            page_title=entity_ctx.page_title,
                            page_rank=entity_ctx.page_rank,
                            page_views=entity_ctx.page_views,
                            num_categories=len(entity_ctx.categories),
                            num_anchors=entity_ctx.length_anchors,
                            tweet_text=tweet,
                            entity_context=entity_ctx,
                        )
                    )
            else:
                page_info = context.get(page_id)
                if page_info:
                    title, rank, views, n_cats, n_anchors = page_info
                    candidates.append(
                        Candidate(
                            ngram=ngram,
                            page_id=page_id,
                            anchor_score=score,
                            anchor_type=typ,
                            page_title=title,
                            page_rank=rank,
                            page_views=views,
                            num_categories=n_cats,
                            num_anchors=n_anchors,
                            tweet_text=tweet,
                        )
                    )
    return candidates
