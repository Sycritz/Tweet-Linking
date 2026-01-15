import os, sys

sys.path.append(os.path.join(os.path.dirname(__file__), "Provided-Resources"))
import SerializedListNew_pb2
import DictionaryWithTitle_pb2

from src.utils import InvertedIndex, PageContext, Candidate
from src.preprocessing.preprocessing import clean_tweet, get_ngrams


def generate_candidates(
    tweet: str, index: InvertedIndex, context: PageContext, top_k: int = 5
) -> list[Candidate]:
    cleaned = clean_tweet(tweet)
    ngrams = get_ngrams(cleaned)
    candidates = []

    for ngram in ngrams:
        postings = index.get(ngram)
        for page_id, score, typ in postings[:top_k]:
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
                    )
                )
    return candidates
