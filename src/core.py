from typing import Optional, List, Tuple, Dict
from dataclasses import dataclass, field
import lmdb
import os

# Centralized settings
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

from src.proto import SerializedListNew_pb2
from src.proto import DictionaryWithTitle_pb2


@dataclass
class EntityContext:
    page_title: str
    page_rank: float
    page_views: float
    anchors: Dict[str, int] = field(default_factory=dict)
    org_anchors: Dict[str, int] = field(default_factory=dict)
    categories: Dict[str, int] = field(default_factory=dict)
    called_pages: Dict[str, int] = field(default_factory=dict)
    length_anchors: int = 0


@dataclass
class Candidate:
    ngram: str
    page_id: str
    anchor_score: int
    anchor_type: int
    page_title: str
    page_rank: float
    page_views: float
    num_categories: int
    num_anchors: int
    tweet_text: str = ""
    ngram_position: int = 0
    entity_context: Optional[EntityContext] = None


class InvertedIndex:
    def __init__(self, path: str):
        self.env = lmdb.open(path, readonly=True, lock=False)

    def get(self, ngram: str) -> List[Tuple[str, int, int]]:
        results = []
        with self.env.begin() as txn:
            val = txn.get(ngram.encode('utf-8'))
            if val:
                posting = SerializedListNew_pb2.SerializedListNew()
                posting.ParseFromString(val)
                for el in posting.Elements:
                    results.append((el.docId, el.score, el.typ))
        return results

    def close(self):
        self.env.close()


class PageContext:
    def __init__(self, path: str):
        self.env = lmdb.open(path, readonly=True, lock=False)

    def get(self, page_id: str) -> Optional[Tuple[str, float, float, int, int]]:
        with self.env.begin() as txn:
            val = txn.get(page_id.encode('utf-8'))
            if val:
                dico = DictionaryWithTitle_pb2.Dico()
                dico.ParseFromString(val)
                return (
                    dico.PageTitle,
                    dico.PageRank,
                    dico.PageViews,
                    len(dico.Categories),
                    dico.length_anchors
                )
        return None

    def get_full_context(self, page_id: str) -> Optional[EntityContext]:
        with self.env.begin() as txn:
            val = txn.get(page_id.encode('utf-8'))
            if val:
                dico = DictionaryWithTitle_pb2.Dico()
                dico.ParseFromString(val)
                anchors = {entry.key: entry.value for entry in dico.Anchors}
                org_anchors = {entry.key: entry.value for entry in dico.OrgAnchors}
                categories = {entry.key: entry.value for entry in dico.Categories}
                called_pages = {entry.key: entry.value for entry in dico.CalledPages}
                return EntityContext(
                    page_title=dico.PageTitle,
                    page_rank=dico.PageRank,
                    page_views=dico.PageViews,
                    anchors=anchors,
                    org_anchors=org_anchors,
                    categories=categories,
                    called_pages=called_pages,
                    length_anchors=dico.length_anchors
                )
        return None

    def close(self):
        self.env.close()
