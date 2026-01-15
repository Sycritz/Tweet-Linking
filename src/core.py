from typing import Optional, List, Tuple
from dataclasses import dataclass
import lmdb
import os

# Centralized settings
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

from src.proto import SerializedListNew_pb2
from src.proto import DictionaryWithTitle_pb2

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

    def close(self):
        self.env.close()
