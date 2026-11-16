"""Focused tests for Day 7 retrieval methods and evidence matching."""
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'starter'))

from baseline import hit
from day7_pipeline import BM25Index, reciprocal_rank_fusion, validate_benchmark


def chunk(chunk_id, text, page, story_id):
    return {
        'id': chunk_id,
        'source_id': 'doc',
        'source': 'doc.pdf',
        'page': page,
        'text': text,
        'metadata': {'story_id': story_id},
    }


class Day7PipelineTests(unittest.TestCase):
    def setUp(self):
        self.chunks = [
            chunk('a', 'Simone Biles is a gymnast and won medals.', 7, 1),
            chunk('b', 'Anna Devolld helps pollinators in gardens.', 13, 4),
            chunk('c', 'Simone Biles trains in gymnastics.', 8, 1),
        ]

    def test_bm25_retrieves_relevant_chunk_and_applies_filter(self):
        index = BM25Index(self.chunks)
        rows = index.search('What sport is Simone Biles known for?', top_k=3, filters={'story_id': 1})
        self.assertEqual(rows[0]['id'], 'a')
        self.assertTrue(all(row['metadata']['story_id'] == 1 for row in rows))

    def test_rrf_combines_rankings_without_duplicate_chunks(self):
        dense = [{'id': 'a', 'score': 0.9}, {'id': 'b', 'score': 0.8}]
        lexical = [{'id': 'b', 'score': 3.2}, {'id': 'c', 'score': 1.1}]
        rows = reciprocal_rank_fusion(dense, lexical, top_k=3, rrf_constant=1)
        self.assertEqual([row['id'] for row in rows], ['b', 'a', 'c'])
        self.assertEqual(len({row['id'] for row in rows}), 3)

    def test_evidence_matching_normalizes_line_breaks(self):
        row = chunk('a', 'He co-founded The Hollies with Allan Clarke.', 9, 2)
        gold = [{'source_id': 'doc', 'page': 9, 'evidence': 'co-founded The\nHollies'}]
        self.assertTrue(hit(row, gold))

    def test_benchmark_rejects_answerable_evidence_outside_corpus(self):
        questions = [{
            'id': 'q', 'split': 'dev', 'question': 'Question?', 'answerable': True,
            'gold': [{'source_id': 'doc', 'page': 1, 'evidence': 'not here'}],
        }]
        with self.assertRaises(ValueError):
            validate_benchmark(self.chunks, questions)


if __name__ == '__main__':
    unittest.main()
