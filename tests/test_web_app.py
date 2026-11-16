"""HTTP smoke tests for the local Day 7 question and retrieval UI."""
import json
import os
import sys
import tempfile
import threading
import unittest
from http.server import ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch
from urllib.request import Request, urlopen
from urllib.error import HTTPError

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'starter'))

from web_app import create_handler


class WebAppTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        chunks = [{
            'id': 'c1',
            'source_id': 'doc',
            'source': 'doc.pdf',
            'page': 7,
            'text': 'Simone Biles is a gymnast.',
            'metadata': {'story_id': 1},
        }]
        questions = [{
            'id': 'q1',
            'split': 'test',
            'question': 'Which sport is Simone Biles known for?',
            'answerable': True,
            'gold': [{'source_id': 'doc', 'page': 7, 'evidence': 'a gymnast'}],
        }]
        handler = create_handler(
            chunks, questions, root / 'chroma', 'test_collection', 'test-model',
            root / 'traces.jsonl', 3, 1200, 3,
        )
        self.server = ThreadingHTTPServer(('127.0.0.1', 0), handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.base = f'http://127.0.0.1:{self.server.server_port}'

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)
        self.temp.cleanup()

    def test_ui_loads_and_questions_hide_gold(self):
        with urlopen(self.base + '/') as response:
            page = response.read().decode('utf-8')
        self.assertIn('Hỏi agent thật', page)
        self.assertIn('Chỉ tìm bằng chứng', page)

        with urlopen(self.base + '/api/questions?split=test') as response:
            questions = json.loads(response.read())
        self.assertEqual(questions[0]['question'], 'Which sport is Simone Biles known for?')
        self.assertNotIn('gold', questions[0])

    def test_retrieval_only_endpoint_returns_ranked_evidence(self):
        request = Request(
            self.base + '/api/search',
            data=json.dumps({
                'question': 'Which sport is Simone Biles known for?',
                'method': 'bm25',
            }).encode('utf-8'),
            headers={'Content-Type': 'application/json'},
            method='POST',
        )
        with urlopen(request) as response:
            payload = json.loads(response.read())
        self.assertEqual(payload['mode'], 'retrieval_only')
        self.assertEqual(payload['results'][0]['id'], 'c1')
        self.assertEqual(payload['results'][0]['page'], 7)

    def test_agent_endpoint_reports_missing_model_configuration(self):
        request = Request(
            self.base + '/api/ask',
            data=json.dumps({
                'question': 'Which sport is Simone Biles known for?',
                'method': 'bm25',
            }).encode('utf-8'),
            headers={'Content-Type': 'application/json'},
            method='POST',
        )
        with patch.dict(os.environ, {'OPENAI_API_KEY': '', 'OPENAI_MODEL': ''}):
            with self.assertRaises(HTTPError) as raised:
                urlopen(request)
        self.assertEqual(raised.exception.code, 503)
        payload = json.loads(raised.exception.read())
        self.assertIn('OPENAI_API_KEY', payload['error'])


if __name__ == '__main__':
    unittest.main()
