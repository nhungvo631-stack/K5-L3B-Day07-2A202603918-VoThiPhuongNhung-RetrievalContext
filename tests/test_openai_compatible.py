"""Verify tool-call normalization without contacting a model provider."""
import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'starter'))

from openai_compatible import OpenAICompatibleController


class FakeResponse:
    def __init__(self, value):
        self.payload = json.dumps(value).encode('utf-8')

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False

    def read(self):
        return self.payload


class OpenAICompatibleTests(unittest.TestCase):
    def test_real_model_search_tool_call_is_normalized_and_traced(self):
        payload = {
            'model': 'test-tool-model',
            'choices': [{
                'message': {
                    'tool_calls': [{
                        'function': {
                            'name': 'search',
                            'arguments': '{"query":"gymnastics"}',
                        },
                    }],
                },
            }],
            'usage': {'prompt_tokens': 10, 'completion_tokens': 3, 'total_tokens': 13},
        }
        controller = OpenAICompatibleController(
            base_url='https://example.invalid/v1',
            api_key='local-test-key',
            model='test-tool-model',
        )
        with patch('openai_compatible.urllib.request.urlopen', return_value=FakeResponse(payload)) as mocked:
            action = controller.decide('Which sport?', [])

        self.assertEqual(action, {'action': 'search', 'query': 'gymnastics'})
        request = mocked.call_args.args[0]
        self.assertEqual(request.full_url, 'https://example.invalid/v1/chat/completions')
        sent = json.loads(request.data)
        self.assertEqual(sent['tools'][0]['function']['name'], 'search')
        self.assertEqual(controller.last_call['tool_call']['arguments']['query'], 'gymnastics')
        self.assertNotIn('local-test-key', json.dumps(controller.last_call))

    def test_final_answer_tool_call_preserves_citations(self):
        payload = {
            'choices': [{
                'message': {
                    'tool_calls': [{
                        'function': {
                            'name': 'submit_answer',
                            'arguments': '{"answer":"Gymnastics [c1]","citations":["c1"]}',
                        },
                    }],
                },
            }],
        }
        controller = OpenAICompatibleController(
            api_key='local-test-key', model='test-tool-model'
        )
        with patch('openai_compatible.urllib.request.urlopen', return_value=FakeResponse(payload)):
            action = controller.decide('Which sport?', [])
        self.assertEqual(
            action,
            {'action': 'answer', 'answer': 'Gymnastics [c1]', 'citations': ['c1']},
        )


if __name__ == '__main__':
    unittest.main()
