"""Verify local model configuration loading without storing credentials."""
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'starter'))

from agent_config import load_agent_config


class AgentConfigTests(unittest.TestCase):
    def test_loads_env_file_without_overriding_existing_environment(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            env_file = Path(temp_dir) / '.env'
            env_file.write_text(
                'OPENAI_MODEL=example-model\nOPENAI_API_KEY=local-test-key\n',
                encoding='utf-8',
            )
            with patch.dict(os.environ, {'OPENAI_MODEL': 'process-model'}, clear=True):
                self.assertTrue(load_agent_config(env_file))
                self.assertEqual(os.environ['OPENAI_MODEL'], 'process-model')
                self.assertEqual(os.environ['OPENAI_API_KEY'], 'local-test-key')


if __name__ == '__main__':
    unittest.main()
