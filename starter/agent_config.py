"""Load local-only Day 7 model configuration from starter/.env."""
import os
from pathlib import Path


DEFAULT_ENV_FILE = Path(__file__).resolve().parent / '.env'
ALLOWED_KEYS = {'OPENAI_BASE_URL', 'OPENAI_MODEL', 'OPENAI_API_KEY'}


def load_agent_config(env_file=DEFAULT_ENV_FILE):
    env_file = Path(env_file)
    if not env_file.is_file():
        return False
    for line_number, raw_line in enumerate(env_file.read_text(encoding='utf-8').splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith('#'):
            continue
        key, separator, value = line.partition('=')
        key, value = key.strip(), value.strip()
        if not separator or key not in ALLOWED_KEYS:
            raise ValueError(f'{env_file}:{line_number}: cấu hình .env không hợp lệ')
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ('"', "'"):
            value = value[1:-1]
        elif value.startswith(("'", '"')) or value.endswith(("'", '"')):
            raise ValueError(f'{env_file}:{line_number}: dấu quote không khớp')
        os.environ.setdefault(key, value)
    return True
