"""Local web UI for asking the Day 7 retrieval agent."""
import argparse
import json
import os
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from agent_config import load_agent_config
from agentic import run_agent
from day7_pipeline import (
    AgentRetriever,
    BM25Index,
    COLLECTION_NAME,
    DEFAULT_OUTPUT,
    EMBEDDING_MODEL,
    RetrievalSuite,
    load_json,
    save_json,
)
from openai_compatible import OpenAICompatibleController

WEB_DIR = Path(__file__).resolve().parent / 'web'
MAX_REQUEST_BYTES = 16_384


def create_handler(chunks, questions, index_dir, collection, embedding_model,
                   trace_path, top_k, budget, max_searches):
    class Day7Handler(BaseHTTPRequestHandler):
        server_version = 'Day7LocalUI/1.0'

        def _json(self, status, value):
            body = json.dumps(value, ensure_ascii=False).encode('utf-8')
            self.send_response(status)
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self.send_header('Content-Length', str(len(body)))
            self.send_header('Cache-Control', 'no-store')
            self.end_headers()
            self.wfile.write(body)

        def _body(self):
            raw_length = self.headers.get('Content-Length')
            if raw_length is None:
                raise ValueError('Content-Length is required')
            length = int(raw_length)
            if length < 1 or length > MAX_REQUEST_BYTES:
                raise ValueError('request body must be 1..16384 bytes')
            try:
                return json.loads(self.rfile.read(length).decode('utf-8'))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ValueError('request body must be valid UTF-8 JSON') from exc

        def _trace_error(self, request_path, question, method, error, started):
            if request_path != '/api/ask':
                return
            trace_path.parent.mkdir(parents=True, exist_ok=True)
            trace = {
                'mode': 'agent',
                'question': question,
                'method': method,
                'status': 'error',
                'error': {'type': type(error).__name__, 'message': str(error)},
                'elapsed_ms': (time.perf_counter() - started) * 1000,
            }
            with trace_path.open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(trace, ensure_ascii=False) + '\n')

        def do_GET(self):
            parsed = urlparse(self.path)
            if parsed.path == '/':
                body = (WEB_DIR / 'index.html').read_bytes()
                self.send_response(200)
                self.send_header('Content-Type', 'text/html; charset=utf-8')
                self.send_header('Content-Length', str(len(body)))
                self.send_header('Cache-Control', 'no-store')
                self.end_headers()
                self.wfile.write(body)
                return
            if parsed.path == '/api/health':
                self._json(200, {
                    'corpus_ready': bool(chunks),
                    'question_count': len(questions),
                    'chat_model_ready': bool(
                        os.environ.get('OPENAI_API_KEY') and os.environ.get('OPENAI_MODEL')
                    ),
                    'chat_model': os.environ.get('OPENAI_MODEL'),
                    'embedding_model': embedding_model,
                    'index_dir': str(index_dir),
                    'index_exists': Path(index_dir).exists(),
                })
                return
            if parsed.path == '/api/questions':
                query = parse_qs(parsed.query)
                split = query.get('split', ['dev'])[0]
                if split not in ('dev', 'test'):
                    self._json(400, {'error': 'split must be dev or test'})
                    return
                self._json(200, [
                    {'id': q['id'], 'split': q['split'], 'question': q['question'],
                     'answerable': q['answerable']}
                    for q in questions if q['split'] == split
                ])
                return
            self._json(404, {'error': 'not found'})

        def do_POST(self):
            parsed = urlparse(self.path)
            if parsed.path not in ('/api/search', '/api/ask'):
                self._json(404, {'error': 'not found'})
                return
            question = None
            method = None
            started = time.perf_counter()
            try:
                request = self._body()
                question = request.get('question')
                method = request.get('method', 'bm25')
                filters = request.get('filters') or {}
                if not isinstance(question, str) or not question.strip():
                    raise ValueError('question must be a non-empty string')
                if method not in ('bm25', 'dense', 'hybrid'):
                    raise ValueError('method must be bm25, dense, or hybrid')
                if not isinstance(filters, dict):
                    raise ValueError('filters must be a JSON object')
                started = time.perf_counter()
                controller = OpenAICompatibleController() if parsed.path == '/api/ask' else None
                suite = RetrievalSuite(
                    chunks, index_dir, [method], embedding_model, collection
                )
                if parsed.path == '/api/search':
                    results = suite.search(method, question.strip(), filters, top_k)
                    response = {
                        'mode': 'retrieval_only',
                        'method': method,
                        'question': question.strip(),
                        'results': results,
                        'elapsed_ms': (time.perf_counter() - started) * 1000,
                    }
                    self._json(200, response)
                    return

                result = run_agent(
                    question.strip(),
                    controller,
                    AgentRetriever(suite, method),
                    filters=filters,
                    max_searches=max_searches,
                    top_k=top_k,
                    budget=budget,
                )
                response = {
                    'mode': 'agent',
                    'method': method,
                    'question': question.strip(),
                    **result,
                    'elapsed_ms': (time.perf_counter() - started) * 1000,
                }
                trace_path.parent.mkdir(parents=True, exist_ok=True)
                with trace_path.open('a', encoding='utf-8') as stream:
                    stream.write(json.dumps(response, ensure_ascii=False) + '\n')
                self._json(200, response)
            except RuntimeError as exc:
                self._trace_error(parsed.path, question, method, exc, started)
                self._json(503, {'error': str(exc)})
            except ValueError as exc:
                self._trace_error(parsed.path, question, method, exc, started)
                self._json(400, {'error': str(exc)})
            except Exception as exc:
                self._trace_error(parsed.path, question, method, exc, started)
                self._json(502, {'error': f'{type(exc).__name__}: {exc}'})

        def log_message(self, fmt, *args):
            print('%s - %s' % (self.address_string(), fmt % args))

    return Day7Handler


def main():
    load_agent_config()
    parser = argparse.ArgumentParser(description='Serve the local Day 7 agent UI')
    parser.add_argument('--host', default='127.0.0.1', choices=('127.0.0.1', 'localhost'))
    parser.add_argument('--port', type=int, default=8765)
    parser.add_argument('--chunks', type=Path, default=DEFAULT_OUTPUT / 'chunks.json')
    parser.add_argument('--questions', type=Path, default=DEFAULT_OUTPUT / 'questions.json')
    parser.add_argument('--index-dir', type=Path, default=DEFAULT_OUTPUT / 'chroma')
    parser.add_argument('--collection', default=COLLECTION_NAME)
    parser.add_argument('--embedding-model', default=EMBEDDING_MODEL)
    parser.add_argument('--trace', type=Path, default=DEFAULT_OUTPUT / 'ui-traces.jsonl')
    parser.add_argument('--top-k', type=int, default=3)
    parser.add_argument('--budget', type=int, default=1200)
    parser.add_argument('--max-searches', type=int, default=3)
    args = parser.parse_args()
    if not 1024 <= args.port <= 65535:
        raise ValueError('port must be between 1024 and 65535')
    if args.top_k < 1 or args.budget < 1 or args.max_searches < 1:
        raise ValueError('top-k, budget, and max-searches must be positive')
    chunks = load_json(args.chunks)
    questions = load_json(args.questions)
    handler = create_handler(
        chunks, questions, args.index_dir, args.collection, args.embedding_model,
        args.trace, args.top_k, args.budget, args.max_searches,
    )
    server = ThreadingHTTPServer((args.host, args.port), handler)
    print(f'Day 7 UI ready: http://127.0.0.1:{args.port}')
    print('Press Ctrl+C to stop.')
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print('\nStopping Day 7 UI.')
    finally:
        server.server_close()


if __name__ == '__main__':
    main()
