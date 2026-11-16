"""Run the real tool-calling agent and save raw traces without gold labels."""
import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from agent_config import load_agent_config
from agentic import run_agent
from day7_pipeline import (
    AgentRetriever, COLLECTION_NAME, DEFAULT_OUTPUT, EMBEDDING_MODEL, RetrievalSuite,
    load_json, save_json,
)
from openai_compatible import OpenAICompatibleController


def main():
    load_agent_config()
    parser = argparse.ArgumentParser(description='Run Day 7 agent with a real OpenAI-compatible model')
    parser.add_argument('--chunks', type=Path, default=DEFAULT_OUTPUT / 'chunks.json')
    parser.add_argument('--questions', type=Path, default=DEFAULT_OUTPUT / 'questions.json')
    parser.add_argument('--index-dir', type=Path, default=DEFAULT_OUTPUT / 'chroma')
    parser.add_argument('--collection', default=COLLECTION_NAME)
    parser.add_argument('--embedding-model', default=EMBEDDING_MODEL)
    parser.add_argument('--method', choices=('dense', 'bm25', 'hybrid'), default='dense')
    parser.add_argument('--split', choices=('dev', 'test'), required=True)
    parser.add_argument('--top-k', type=int, default=3)
    parser.add_argument('--budget', type=int, default=1200)
    parser.add_argument('--max-searches', type=int, default=3)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.top_k < 1 or args.budget < 1 or args.max_searches < 1:
        raise ValueError('top-k, budget và max-searches phải > 0')

    questions = [
        row for row in load_json(args.questions) if row['split'] == args.split
    ]
    chunks = load_json(args.chunks)
    suite = RetrievalSuite(
        chunks, args.index_dir, [args.method], args.embedding_model, args.collection
    )
    base_url = os.environ.get('OPENAI_BASE_URL')
    controller = OpenAICompatibleController(base_url=base_url)
    cases = []
    traces = []
    for question in questions:
        cases.append({
            'id': question['id'],
            'split': question['split'],
            'answerable': question['answerable'],
        })
        started = time.perf_counter()
        controller.last_call = None
        try:
            result = run_agent(
                question['question'],
                controller,
                AgentRetriever(suite, args.method),
                filters=question.get('filter'),
                max_searches=args.max_searches,
                top_k=args.top_k,
                budget=args.budget,
            )
            trace = {
                'query_id': question['id'],
                'split': question['split'],
                'question': question['question'],
                **result,
                'method': args.method,
                'elapsed_ms': (time.perf_counter() - started) * 1000,
            }
        except Exception as exc:
            trace = {
                'query_id': question['id'],
                'split': question['split'],
                'question': question['question'],
                'status': 'error',
                'answer': None,
                'citations': [],
                'events': [],
                'model_calls': [controller.last_call] if controller.last_call else [],
                'method': args.method,
                'elapsed_ms': (time.perf_counter() - started) * 1000,
                'error': {'type': type(exc).__name__, 'message': str(exc)},
            }
        traces.append(trace)
        print(json.dumps({
            'query_id': question['id'],
            'status': trace['status'],
            'elapsed_ms': round(trace['elapsed_ms'], 2),
        }, ensure_ascii=False))

    args.output.mkdir(parents=True, exist_ok=True)
    save_json(args.output / 'cases.json', cases)
    save_json(args.output / 'traces.json', traces)


if __name__ == '__main__':
    main()
