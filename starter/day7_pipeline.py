"""Prepare the Day 7 corpus and compare lexical, dense, and hybrid retrieval."""
import argparse
from functools import lru_cache
import json
import math
import os
import re
import statistics
import time
import unicodedata
from collections import Counter
from pathlib import Path

from pypdf import PdfReader

try:
    from .baseline import TfidfIndex, hit, matches_filter, normalized_text, pack_context, tokenize
except ImportError:
    from baseline import TfidfIndex, hit, matches_filter, normalized_text, pack_context, tokenize


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PDF = ROOT / 'data' / 'FBL_PassItOn_Stories_Volume_2.pdf'
DEFAULT_MANIFEST = ROOT / 'data' / 'manifest.json'
DEFAULT_QUESTIONS = ROOT / 'data' / 'questions.json'
DEFAULT_ABSENCE_QUESTIONS = ROOT / 'data' / 'day7_absence_questions.json'
DEFAULT_OUTPUT = ROOT / 'starter' / 'results' / 'day7'
EMBEDDING_MODEL = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
COLLECTION_NAME = 'day6_strategy_a_rebuilt'


def load_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def load_questions(answerable_path=DEFAULT_QUESTIONS, absence_path=DEFAULT_ABSENCE_QUESTIONS):
    answerable = []
    for raw in load_json(answerable_path):
        answerable.append({
            'id': raw['id'],
            'split': raw['split'],
            'question': raw['question'],
            'expected_answer': raw['expected_answer'],
            'answerable': True,
            'gold': [{
                'source_id': raw['gold_source_id'],
                'page': raw['gold_page'],
                'evidence': raw['evidence_quote'],
            }],
            **({'filter': raw['filter']} if raw.get('filter') else {}),
        })
    unanswerable = load_json(absence_path)
    questions = answerable + unanswerable
    ids = [question['id'] for question in questions]
    if len(ids) != len(set(ids)):
        raise ValueError('question ID trùng')
    for split in ('dev', 'test'):
        selected = [q for q in questions if q['split'] == split]
        answerable_count = sum(q['answerable'] for q in selected)
        absence_count = len(selected) - answerable_count
        if answerable_count < (15 if split == 'dev' else 5) or absence_count < 2:
            raise ValueError(f'{split}: benchmark cần đủ câu có đáp án và câu thiếu evidence')
    return questions


def extract_chunks(pdf_path=DEFAULT_PDF, manifest_path=DEFAULT_MANIFEST,
                   chunk_size=500, overlap=50):
    if chunk_size < 1 or overlap < 0 or overlap >= chunk_size:
        raise ValueError('chunk_size/overlap không hợp lệ')
    manifest = load_json(manifest_path)
    reader = PdfReader(str(pdf_path))
    content_pages = manifest.get('content_pages')
    if not isinstance(content_pages, list) or not content_pages:
        raise ValueError('manifest thiếu content_pages')
    if len(reader.pages) != manifest['snapshot']['physical_page_count']:
        raise ValueError('số trang PDF không khớp manifest')

    source_id = manifest['source_id']
    source = manifest.get('source_url', manifest['title'])
    chunks = []
    stride = chunk_size - overlap
    for page_info in content_pages:
        page = page_info['page']
        text = normalized_text(reader.pages[page - 1].extract_text() or '')
        if not text:
            raise ValueError(f'không trích xuất được text ở trang {page}')
        start = 0
        index = 0
        while start < len(text):
            part = text[start:start + chunk_size]
            chunks.append({
                'id': f'{source_id}:p{page}:c{index}',
                'source_id': source_id,
                'source': source,
                'page': page,
                'text': part,
                'metadata': {
                    'document_id': source_id,
                    'story_id': page_info['story_id'],
                    'story_title': page_info['story_title'],
                    'chunk_index': index,
                },
            })
            if start + chunk_size >= len(text):
                break
            start += stride
            index += 1
    if len({chunk['id'] for chunk in chunks}) != len(chunks):
        raise ValueError('chunk ID trùng')
    return chunks


def validate_benchmark(chunks, questions):
    if not chunks:
        raise ValueError('corpus rỗng')
    for question in questions:
        if question.get('split') not in ('dev', 'test') or not question.get('question', '').strip():
            raise ValueError(f"{question.get('id')}: question/split không hợp lệ")
        if question.get('answerable') is True:
            if not question.get('gold'):
                raise ValueError(f"{question['id']}: câu có đáp án cần gold")
            for evidence in question['gold']:
                if not any(hit(chunk, [evidence]) for chunk in chunks):
                    raise ValueError(f"{question['id']}: evidence không khớp chunk đúng nguồn/trang")
        elif question.get('answerable') is not False or not question.get('absence_reason'):
            raise ValueError(f"{question['id']}: câu thiếu evidence cần absence_reason")


def prepare(args):
    chunks = extract_chunks(args.pdf, args.manifest, args.chunk_size, args.overlap)
    questions = load_questions(args.questions, args.absence_questions)
    validate_benchmark(chunks, questions)
    output = Path(args.output)
    save_json(output / 'chunks.json', chunks)
    save_json(output / 'questions.json', questions)
    counts = {
        split: {
            'answerable': sum(q['split'] == split and q['answerable'] for q in questions),
            'unanswerable': sum(q['split'] == split and not q['answerable'] for q in questions),
        }
        for split in ('dev', 'test')
    }
    summary = {
        'source_id': load_json(args.manifest)['source_id'],
        'chunk_count': len(chunks),
        'chunk_size_chars': args.chunk_size,
        'overlap_chars': args.overlap,
        'benchmark_counts': counts,
    }
    save_json(output / 'prepare_summary.json', summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


@lru_cache(maxsize=2)
def load_encoder(model_name=EMBEDDING_MODEL):
    os.environ.setdefault('HF_HUB_OFFLINE', '1')
    os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError(
            'Thiếu sentence-transformers; cài starter/requirements-day7.txt trong Python environment đang chạy.'
        ) from exc
    return SentenceTransformer(model_name, device='cpu')


def open_chroma(index_dir, collection_name=COLLECTION_NAME):
    try:
        import chromadb
    except ImportError as exc:
        raise RuntimeError(
            'Thiếu chromadb; cài starter/requirements-day7.txt trong Python environment đang chạy.'
        ) from exc
    client = chromadb.PersistentClient(path=str(index_dir))
    return client, collection_name


def build_index(chunks, index_dir, model_name=EMBEDDING_MODEL, collection_name=COLLECTION_NAME):
    client, collection_name = open_chroma(index_dir, collection_name)
    metadata = {'hnsw:space': 'cosine', 'embedding_model': model_name}
    collection = client.get_or_create_collection(
        name=collection_name, metadata=metadata, embedding_function=None
    )
    if collection.metadata.get('embedding_model') != model_name:
        raise ValueError('embedding model của collection không khớp')
    if collection.count():
        existing_ids = set(collection.get(include=['metadatas'])['ids'])
        expected_ids = {chunk['id'] for chunk in chunks}
        if existing_ids != expected_ids:
            raise ValueError(
                'index đã có dữ liệu corpus khác; chọn một --index-dir mới, không xóa index hiện có'
            )
        return collection.count()

    encoder = load_encoder(model_name)
    batch_size = 64
    for offset in range(0, len(chunks), batch_size):
        batch = chunks[offset:offset + batch_size]
        vectors = encoder.encode(
            [chunk['text'] for chunk in batch],
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        collection.add(
            ids=[chunk['id'] for chunk in batch],
            documents=[chunk['text'] for chunk in batch],
            metadatas=[{
                'source_id': chunk['source_id'],
                'source': chunk['source'],
                'page': chunk['page'],
                **chunk['metadata'],
            } for chunk in batch],
            embeddings=vectors.tolist(),
        )
    if collection.count() != len(chunks):
        raise RuntimeError('số chunk trong Chroma không khớp corpus')
    return collection.count()


class BM25Index:
    def __init__(self, chunks, k1=1.5, b=0.75):
        self.chunks = chunks
        self.k1 = k1
        self.b = b
        self.tokens = [tokenize(chunk['text']) for chunk in chunks]
        self.lengths = [len(row) for row in self.tokens]
        self.average_length = sum(self.lengths) / max(1, len(self.lengths))
        document_frequency = Counter(
            token for row in self.tokens for token in set(row)
        )
        self.idf = {
            token: math.log(1 + (len(chunks) - frequency + 0.5) / (frequency + 0.5))
            for token, frequency in document_frequency.items()
        }
        self.counts = [Counter(row) for row in self.tokens]

    def search(self, question, top_k=3, filters=None):
        query = set(tokenize(question))
        results = []
        for chunk, counts, length in zip(self.chunks, self.counts, self.lengths):
            if not matches_filter(chunk, filters):
                continue
            score = 0.0
            for token in query:
                frequency = counts.get(token, 0)
                if not frequency:
                    continue
                denominator = frequency + self.k1 * (
                    1 - self.b + self.b * length / (self.average_length or 1)
                )
                score += self.idf.get(token, 0) * frequency * (self.k1 + 1) / denominator
            if score > 0:
                results.append({**chunk, 'score': score, 'strategy': 'bm25'})
        return sorted(results, key=lambda row: (-row['score'], row['id']))[:top_k]


class DenseIndex:
    def __init__(self, chunks, index_dir, encoder, collection_name=COLLECTION_NAME):
        client, collection_name = open_chroma(index_dir, collection_name)
        self.collection = client.get_collection(collection_name, embedding_function=None)
        if self.collection.count() != len(chunks):
            raise ValueError('Chroma index không khớp số chunk của corpus')
        self.chunks = {chunk['id']: chunk for chunk in chunks}
        self.encoder = encoder

    def search(self, question, top_k=3, filters=None):
        query = self.encoder.encode(
            [question], convert_to_numpy=True, normalize_embeddings=True,
            show_progress_bar=False,
        )[0].tolist()
        where = None
        if filters:
            clauses = [{key: {'$eq': value}} for key, value in filters.items()]
            where = clauses[0] if len(clauses) == 1 else {'$and': clauses}
        result = self.collection.query(
            query_embeddings=[query],
            n_results=top_k,
            where=where,
            include=['distances'],
        )
        rows = []
        for chunk_id, distance in zip(result['ids'][0], result['distances'][0]):
            chunk = self.chunks[chunk_id]
            rows.append({**chunk, 'score': 1 - distance, 'strategy': 'dense'})
        return rows


def reciprocal_rank_fusion(dense_rows, lexical_rows, top_k=3, rrf_constant=60):
    fused = {}
    for ranked in (dense_rows, lexical_rows):
        for rank, row in enumerate(ranked, start=1):
            entry = fused.setdefault(row['id'], {'chunk': row, 'score': 0.0})
            entry['score'] += 1 / (rrf_constant + rank)
    rows = [
        {**entry['chunk'], 'score': entry['score'], 'strategy': 'hybrid_rrf'}
        for entry in fused.values()
    ]
    return sorted(rows, key=lambda row: (-row['score'], row['id']))[:top_k]


class RetrievalSuite:
    def __init__(self, chunks, index_dir, methods, model_name=EMBEDDING_MODEL,
                 collection_name=COLLECTION_NAME, rrf_constant=60):
        self.chunks = chunks
        self.tfidf = TfidfIndex(chunks) if 'tfidf' in methods else None
        self.bm25 = BM25Index(chunks) if any(m in methods for m in ('bm25', 'hybrid')) else None
        self.dense = None
        if any(m in methods for m in ('dense', 'hybrid')):
            self.dense = DenseIndex(
                chunks, index_dir, load_encoder(model_name), collection_name
            )
        self.rrf_constant = rrf_constant

    def search(self, method, question, filters=None, top_k=3):
        if method == 'tfidf':
            return self.tfidf.search(question, top_k, filters)
        if method == 'bm25':
            return self.bm25.search(question, top_k, filters)
        if method == 'dense':
            return self.dense.search(question, top_k, filters)
        if method == 'hybrid':
            candidate_k = max(top_k, 10)
            dense_rows = self.dense.search(question, candidate_k, filters)
            lexical_rows = self.bm25.search(question, candidate_k, filters)
            return reciprocal_rank_fusion(
                dense_rows, lexical_rows, top_k, self.rrf_constant
            )
        raise ValueError(f'phương pháp retrieval không hợp lệ: {method}')


class AgentRetriever:
    def __init__(self, suite, method):
        self.suite = suite
        self.method = method

    def search(self, query, *, filters=None, top_k=3):
        return self.suite.search(self.method, query, filters, top_k)


def evaluate(suite, questions, method, split, top_k=3, budget=800):
    selected = [q for q in questions if q['split'] == split and q['answerable']]
    traces = []
    for question in selected:
        started = time.perf_counter()
        filters = question.get('filter')
        results = suite.search(method, question['question'], filters, top_k)
        context, context_ids = pack_context(results, budget)
        annotated = [{**row, 'evidence_hit': hit(row, question['gold'])} for row in results]
        traces.append({
            'query_id': question['id'],
            'split': split,
            'question': question['question'],
            'method': method,
            'top_k': top_k,
            'filters': filters or {},
            'budget_chars': budget,
            'results': annotated,
            'context': context,
            'context_ids': context_ids,
            'context_chars': len(context),
            'elapsed_ms': (time.perf_counter() - started) * 1000,
            'hit_at_1': bool(annotated and annotated[0]['evidence_hit']),
            'evidence_hit_at_3': any(row['evidence_hit'] for row in annotated[:3]),
            'context_evidence': any(
                row['evidence_hit'] and row['id'] in context_ids for row in annotated
            ),
        })
    n = len(traces)
    summary = {
        'method': method,
        'split': split,
        'answerable_n': n,
        'hit_at_1': {
            'n': sum(trace['hit_at_1'] for trace in traces),
            'N': n,
            'rate': sum(trace['hit_at_1'] for trace in traces) / n if n else None,
        },
        'evidence_hit_at_3': {
            'n': sum(trace['evidence_hit_at_3'] for trace in traces),
            'N': n,
            'rate': sum(trace['evidence_hit_at_3'] for trace in traces) / n if n else None,
        },
        'context_evidence_rate': {
            'n': sum(trace['context_evidence'] for trace in traces),
            'N': n,
            'rate': sum(trace['context_evidence'] for trace in traces) / n if n else None,
        },
        'mean_context_chars': statistics.mean(t['context_chars'] for t in traces) if traces else 0,
        'median_query_ms': statistics.median(t['elapsed_ms'] for t in traces) if traces else 0,
        'embedding_cost_usd': 0,
        'llm_cost_usd': None,
    }
    return summary, traces


def main():
    parser = argparse.ArgumentParser(description='Day 7 retrieval pipeline')
    subparsers = parser.add_subparsers(dest='command', required=True)

    prepare_parser = subparsers.add_parser('prepare', help='trích chunk và kiểm tra benchmark')
    prepare_parser.add_argument('--pdf', type=Path, default=DEFAULT_PDF)
    prepare_parser.add_argument('--manifest', type=Path, default=DEFAULT_MANIFEST)
    prepare_parser.add_argument('--questions', type=Path, default=DEFAULT_QUESTIONS)
    prepare_parser.add_argument('--absence-questions', type=Path, default=DEFAULT_ABSENCE_QUESTIONS)
    prepare_parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    prepare_parser.add_argument('--chunk-size', type=int, default=500)
    prepare_parser.add_argument('--overlap', type=int, default=50)

    index_parser = subparsers.add_parser('index', help='xây Chroma index mới, không sửa index Day 6')
    index_parser.add_argument('--chunks', type=Path, default=DEFAULT_OUTPUT / 'chunks.json')
    index_parser.add_argument('--index-dir', type=Path, default=DEFAULT_OUTPUT / 'chroma')
    index_parser.add_argument('--model', default=EMBEDDING_MODEL)
    index_parser.add_argument('--collection', default=COLLECTION_NAME)

    evaluate_parser = subparsers.add_parser('evaluate', help='đánh giá một retrieval method trên một split')
    evaluate_parser.add_argument('--chunks', type=Path, default=DEFAULT_OUTPUT / 'chunks.json')
    evaluate_parser.add_argument('--questions', type=Path, default=DEFAULT_OUTPUT / 'questions.json')
    evaluate_parser.add_argument('--index-dir', type=Path, default=DEFAULT_OUTPUT / 'chroma')
    evaluate_parser.add_argument('--model', default=EMBEDDING_MODEL)
    evaluate_parser.add_argument('--collection', default=COLLECTION_NAME)
    evaluate_parser.add_argument('--method', choices=('tfidf', 'bm25', 'dense', 'hybrid'), required=True)
    evaluate_parser.add_argument('--split', choices=('dev', 'test'), required=True)
    evaluate_parser.add_argument('--top-k', type=int, default=3)
    evaluate_parser.add_argument('--budget', type=int, default=800)
    evaluate_parser.add_argument('--rrf-constant', type=int, default=60)
    evaluate_parser.add_argument('--output', type=Path, required=True)

    args = parser.parse_args()
    if args.command == 'prepare':
        prepare(args)
    elif args.command == 'index':
        chunks = load_json(args.chunks)
        count = build_index(chunks, args.index_dir, args.model, args.collection)
        print(json.dumps({
            'collection': args.collection,
            'chunk_count': count,
            'embedding_model': args.model,
            'index_dir': str(args.index_dir),
        }, ensure_ascii=False, indent=2))
    else:
        if args.top_k < 1 or args.budget < 1 or args.rrf_constant < 1:
            raise ValueError('top-k, budget và rrf-constant phải > 0')
        chunks = load_json(args.chunks)
        questions = load_json(args.questions)
        validate_benchmark(chunks, questions)
        suite = RetrievalSuite(
            chunks, args.index_dir, [args.method], args.model,
            args.collection, args.rrf_constant,
        )
        summary, traces = evaluate(
            suite, questions, args.method, args.split, args.top_k, args.budget
        )
        args.output.mkdir(parents=True, exist_ok=True)
        save_json(args.output / 'summary.json', summary)
        save_json(args.output / 'traces.json', traces)
        print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
