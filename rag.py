"""RAG 代码检索引擎：向量（ChromaDB）+ BM25 混合检索 + Cross-Encoder 重排。

用法：
    rag.index_codebase()            # 扫描并索引工作区代码
    rag.retrieve("登录校验逻辑")     # 混合检索，返回 top-k 相关代码片段

模型懒加载：首次调用时才下载/加载 embedding 与 rerank 模型。
"""
from __future__ import annotations

import re
from pathlib import Path

from config import settings

# ---- 可调参数 ----
CHUNK_LINES = 40       # 每个 chunk 的行数
CHUNK_OVERLAP = 10     # 相邻 chunk 重叠行数
EMBED_MODEL = "all-MiniLM-L6-v2"                       # 向量模型（小、通用）
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"  # 重排模型
VECTOR_TOP_K = 20      # 向量检索初筛数量
BM25_TOP_K = 20        # BM25 检索初筛数量
CHROMA_DIR = settings.workspace_root / "chroma"

# ---- 懒加载单例 ----
_embedder = None
_reranker = None
_bm25 = None
_meta: list[dict] = []    # chunk 元数据 [{path, start, end}]，与语料对齐
_corpus: list[str] = []   # chunk 文本，与 _meta 对齐

_TOKEN_RE = re.compile(r"[a-zA-Z0-9_]+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def _get_embedder():
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer
        _embedder = SentenceTransformer(EMBED_MODEL)
    return _embedder


def _get_reranker():
    global _reranker
    if _reranker is None:
        from sentence_transformers import CrossEncoder
        _reranker = CrossEncoder(RERANK_MODEL)
    return _reranker


def _iter_code_files(root: Path):
    """遍历代码文件，跳过 venv/缓存/向量库等。"""
    skip = {".git", "__pycache__", ".venv", "chroma", "node_modules",
            "bin", "lib", "include", "share", "site-packages"}
    for p in sorted(root.rglob("*.py")):
        if any(part in skip for part in p.parts):
            continue
        yield p


def _split(lines: list[str]):
    """按行切分，返回 [(start_line, end_line, text)]，start 为 1 基行号。"""
    chunks = []
    i = 0
    n = len(lines)
    while i < n:
        end = min(i + CHUNK_LINES, n)
        chunks.append((i + 1, end, "\n".join(lines[i:end])))
        if end >= n:
            break
        i += CHUNK_LINES - CHUNK_OVERLAP
    return chunks


def _chroma_collection():
    import chromadb
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return client.get_or_create_collection("code", metadata={"hnsw:space": "cosine"})


def index_codebase(root: Path | None = None) -> int:
    """扫描并索引代码库，重建 Chroma 与 BM25。返回 chunk 总数。"""
    global _bm25, _meta, _corpus
    from rank_bm25 import BM25Okapi

    root = (root or settings.workspace_root).resolve()
    corpus: list[str] = []
    meta: list[dict] = []
    ids: list[str] = []

    for path in _iter_code_files(root):
        try:
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            continue
        rel = str(path.relative_to(root))
        for start, end, text in _split(lines):
            corpus.append(text)
            meta.append({"path": rel, "start": start, "end": end})
            ids.append(f"{rel}:{start}-{end}")

    if not corpus:
        return 0

    embeddings = _get_embedder().encode(corpus, normalize_embeddings=True).tolist()

    # 重建向量库
    collection = _chroma_collection()
    try:
        collection.delete(where={})
    except Exception:  # noqa: BLE001 —— 空集合删除可能抛错，忽略
        pass
    collection.add(ids=ids, embeddings=embeddings, documents=corpus, metadatas=meta)

    # 重建 BM25
    _meta = meta
    _corpus = corpus
    _bm25 = BM25Okapi([_tokenize(t) for t in corpus])
    return len(corpus)


def retrieve(query: str, k: int = 5) -> list[dict]:
    """混合检索 + 重排，返回 top-k 代码片段 [{path, start, end, text, score}]。"""
    if _bm25 is None:
        if index_codebase() == 0:
            return []

    # 1) 向量检索
    qv = _get_embedder().encode([query], normalize_embeddings=True).tolist()
    v_res = _chroma_collection().query(query_embeddings=qv, n_results=VECTOR_TOP_K)

    # 2) BM25 检索
    b_scores = _bm25.get_scores(_tokenize(query))
    b_top = sorted(range(len(b_scores)), key=lambda i: b_scores[i], reverse=True)[:BM25_TOP_K]

    # 3) RRF 融合
    id_to_idx = {f"{m['path']}:{m['start']}-{m['end']}": i for i, m in enumerate(_meta)}
    rrf: dict[int, float] = {}
    for rank, cid in enumerate(v_res["ids"][0]):
        i = id_to_idx.get(cid)
        if i is not None:
            rrf[i] = rrf.get(i, 0) + 1.0 / (60 + rank + 1)
    for rank, i in enumerate(b_top):
        rrf[i] = rrf.get(i, 0) + 1.0 / (60 + rank + 1)

    merged = sorted(rrf.items(), key=lambda x: x[1], reverse=True)[:15]

    # 4) Cross-Encoder 重排
    candidates = [(i, _corpus[i]) for i, _ in merged]
    if candidates:
        pairs = [[query, text] for _, text in candidates]
        scores = _get_reranker().predict(pairs)
        ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
    else:
        ranked = [(c, 0.0) for c in candidates]

    out = []
    for (i, text), score in ranked[:k]:
        m = _meta[i]
        out.append({"path": m["path"], "start": m["start"], "end": m["end"],
                    "text": text, "score": float(score)})
    return out
