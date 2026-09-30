"""E5 dense retrieval using an external corpus and FAISS index."""

import argparse
import os
from pathlib import Path
from threading import Lock
from typing import Any, Sequence


MAX_BATCH_QUERIES = 512
_RETRIEVAL_BACKENDS = frozenset({"faiss_cpu_flat", "faiss_gpu_sharded_fp16"})


def _validated_queries(values: Sequence[object]) -> list[str]:
    if isinstance(values, (str, bytes, bytearray)):
        raise ValueError("queries must be a sequence of strings")
    queries = [str(value).strip() for value in values]
    if not queries or len(queries) > MAX_BATCH_QUERIES:
        raise ValueError(
            f"queries must contain between 1 and {MAX_BATCH_QUERIES} items"
        )
    if any(not value for value in queries):
        raise ValueError("queries must not contain empty strings")
    return queries


def create_app(
    retriever: Any,
    *,
    default_topk: int,
    backend: str = "faiss_cpu_flat",
):
    """Create the FastAPI application around one already-loaded retriever."""

    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel

    if isinstance(default_topk, bool) or int(default_topk) <= 0:
        raise ValueError("default_topk must be a positive integer")
    if backend not in _RETRIEVAL_BACKENDS:
        raise ValueError(f"unsupported retrieval backend: {backend!r}")
    topk_default = int(default_topk)
    retrieval_lock = Lock()
    app = FastAPI()

    class QueryRequest(BaseModel):
        query: str
        topk: int | None = None
        return_scores: bool = False

    class BatchQueryRequest(BaseModel):
        queries: list[str]
        topk: int | None = None
        return_scores: bool = False

    def resolve_topk(value: int | None) -> int:
        resolved = topk_default if value is None else int(value)
        if resolved <= 0:
            raise HTTPException(status_code=422, detail="topk must be positive")
        return resolved

    @app.get("/health")
    def health() -> dict[str, object]:
        return {
            "ready": True,
            "service": "searchqa_batched_retriever",
            "backend": backend,
            "max_batch_queries": MAX_BATCH_QUERIES,
        }

    @app.post("/retrieve")
    def retrieve(request: QueryRequest) -> dict[str, object]:
        query = str(request.query or "").strip()
        if not query:
            raise HTTPException(status_code=422, detail="query must be non-empty")
        topk = resolve_topk(request.topk)
        if request.return_scores:
            with retrieval_lock:
                documents, scores = retriever.search(query, topk, True)
            combined = [
                {"document": document, "score": float(score)}
                for document, score in zip(documents, scores)
            ]
            return {"result": [combined]}
        with retrieval_lock:
            documents = retriever.search(query, topk, False)
        return {"result": [documents]}

    @app.post("/retrieve_batch")
    def retrieve_batch(request: BatchQueryRequest) -> dict[str, object]:
        try:
            queries = _validated_queries(request.queries)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        topk = resolve_topk(request.topk)
        if request.return_scores:
            with retrieval_lock:
                document_rows, score_rows = retriever.batch_search(queries, topk, True)
            results = [
                [
                    {"document": document, "score": float(score)}
                    for document, score in zip(documents, scores)
                ]
                for documents, scores in zip(document_rows, score_rows)
            ]
        else:
            with retrieval_lock:
                results = retriever.batch_search(queries, topk, False)
        if len(results) != len(queries):
            raise RuntimeError(
                "retriever returned a partial batch: "
                f"expected={len(queries)} actual={len(results)}"
            )
        return {"results": results}

    return app



class E5Retriever:
    """Mean-pooled, normalized FP16 E5 queries and eight FP16 Flat shards."""
    def __init__(self, *, index_path, corpus_path, model_path, gpu_map, batch_size=512):
        import faiss
        import torch
        import datasets
        from transformers import AutoModel, AutoTokenizer
        if len(gpu_map) != 8:
            raise ValueError("The main retrieval protocol requires eight logical GPU shards")
        if any(index < 0 or index >= faiss.get_num_gpus() for index in gpu_map):
            raise ValueError("Every retrieval GPU must be visible to FAISS")
        cpu_index = faiss.read_index(str(index_path))
        if type(cpu_index).__name__ not in {"IndexFlat", "IndexFlatIP", "IndexFlatL2"}:
            raise ValueError("Main retrieval requires a Flat index")
        self.resources = [faiss.StandardGpuResources() for _ in gpu_map]
        options = faiss.GpuMultipleClonerOptions()
        options.shard, options.useFloat16 = True, True
        self.index = faiss.index_cpu_to_gpu_multiple_py(self.resources, cpu_index, co=options, gpus=list(gpu_map))
        self.corpus = datasets.load_dataset("json", data_files=str(corpus_path), split="train", num_proc=4)
        if len(self.corpus) != self.index.ntotal:
            raise ValueError("Corpus length and index vector count must match")
        self.device = torch.device("cuda:0")
        self.model = AutoModel.from_pretrained(model_path, trust_remote_code=True).eval().to(self.device).half()
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True, trust_remote_code=True)
        self.batch_size = batch_size

    def _encode(self, queries):
        import numpy as np
        import torch
        encoded = self.tokenizer([f"query: {query}" for query in queries], max_length=256,
                                 padding=True, truncation=True, return_tensors="pt")
        encoded = {key: value.to(self.device) for key, value in encoded.items()}
        with torch.no_grad():
            hidden = self.model(**encoded, return_dict=True).last_hidden_state
            mask = encoded["attention_mask"]
            hidden = hidden.masked_fill(~mask[..., None].bool(), 0.0)
            pooled = hidden.sum(dim=1) / mask.sum(dim=1)[..., None]
            pooled = torch.nn.functional.normalize(pooled, dim=-1)
        return pooled.cpu().numpy().astype(np.float32, order="C")

    def batch_search(self, queries, topk, return_scores=False):
        if not 0 < topk <= self.index.ntotal:
            raise ValueError("topk exceeds the index size")
        documents, scores = [], []
        for start in range(0, len(queries), self.batch_size):
            values, indices = self.index.search(self._encode(queries[start:start + self.batch_size]), topk)
            if (indices < 0).any():
                raise RuntimeError("FAISS returned an invalid document ID")
            documents.extend([[self.corpus[int(index)] for index in row] for row in indices])
            scores.extend(values.tolist())
        return (documents, scores) if return_scores else documents

    def search(self, query, topk, return_scores=False):
        result = self.batch_search([query], topk, return_scores)
        return (result[0][0], result[1][0]) if return_scores else result[0]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index-path", required=True)
    parser.add_argument("--corpus-path", required=True)
    parser.add_argument("--model-path", default="intfloat/e5-base-v2")
    parser.add_argument("--gpu-map", default="0,1,2,3,4,5,6,7", help="Eight visible GPU IDs, repeats allowed")
    parser.add_argument("--port", type=int, default=8042)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--topk", type=int, default=3)
    parser.add_argument("--faiss-threads", type=int, default=32)
    args = parser.parse_args(argv)
    import faiss
    import uvicorn
    faiss.omp_set_num_threads(args.faiss_threads)
    retriever = E5Retriever(index_path=args.index_path, corpus_path=args.corpus_path,
                            model_path=args.model_path, gpu_map=[int(x) for x in args.gpu_map.split(",")])
    uvicorn.run(create_app(retriever, default_topk=args.topk, backend="faiss_gpu_sharded_fp16"),
                host=args.host, port=args.port, workers=1)


if __name__ == "__main__":
    main()
