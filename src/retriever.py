"""RAG 파이프라인 — 문서 로딩·인덱싱.

설계 메모
---------
청킹하지 않는다. 문서 한 건이 1~3줄뿐이라 분할하면 맥락만 잃는다.
  - 생물·주민: JSON 레코드 1건 = 문서 1건
  - 성격 가이드: 마크다운만 `## [P-0N]` 헤딩 단위로 쪼갠다

JSON 을 그대로 임베딩하지 않고 자연어 문장으로 바꿔서 넣는다.
`{"name":"무당벌레","location":"꽃 주변"}` 보다
`무당벌레는 꽃 주변에 나타나는 곤충입니다` 쪽이 질의와 훨씬 잘 붙는다.

모듈 최상단에서 임베딩 클라이언트를 만들지 않는다. 전부 build_* 팩토리로
주입받아, import 만으로 Bedrock 을 호출하지 않고 가짜 객체로 테스트할 수 있게 한다.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter
from pydantic import BaseModel, Field

from src.llm import build_llm
from src.schemas import RetrievedDoc

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"

COLLECTION_NAME = "acnh_docs"
PERSIST_DIR = str(PROJECT_ROOT / "chroma_db")


# ── 문서 로딩 · 문장화 ─────────────────────────────────────────────────────


def _load_json(filename: str):
    with open(DATA_DIR / filename, encoding="utf-8") as f:
        return json.load(f)


def _critter_text(row: dict, category: str) -> str:
    """생물 레코드 한 건을 검색 친화적인 한 문단으로 만든다."""
    parts = [f"{row['name']}({row['id']})는 {row['location']}에 나타나는 {category}입니다."]
    if row.get("shadow"):
        parts.append(f"그림자 크기는 '{row['shadow']}'입니다.")
    parts.append(f"출현 시간대는 {row['time']}입니다.")
    parts.append(f"판매가는 {row['price']}벨입니다.")
    return " ".join(parts)


def _villager_text(row: dict) -> str:
    """주민 레코드 한 건을 한 문단으로 만든다."""
    return (
        f"{row['name']}({villager_doc_id(row['id'])})는 {row['species']} 주민입니다. "
        f"성격은 {row['personality']}이고, 생일은 {row['birthday']}, "
        f"좋아하는 색은 {row['favorite_color']}입니다."
    )


def villager_doc_id(raw_id: int) -> str:
    """주민의 정수 id 를 doc_id 규칙(V-01~V-20)으로 바꾼다."""
    return f"V-{raw_id:02d}"


def load_documents() -> list[Document]:
    """data/ 의 네 소스를 모두 읽어 Document 목록으로 돌려준다.

    이 함수는 외부 호출이 전혀 없어 단위 테스트가 가능하다.
    metadata 의 doc_type 은 retrieve_docs 의 필터 인자로 쓰인다.
    """
    docs: list[Document] = []

    for filename, category, doc_type in [
        ("insects.json", "곤충", "insect"),
        ("fishes.json", "물고기", "fish"),
    ]:
        for row in _load_json(filename):
            docs.append(
                Document(
                    page_content=_critter_text(row, category),
                    metadata={"doc_id": row["id"], "doc_type": doc_type, "name": row["name"]},
                )
            )

    for row in _load_json("villagers.json"):
        docs.append(
            Document(
                page_content=_villager_text(row),
                metadata={
                    "doc_id": villager_doc_id(row["id"]),
                    "doc_type": "villager",
                    "name": row["name"],
                },
            )
        )

    docs.extend(_load_personalities())
    return docs


def _load_personalities() -> list[Document]:
    """personalities.md 를 `## [P-0N] 이름` 섹션 단위로 쪼갠다."""
    md = (DATA_DIR / "personalities.md").read_text(encoding="utf-8")
    sections = MarkdownHeaderTextSplitter(
        headers_to_split_on=[("##", "heading")]
    ).split_text(md)

    docs = []
    for sec in sections:
        heading = sec.metadata.get("heading", "")
        # heading 예: "[P-01] 느끼함"
        if not heading.startswith("["):
            continue  # 문서 상단의 제목·안내문은 건너뛴다
        doc_id, _, name = heading.partition("]")
        doc_id, name = doc_id.lstrip("["), name.strip()
        docs.append(
            Document(
                page_content=f"{name} — {sec.page_content}",
                metadata={"doc_id": doc_id, "doc_type": "personality", "name": name},
            )
        )
    return docs


# ── 인덱싱 ─────────────────────────────────────────────────────────────────


def build_embeddings(embeddings=None):
    """Bedrock 임베딩 클라이언트를 만든다. 주입받으면 그대로 쓴다."""
    if embeddings is not None:
        return embeddings
    from langchain_aws import BedrockEmbeddings

    from src.llm import bedrock_retry_config

    return BedrockEmbeddings(
        model_id=os.getenv("BEDROCK_EMBED_MODEL_ID", "amazon.titan-embed-text-v2:0"),
        region_name=os.getenv("AWS_DEFAULT_REGION", "us-east-1"),
        config=bedrock_retry_config(),
    )


def build_vectorstore(embeddings=None, persist_directory: str | None = None):
    """Chroma 벡터스토어를 연다. 없으면 새로 만든다."""
    from langchain_chroma import Chroma

    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=build_embeddings(embeddings),
        persist_directory=persist_directory or PERSIST_DIR,
    )


def index_documents(vectorstore=None, docs: list[Document] | None = None) -> int:
    """문서를 색인하고 색인된 건수를 돌려준다.

    doc_id 를 Chroma 의 id 로 그대로 쓴다. 덕분에 여러 번 실행해도
    중복이 쌓이지 않고 같은 id 를 덮어쓴다(upsert).
    """
    docs = load_documents() if docs is None else docs
    vs = build_vectorstore() if vectorstore is None else vectorstore
    vs.add_documents(docs, ids=[d.metadata["doc_id"] for d in docs])
    return len(docs)


# ── 토크나이저 (BM25용) ────────────────────────────────────────────────────

_kiwi = None


def _get_kiwi():
    """Kiwi 인스턴스는 생성이 느려 한 번만 만들어 재사용한다."""
    global _kiwi
    if _kiwi is None:
        from kiwipiepy import Kiwi

        _kiwi = Kiwi()
    return _kiwi


def tokenize(text: str) -> list[str]:
    """구두점 분리 토큰과 kiwi 형태소의 합집합을 돌려준다.

    둘 중 하나만 쓰면 각각 다음과 같이 실패한다(실측):
      - 구두점 분리만: '주민' 질의가 문서의 '주민입니다' 와 안 맞는다(조사 미분리).
      - kiwi 만: '1호' 가 '1' + '호' 로 쪼개져 흔한 토큰에 묻힌다.
    합집합으로 두 경우를 모두 살린다. 문서가 128건뿐이라 색인 비용은 무시할 수준.
    """
    surface = [w for w in re.split(r"[^0-9A-Za-z가-힣]+", text.lower()) if w]
    morphs = [t.form.lower() for t in _get_kiwi().tokenize(text) if len(t.form) > 1]
    return list(dict.fromkeys(surface + morphs))


# ── 3단계 검색 ─────────────────────────────────────────────────────────────

RRF_K = 60           # RRF 상수. 관행값 60을 그대로 쓴다.
MIN_RELEVANCE = 4    # 리랭커 점수(0~10) 하한. 미만이면 '관련 문서 없음' 취급.


class _Variants(BaseModel):
    """쿼리 확장 결과."""

    queries: list[str] = Field(description="원 질의와 뜻이 같은 검색어 2개")


class _Ranked(BaseModel):
    """리랭킹 결과 한 건."""

    doc_id: str
    score: int = Field(description="질의와의 관련도 0~10. 무관하면 0")


class _RankedList(BaseModel):
    results: list[_Ranked]


@dataclass
class SearchResult:
    """검색 결과와 그 과정. 호출자가 trace 를 만들 수 있도록 과정을 함께 돌려준다."""

    docs: list[RetrievedDoc]
    expanded: list[str]     # 실제로 검색에 쓴 질의들
    retried: bool = False   # 빈 결과로 쿼리를 재작성해 재시도했는가


class AcnhRetriever:
    """쿼리 확장 → 하이브리드 검색 → 리랭킹의 3단계 파이프라인."""

    def __init__(self, vectorstore, docs: list[Document], llm=None,
                 use_expansion: bool = True, use_rerank: bool = True):
        self.vs = vectorstore
        self.docs = docs
        self.llm = llm
        self.use_expansion = use_expansion
        self.use_rerank = use_rerank

        from rank_bm25 import BM25Okapi

        self._bm25 = BM25Okapi([tokenize(d.page_content) for d in docs])
        self._by_id = {d.metadata["doc_id"]: d for d in docs}

    # ── 1단계: 쿼리 확장 ──
    def expand(self, query: str) -> list[str]:
        """원 질의에 같은 뜻의 검색어 2개를 더한다."""
        if not self.use_expansion or self.llm is None:
            return [query]
        prompt = (
            "너는 '모여봐요 동물의 숲' 도감 검색기의 질의 확장기다.\n"
            "아래 질문과 같은 것을 찾는 검색어 2개를 만들어라. "
            "고유명사(주민 이름·생물 이름)는 절대 바꾸거나 빼지 말고 그대로 남겨라.\n\n"
            f"질문: {query}"
        )
        try:
            out = self.llm.with_structured_output(_Variants).invoke(prompt)
            return list(dict.fromkeys([query, *out.queries]))
        except Exception:
            return [query]  # 확장 실패는 치명적이지 않다. 원 질의로 진행.

    # ── 2단계: 하이브리드 검색 (BM25 + 임베딩, RRF 융합) ──
    def hybrid(self, queries: list[str], doc_type: str | None = None,
               k: int = 10) -> list[tuple[Document, float]]:
        """각 질의로 두 방식을 돌리고 RRF 로 순위를 합친다.

        점수 체계가 다른 두 랭킹(거리 vs BM25 점수)을 직접 더할 수 없으므로,
        순위만 쓰는 RRF 로 융합한다.
        """
        fused: dict[str, float] = {}

        def add(ranked_ids: list[str]) -> None:
            for rank, doc_id in enumerate(ranked_ids):
                fused[doc_id] = fused.get(doc_id, 0.0) + 1.0 / (RRF_K + rank)

        flt = {"doc_type": doc_type} if doc_type else None
        for q in queries:
            hits = self.vs.similarity_search(q, k=k, filter=flt)
            add([h.metadata["doc_id"] for h in hits])

            scores = self._bm25.get_scores(tokenize(q))
            pairs = [
                (self.docs[i].metadata["doc_id"], s)
                for i, s in enumerate(scores)
                if s > 0 and (not doc_type or self.docs[i].metadata["doc_type"] == doc_type)
            ]
            pairs.sort(key=lambda x: -x[1])
            add([doc_id for doc_id, _ in pairs[:k]])

        ordered = sorted(fused.items(), key=lambda x: -x[1])[:k]
        return [(self._by_id[doc_id], score) for doc_id, score in ordered]

    # ── 3단계: 리랭킹 (관련도 필터 겸함) ──
    def rerank(self, query: str, candidates: list[tuple[Document, float]],
               top_n: int) -> list[RetrievedDoc]:
        """LLM 이 후보에 관련도를 매겨 재정렬하고, 하한 미만은 버린다.

        단순 재정렬이 아니라 '관련 문서 없음'을 판정하는 역할도 한다.
        도감에 없는 생물·주민을 물었을 때 무관한 문서가 딸려와 환각의 근거가
        되는 것을 여기서 막는다(SERVICE.md 정책 2).
        """
        if not candidates:
            return []
        if not self.use_rerank or self.llm is None:
            return [
                RetrievedDoc(doc_id=d.metadata["doc_id"], text=d.page_content,
                             score=s, doc_type=d.metadata["doc_type"])
                for d, s in candidates[:top_n]
            ]

        listing = "\n".join(f"[{d.metadata['doc_id']}] {d.page_content}" for d, _ in candidates)
        prompt = (
            "아래 문서들이 질문에 답하는 데 실제로 쓸모 있는지 0~10으로 채점하라.\n\n"
            "채점 기준\n"
            "- 9~10: 이 문서만으로 질문에 답할 수 있다.\n"
            "- 5~8 : 답의 일부를 담고 있다.\n"
            "- 1~4 : 주제만 같고 질문에 답하지는 못한다.\n"
            "- 0    : 질문이 묻는 대상이 문서에 아예 없다.\n\n"
            "두 가지를 구분하라.\n"
            "- 여러 문서를 이어붙여야 답이 완성되는 질문이라면, 그 연결고리가 되는 "
            "문서에도 높은 점수를 준다. 예: '쭈니 선물'을 물으면 쭈니의 성격을 알려주는 "
            "주민 문서와 그 성격의 선물 목록 문서가 둘 다 필요하다.\n"
            "- 반대로 '전체에 공통으로 적용되는 규칙'을 물었는데 개별 사례 하나만 담은 "
            "문서라면, 주제가 같아도 질문에 답하지 못하므로 낮은 점수다.\n\n"
            f"질문: {query}\n\n문서:\n{listing}"
        )
        try:
            out = self.llm.with_structured_output(_RankedList).invoke(prompt)
        except Exception:
            out = None
        if out is None:
            return [
                RetrievedDoc(doc_id=d.metadata["doc_id"], text=d.page_content,
                             score=s, doc_type=d.metadata["doc_type"])
                for d, s in candidates[:top_n]
            ]

        kept = sorted(
            (r for r in out.results if r.score >= MIN_RELEVANCE and r.doc_id in self._by_id),
            key=lambda r: -r.score,
        )[:top_n]
        return [
            RetrievedDoc(
                doc_id=r.doc_id,
                text=self._by_id[r.doc_id].page_content,
                score=float(r.score),
                doc_type=self._by_id[r.doc_id].metadata["doc_type"],
            )
            for r in kept
        ]

    # ── 전체 파이프라인 + 재시도 미들웨어 ──
    def search(self, query: str, doc_type: str | None = None, k: int = 5) -> SearchResult:
        """3단계를 모두 태우고, 빈 결과면 쿼리를 재작성해 1회만 재시도한다.

        두 번째도 비면 빈 목록을 그대로 돌려준다. 호출자는 지어내지 말고
        '확인되지 않는다'로 안내해야 한다(정책 2).
        """
        variants = self.expand(query)
        docs = self.rerank(query, self.hybrid(variants, doc_type), k)
        if docs:
            return SearchResult(docs=docs, expanded=variants)

        rewritten = self._rewrite(query)
        if rewritten == query:
            return SearchResult(docs=[], expanded=variants, retried=False)
        docs = self.rerank(rewritten, self.hybrid([rewritten], doc_type), k)
        return SearchResult(docs=docs, expanded=variants + [rewritten], retried=True)

    def _rewrite(self, query: str) -> str:
        """재시도용으로 질의를 다르게 바꿔 본다."""
        if self.llm is None:
            return query
        try:
            out = self.llm.with_structured_output(_Variants).invoke(
                "아래 검색이 결과를 하나도 못 찾았다. 표현을 크게 바꿔 다시 검색할 "
                "질의를 2개 제안하라. 핵심 명사는 유지하되 설명적으로 풀어써라.\n\n"
                f"실패한 질의: {query}"
            )
            return out.queries[0] if out.queries else query
        except Exception:
            return query


def build_retriever(vectorstore=None, docs=None, llm=None, **kwargs) -> AcnhRetriever:
    """검색기를 조립한다. 모든 의존을 주입받을 수 있다."""
    docs = load_documents() if docs is None else docs
    vs = build_vectorstore() if vectorstore is None else vectorstore
    return AcnhRetriever(vs, docs, llm=build_llm(llm), **kwargs)


if __name__ == "__main__":
    from collections import Counter

    from dotenv import load_dotenv

    load_dotenv(PROJECT_ROOT / ".env")

    documents = load_documents()
    counts = Counter(d.metadata["doc_type"] for d in documents)
    print(f"문서 {len(documents)}건: " + " · ".join(f"{k} {v}" for k, v in counts.items()))

    print(f"색인 중... (Chroma: {PERSIST_DIR})")
    n = index_documents(docs=documents)
    print(f"색인 완료: {n}건\n")

    # 색인이 쓸 만한지 바로 확인한다. LLM 을 끄고 하이브리드 검색만 본다.
    retriever = build_retriever(docs=documents, llm=None,
                                use_expansion=False, use_rerank=False)
    for q, want in [("무당벌레 어디서 잡혀?", "I-031"), ("1호 선물 추천해줘", "V-16")]:
        hits = retriever.search(q, k=3).docs
        top = hits[0].doc_id if hits else "없음"
        print(f"  {'✓' if top == want else '✗'} {q!r} → 1위 {top} (기대 {want})")
