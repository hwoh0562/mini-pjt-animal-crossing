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
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter

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

    return BedrockEmbeddings(
        model_id=os.getenv("BEDROCK_EMBED_MODEL_ID", "amazon.titan-embed-text-v2:0"),
        region_name=os.getenv("AWS_DEFAULT_REGION", "us-east-1"),
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


if __name__ == "__main__":
    from collections import Counter

    from dotenv import load_dotenv

    load_dotenv(PROJECT_ROOT / ".env")

    documents = load_documents()
    counts = Counter(d.metadata["doc_type"] for d in documents)
    print(f"문서 {len(documents)}건: " + " · ".join(f"{k} {v}" for k, v in counts.items()))

    print(f"색인 중... (Chroma: {PERSIST_DIR})")
    n = index_documents(docs=documents)
    print(f"색인 완료: {n}건")
