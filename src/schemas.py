"""응답 규약(Pydantic)과 에이전트 상태(TypedDict)를 정의한다.

이 모듈은 다른 프로젝트 모듈을 import 하지 않는다.
의존 그래프의 최하위에 두어 순환 import 를 구조적으로 막기 위함이다.
"""
from __future__ import annotations

import operator
from typing import Annotated, Literal, TypedDict

from pydantic import BaseModel, Field

# trace 의 step 이름은 SERVICE.md §3 응답 규약에서 6개로 고정했다.
# 임의의 이름을 추가하지 말 것. 평가 리포트가 이 값으로 단계를 집계한다.
TraceStepName = Literal[
    "guardrail_in",     # 입력 가드레일 판정
    "retrieve",         # retrieve_docs 검색
    "retrieve_retry",   # 검색 결과가 비어 쿼리를 재작성해 재시도
    "tool",             # 그 외 도구 호출 (구조화 필터 · 계산 · 액션)
    "generate",         # 최종 답변 생성 (구조화 출력)
    "guardrail_out",    # 출력 가드레일 판정
]


# ── API 응답 규약 (산출물 규약 §4-2 · 변경 금지) ───────────────────────────


class Context(BaseModel):
    """답변의 근거 문서 한 건.

    RAG 검색 결과뿐 아니라 구조화 필터 도구(search_villagers 등)가 돌려준
    문서도 여기에 담는다. 근거의 출처가 도구 종류와 무관하게 일관되어야
    RAGAS 의 context_precision 이 떨어지지 않는다.
    """

    doc_id: str = Field(description="I-001 · F-007 · V-02 · P-01 형식의 문서 식별자")
    text: str = Field(description="문서 원문")


class TraceStep(BaseModel):
    """에이전트가 거친 단계 하나의 기록."""

    step: TraceStepName
    input: str
    output: str


class QueryResponse(BaseModel):
    """POST /query 의 응답 본문.

    세 필드 모두 비우거나 생략하지 않는다.
    contexts 는 RAGAS 산출의 입력이고, trace 는 Observability 채점의 입력이다.
    """

    answer: str
    contexts: list[Context] = Field(default_factory=list)
    trace: list[TraceStep] = Field(default_factory=list)


# ── 생성 단계의 구조화 출력 ────────────────────────────────────────────────


class Answer(BaseModel):
    """최종 답변 생성(generate 단계)에서 LLM 이 채우는 구조.

    cited_doc_ids 를 스키마에 넣어 출처 명시를 강제한다.
    출력 가드레일이 이 값이 비었는지로 '근거 없는 답변'을 걸러낸다.
    """

    answer_text: str = Field(description="사용자에게 보여줄 답변 본문")
    cited_doc_ids: list[str] = Field(
        default_factory=list,
        description="답변의 근거가 된 문서 id 목록. 문서를 인용하지 않았다면 빈 목록",
    )


# ── 검색 결과 ──────────────────────────────────────────────────────────────


class RetrievedDoc(BaseModel):
    """retrieve_docs 가 돌려주는 검색 결과 한 건."""

    doc_id: str
    text: str
    score: float
    doc_type: str = Field(description="insect · fish · villager · personality")


# ── 에이전트 상태 ──────────────────────────────────────────────────────────


class AgentState(TypedDict):
    """LangGraph 그래프가 들고 다니는 상태.

    trace 와 contexts 에 operator.add 리듀서를 걸어두면, 각 노드는 자기 몫
    한 건만 반환하면 되고 LangGraph 가 순서대로 이어붙인다. 별도의 수집기가
    필요 없다.

    필드를 늘리고 싶어질 때 먼저 아래를 검토할 것:
      - retry_used  → 재시도는 retriever 내부에서 끝내므로 상태에 둘 필요 없음
      - cited_doc_ids → 출력 가드레일이 answer 문자열과 contexts 를 대조하면 됨
    """

    # add_messages 리듀서는 agent.py 에서 붙인다(여기서 langgraph 를 import 하지
    # 않기 위함). 실제 타입은 list[AnyMessage] 이다.
    messages: Annotated[list, operator.add]
    trace: Annotated[list[TraceStep], operator.add]
    contexts: Annotated[list[Context], operator.add]

    # 입력 가드레일 차단 사유. None 이면 통과.
    blocked_reason: str | None
    # generate 단계가 채우는 최종 답변.
    answer: str
