"""FastAPI 서버 — POST /query.

승인(HITL)을 단일 엔드포인트로 처리하는 방법
    산출물 규약상 요청 본문은 {"question": "..."} 하나뿐이라 thread_id 도
    approve 플래그도 실을 수 없다. 그래서 상태로 분기한다.

      1턴  "무 148벨인데 전부 팔아줘"
           → 그래프가 interrupt 로 멈춤. 확인 문구를 answer 로 돌려준다.
      2턴  "응 팔아줘"
           → get_state().next 가 비어 있지 않으면 승인 응답으로 해석하고 재개.

    단일 사용자용 더미 서비스라 thread_id 를 하나로 고정한다. 사용자가 여럿인
    서비스라면 인증 주체별로 thread_id 를 나눠야 한다.
"""
from __future__ import annotations

from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI
from langgraph.store.memory import InMemoryStore
from pydantic import BaseModel, Field

from src.agent import ask, build_checkpointer, build_graph, is_awaiting_approval, resume
from src.guardrails import parse_consent
from src.llm import enable_llm_cache
from src.retriever import PROJECT_ROOT
from src.schemas import QueryResponse, TraceStep
from src.tools import seed_player_state

THREAD_ID = "default"
CONFIG = {"configurable": {"thread_id": THREAD_ID}}


class QueryRequest(BaseModel):
    question: str = Field(description="사용자 질의")


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_dotenv(PROJECT_ROOT / ".env")
    enable_llm_cache()

    store = InMemoryStore()
    seed_player_state(store)
    app.state.store = store
    app.state.graph = build_graph(store=store, checkpointer=build_checkpointer())
    yield


app = FastAPI(title="동물의 숲 무인도 생활 어시스턴트", lifespan=lifespan)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest) -> QueryResponse:
    graph = app.state.graph

    if is_awaiting_approval(graph, CONFIG):
        consent = parse_consent(request.question)
        if consent is None:
            # 애매하면 재개하지 않는다. 승인 대기 상태를 그대로 유지한다.
            return QueryResponse(
                answer="진행 여부를 알기 어렵습니다. '응' 또는 '취소'로 답해 주세요.",
                trace=[TraceStep(step="guardrail_in",
                                 input=request.question,
                                 output="승인 의사 불명 — 재개하지 않음")],
            )
        return resume(graph, consent, CONFIG)

    return ask(graph, request.question, CONFIG)
