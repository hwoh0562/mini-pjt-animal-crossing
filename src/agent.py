"""LangGraph 에이전트 그래프.

흐름
    START → guardrail_in ─[차단]────────────────────────┐
                 │                                      │
              [통과]                                    │
                 ↓                                      │
               agent ─[도구 호출 있음]→ tools ─┐        │
                 ↑                              │        │
                 └──────────────────────────────┘        │
                 └─[도구 호출 없음]→ generate ──→ guardrail_out → END

입력 가드레일에 걸리면 LLM 을 한 번도 호출하지 않고 안내 문구만 내보낸다.
차단이 결정적이라 차단율 100% 를 보장할 수 있고, 토큰도 들지 않는다.

agent 노드와 generate 노드를 나눈 이유
    bind_tools 와 with_structured_output 은 둘 다 tool-calling 메커니즘을 쓰기
    때문에 한 호출에 같이 쓸 수 없다. 그래서 도구를 고르는 루프(agent)와
    최종 답변을 Pydantic 으로 뽑는 단계(generate)를 분리했다.
    덕분에 Answer 스키마의 cited_doc_ids 로 출처 명시를 강제할 수 있다.

프리빌트 ToolNode 를 쓰지 않은 이유
    ToolNode 는 ToolMessage 만 남기고 contexts / trace 를 채우지 못한다.
    도구가 돌려준 docs 를 모아 상태에 넣어야 응답 규약을 지킬 수 있다.
"""
from __future__ import annotations

import json
from datetime import datetime

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END, START, StateGraph

from src.guardrails import check_input, check_output
from src.llm import build_llm
from src.schemas import AgentState, Answer, Context, QueryResponse, TraceStep
from src.tools import build_tools

SYSTEM_PROMPT = """너는 '모여봐요 동물의 숲' 플레이어를 돕는 섬생활 어시스턴트다.

반드시 지킬 것
- 사실은 도구로 확인한 내용만 말한다. 도구가 '확인되지 않습니다'를 돌려주면
  지어내지 말고 보유한 자료에 없다고 안내한다.
- 답변에는 근거 문서 id(V-02 · I-031 · P-01 등)를 함께 밝힌다.
- 보유하지 않은 자료: 해산물, 생물의 출현 월·계절·반구·날씨. 이런 조건을
  물으면 추측하지 말고 자료가 없다고 안내한다.
- 게임이 공개하지 않거나 무작위인 값(미래 무 시세, 주민 친밀도 내부 수치)은
  확정적인 숫자로 답하지 않는다.
- 조건에 맞는 대상이 없으면 '없다'가 정답이다. 다른 것으로 대신 채우지 마라.
- 정중한 존댓말로 답한다.

현재 시각은 {hour}시다. 시각이 필요한 도구에는 이 값을 쓰되, 사용자가 다른
시각을 말하면 그 값을 우선한다."""

GENERATE_PROMPT = """위 대화의 도구 결과만을 근거로 사용자 질문에 최종 답변하라.

- answer_text: 사용자에게 보여줄 답변. 근거 문서 id 를 문장 안에 함께 적는다.
- cited_doc_ids: 실제로 근거로 삼은 문서 id 목록. 도구가 문서를 돌려주지
  않았다면 빈 목록으로 둔다.

도구가 자료를 못 찾았다면 그 사실을 그대로 안내하라. 지어내지 마라."""

# 도구 이름 → trace step. retrieve_docs 만 'retrieve', 나머지는 'tool'.
# step 이름은 SERVICE.md 에서 6개로 고정했으므로 새로 만들지 않는다.
_RETRIEVE_TOOLS = {"retrieve_docs"}


def _summarize(value, limit: int = 300) -> str:
    """trace 에 넣을 수 있도록 도구 인자·결과를 짧은 문자열로 만든다."""
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
    return text if len(text) <= limit else text[:limit] + "…"


def build_graph(llm=None, tools=None, store=None, checkpointer=None, now_hour: int | None = None):
    """에이전트 그래프를 조립한다. 모든 의존을 주입받을 수 있다.

    Args:
        llm: 주입하지 않으면 .env 의 BEDROCK_MODEL_ID 로 만든다.
        tools: 주입하지 않으면 build_tools 로 만든다.
        store: 플레이어 상태가 사는 LangGraph Store.
        checkpointer: HITL 을 쓰려면 필요하다.
        now_hour: 시스템 시각 대신 쓸 시각(테스트 고정용).
    """
    # max_tokens 를 넉넉히 준다. 구조화 출력은 tool-call 로 직렬화되므로 중간에
    # 잘리면 인자가 통째로 비어 Pydantic 검증이 실패한다(실제로 겪음).
    model = build_llm(llm, max_tokens=4096)
    tool_list = build_tools(llm=model, store=store) if tools is None else tools
    tools_by_name = {t.name: t for t in tool_list}
    model_with_tools = model.bind_tools(tool_list)

    def _system_message() -> SystemMessage:
        hour = datetime.now().hour if now_hour is None else now_hour
        return SystemMessage(SYSTEM_PROMPT.format(hour=hour))

    # ── agent: 도구를 고르는 ReAct 루프 ──
    def agent_node(state: AgentState) -> dict:
        response = model_with_tools.invoke([_system_message(), *state["messages"]])
        return {"messages": [response]}

    # ── tools: 도구 실행 + contexts / trace 수집 ──
    def tool_node(state: AgentState) -> dict:
        last = state["messages"][-1]
        messages, contexts, trace = [], [], []

        for call in last.tool_calls:
            name, args = call["name"], call["args"]
            tool = tools_by_name.get(name)
            if tool is None:
                result = {"data": {"error": f"알 수 없는 도구입니다: {name}"}, "docs": []}
            else:
                try:
                    result = tool.invoke(args)
                except Exception as exc:  # 도구 실패가 그래프 전체를 죽이지 않게 한다
                    result = {"data": {"error": f"{type(exc).__name__}: {exc}"}, "docs": []}

            messages.append(ToolMessage(
                content=_summarize(result["data"], limit=4000),
                tool_call_id=call["id"],
                name=name,
            ))
            contexts.extend(Context(**d) for d in result.get("docs", []))
            trace.append(TraceStep(
                step="retrieve" if name in _RETRIEVE_TOOLS else "tool",
                input=f"{name}({_summarize(args, 150)})",
                output=_summarize(result["data"], 200),
            ))

        return {"messages": messages, "contexts": contexts, "trace": trace}

    # ── generate: 구조화 출력으로 최종 답변 ──
    def generate_node(state: AgentState) -> dict:
        messages = [_system_message(), *state["messages"], HumanMessage(GENERATE_PROMPT)]
        try:
            answer: Answer = model.with_structured_output(Answer).invoke(messages)
            if answer is None:
                raise ValueError("구조화 출력이 비었습니다")
        except Exception:
            # 구조화 출력이 실패해도 답변 자체는 내보낸다. 평가에서 한 케이스가
            # 예외로 통째로 날아가는 것보다, 인용 없이라도 답하는 편이 낫다.
            # 출처가 비므로 출력 가드레일이 이 답변을 걸러낼 수 있다.
            plain = model.invoke(messages)
            answer = Answer(answer_text=plain.content if isinstance(plain.content, str)
                            else str(plain.content), cited_doc_ids=[])
        cited = ", ".join(answer.cited_doc_ids) or "(없음)"
        return {
            "answer": answer.answer_text,
            "trace": [TraceStep(step="generate",
                                input="최종 답변 생성",
                                output=f"인용 {cited} · {_summarize(answer.answer_text, 150)}")],
        }

    # ── guardrail_in: 규칙 기반 입력 필터 ──
    def guard_in_node(state: AgentState) -> dict:
        question = state["messages"][-1].content
        decision = check_input(question)
        return {
            "blocked_reason": decision.reason or None,
            "answer": decision.message,
            "trace": [TraceStep(step="guardrail_in",
                                input=_summarize(question, 150),
                                output=f"block:{decision.reason}" if decision.blocked else "pass")],
        }

    # ── guardrail_out: 출력 검사 · 출처 보정 ──
    def guard_out_node(state: AgentState) -> dict:
        doc_ids = [c.doc_id for c in state.get("contexts", [])]
        fixed, decision = check_output(state.get("answer", ""), doc_ids)
        note = decision.reason or "pass"
        return {
            "answer": fixed,
            "trace": [TraceStep(step="guardrail_out", input="답변 검사", output=note)],
        }

    def route_guard(state: AgentState) -> str:
        # 차단됐으면 LLM 을 한 번도 호출하지 않고 안내 문구만 내보낸다.
        return "blocked" if state.get("blocked_reason") else "agent"

    def route(state: AgentState) -> str:
        last = state["messages"][-1]
        return "tools" if getattr(last, "tool_calls", None) else "generate"

    graph = StateGraph(AgentState)
    graph.add_node("guardrail_in", guard_in_node)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tool_node)
    graph.add_node("generate", generate_node)
    graph.add_node("guardrail_out", guard_out_node)

    graph.add_edge(START, "guardrail_in")
    graph.add_conditional_edges("guardrail_in", route_guard,
                                {"blocked": "guardrail_out", "agent": "agent"})
    graph.add_conditional_edges("agent", route, {"tools": "tools", "generate": "generate"})
    graph.add_edge("tools", "agent")
    graph.add_edge("generate", "guardrail_out")
    graph.add_edge("guardrail_out", END)

    return graph.compile(checkpointer=checkpointer, store=store)


# ── 실행 헬퍼 ──────────────────────────────────────────────────────────────


def to_response(state: dict) -> QueryResponse:
    """그래프 최종 상태를 API 응답 규약으로 옮긴다.

    같은 문서가 여러 도구에서 중복으로 올라올 수 있으므로 doc_id 로 한 번
    걸러낸다. 순서는 처음 등장한 순서를 유지한다.
    """
    seen, contexts = set(), []
    for ctx in state.get("contexts", []):
        if ctx.doc_id not in seen:
            seen.add(ctx.doc_id)
            contexts.append(ctx)
    return QueryResponse(
        answer=state.get("answer", ""),
        contexts=contexts,
        trace=state.get("trace", []),
    )


def ask(graph, question: str, config: dict | None = None) -> QueryResponse:
    """질문 한 건을 실행하고 응답 규약으로 돌려준다."""
    state = graph.invoke({"messages": [HumanMessage(question)]}, config or {})
    return to_response(state)


if __name__ == "__main__":
    from dotenv import load_dotenv
    from langgraph.store.memory import InMemoryStore

    from src.llm import enable_llm_cache
    from src.retriever import PROJECT_ROOT
    from src.tools import seed_player_state

    load_dotenv(PROJECT_ROOT / ".env")
    enable_llm_cache()

    player_store = InMemoryStore()
    seed_player_state(player_store)
    compiled = build_graph(store=player_store, now_hour=20)

    for q in ["지금 잡을 수 있는 곤충 알려줘", "늑대 주민 누구누구 있어?"]:
        res = ask(compiled, q)
        print(f"\nQ. {q}")
        print(f"A. {res.answer}")
        print(f"   contexts: {[c.doc_id for c in res.contexts]}")
        print(f"   trace   : {[t.step for t in res.trace]}")
