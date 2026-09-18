"""터미널 대화형 테스트.

실행:  python -m src.chat

한 세션 안에서 대화가 이어지므로 멀티턴("쭈니 선물 추천해줘" → "걔 성격이
뭐였지?")과 승인 흐름("무 전부 팔아줘" → "응")을 그대로 확인할 수 있다.
답변과 함께 근거 문서와 trace 단계를 같이 찍어, 어느 경로로 답이 나왔는지
한 화면에서 보이게 했다.
"""
from __future__ import annotations

import sys

from dotenv import load_dotenv
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore

from src.agent import ask, build_graph, is_awaiting_approval, resume
from src.guardrails import parse_consent
from src.llm import enable_llm_cache
from src.retriever import PROJECT_ROOT
from src.tools import seed_player_state

CONFIG = {"configurable": {"thread_id": "chat"}}


def show(response) -> None:
    print(f"\n{response.answer}\n")
    if response.contexts:
        print(f"  근거  {' · '.join(c.doc_id for c in response.contexts)}")
    print(f"  단계  {' → '.join(t.step for t in response.trace)}\n")


def main() -> None:
    # Windows 콘솔 기본 인코딩이 UTF-8 이 아닐 수 있어 한글이 깨진다.
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    load_dotenv(PROJECT_ROOT / ".env")
    enable_llm_cache()

    store = InMemoryStore()
    seed_player_state(store)
    graph = build_graph(store=store, checkpointer=InMemorySaver())

    print("동물의 숲 섬생활 어시스턴트 (종료: Ctrl+C 또는 빈 줄)\n"
          "예) 지금 잡을 수 있는 곤충 알려줘 / 쭈니한테 무슨 선물 주면 좋아?\n"
          "    무 지금 148벨인데 전부 팔아줘  → 승인을 물어본다\n")

    while True:
        waiting = is_awaiting_approval(graph, CONFIG)
        prompt = "승인(응/취소)> " if waiting else "질문> "
        try:
            text = input(prompt).strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not text:
            break
        # 파이프로 입력받으면 stdin 이 화면에 반향되지 않아 기록에 질문이 빠진다.
        # 시연 로그를 파일로 남길 때 대화가 반쪽이 되므로 직접 찍어 준다.
        # 터미널에서는 이미 보이므로 중복 출력하지 않는다.
        if not sys.stdin.isatty():
            print(text)

        try:
            if waiting:
                consent = parse_consent(text)
                if consent is None:
                    print("\n진행 여부를 알기 어렵습니다. '응' 또는 '취소'로 답해 주세요.\n")
                    continue
                show(resume(graph, consent, CONFIG))
            else:
                show(ask(graph, text, CONFIG))
        except Exception as exc:
            # 토큰 한도 등으로 한 번 실패해도 세션을 끊지 않는다.
            print(f"\n[오류] {type(exc).__name__}: {str(exc)[:200]}\n")


if __name__ == "__main__":
    main()
