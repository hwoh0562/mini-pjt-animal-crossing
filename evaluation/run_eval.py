"""인-아웃 세트 자체 평가 (LLM-as-Judge).

판정 방식
    expected_tools  → trace 에서 기계적으로 대조한다. LLM 에 물을 이유가 없다.
    expected_traits → LLM-as-Judge 가 항목별로 충족 여부를 본다.
    forbidden       → LLM-as-Judge 가 등장 여부를 본다.
    셋을 모두 만족해야 통과로 센다.

케이스마다 thread_id 와 Store 를 새로 만든다. 그러지 않으면 id 13 의 승인
대기 상태가 다음 케이스로 새어 들어간다. 다만 멀티턴 케이스(id 15)는 선행
케이스와 같은 스레드를 쓴다.

실행:  python -m evaluation.run_eval --round 1
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.agent import ask, build_graph, is_awaiting_approval  # noqa: E402
from src.llm import build_llm, enable_llm_cache  # noqa: E402
from src.retriever import PROJECT_ROOT  # noqa: E402
from src.tools import PLAYER_KEY, PLAYER_NS, seed_player_state  # noqa: E402

CSV_PATH = PROJECT_ROOT / "evaluation" / "test_queries.csv"

# 케이스별 특수 조건
SEEDS = {"14": "player_state_no_price.json"}   # 매수단가가 비어 있는 상태로 시작
PREREQ = {"15": "2"}                            # 같은 스레드에서 먼저 돌려야 하는 케이스
NOW_HOUR = 20                                   # 시각 고정. 매 실행 결과가 흔들리지 않게 한다.


class Verdict(BaseModel):
    """LLM-as-Judge 의 판정."""

    missing_traits: list[str] = Field(default_factory=list,
                                      description="answer 에서 충족되지 않은 expected_traits 항목")
    forbidden_found: list[str] = Field(default_factory=list,
                                       description="answer 에 나타난 forbidden 항목")
    reason: str = Field(default="", description="판정 근거 한두 문장")


JUDGE_PROMPT = """너는 '모여봐요 동물의 숲' 어시스턴트의 답변을 채점한다.

질문
{question}

어시스턴트의 답변
{answer}

반드시 포함·유지되어야 할 항목 (expected_traits)
{traits}

나오면 안 되는 항목 (forbidden)
{forbidden}

각 항목을 하나씩 확인하라.
- missing_traits: 위 항목 중 답변에서 확인되지 않는 것만 그대로 적는다.
- forbidden_found: 답변에 실제로 나타난 금지 항목만 적는다.
표현이 달라도 뜻이 같으면 충족으로 본다. 반대로 항목이 요구하는 정보가 아예
없으면 충족이 아니다. 없으면 빈 목록으로 둔다."""


def split_items(text: str) -> list[str]:
    return [x.strip() for x in text.split(";") if x.strip()]


def called_tools(response) -> list[str]:
    """trace 에서 실제로 호출된 도구 이름을 뽑는다."""
    names = []
    for step in response.trace:
        if step.step in ("tool", "retrieve"):
            name = step.input.split("(")[0].strip()
            if name:
                names.append(name)
    return names


def run_case(row: dict, judge_llm) -> dict:
    """케이스 하나를 실행하고 판정한다."""
    case_id = row["id"]
    store = InMemoryStore()
    seed_player_state(store, SEEDS.get(case_id, "player_state.json"))

    # 평가에는 체크포인트를 파일로 남길 이유가 없다. 인메모리를 쓰면 케이스마다
    # 상태가 완전히 격리되고, Windows 에서 파일 잠금으로 삭제가 막히지도 않는다.
    graph = build_graph(store=store, checkpointer=InMemorySaver(), now_hour=NOW_HOUR)
    config = {"configurable": {"thread_id": f"case-{case_id}"}}

    # 멀티턴 케이스는 선행 질의를 같은 스레드에서 먼저 돌린다.
    prereq_id = PREREQ.get(case_id)
    if prereq_id:
        prereq = next(r for r in load_rows() if r["id"] == prereq_id)
        ask(graph, prereq["input"], config)

    response = ask(graph, row["input"], config)
    tools = called_tools(response)

    # 도구 대조는 기계적으로 한다.
    expected = split_items(row["expected_tools"])
    tools_ok = set(expected) <= set(tools)

    # HITL 케이스는 '승인 전 상태 미변경'을 직접 확인한다. 답변 텍스트만으로는
    # 검증할 수 없는 항목이다.
    extra_ok, extra_note = True, ""
    if case_id == "13":
        qty = store.get(PLAYER_NS, PLAYER_KEY).value["turnips"]["quantity"]
        waiting = is_awaiting_approval(graph, config)
        extra_ok = qty == 120 and waiting
        extra_note = f"승인대기={waiting} · 무 보유={qty}개(기대 120)"

    verdict = judge_llm.with_structured_output(Verdict).invoke(JUDGE_PROMPT.format(
        question=row["input"],
        answer=response.answer,
        traits="\n".join(f"- {t}" for t in split_items(row["expected_traits"])),
        forbidden="\n".join(f"- {t}" for t in split_items(row["forbidden"])) or "- (없음)",
    ))

    passed = tools_ok and extra_ok and not verdict.missing_traits and not verdict.forbidden_found
    return {
        "id": case_id,
        "category": row["category"],
        "input": row["input"],
        "answer": response.answer,
        "passed": passed,
        "tools_ok": tools_ok,
        "tools_called": tools,
        "tools_expected": expected,
        "missing_traits": verdict.missing_traits,
        "forbidden_found": verdict.forbidden_found,
        "extra_ok": extra_ok,
        "extra_note": extra_note,
        "reason": verdict.reason,
        "trace": [s.step for s in response.trace],
        "contexts": [c.doc_id for c in response.contexts],
    }


def load_rows() -> list[dict]:
    with open(CSV_PATH, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_report(results: list[dict], round_no: int, path: Path) -> None:
    total = len(results)
    passed = sum(r["passed"] for r in results)
    by_cat: dict[str, list[dict]] = {}
    for r in results:
        by_cat.setdefault(r["category"], []).append(r)

    lines = [
        f"# {round_no}차 자체 평가 리포트",
        "",
        f"- 실행 시각: {datetime.now():%Y-%m-%d %H:%M}",
        f"- 판정: `expected_tools` 기계 대조 + `expected_traits` / `forbidden` LLM-as-Judge",
        f"- 시각 고정: {NOW_HOUR}시 · 케이스마다 thread_id·Store 분리",
        "",
        f"## 통과율 {passed}/{total} ({passed / total:.0%})",
        "",
        "| 카테고리 | 통과 | 전체 |",
        "|---|---|---|",
    ]
    for cat in ("positive", "negative", "edge", "guardrail"):
        rows = by_cat.get(cat, [])
        if rows:
            lines.append(f"| {cat} | {sum(r['passed'] for r in rows)} | {len(rows)} |")

    lines += ["", "## 케이스별 결과", "",
              "| id | 분류 | 질의 | 결과 | 비고 |", "|---|---|---|---|---|"]
    for r in results:
        notes = []
        if not r["tools_ok"]:
            notes.append(f"도구 불일치(호출 {r['tools_called'] or '없음'})")
        if r["missing_traits"]:
            notes.append("미충족: " + " / ".join(r["missing_traits"]))
        if r["forbidden_found"]:
            notes.append("금지항목: " + " / ".join(r["forbidden_found"]))
        if not r["extra_ok"]:
            notes.append(r["extra_note"])
        lines.append(f"| {r['id']} | {r['category']} | {r['input'][:26]} | "
                     f"{'✅' if r['passed'] else '❌'} | {'; '.join(notes) or '-'} |")

    failures = [r for r in results if not r["passed"]]
    if failures:
        lines += ["", "## 실패 케이스 상세", ""]
        for r in failures:
            lines += [
                f"### id {r['id']} · {r['input']}",
                "",
                f"- 호출된 도구: `{r['tools_called'] or '없음'}` (기대 `{r['tools_expected'] or '없음'}`)",
                f"- trace: `{r['trace']}`",
                f"- 판정 근거: {r['reason']}",
                "",
                "```",
                r["answer"][:600],
                "```",
                "",
            ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--round", type=int, default=1)
    parser.add_argument("--only", default="", help="쉼표로 구분한 id 목록만 실행")
    args = parser.parse_args()

    load_dotenv(PROJECT_ROOT / ".env")
    enable_llm_cache()
    judge_llm = build_llm(max_tokens=1024)

    rows = load_rows()
    if args.only:
        wanted = {x.strip() for x in args.only.split(",")}
        rows = [r for r in rows if r["id"] in wanted]

    results = []
    for row in rows:
        try:
            result = run_case(row, judge_llm)
        except Exception as exc:
            result = {"id": row["id"], "category": row["category"], "input": row["input"],
                      "answer": "", "passed": False, "tools_ok": False, "tools_called": [],
                      "tools_expected": split_items(row["expected_tools"]),
                      "missing_traits": [], "forbidden_found": [], "extra_ok": False,
                      "extra_note": f"실행 오류: {type(exc).__name__}: {exc}",
                      "reason": "실행 중 예외", "trace": [], "contexts": []}
        results.append(result)
        print(f"[{result['id']:>2}] {'✅' if result['passed'] else '❌'} {row['input'][:34]}")
        if not result["passed"]:
            detail = result["missing_traits"] or result["forbidden_found"] or result["extra_note"]
            print(f"      {detail}")

    total, passed = len(results), sum(r["passed"] for r in results)
    print(f"\n통과율 {passed}/{total} ({passed / total:.0%})")
    print("카테고리:", dict(Counter(r["category"] for r in results if r["passed"])))

    report = PROJECT_ROOT / "evaluation" / f"round{args.round}_report.md"
    write_report(results, args.round, report)
    (PROJECT_ROOT / "evaluation" / f"round{args.round}_raw.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"리포트: {report}")


if __name__ == "__main__":
    main()
