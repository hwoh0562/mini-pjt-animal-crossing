"""RAGAS 지표 산출.

test_queries.csv 와 세트를 나눈 이유
    RAGAS 의 context_recall 은 정답(reference)을 요구하는데 test_queries.csv
    스키마에는 그 컬럼이 없다. 또 가드레일·계산 케이스는 검색을 타지 않아
    retrieved_contexts 가 비어 지표가 의미를 잃는다. 그래서 검색을 타는
    질의만 ragas_set.csv 로 따로 관리한다.

판정자와 임베딩도 Bedrock 을 쓴다. RAGAS 기본값은 OpenAI 라 명시적으로 주입한다.

실행:  python -m evaluation.run_ragas
"""
from __future__ import annotations

import asyncio
import csv
import json
import sys
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.agent import ask, build_graph  # noqa: E402
from src.llm import build_llm, enable_llm_cache  # noqa: E402
from src.retriever import PROJECT_ROOT, build_embeddings  # noqa: E402
from src.tools import seed_player_state  # noqa: E402

SET_PATH = PROJECT_ROOT / "evaluation" / "ragas_set.csv"
NOW_HOUR = 20

# SERVICE.md §5 의 목표치
TARGETS = {
    "faithfulness": 0.85,
    "context_recall": 0.80,
    "context_precision": 0.75,
    "answer_relevancy": 0.75,
}


def collect() -> list[dict]:
    """각 질의를 에이전트에 태워 답변과 근거 문서를 모은다."""
    with open(SET_PATH, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    samples = []
    for row in rows:
        store = InMemoryStore()
        seed_player_state(store)
        graph = build_graph(store=store, checkpointer=InMemorySaver(), now_hour=NOW_HOUR)
        response = ask(graph, row["question"], {"configurable": {"thread_id": row["id"]}})

        got = [c.doc_id for c in response.contexts]
        want = [x for x in row["expected_doc_ids"].split(";") if x]
        print(f"[{row['id']:>3}] {row['question'][:30]:<32} "
              f"contexts={got or '없음'} (기대 {want})")

        samples.append({
            "user_input": row["question"],
            "retrieved_contexts": [c.text for c in response.contexts] or ["(검색 결과 없음)"],
            "response": response.answer,
            "reference": row["reference"],
            "_id": row["id"],
            "_got": got,
            "_want": want,
        })
    return samples


async def score_all(samples: list[dict]) -> dict[str, float]:
    """네 지표를 샘플별로 매기고 평균을 낸다.

    RAGAS 의 evaluate()/single_turn_ascore() 는 내부에서 asyncio.wait_for 를
    쓰는데, Python 3.14 에서 `Timeout should be used inside a task` 로 전부
    실패한다(40개 Job 모두 nan). 타임아웃을 두르지 않는 _single_turn_ascore 를
    직접 호출해 우회한다. 지표 계산 로직 자체는 동일하다.
    """
    from ragas.dataset_schema import SingleTurnSample
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import (answer_relevancy, context_precision,
                               context_recall, faithfulness)

    llm = LangchainLLMWrapper(build_llm(max_tokens=2048))
    embeddings = LangchainEmbeddingsWrapper(build_embeddings())

    # answer_relevancy 는 '답변으로부터 질문을 역생성 → 원 질문과 코사인 유사도'
    # 로 계산한다. RAGAS 기본 프롬프트가 영어라 역생성 질문이 영어로 나오는데,
    # Titan 임베딩에서 영↔한 유사도는 0.28 수준(한↔한 패러프레이즈는 0.61~0.76)
    # 이라 답변 품질이 아니라 언어 불일치를 재게 된다. 한국어로 뽑도록 지시한다.
    answer_relevancy.question_generation.instruction += (
        "\n\nIMPORTANT: Write the generated question in Korean. "
        "It must be in the same language as the given answer."
    )

    metrics = [faithfulness, answer_relevancy, context_precision, context_recall]
    for metric in metrics:
        metric.llm = llm
        if hasattr(metric, "embeddings"):
            metric.embeddings = embeddings

    per_metric: dict[str, list[float]] = {m.name: [] for m in metrics}
    for sample in samples:
        turn = SingleTurnSample(
            **{k: v for k, v in sample.items() if not k.startswith("_")})
        for metric in metrics:
            try:
                per_metric[metric.name].append(float(await metric._single_turn_ascore(turn, None)))
            except Exception as exc:
                print(f"  [{sample['_id']}] {metric.name} 실패: {type(exc).__name__}")
        print(f"  [{sample['_id']}] 채점 완료")

    return {name: sum(v) / len(v) for name, v in per_metric.items() if v}


def main() -> None:
    load_dotenv(PROJECT_ROOT / ".env")
    enable_llm_cache()

    samples = collect()

    print("\nRAGAS 평가 중...")
    # asyncio.run() 은 종료 단계의 shutdown_default_executor 에서도 타임아웃을
    # 두르기 때문에, ragas 가 적용한 nest_asyncio 와 겹쳐 또 터진다.
    # 루프를 직접 만들고 닫아 그 경로를 피한다.
    loop = asyncio.new_event_loop()
    try:
        scores = loop.run_until_complete(score_all(samples))
    finally:
        loop.close()
    print()
    lines = [
        "# RAGAS 평가 결과",
        "",
        f"- 실행 시각: {datetime.now():%Y-%m-%d %H:%M}",
        f"- 대상: `ragas_set.csv` {len(samples)}건 (검색을 타는 질의만)",
        "- 판정 LLM · 임베딩 모두 Bedrock (RAGAS 기본값인 OpenAI 대신 주입)",
        "",
        "| 지표 | 점수 | 목표 | 충족 |",
        "|---|---|---|---|",
    ]
    for name, target in TARGETS.items():
        score = scores.get(name)
        if score is None:
            continue
        print(f"  {name:<20} {score:.3f}  (목표 {target})  {'✅' if score >= target else '❌'}")
        lines.append(f"| `{name}` | {score:.3f} | ≥ {target} | "
                     f"{'✅' if score >= target else '❌'} |")

    lines += ["", "## 문서 검색 정확도", "",
              "| id | 질의 | 검색된 문서 | 기대 문서 |", "|---|---|---|---|"]
    for s in samples:
        hit = "✅" if set(s["_want"]) <= set(s["_got"]) else "❌"
        lines.append(f"| {s['_id']} | {s['user_input'][:24]} | "
                     f"{' '.join(s['_got']) or '없음'} | {' '.join(s['_want'])} {hit} |")

    report = PROJECT_ROOT / "evaluation" / "ragas_report.md"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    (PROJECT_ROOT / "evaluation" / "ragas_raw.json").write_text(
        json.dumps({"scores": scores, "samples": samples}, ensure_ascii=False, indent=2),
        encoding="utf-8")
    print(f"\n리포트: {report}")


if __name__ == "__main__":
    main()
