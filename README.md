# 미니 PJT: 동물의 숲 무인도 생활 어시스턴트

'모여봐요 동물의 숲' 플레이어가 생물 출현 조건·주민 선물·무 거래 손익을 자연어로 묻고,
근거 문서와 함께 답을 받는 Agentic RAG 어시스턴트입니다.

## 무엇을 푸나

생물 100종의 출현 조건과 주민 20명의 프로필이 방대한 표로 흩어져 있어, 플레이 도중
게임을 멈추고 위키에서 조건을 직접 필터링해야 하는 문제를 해결합니다.

기존 공략 사이트는 **표를 보여주고 필터링은 사용자 몫**으로 남기지만, 이 에이전트는
조건이 이미 적용된 결과만 근거 문서 id와 함께 돌려줍니다.

```
"지금 잡을 수 있는 곤충 알려줘"        → 20시 기준 26종 중 판매가 상위 15종
"쭈니한테 무슨 선물 주면 좋아?"        → V-02(느끼함) + P-01(느끼함 선물 목록)
"무 지금 148벨인데 전부 팔아줘"        → 승인 요청 후에만 실행
```

## 활용한 패턴 (Day 1~7 · 10개)

| # | 패턴 | Day | 이 서비스에서의 적용 |
|---|---|---|---|
| 1 | LCEL + Pydantic 구조화 출력 | 1 | `generate` 노드가 `Answer(answer_text, cited_doc_ids)` 로 파싱. 출처 명시를 스키마로 강제 |
| 2 | ReAct (도구 자율 선택) | 3 | `bind_tools` 로 도구 6개를 묶고 모델이 필요한 것을 고름 |
| 3 | RAG (쿼리 확장·하이브리드·리랭킹) | 2 | `retrieve_docs` 3단계 파이프라인 |
| 4 | 도구 다중 결합 | 4 | "무 152벨인데 이득이야?" → `get_player_state` + `calculate_turnip_profit` |
| 6 | 가드레일 | 5 | 규칙 기반 입력 필터 + 출력 검사. 차단 시 LLM 호출 0회 |
| 7 | HITL (위험 작업 승인) | 5 | `sell_all_turnips` 실행 전 `interrupt()` |
| 8 | 미들웨어 (재시도) | 5 | 검색 결과가 비면 쿼리를 재작성해 1회 재시도 |
| 10 | 장기 메모리 | 7 | 플레이어 상태를 LangGraph `Store` 에 보관 |
| 11 | Observability · Trace | 7 | 응답의 `trace` 6단계 + LangSmith |
| 12 | 평가 (RAGAS · LLM-as-Judge) | 7 | `run_eval.py` · `run_ragas.py` |

**미사용**: 5(MCP) — 데이터가 전부 로컬 파일이라 외부 서버로 노출할 대상이 없음.
9(Multi-Agent Supervisor) — 도구 6개를 단일 그래프가 다루는 규모라 분할 이득이 없음.

## 아키텍처

```
POST /query
    │
    ▼
START → begin → guardrail_in ─[차단]──────────────────┐
                     │                                │
                  [통과]                              │
                     ▼                                │
                   agent ─[도구 호출]→ tools ─┐       │
                     ▲                        │       │
                     └────────────────────────┘       │
                     └─[호출 없음]→ generate ─→ guardrail_out → END
```

| 노드 | 하는 일 |
|---|---|
| `begin` | 요청 단위로 `trace`·`contexts` 초기화. 이번 질문을 상태에 기록 |
| `guardrail_in` | 규칙 기반 차단. 걸리면 LLM 을 한 번도 호출하지 않음 |
| `agent` | `bind_tools` 로 도구 선택 (ReAct 루프) |
| `tools` | 도구 실행 + `contexts`/`trace` 수집. 위험 도구면 `interrupt()` |
| `generate` | `with_structured_output` 으로 최종 답변 |
| `guardrail_out` | 출처 보정 · 프롬프트 유출 검사 |

**모듈 의존은 한 방향으로만 흐릅니다.** 역방향 import 가 없어 순환이 구조적으로 불가능합니다.

```
schemas.py → llm.py → guardrails.py → retriever.py → tools.py → agent.py → api.py
   (의존 0)   (의존 0)    (의존 0)
```

어떤 모듈도 최상단에서 LLM·Chroma 클라이언트를 만들지 않습니다. 전부 `build_*(llm=None)`
팩토리로 주입받아, import 만으로 Bedrock 을 호출하지 않고 가짜 객체로 테스트할 수 있습니다.

### 데이터 (전부 더미)

| 파일 | 내용 | doc_id |
|---|---|---|
| `data/insects.json` | 곤충 50종 | `I-001`~`I-050` |
| `data/fishes.json` | 물고기 50종 | `F-001`~`F-050` |
| `data/villagers.json` | 주민 20명 | `V-01`~`V-20` |
| `data/personalities.md` | 성격 7종 선물 가이드 + 공통 규칙 | `P-00`~`P-07` |

생성 스크립트에 `random.seed(42)` 를 고정해 **재실행해도 같은 데이터**가 나옵니다.
1차·2차 평가 사이에 데이터가 바뀌면 개선폭 측정이 무의미해지기 때문입니다.

## 실행 방법

```bash
# 1. 의존성 (Python 3.14 · Windows 기준으로 검증)
pip install --only-binary=:all: -r requirements.txt

# 2. 자격증명
cp .env.example .env    # AWS 키와 BEDROCK_MODEL_ID 를 채운다

# 3. 데이터 생성 + 색인 (128건)
python data/make_data.py
python -m src.retriever

# 4. 터미널에서 대화로 확인
python -m src.chat

# 5. API 서버
uvicorn src.api:app --reload --port 8000

# 6. 평가
python -m evaluation.run_eval --round 2
python -m evaluation.run_ragas
```

대화형 CLI는 한 세션 안에서 대화가 이어지므로 멀티턴과 승인 흐름을 그대로 확인할 수
있습니다. 답변과 함께 근거 문서·trace 단계가 같이 출력됩니다.

```
질문> 쭈니한테 무슨 선물 주면 좋아?
쭈니는 느끼함 성격의 다람쥐 주민입니다(V-02)...

  근거  V-02 · P-01
  단계  guardrail_in → retrieve → generate → guardrail_out

질문> 무 지금 148벨인데 전부 팔아줘
무 120개를 개당 148벨에 매도하면 17,760벨을 받습니다. 진행할까요?

승인(응/취소)> 응
```

```bash
curl -X POST http://127.0.0.1:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "쭈니한테 무슨 선물 주면 좋아?"}'
```

### Docker

```bash
docker build -t acnh-assistant .
docker run --env-file .env -p 8000:8000 acnh-assistant
```

`.env` 는 `.dockerignore` 로 이미지에서 제외되며, 실행 시점에 `--env-file` 로 주입합니다.
색인도 빌드가 아니라 컨테이너 시작 시점에 합니다. 빌드 때 키를 넣으면 이미지에 남기
때문입니다. 첫 기동에 약 70초(색인 128건 포함), 이후 재시작은 색인을 건너뜁니다.

색인을 컨테이너 간에 유지하려면 볼륨을 붙입니다.

```bash
docker run --env-file .env -p 8000:8000 -v acnh-chroma:/app/chroma_db acnh-assistant
```

**베이스 이미지는 `python:3.12-slim` 입니다.** 로컬 개발은 3.14 지만, 3.14 는 휠이 없는
패키지가 있어 소스 빌드로 넘어가면 C++ 컴파일러가 필요합니다. 3.12 Linux 에서는 이 스택
전체가 휠로 설치돼 `--only-binary` 플래그도 필요 없었습니다.

```json
{
  "answer": "쭈니는 느끼함 성격의 다람쥐 주민입니다(V-02). ...",
  "contexts": [{"doc_id": "V-02", "text": "..."}, {"doc_id": "P-01", "text": "..."}],
  "trace": [{"step": "guardrail_in", "input": "...", "output": "pass"}, ...]
}
```

`trace[].step` 은 `guardrail_in · retrieve · retrieve_retry · tool · generate · guardrail_out`
6개로 고정돼 있습니다.

## RAGAS 평가 결과

`evaluation/ragas_set.csv` 10건(검색을 타는 질의만) 기준. 판정 LLM 과 임베딩 모두 Bedrock.

- context_recall: **1.000** (목표 0.80)
- context_precision: **0.900** (목표 0.75)
- faithfulness: **0.882** (목표 0.85)
- answer_relevancy: **0.914** (목표 0.75)

문서 검색 정확도 **10/10** — 10건 모두 기대 문서를 빠짐없이 가져왔습니다.

## 인-아웃 세트 통과율 (자체 평가)

- **1차 (Day 9 종료): 15 / 20 통과 (75%)**
- **2차 (Day 10 개선 후): 20 / 20 통과 (100%)**
- **개선폭: +5건**

| 카테고리 | 1차 | 2차 |
|---|---|---|
| positive | 6/8 | 8/8 |
| negative | 3/4 | 4/4 |
| edge | 3/5 | 5/5 |
| guardrail | 3/3 | 3/3 |

주요 개선 사항:

1. **`search_villagers` 부분 일치 반환** — 조건을 여러 개 걸어 결과가 비면 조건별 결과를
   함께 돌려줍니다. id 17 이 실패한 진짜 이유는 프롬프트가 아니라 도구였습니다. 빈 목록만
   받은 에이전트는 "느끼함 주민이 누구인지" 알 방법이 없어 근거를 댈 수 없었습니다.
2. **출처 줄 자동 보정** — 근거가 있으면 `출처:` 줄을 항상 덧붙입니다. 본문에 `(V-02)` 를
   흘려 적는 것만으로는 출처를 밝혔다고 보기 어렵습니다.
3. **프롬프트 4건** — 빈 결과에 근거 제시 / 거절 시 대안 제시 / 손익 판단은 사용자 몫 /
   검색 없이 되묻기 금지.
4. **`retrieve_docs` 경계 명시** — 주민 선물 질문에 `doc_type='villager'` 로 좁히면 성격
   가이드가 걸러져 구체적인 선물을 답할 수 없습니다.

판정 방식은 **사실은 코드, 서술은 LLM** 으로 나눴습니다. `expected_tools` 는 `trace` 에서
기계적으로 대조하고, id 13(승인 전 상태 미변경)과 id 4(시간대 위반 종 혼입)는 Store 와
데이터를 직접 확인합니다. `expected_traits`·`forbidden` 만 LLM-as-Judge 가 봅니다.

## 트라이앤에러 회고

### 실패한 접근

**`langchain>=0.3` 상한을 안 걸어 1.x 가 설치됐다.**
기존 과제의 `requirements.txt` 를 그대로 옮겼는데 상한이 없어 langchain 1.4 / langgraph 1.2
가 깔렸습니다. 그 결과 `langchain-community` 에서 `chat_models.vertexai` 가 제거돼 **ragas 가
버전과 무관하게 import 단계에서 실패**했습니다. 처음엔 ragas 구버전을 찾아 헤맸지만, 원인은
ragas 가 아니라 스택 세대였습니다. `<1.0` 상한을 걸어 0.3 세대로 맞추자 전부 해결됐습니다.
Day 1~7 실습 코드도 0.3 세대 기준이라 애초에 이쪽이 맞았습니다.

**리랭킹을 precision 만 보고 조였다가 회귀했다.**
RAGAS `context_precision` 이 0.725 로 떨어져 "느슨하게 관련된 문서에 점수를 주지 마라"는
예시를 넣었더니, 이번엔 "쭈니 선물" 질의에서 `P-01`(느끼함 선물 목록)이 탈락했습니다.
멀티홉 연결고리까지 깎아버린 것입니다. "공통 규칙 질의"와 "여러 문서를 이어야 답이 되는
질의"를 구분해 다시 썼습니다.

**HITL 이 아예 발동하지 않았다.**
`interrupt()` 를 붙였는데 승인 절차가 시작되지 않았습니다. LLM 이 `sell_all_turnips` 를
호출하지 않고 스스로 "매도를 진행하시겠습니까?"라고 되물었기 때문입니다. 도구를 안 부르니
승인할 대상이 없었습니다. 모델 입장에선 조심스러운 행동이지만, 이 구조에서는 **모델이
망설이면 승인 절차 자체가 시작되지 않습니다.** 프롬프트에 "확인 질문을 직접 하지 마라.
승인은 시스템이 끼워 넣는다"를 명시해 해결했습니다.

**요청 사이에 상태가 새어 나갔다.**
체크포인터가 스레드 상태를 보존하는데 `trace`·`contexts` 리듀서가 append 라, 앞 요청의
근거가 다음 응답에 섞였습니다. 가드레일 차단 응답에 무관한 `contexts` 2건이 붙고 trace 가
8단계까지 늘었습니다. 처음엔 응답에서 잘라내는 미봉책을 썼지만 `guard_out` 노드가 여전히
누적분을 보고 있었습니다. `append_or_reset` 리듀서와 `begin` 노드로 상태 자체를 요청 단위로
비우는 것이 근본 해결이었습니다. **단건 테스트에서는 안 잡히고 연속 호출에서만 드러나는
버그**였습니다.

### 최종 채택한 접근

**하이브리드 검색 — 실측으로 필요성이 증명됐다.**
주민 중 이름이 `1호` 인 고양이가 있습니다. 임베딩 단독에서는 유사도 1.439 로 사실상
무매칭이었는데(다른 질의 1위는 0.59~0.92), BM25 를 더하자 RRF 최고점 0.0333 으로
양쪽 모두 1위가 됐습니다. 토크나이저는 **구두점 분리 토큰과 kiwi 형태소의 합집합**을
씁니다. 구두점 분리만 쓰면 `주민` 이 `주민입니다` 와 안 맞고, kiwi 만 쓰면 `1호` 가
`1` + `호` 로 쪼개져 흔한 토큰에 묻히기 때문입니다.

**리랭커에 관련도 하한을 둬 '관련 문서 없음'을 판정하게 했다.**
단순 재정렬로만 썼다면 `그리핀나비`·`홍길동`·`가리비` 질의에 무관한 문서 5건이 딸려와
환각의 근거가 됐을 것입니다. 0~10 채점에 하한 4점을 걸어 negative 케이스가 빈 결과를
반환합니다.

**규칙 기반 가드레일.** 차단율 100% 를 목표로 걸었는데 LLM 판정은 확률적이라 100% 를
보장하지 못합니다. 규칙에 걸리면 LLM 을 한 번도 호출하지 않아 결정적이고 토큰도 들지
않습니다. 오탐을 막기 위해 단어 하나가 아니라 복합어로만 잡습니다 — `무 주식`(게임 용어)은
통과하고 `삼성전자 주가` 는 차단됩니다.

**`interrupt()` 를 도구 밖에 뒀다.** 도구 함수는 LangGraph 를 모르는 순수 함수로 남아
가짜 `store` 만 꽂으면 그래프 없이 단위 테스트가 됩니다. 승인 판정은 툴 노드 **맨 앞**에
둡니다. 재개하면 노드가 처음부터 다시 실행되므로, 뒤에 뒀다면 조회 도구가 두 번
호출됐을 것입니다.

**플레이어 상태는 Store 에만 쓴다.** 시드 JSON 은 최초 적재용이며 런타임에 덮어쓰지
않습니다. 파일에 쓰면 평가를 돌릴 때마다 시드가 변해 1차/2차 비교가 깨집니다.

### 남은 한계

- **`context_precision` 은 집계형 질의를 과소평가합니다.** 이 지표는 검색된 문서를 하나씩
  놓고 "이것만으로 정답에 도달할 수 있는가"를 판정자에게 묻습니다. R6("늑대 주민 누구누구
  있어?")은 늑대 5명을 정확히 다 찾아 답도 완벽했지만, 정답이 "다섯 명"이라는 집합이라
  주민 한 명짜리 문서는 각각 무용하다고 판정돼 0.0 이 나왔습니다. 검색 결함이 아니라
  지표가 다건 집계를 다루는 방식의 한계입니다.
- **Bedrock 일일 토큰 한도** — 평가 한 바퀴가 LLM 호출 90~180회라 하루에 여러 번 돌리면
  `ThrottlingException` 이 납니다. 적응형 재시도(`max_attempts=10, mode=adaptive`)와
  `SQLiteCache` 로 완화했지만, 반복 실행은 여전히 쿼터에 묶입니다.
- **Python 3.14 + RAGAS 호환** — `evaluate()` 와 `single_turn_ascore()` 가 내부에서
  `asyncio.wait_for` 를 써 `Timeout should be used inside a task` 로 전부 실패합니다.
  타임아웃을 두르지 않는 `_single_turn_ascore` 를 직접 호출해 우회했습니다.
  지표 계산 로직 자체는 건드리지 않았지만, 상위 API 를 쓰지 못하는 상태입니다.
- **`answer_relevancy` 의 언어 의존** — 이 지표는 답변에서 질문을 역생성해 원 질문과
  코사인 유사도를 잽니다. RAGAS 기본 프롬프트가 영어라 역생성 질문이 영어로 나왔고,
  Titan 임베딩에서 영↔한 유사도는 0.278(한↔한 패러프레이즈는 0.607~0.764)이라 완벽한
  답변도 0.25 근처가 나왔습니다. 한국어로 뽑도록 지시해 0.419 → 0.922 로 정상화했지만,
  다국어 환경에서는 같은 함정이 반복될 수 있습니다.
- **Chroma HNSW 의 근사 탐색** — 색인을 새로 만들면 순위가 미세하게 달라집니다. Docker
  이미지에서 실제로 `1호` 질의의 하이브리드 1위가 `V-16` → `P-06` 으로 바뀌었습니다.
  최종 순서는 리랭킹 단계가 정하므로 서비스 동작에는 영향이 없지만, 색인 직후 스모크
  테스트가 1위를 단정하던 것은 과도한 가정이라 상위 3건 포함 여부로 낮췄습니다.
- **Docker 이미지의 질의 검증 미완** — 빌드·기동·색인·`/health` 까지 확인했으나,
  LLM 을 타는 `/query` 는 Bedrock 일일 토큰 한도로 응답을 받지 못했습니다. Docker 쪽
  문제가 아니라 쿼터 문제이며, 같은 코드가 로컬에서는 20/20 으로 통과합니다.
- **보유하지 않은 데이터** — 해산물, 생물의 출현 월·계절·반구·날씨. 질의가 오면 추측하지
  않고 "확인되지 않는다"로 안내합니다.
- **단일 사용자 전제** — `POST /query` 요청 본문에 `thread_id` 가 없어 스레드를 하나로
  고정했습니다. 다중 사용자로 확장하려면 인증 주체별로 스레드를 나눠야 합니다.

### 향후 개선 방향

- 생물 데이터에 출현 월·반구를 추가해 계절 질의를 지원
- LLM 응답 캐싱을 평가 파이프라인 전반에 적용해 토큰 한도 압박 완화
- `faithfulness` 가 간헐적으로 `OutputParserException` 을 내는 샘플의 원인 규명

## 핵심 코드 위치

| 위치 | 내용 |
|---|---|
| [`src/agent.py:106`](src/agent.py#L106) | `build_graph` — 그래프 조립 |
| [`src/agent.py:150`](src/agent.py#L150) | `tool_node` — 도구 실행 + `contexts`/`trace` 수집 + `interrupt()` |
| [`src/agent.py:201`](src/agent.py#L201) | `generate_node` — Pydantic 구조화 출력 (패턴 1) |
| [`src/retriever.py:231`](src/retriever.py#L231) | `AcnhRetriever` — 검색 3단계 (패턴 3) |
| [`src/retriever.py:355`](src/retriever.py#L355) | `search` — 재시도 미들웨어 포함 (패턴 8) |
| [`src/tools.py:85`](src/tools.py#L85) | `build_tools` — 도메인 도구 6개 |
| [`src/tools.py:180`](src/tools.py#L180) | `search_villagers` — 부분 일치 반환 |
| [`src/guardrails.py:76`](src/guardrails.py#L76) | `check_input` — 규칙 기반 차단 (패턴 6) |
| [`src/guardrails.py:138`](src/guardrails.py#L138) | `parse_consent` — 승인 의사 판정 (패턴 7) |
| [`src/schemas.py:104`](src/schemas.py#L104) | `AgentState` — `append_or_reset` 리듀서 |
| [`src/api.py:60`](src/api.py#L60) | `POST /query` — 승인 2턴 분기 |
| [`evaluation/run_eval.py`](evaluation/run_eval.py) | LLM-as-Judge 자체 평가 (패턴 12) |
| [`evaluation/run_ragas.py`](evaluation/run_ragas.py) | RAGAS 4지표 |

서비스 스펙은 [`SERVICE.md`](SERVICE.md), 평가 세트는
[`evaluation/test_queries.csv`](evaluation/test_queries.csv) 에 있습니다.
