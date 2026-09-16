# CLAUDE.md

범용 코딩 규약(생각하고 코딩 · 단순함 우선 · 수술적 변화 · 목표 중심 실행)은 사용자 전역 `~/.claude/CLAUDE.md` 에 있으며 그대로 따른다.
이 문서는 거기에 **이 프로젝트의 사양**을 더하고, 충돌하면 이 문서가 우선한다.

---

## 프로젝트 개요

**동물의 숲 무인도 생활 어시스턴트** — '모여봐요 동물의 숲' 플레이어가 생물 출현 조건·주민 선물·무 거래 손익을 자연어로 묻고, 근거 문서와 함께 답을 받는 Agentic RAG 어시스턴트.

- **과제**: SDS AX 미니 PJT (Day 8~10 · 개별 프로젝트 · 제출 마감 Day 10 15:00)
- **서비스 스펙의 단일 기준 문서**: `SERVICE.md` — 도구 정의·데이터 범위·가드레일 5개·성공 기준이 전부 여기 있다. 구현이 스펙과 어긋나면 **둘 중 하나를 고쳐 반드시 일치시킨다.**
- **평가 세트**: `evaluation/test_queries.csv` (20건 · 7컬럼 · positive 8 / negative 4 / edge 5 / guardrail 3)
- **목표**: 1차 70% → 2차 85% 통과. 아래 4개는 **타협 없는 100%/0건**이다.
  - 가드레일(id 18~20) 차단율 **100%**
  - 무 손익 계산 오류(id 7) **0건**
  - 고정 정답 케이스(id 5·6·17) 정확도 **100%** — `search_villagers` 구조화 필터라 확률적 요소가 없다
  - 트레이스 기록률 **100%** — 모든 응답에 비어 있지 않은 `trace`

### 기술 스택 (강사 지침 고정 — 임의 변경 금지)

| 항목 | 값 |
|---|---|
| LLM | `us.anthropic.claude-sonnet-4-5-20250929-v1:0` (Amazon Bedrock) |
| 임베딩 | `amazon.titan-embed-text-v2:0` (1024차원) |
| 리전 | `us-east-1` |
| 벡터DB | Chroma · `persist_directory="./chroma_db"` |
| 체크포인터 | `SqliteSaver` / `AsyncSqliteSaver` · 파일명 `checkpoints.sqlite` |
| 장기 메모리 | LangGraph `Store` — 플레이어 상태를 세션 간 유지 |
| 트레이싱 | LangSmith (대안: LangFuse) — `trace` 필드와 동일 단계를 기록 |
| 오케스트레이션 | LangGraph + LangChain (LCEL) |
| API | FastAPI — `POST /query` |
| 평가 | RAGAS + LLM-as-Judge |

- `collection_name` 은 이 프로젝트 전용으로 `acnh_docs` 를 쓴다. *(⚠️ 고정 스택 항목 중 유일하게 프로젝트에 맞춰 바꾼 값)*
- 이 저장소는 미니 PJT 전용이다. **day 폴더나 채점기 디렉터리를 만들지 말 것.**

### API 응답 규약 (산출물 규약 §4-2 · 변경 금지)

```json
{
  "answer": "근거 기반 응답",
  "contexts": [{"doc_id": "V-02", "text": "..."}],
  "trace": [{"step": "retrieve", "input": "...", "output": "..."}]
}
```

`contexts` 와 `trace` 는 RAGAS 산출과 Observability 채점의 입력이다. **비워두거나 생략하지 않는다.**

`trace[].step` 은 **아래 6개로 고정**(SERVICE.md §3 응답 규약). 임의의 이름을 만들지 말 것.

```
guardrail_in · retrieve · retrieve_retry · tool · generate · guardrail_out
```

`contexts` 는 RAG 결과만이 아니라 **구조화 필터 도구가 반환한 `doc_id` 도 포함**한다. 근거의 출처가 도구 종류와 무관하게 일관되어야 하고, 그래야 `context_precision` 이 떨어지지 않는다.

### 디렉토리 구조

```
mini-pjt/
├── src/                        # ⬜ 미착수 · 의존은 아래 순서로만 흐른다(역방향 import 금지)
│   ├── schemas.py              # 1. 의존 0 — Pydantic 응답 모델 + AgentState TypedDict
│   ├── guardrails.py           # 2. 의존 0 — 차단 규칙 목록 + 판정 (순수 함수)
│   ├── retriever.py            # 3. RAG — 쿼리 확장 → 하이브리드(BM25+임베딩) → 리랭킹 + 재시도
│   ├── tools.py                # 4. 도메인 도구 6개 · build_tools(llm=None, retriever=None)
│   ├── agent.py                # 5. LangGraph 그래프 · build_graph(llm=None, ...)
│   └── api.py                  # 6. FastAPI · POST /query
├── data/                       # ✅ 완료
│   ├── make_data.py            #    더미 데이터 생성 스크립트 (단일 생성 지점)
│   ├── insects.json            #    곤충 50종  (doc_id: I-001~I-050)
│   ├── fishes.json             #    물고기 50종 (doc_id: F-001~F-050)
│   ├── villagers.json          #    주민 20명   (doc_id: V-01~V-20)
│   ├── personalities.md        #    성격 7종 선물 가이드 (doc_id: P-00~P-07)
│   ├── player_state.json       #    플레이어 더미 상태
│   └── player_state_no_price.json  # 매수단가 null 상태 (test id=14 전용)
├── evaluation/
│   ├── test_queries.csv        # ✅ 완료 · 채점 대상
│   ├── ragas_set.csv           # ⬜ RAGAS용 ground_truth 별도 관리
│   ├── run_eval.py             # ⬜ LLM-as-Judge 채점 러너
│   ├── round1_report.md        # ⬜ Day 9
│   └── round2_report.md        # ⬜ Day 10
├── SERVICE.md                  # ✅ 완료 · 채점 대상
├── README.md                   # ⬜ 템플릿 §4-5 준수
├── requirements.txt            # ⬜
└── Dockerfile                  # ⬜ 선택
```

### 주요 명령어

```bash
# 현재 동작하는 것
python data/make_data.py                      # 더미 데이터 재생성 (시드 고정 · 항상 같은 결과)

# 구현 후 사용할 것
uvicorn src.api:app --reload --port 8000      # API 서버
python evaluation/run_eval.py --round 1       # 자체 평가 → round1_report.md
```

---

## 이 프로젝트의 불변 규칙

아래는 전역 `~/.claude/CLAUDE.md` 의 일반 지침보다 **우선한다.**

### 데이터
- `data/*.json` 과 `personalities.md` 는 **생성물이다. 직접 편집하지 말 것.** 값을 바꾸려면 `make_data.py` 를 고치고 다시 실행한다.
- **`player_state*.json` 은 시드 전용이다. 런타임에 절대 덮어쓰지 않는다.** 최초 1회 LangGraph `Store` 에 적재한 뒤, 조회와 갱신(`sell_all_turnips` · 사용자가 알려준 매수단가)은 **전부 Store에서만** 일어난다. 파일에 쓰면 평가를 돌릴 때마다 시드가 변해 1차/2차 비교가 깨진다.
- `make_data.py` 의 `random.seed(42)` 와 `ISLAND_TODAY` 고정값을 **제거하지 말 것.** 1차/2차 평가의 데이터가 달라지면 개선폭 측정이 무의미해진다.
- 더미 데이터에는 "교육용 더미 데이터입니다"에 해당하는 안내를 남긴다 (`personalities.md` 상단 참고).
- **보유하지 않은 데이터**: 해산물, 출현 월·계절·반구·날씨. 이 조건을 요구하는 기능을 추가하지 말고, 질의가 오면 "보유하지 않은 정보"로 안내한다 (SERVICE.md 정책 2).

### 코드 (강사 지침 §코드 규약)
- 주석과 문서는 **한국어로 상세하게** 쓴다.
- **모듈 최상단에서 실제 모델을 만들지 않는다.** 모델 생성은 함수 안에서 하고, `llm=None` 인자를 받아 주입 가능하게 둔다. 평가 코드가 가짜 모델을 꽂아 실호출 없이 배선을 검증한다.
- 실제 API 키를 커밋하지 않는다. 자격증명은 환경변수로만 읽는다.

### 구현 범위 (SERVICE.md §3)

**활용 패턴 10개.** 필수 4개(1·3·11·12)를 먼저 끝낸다. 미사용은 5(MCP)·9(Multi-Agent)이며 **추가하지 말 것.**

| # | 패턴 | # | 패턴 |
|---|---|---|---|
| 1 | LCEL + Pydantic 구조화 출력 **(필수)** | 7 | HITL — `interrupt()` 승인 |
| 2 | ReAct — 도구 6개 자율 선택 | 8 | 미들웨어 — 검색 실패 시 재시도 |
| 3 | RAG 3단계 **(필수)** | 10 | 장기 메모리 — LangGraph `Store` |
| 4 | 도구 다중 결합 | 11 | Observability · Trace **(필수)** |
| 6 | 가드레일 — 입력/출력 양쪽 | 12 | 평가 — RAGAS · LLM-as-Judge **(필수)** |

**도구 선택 기준** — docstring에 이 경계를 그대로 쓴다.
> **대상 이름을 알면 `retrieve_docs`, 조건만 알면 구조화 필터**(`check_critter_availability` · `search_villagers`).

집계·필터 질의("늑대 주민 전부", "9월 생일")를 유사도 검색으로 처리하면 누락이 생긴다. 각 도구 docstring에 **"언제 쓰지 말 것"** 을 반드시 명시한다.

**검색 재시도** — `retrieve_docs` 가 빈 결과를 내면 쿼리를 재작성해 **1회만** 재시도하고, `trace` 에 `retrieve_retry` 를 남긴다. 두 번째도 비면 지어내지 말고 "확인되지 않는다"로 넘긴다(가드레일 2). 재시도는 그래프 노드가 아니라 **`retriever.py` 내부**에 둔다.

### 스펙 동기화
- `SERVICE.md` · `evaluation/test_queries.csv` 는 **채점 대상 문서**다. 도구 이름·시그니처·데이터 범위·가드레일을 바꾸면 **양쪽 모두** 갱신한다.
- `doc_id` 규칙(`I-` 곤충 / `F-` 물고기 / `V-` 주민 / `P-` 성격)은 `contexts` 응답과 CSV의 `note` 에 이미 박혀 있다. 바꾸지 않는다.

### 가드레일 (SERVICE.md §4 · 구현 필수)
1. 어뷰징(타임슬립·복사 버그·치트) 방법 설명 거절
2. 근거 없는 답변 금지 — 문서에 없으면 지어내지 않고 "확인되지 않는다"
3. 불확실 수치(미래 무 시세·친밀도 내부값) 단정 금지
4. `sell_all_turnips` 는 **실행 전 사용자 승인 필수** (LangGraph `interrupt()`)
5. 도메인 이탈(실제 주식·코인·정치) 및 프롬프트 인젝션 차단

---

## 작업 시작 전 확인

1. 지금 고치려는 것이 `SERVICE.md` 의 어느 항목에 해당하는가?
2. 그 변경이 `test_queries.csv` 의 어떤 id를 깨뜨리는가?
3. 데이터가 필요하면 `make_data.py` 에 추가하는가, 아니면 없는 정보로 안내하는가?

세 질문에 답이 안 나오면 **코드를 쓰기 전에 사용자에게 묻는다.**
