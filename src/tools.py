"""도메인 도구 6개.

도구 선택 기준 (SERVICE.md §3)
  대상 이름을 알면  → retrieve_docs
  조건만 알면      → check_critter_availability · search_villagers

집계·필터 질의("늑대 주민 전부", "9월 생일")를 유사도 검색으로 처리하면
조건을 만족하는 대상이 누락된다. 그래서 구조화 필터를 따로 둔다.

모든 도구는 {"data": ..., "docs": [{doc_id, text}]} 형태로 돌려준다.
툴 노드가 docs 를 모아 응답의 contexts 를 채우므로, 근거의 출처가 도구
종류와 무관하게 일관된다.

이 모듈의 도구 함수는 LangGraph 런타임을 모른다. interrupt() 는 툴 노드에
두어, 가짜 store 만 꽂으면 그래프 없이 단위 테스트가 되도록 했다.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from langchain_core.tools import tool

from src.retriever import DATA_DIR, build_retriever, villager_doc_id

# Store 안에서 플레이어 상태가 사는 자리
PLAYER_NS = ("player",)
PLAYER_KEY = "state"

# 승인 없이 실행하면 안 되는 도구. 툴 노드가 이 목록을 보고 interrupt() 한다.
DANGEROUS_TOOLS = {"sell_all_turnips"}


def _result(data, docs=None) -> dict:
    return {"data": data, "docs": docs or []}


def _load(filename: str):
    with open(DATA_DIR / filename, encoding="utf-8") as f:
        return json.load(f)


# ── 시간대 파싱 ────────────────────────────────────────────────────────────


def _to_hour24(token: str) -> int:
    """'PM 4시' → 16, 'AM 9시' → 9 로 바꾼다."""
    m = re.match(r"(AM|PM)\s*(\d+)\s*시", token.strip())
    if not m:
        raise ValueError(f"시간 형식을 해석할 수 없습니다: {token!r}")
    period, hour = m.group(1), int(m.group(2))
    if period == "AM":
        return 0 if hour == 12 else hour
    return 12 if hour == 12 else hour + 12


def hour_matches(hour: int, time_str: str) -> bool:
    """주어진 시각이 출현 시간대에 드는지 본다.

    'PM 4시 ~ AM 9시' 처럼 자정을 넘는 구간을 다뤄야 하므로, 시작이 끝보다
    크면 하루를 넘어간 것으로 보고 조건을 뒤집는다.
    """
    time_str = time_str.strip()
    if time_str == "하루 종일":
        return True
    start_s, _, end_s = time_str.partition("~")
    start, end = _to_hour24(start_s), _to_hour24(end_s)
    if start <= end:
        return start <= hour < end
    return hour >= start or hour < end


def _month_of(birthday: str) -> int:
    """'9월 29일' → 9"""
    return int(birthday.split("월")[0])


# ── 도구 조립 ──────────────────────────────────────────────────────────────


def build_tools(llm=None, retriever=None, store=None) -> list:
    """도구 6개를 만들어 돌려준다. 모든 의존을 주입받는다.

    모듈 최상단에서 만들지 않는 이유는, import 만으로 Bedrock 이나 Chroma 에
    붙지 않게 하고 평가 코드가 가짜 객체를 꽂을 수 있게 하기 위함이다.
    """
    _retriever = retriever if retriever is not None else build_retriever(llm=llm)

    insects = _load("insects.json")
    fishes = _load("fishes.json")
    villagers = _load("villagers.json")

    # ── 1. RAG 검색 ──
    @tool
    def retrieve_docs(query: str, doc_type: str | None = None) -> dict:
        """이름을 아는 대상의 설명을 도감·가이드에서 찾는다.

        "무당벌레 어디서 잡혀?", "쭈니는 어떤 주민이야?", "무뚝뚝 성격은 뭘 좋아해?"
        처럼 찾으려는 대상의 이름을 이미 아는 질문에 쓴다.

        쓰지 말아야 할 때:
          - "지금 뭐 잡혀?" 처럼 조건만 주어진 경우 → check_critter_availability
          - "늑대 주민 전부", "9월 생일" 같은 집계·필터 → search_villagers
          이 도구는 유사도 상위 몇 건만 돌려주므로 '조건을 만족하는 전원'을
          보장하지 못한다.

        Args:
            query: 찾으려는 내용
            doc_type: insect · fish · villager · personality 중 하나로 범위를 좁힐 때
        """
        res = _retriever.search(query, doc_type=doc_type)
        if not res.docs:
            return _result({"found": False,
                            "message": "보유한 자료에서 확인되지 않습니다."})
        return _result(
            [{"doc_id": d.doc_id, "text": d.text} for d in res.docs],
            [{"doc_id": d.doc_id, "text": d.text} for d in res.docs],
        )

    # ── 2. 생물 구조화 필터 ──
    @tool
    def check_critter_availability(category: str, hour: int,
                                   location: str | None = None,
                                   min_price: int | None = None) -> dict:
        """지금(또는 지정한 시각)에 잡을 수 있는 생물을 조건으로 추려낸다.

        "지금 잡을 수 있는 곤충", "저녁 8시에 낚이는 물고기", "지금 잡히는 것 중
        제일 비싼 것" 처럼 이름이 아니라 조건이 주어진 질문에 쓴다.

        쓰지 말아야 할 때:
          - 특정 생물의 이름을 알고 그 정보를 물을 때 → retrieve_docs
          - 주민에 대한 질문 → search_villagers

        보유하지 않은 조건: 출현 월·계절·반구·날씨. 해산물 자료도 없다.
        이런 조건을 요구받으면 이 도구를 쓰지 말고 자료가 없다고 안내하라.

        Args:
            category: '곤충' 또는 '물고기'
            hour: 0~23 의 24시간제 시각
            location: '꽃 주변' '강' 처럼 장소로 더 좁힐 때
            min_price: 이 판매가 이상만 볼 때
        """
        cat = category.strip()
        rows = insects if cat in ("곤충", "insect") else fishes if cat in ("물고기", "fish") else None
        if rows is None:
            return _result({"error": f"category 는 '곤충' 또는 '물고기' 여야 합니다: {category!r}"})

        hits = [r for r in rows if hour_matches(hour, r["time"])]
        if location:
            hits = [r for r in hits if location in r["location"]]
        if min_price is not None:
            hits = [r for r in hits if r["price"] >= min_price]
        hits.sort(key=lambda r: -r["price"])

        docs = [{"doc_id": r["id"],
                 "text": f"{r['name']}({r['id']}) · {r['location']} · {r['time']} · {r['price']}벨"}
                for r in hits]
        return _result({"count": len(hits), "hour": hour, "critters": hits}, docs)

    # ── 3. 주민 구조화 필터 ──
    @tool
    def search_villagers(species: str | None = None,
                         personality: str | None = None,
                         birth_month: int | None = None) -> dict:
        """조건에 맞는 주민을 빠짐없이 찾는다.

        "늑대 주민 누구누구 있어?", "9월에 생일인 주민?", "느끼함 성격 중 3월 생일?"
        처럼 조건으로 주민을 추리는 질문에 쓴다. 조건을 만족하는 주민 전원을
        돌려주며, 없으면 빈 목록을 돌려준다. 빈 목록은 '해당 없음'이라는 확정된
        답이므로 다른 주민으로 대신 채우지 마라.

        쓰지 말아야 할 때:
          - 주민 이름을 이미 아는 경우 → retrieve_docs
            ("쭈니 선물 추천"은 이 도구가 아니라 retrieve_docs 다)

        Args:
            species: 종족. 예: '늑대' '고양이' '사슴'
            personality: 성격. 느끼함·성숙함·아이돌·친절함·무뚝뚝·먹보·운동광 중 하나
            birth_month: 생일 월(1~12)
        """
        hits = villagers
        if species:
            hits = [v for v in hits if v["species"] == species]
        if personality:
            hits = [v for v in hits if v["personality"] == personality]
        if birth_month is not None:
            hits = [v for v in hits if _month_of(v["birthday"]) == birth_month]

        docs = [{"doc_id": villager_doc_id(v["id"]),
                 "text": (f"{v['name']}({villager_doc_id(v['id'])}) · {v['species']} · "
                          f"{v['personality']} · {v['birthday']} · {v['favorite_color']}")}
                for v in hits]
        return _result({"count": len(hits),
                        "villagers": [{**v, "doc_id": villager_doc_id(v["id"])} for v in hits]},
                       docs)

    # ── 4. 플레이어 상태 조회 ──
    @tool
    def get_player_state() -> dict:
        """플레이어의 보유 벨·무 보유량·매수단가·거래 이력을 조회한다.

        "내 벨 얼마야?", "지금 무 팔면 이득이야?" 처럼 플레이어 본인의 상태가
        필요한 질문에 쓴다. 무 손익을 계산하기 전에 먼저 이 도구로 보유 수량과
        매수단가를 확인하라.

        매수단가(buy_price)가 null 이면 기록이 없다는 뜻이다. 임의의 값을
        가정해 계산하지 말고 사용자에게 매수단가와 현재 시세를 되물어라.
        """
        return _result(_read_state(store))

    # ── 5. 무 손익 계산 ──
    @tool
    def calculate_turnip_profit(quantity: int, buy_price: int, current_price: int) -> dict:
        """무 매도 손익을 계산한다.

        Args:
            quantity: 보유 개수
            buy_price: 개당 매수단가(벨)
            current_price: 현재 너굴상점 매입가(벨)
        """
        cost = quantity * buy_price
        revenue = quantity * current_price
        profit = revenue - cost
        return _result({
            "quantity": quantity,
            "buy_price": buy_price,
            "current_price": current_price,
            "cost": cost,
            "revenue": revenue,
            "profit": profit,
            "roi_percent": round(profit / cost * 100, 1) if cost else None,
        })

    # ── 6. 무 전량 매도 (위험 작업) ──
    @tool
    def sell_all_turnips(current_price: int) -> dict:
        """보유한 무를 전량 매도 처리하고 플레이어 상태를 갱신한다.

        되돌릴 수 없는 작업이라 반드시 사용자 승인을 받은 뒤에만 실행된다.
        승인 절차는 그래프가 처리하므로, 사용자가 매도를 요청하면 이 도구를
        호출하면 된다. 승인 전 단계에서는 예상 수령 벨만 안내된다.

        게임 본체를 조작하지는 않는다. 어시스턴트가 관리하는 기록을 바꾼다.

        Args:
            current_price: 현재 너굴상점 매입가(벨)
        """
        state = _read_state(store)
        qty = state["turnips"]["quantity"]
        if qty <= 0:
            return _result({"sold": 0, "message": "보유한 무가 없습니다."})

        buy_price = state["turnips"].get("buy_price")
        revenue = qty * current_price
        profit = revenue - qty * buy_price if buy_price else None

        state["bells"] += revenue
        state["turnips"] = {"quantity": 0, "buy_price": None, "bought_at": None}
        state.setdefault("history", []).append({
            "date": state.get("today"), "action": "sell",
            "quantity": qty, "unit_price": current_price, "profit": profit,
        })
        _write_state(store, state)
        return _result({"sold": qty, "unit_price": current_price,
                        "revenue": revenue, "profit": profit,
                        "new_bells": state["bells"]})

    return [retrieve_docs, check_critter_availability, search_villagers,
            get_player_state, calculate_turnip_profit, sell_all_turnips]


# ── 플레이어 상태 (Store) ──────────────────────────────────────────────────


def seed_player_state(store, seed_file: str = "player_state.json") -> dict:
    """시드 JSON 을 Store 에 한 번 적재한다.

    시드 파일은 런타임에 절대 덮어쓰지 않는다. 평가를 몇 번 돌려도 시드가
    그대로 남아야 1차·2차 비교가 성립하기 때문이다.
    """
    with open(DATA_DIR / seed_file, encoding="utf-8") as f:
        state = json.load(f)
    store.put(PLAYER_NS, PLAYER_KEY, state)
    return state


def _read_state(store) -> dict:
    if store is None:
        raise RuntimeError("store 가 주입되지 않았습니다. build_tools(store=...) 를 쓰세요.")
    item = store.get(PLAYER_NS, PLAYER_KEY)
    if item is None:
        return seed_player_state(store)
    return json.loads(json.dumps(item.value))  # 호출자가 원본을 건드리지 않도록 복사


def _write_state(store, state: dict) -> None:
    store.put(PLAYER_NS, PLAYER_KEY, state)
