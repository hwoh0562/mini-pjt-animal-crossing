"""make_data.py - 동물의 숲 무인도 생활 어시스턴트용 더미 데이터 생성 스크립트"""
import copy
import datetime
import json
import os
import random

# 재실행해도 같은 데이터가 나오도록 고정한다.
# 시드가 없으면 1차/2차 평가의 데이터가 달라져 개선폭 비교가 무의미해진다.
random.seed(42)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 1. 마을 주민 데이터 (실제 인기 주민 20명 세팅)
VILLAGERS = [
    {"id": 1, "name": "잭슨", "species": "고양이", "personality": "느끼함", "birthday": "10월 1일", "favorite_color": "블랙"},
    {"id": 2, "name": "쭈니", "species": "다람쥐", "personality": "느끼함", "birthday": "9월 29일", "favorite_color": "화이트"},
    {"id": 3, "name": "미애", "species": "아기곰", "personality": "성숙함", "birthday": "3월 10일", "favorite_color": "핑크"},
    {"id": 4, "name": "애플", "species": "햄스터", "personality": "아이돌", "birthday": "9월 24일", "favorite_color": "레드"},
    {"id": 5, "name": "뽀야미", "species": "햄스터", "personality": "친절함", "birthday": "1월 30일", "favorite_color": "핑크"},
    {"id": 6, "name": "시베리아", "species": "늑대", "personality": "무뚝뚝", "birthday": "12월 18일", "favorite_color": "블랙"},
    {"id": 7, "name": "비앙카", "species": "늑대", "personality": "성숙함", "birthday": "9월 17일", "favorite_color": "블루"},
    {"id": 8, "name": "사이다", "species": "고양이", "personality": "친절함", "birthday": "3월 27일", "favorite_color": "그레이"},
    {"id": 9, "name": "피터", "species": "사슴", "personality": "먹보", "birthday": "4월 5일", "favorite_color": "브라운"},
    {"id": 10, "name": "나탈리", "species": "사슴", "personality": "성숙함", "birthday": "1월 4일", "favorite_color": "퍼플"},
    {"id": 11, "name": "귀오미", "species": "오리", "personality": "친절함", "birthday": "3월 7일", "favorite_color": "옐로우"},
    {"id": 12, "name": "부케", "species": "고양이", "personality": "아이돌", "birthday": "2월 27일", "favorite_color": "핑크"},
    {"id": 13, "name": "패치", "species": "아기곰", "personality": "먹보", "birthday": "2월 10일", "favorite_color": "컬러풀"},
    {"id": 14, "name": "모니카", "species": "늑대", "personality": "아이돌", "birthday": "8월 31일", "favorite_color": "오렌지"},
    {"id": 15, "name": "차둘", "species": "양", "personality": "운동광", "birthday": "3월 18일", "favorite_color": "레드"},
    {"id": 16, "name": "1호", "species": "고양이", "personality": "운동광", "birthday": "8월 1일", "favorite_color": "레드"},
    {"id": 17, "name": "대장", "species": "늑대", "personality": "무뚝뚝", "birthday": "12월 19일", "favorite_color": "오렌지"},
    {"id": 18, "name": "릴리", "species": "늑대", "personality": "친절함", "birthday": "3월 24일", "favorite_color": "민트"},
    {"id": 19, "name": "솔미", "species": "사슴", "personality": "친절함", "birthday": "3월 26일", "favorite_color": "브라운"},
    {"id": 20, "name": "아폴로", "species": "독수리", "personality": "무뚝뚝", "birthday": "7월 4일", "favorite_color": "블랙"},
]

# 2. 물고기 50마리 목록
FISH_NAMES = [
    "납줄개", "붕어", "잉어", "비단잉어", "금붕어", "툭눈금붕어", "송사리", "가재", "자라", "늑대거북",
    "올챙이", "개구리", "동사리", "미꾸라지", "메기", "가물치", "블루길", "옐로우퍼치", "블랙배스", "틸라피아",
    "강꼬치고기", "빙어", "은어", "체리연어", "산천어", "열목어", "연어", "왕연어", "참게", "구피",
    "닥터피시", "엔젤피시", "베타", "네온테트라", "레인보우피시", "피라냐", "아로와나", "도라도", "가아", "피라루쿠",
    "해마", "흰동가리", "블루탱", "나비고기", "나폴레옹피시", "쏠배감펭", "복어", "가시복", "멸치", "전갱이"
]

FISH_LOCATIONS = ["강", "연못", "바다", "절벽 위 강", "하구"]
FISH_SHADOWS = ["가장 작음", "작음", "중간", "큼", "가장 큼", "등지느러미"]

FISHES = []
for i, name in enumerate(FISH_NAMES, 1):
    FISHES.append({
        "id": f"F-{i:03d}",
        "name": name,
        "location": random.choice(FISH_LOCATIONS),
        "shadow": random.choice(FISH_SHADOWS),
        "time": random.choice(["하루 종일", "AM 9시 ~ PM 4시", "PM 4시 ~ AM 9시"]),
        "price": random.choice([100, 200, 400, 800, 1500, 3000, 4000, 10000, 15000])
    })

# 3. 곤충 50마리 목록
INSECT_NAMES = [
    "배추흰나비", "노랑나비", "호랑나비", "제비나비", "청띠제비나비", "왕오색나비", "모르포나비", "아구아스나비", "붉은목도리나비", "알렉산드라비단제비나비",
    "나방", "아틀라스나방", "마다가스카르비단제비나방", "방아깨비", "메뚜기", "여치", "베짱이", "삽사리", "사마귀", "난초사마귀",
    "연꽃사마귀", "꿀벌", "말벌", "장수말벌", "개미", "소금쟁이", "물방개", "물장군", "노린재", "인면노린재",
    "무당벌레", "길앞잡이", "비단벌레", "알락하늘소", "루리하늘소", "보석거북벌레", "풍뎅이", "골리앗꽃무지", "사슴벌레", "톱사슴벌레",
    "황금사슴벌레", "뮤엘러리사슴벌레", "왕사슴벌레", "기라파톱사슴벌레", "장수풍뎅이", "코카서스장수풍뎅이", "코끼리장수풍뎅이", "헤라클레스장수풍뎅이", "파리", "모기"
]

INSECT_LOCATIONS = ["날아다님", "꽃 주변", "나무 밑동", "나무 기둥", "땅바닥", "연못", "그루터기"]

INSECTS = []
for i, name in enumerate(INSECT_NAMES, 1):
    INSECTS.append({
        "id": f"I-{i:03d}",
        "name": name,
        "location": random.choice(INSECT_LOCATIONS),
        "time": random.choice(["하루 종일", "AM 8시 ~ PM 5시", "PM 5시 ~ AM 8시", "PM 11시 ~ AM 8시"]),
        "price": random.choice([10, 100, 160, 400, 850, 2000, 3000, 8000, 10000, 12000])
    })

# 4. 성격 7종 선물 가이드 (RAG 대상 마크다운)
# villagers.json 의 personality 값과 1:1로 대응한다.
PERSONALITIES = [
    {
        "id": "P-01", "name": "느끼함",
        "trait": "자기애가 강하고 세련된 것을 좋아합니다. 칭찬에 약합니다.",
        "likes": ["정장·셔츠류 의류", "선글라스·모자 같은 액세서리", "향수와 거울"],
        "dislikes": ["흙 묻은 화석", "잡초·나뭇가지 같은 잡템"],
        "tip": "옷을 선물하면 실제로 갈아입고 나타나는 경우가 많습니다.",
    },
    {
        "id": "P-02", "name": "성숙함",
        "trait": "우아하고 도도합니다. 값어치 있는 물건을 알아봅니다.",
        "likes": ["보석·광석으로 만든 가구", "고급스러운 원피스와 드레스", "화장대·거울류"],
        "dislikes": ["저렴한 잡화", "유치한 캐릭터 소품"],
        "tip": "판매가가 높은 아이템일수록 반응이 좋습니다.",
    },
    {
        "id": "P-03", "name": "아이돌",
        "trait": "발랄하고 목소리가 큽니다. 유행에 민감합니다.",
        "likes": ["화사한 색 의류", "인형·마스코트 소품", "리본·머리 장식"],
        "dislikes": ["어둡고 무거운 색의 가구", "공포 테마 소품"],
        "tip": "선물 자체보다 자주 말을 거는 것이 친밀도에 더 크게 작용합니다.",
    },
    {
        "id": "P-04", "name": "친절함",
        "trait": "온화하고 조용합니다. 소소한 생활용품을 반깁니다.",
        "likes": ["꽃과 화분", "책·티세트 같은 아기자기한 소품", "포근한 러그와 쿠션"],
        "dislikes": ["무섭거나 공격적인 디자인의 아이템"],
        "tip": "직접 만든 DIY 가구에 특히 좋은 반응을 보입니다.",
    },
    {
        "id": "P-05", "name": "무뚝뚝",
        "trait": "까칠하지만 친해지면 가장 살갑습니다. 차분한 것을 선호합니다.",
        "likes": ["앤티크·클래식 계열 가구", "차분한 무채색 의류", "낚싯대·도구류"],
        "dislikes": ["알록달록하고 귀여운 소품", "인형류"],
        "tip": "초반에 말투가 퉁명스러워도 매일 대화하면 태도가 바뀝니다.",
    },
    {
        "id": "P-06", "name": "먹보",
        "trait": "느긋하고 먹는 것을 좋아합니다. 반응이 가장 무던합니다.",
        "likes": ["과일과 음식 아이템", "푹신한 소파·침대", "주방 가구"],
        "dislikes": ["운동기구", "격식 있는 정장"],
        "tip": "섬에서 자라지 않는 다른 종류의 과일을 주면 반응이 좋습니다.",
    },
    {
        "id": "P-07", "name": "운동광",
        "trait": "활동적이고 목소리가 큽니다. 운동 이야기를 즐깁니다.",
        "likes": ["아령·러닝머신 등 운동기구", "트레이닝복과 운동화", "스포츠 음료"],
        "dislikes": ["정적인 실내 장식품", "격식 있는 드레스"],
        "tip": "곤충을 선물해도 무난하게 반응합니다.",
    },
]

# 5. 플레이어 상태 (대화 중 사용자가 알려주면 갱신되는 더미 1인분)
ISLAND_TODAY = datetime.date(2026, 9, 16)  # 기준일 고정 (재현성)
_WEEKDAY = "월화수목금토일"


def _fmt_date(d: datetime.date) -> str:
    return f"{d.isoformat()}({_WEEKDAY[d.weekday()]})"


_THIS_SUN = ISLAND_TODAY - datetime.timedelta(days=(ISLAND_TODAY.weekday() + 1) % 7)
_LAST_SUN = _THIS_SUN - datetime.timedelta(days=7)
_LAST_FRI = _LAST_SUN + datetime.timedelta(days=5)

PLAYER_STATE = {
    "island_name": "너굴섬",
    "today": _fmt_date(ISLAND_TODAY),
    "bells": 42300,
    "turnips": {"quantity": 120, "buy_price": 88, "bought_at": _fmt_date(_THIS_SUN)},
    "history": [
        {"date": _fmt_date(_LAST_SUN), "action": "buy", "quantity": 80, "unit_price": 95},
        {"date": _fmt_date(_LAST_FRI), "action": "sell", "quantity": 80, "unit_price": 143, "profit": 3840},
        {"date": _fmt_date(_THIS_SUN), "action": "buy", "quantity": 120, "unit_price": 88},
    ],
}

# test_queries.csv id=14 전용. 무는 있는데 매수단가를 안 알려준 상태 → 에이전트가 되물어야 한다.
PLAYER_STATE_NO_PRICE = copy.deepcopy(PLAYER_STATE)
PLAYER_STATE_NO_PRICE["turnips"]["buy_price"] = None
PLAYER_STATE_NO_PRICE["history"][-1]["unit_price"] = None


def build_personalities_md(rows: list) -> str:
    """성격 가이드를 RAG 인덱싱용 마크다운으로 만듭니다. 섹션 제목의 [P-0N]이 doc_id입니다."""
    parts = [
        "# 주민 성격별 선물 가이드",
        "",
        "> 이 문서는 미니 PJT용으로 자체 작성한 더미 가이드입니다. 게임의 공식 수치가 아닙니다.",
        "",
        "## [P-00] 공통 규칙",
        "",
        "- 주민이 **좋아하는 색**(`villagers.json` 의 `favorite_color`)과 같은 색 아이템이면 반응이 한 단계 더 좋습니다.",
        "- 생일 당일에 주는 선물은 평소보다 친밀도가 크게 오릅니다.",
        "- 포장지에 싸서 주면 추가로 친밀도가 오릅니다.",
        "- 잡초·나뭇가지처럼 값이 거의 없는 아이템은 어떤 성격에게도 권하지 않습니다.",
        "",
    ]
    for row in rows:
        parts += [
            f"## [{row['id']}] {row['name']}",
            "",
            f"- **성향**: {row['trait']}",
            f"- **좋아하는 선물**: {' · '.join(row['likes'])}",
            f"- **피해야 할 선물**: {' · '.join(row['dislikes'])}",
            f"- **친밀도 팁**: {row['tip']}",
            "",
        ]
    return "\n".join(parts)


def write_json(filename: str, rows: list):
    """한글이 깨지지 않게 UTF-8과 ensure_ascii=False로 씁니다."""
    path = os.path.join(BASE_DIR, filename)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    return path


def write_text(filename: str, text: str):
    path = os.path.join(BASE_DIR, filename)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    return path

if __name__ == "__main__":
    for filename, rows in [("villagers.json", VILLAGERS),
                           ("fishes.json", FISHES),
                           ("insects.json", INSECTS)]:
        write_json(filename, rows)
        print(f"생성 완료: {filename} ({len(rows)}건)")

    for filename, state in [("player_state.json", PLAYER_STATE),
                            ("player_state_no_price.json", PLAYER_STATE_NO_PRICE)]:
        write_json(filename, state)
        print(f"생성 완료: {filename}")

    write_text("personalities.md", build_personalities_md(PERSONALITIES))
    print(f"생성 완료: personalities.md ({len(PERSONALITIES)}종)")