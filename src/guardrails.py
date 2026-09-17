"""가드레일 — 입력·출력 필터.

규칙 기반을 1차로 둔다. SERVICE.md 가 차단율 100% 를 요구하는데 LLM 판정은
확률적이라 100% 를 보장하지 못한다. 규칙에 걸리면 LLM 을 아예 호출하지 않으므로
토큰도 아낀다.

이 모듈은 다른 프로젝트 모듈을 import 하지 않는다. 순수 함수만 두어 외부 호출
없이 전량 테스트할 수 있다.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

# ── 입력 차단 규칙 ─────────────────────────────────────────────────────────
#
# 오탐이 나면 정상 질의가 막혀 통과율이 떨어진다. 그래서 단어 하나가 아니라
# 복합어로만 잡는다. 예를 들어 '시간' 은 "지금 시간에 뭐 잡혀?" 같은 정상
# 질의에 흔하므로 단독으로 쓰지 않고 '시간 조작' 형태로만 본다.

ABUSE_ALTERNATIVE = (
    "대신 지금 시각에 잡을 수 있는 생물이나, 보유하신 무의 손익 계산은 도와드릴 수 있습니다."
)

RULES: list[tuple[str, str, str]] = [
    # (사유 코드, 정규식, 사용자에게 보여줄 안내)
    (
        "abuse_timeslip",
        r"(타임\s*슬립|타임\s*트래블|time\s*travel|시간\s*조작|날짜\s*조작|"
        r"시간\s*(을\s*)?(돌려|바꿔|앞당)|날짜\s*(를\s*)?(돌려|바꿔)|본체\s*시간)",
        "게임 내 시간을 조작하는 방법은 안내해 드리지 않습니다. "
        "무 시세와 이벤트 진행이 뒤틀리고 무가 썩거나 주민이 떠나는 등 "
        "되돌리기 어려운 부작용이 따르기 때문입니다. " + ABUSE_ALTERNATIVE,
    ),
    (
        "abuse_exploit",
        r"(아이템\s*복사|복사\s*버그|복제\s*버그|증식\s*버그|무한\s*복사|벨\s*복사|"
        r"세이브\s*(데이터\s*)?(변조|수정|편집)|세이브\s*에디터|치트\s*(기기|코드)?|"
        r"핵\s*(툴|프로그램)|커스텀\s*펌웨어|에뮬레이터)",
        "버그나 치트를 이용한 방법은 안내해 드리지 않습니다. "
        "세이브 데이터가 손상되거나 온라인 이용이 제한될 수 있고, "
        "다른 플레이어에게도 피해가 가기 때문입니다. " + ABUSE_ALTERNATIVE,
    ),
    (
        "out_of_domain",
        # '무 주식' 은 게임 용어라 오탐이 나지 않도록 '주식' 앞에 '무' 가 오면 제외한다.
        r"((?<!무)(?<!무\s)주식|주가|코스피|코스닥|나스닥|s&p|비트코인|이더리움|"
        r"암호\s*화폐|가상\s*화폐|코인\s*(시세|투자)|환율|금리|"
        r"대통령|국회|선거|정치)",
        "게임 밖의 주식·가상화폐·정치 같은 주제는 다루지 않습니다. "
        "동물의 숲 도감, 주민 선물, 무 거래에 대해 물어봐 주세요.",
    ),
    (
        "prompt_injection",
        r"(시스템\s*프롬프트|system\s*prompt|프롬프트를?\s*(공개|출력|보여|알려)|"
        r"(이전|위의?|앞의?)\s*(지시|명령|규칙|설정)\w*\s*(을|를)?\s*(무시|잊)|"
        r"ignore\s+(all\s+)?(previous|prior|above)|"
        r"너의?\s*(지시문|설정|규칙)\w*\s*(을|를)?\s*(알려|보여|출력))",
        "요청하신 내용은 도와드릴 수 없습니다. 섬 생활에 대한 질문은 계속 받고 있습니다.",
    ),
]

_COMPILED = [(code, re.compile(pattern, re.IGNORECASE), message)
             for code, pattern, message in RULES]


@dataclass
class GuardDecision:
    """가드레일 판정 결과."""

    blocked: bool
    reason: str = ""      # 차단 사유 코드. 통과면 빈 문자열.
    message: str = ""     # 차단 시 사용자에게 보여줄 안내


def check_input(text: str) -> GuardDecision:
    """사용자 질의를 규칙으로 검사한다.

    걸리면 LLM 을 한 번도 호출하지 않고 즉시 차단한다.
    """
    for code, pattern, message in _COMPILED:
        if pattern.search(text):
            return GuardDecision(blocked=True, reason=code, message=message)
    return GuardDecision(blocked=False)


# ── 출력 검사 ──────────────────────────────────────────────────────────────

DOC_ID_RE = re.compile(r"\b[IFVP]-\d{2,3}\b")

# 답변에 새어 나오면 안 되는 내부 문구.
_LEAK_MARKERS = (
    "너는 '모여봐요 동물의 숲' 플레이어를 돕는",
    "반드시 지킬 것",
    "쓰지 말아야 할 때",
)


def check_output(answer: str, context_doc_ids: list[str]) -> tuple[str, GuardDecision]:
    """생성된 답변을 검사하고, 고칠 수 있으면 고쳐서 돌려준다.

    반환: (보정된 답변, 판정)

    - 시스템 프롬프트가 새어 나갔으면 답변을 통째로 버린다.
    - 근거 문서가 있는데 답변이 하나도 인용하지 않았으면, 실패시키는 대신
      출처 줄을 덧붙인다. 정책 2 가 요구하는 것은 '출처 명시'이지
      '답변 폐기'가 아니기 때문이다.
    """
    for marker in _LEAK_MARKERS:
        if marker in answer:
            return (
                "요청하신 내용은 도와드릴 수 없습니다. 섬 생활에 대한 질문은 계속 받고 있습니다.",
                GuardDecision(blocked=True, reason="prompt_leak"),
            )

    if context_doc_ids and "출처" not in answer:
        # 본문에 (V-02) 처럼 id 를 흘려 적는 것만으로는 출처를 밝혔다고 보기
        # 어렵다. 근거가 있으면 '출처:' 줄을 항상 덧붙여 명시적으로 만든다.
        unique = list(dict.fromkeys(context_doc_ids))
        shown = " · ".join(unique[:10]) + (f" 외 {len(unique) - 10}건" if len(unique) > 10 else "")
        return (f"{answer}\n\n출처: {shown}",
                GuardDecision(blocked=False, reason="citation_appended"))

    return answer, GuardDecision(blocked=False)


# ── 승인 의사 판정 (HITL) ──────────────────────────────────────────────────

_YES = re.compile(
    r"(^|\s)(응|어|네|넵|예|웅|그래|좋아|오케이|ok|okay|yes|y)(\s|$|[.!~])|"
    r"(진행|승인|실행|동의|확인)해?|팔아|매도해?|판다|계속"
)
_NO = re.compile(
    r"(아니|아뇨|노|no|취소|그만|하지\s*마|안\s*팔|중단|멈춰|보류|나중에|싫)"
)


def parse_consent(text: str) -> bool | None:
    """승인 대기 상태에서 사용자의 답을 판정한다.

    True=승인 · False=거절 · None=판단 불가(되물어야 함)

    LLM 에 맡기지 않는 이유: 오판 한 번에 되돌릴 수 없는 재화 기록이 바뀐다.
    애매하면 진행하지 않고 되묻는 쪽이 항상 안전하다.
    부정이 긍정보다 우선한다. "응 아니야" 같은 입력은 거절로 본다.
    """
    lowered = text.strip().lower()
    if _NO.search(lowered):
        return False
    if _YES.search(lowered):
        return True
    return None
