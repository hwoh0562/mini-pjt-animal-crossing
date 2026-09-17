"""Bedrock LLM 팩토리와 응답 캐시.

retriever(쿼리 확장·리랭킹)와 agent(ReAct·생성)가 모두 LLM 을 쓴다.
한쪽이 다른 쪽을 import 하지 않도록 생성 지점을 여기로 모은다.

모듈 최상단에서 모델을 만들지 않는다. build_llm(llm=...) 으로 가짜 모델을
주입하면 Bedrock 실호출 없이 배선을 검증할 수 있다.
"""
from __future__ import annotations

import os

DEFAULT_MODEL_ID = "global.anthropic.claude-sonnet-4-5-20250929-v1:0"


def build_llm(llm=None, *, max_tokens: int = 1024, temperature: float = 0.0):
    """ChatBedrockConverse 를 만든다. 주입받으면 그대로 돌려준다.

    모델 ID 는 .env 의 BEDROCK_MODEL_ID 로 주입한다. 하드코딩하지 않는 이유는
    추론 프로파일 접두사(us. / global.)마다 일일 토큰 쿼터가 따로라,
    한쪽이 소진되면 .env 한 줄로 갈아탈 수 있어야 하기 때문이다.
    """
    if llm is not None:
        return llm
    from langchain_aws import ChatBedrockConverse

    return ChatBedrockConverse(
        model=os.getenv("BEDROCK_MODEL_ID", DEFAULT_MODEL_ID),
        region_name=os.getenv("AWS_DEFAULT_REGION", "us-east-1"),
        max_tokens=max_tokens,
        temperature=temperature,
        config=bedrock_retry_config(),
    )


def bedrock_retry_config():
    """ThrottlingException 에 대비한 적응형 재시도 설정.

    Bedrock 은 일일 토큰 한도에 가까워지면 ThrottlingException 을 던진다.
    기본 재시도 4회로는 평가를 한 바퀴 도는 중에 터져 결과가 통째로 망가진다.
    adaptive 모드는 응답을 보고 호출 속도를 스스로 늦춘다.
    """
    from botocore.config import Config

    return Config(retries={"max_attempts": 10, "mode": "adaptive"})


def enable_llm_cache(path: str = "llm_cache.sqlite") -> None:
    """LLM 응답을 SQLite 에 캐싱한다.

    질의 하나당 LLM 호출이 5~8회라 테스트 20건을 반복하면 일일 토큰 한도에
    닿는다. 같은 프롬프트를 다시 보내면 호출 없이 캐시에서 돌려주므로,
    케이스 하나만 고쳐 재실행할 때 나머지는 공짜가 된다.
    """
    from langchain_community.cache import SQLiteCache
    from langchain_core.globals import set_llm_cache

    set_llm_cache(SQLiteCache(database_path=path))
