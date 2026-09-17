# Python 3.12 를 쓰는 이유
#   로컬 개발 환경은 3.14 지만, 3.14 는 아직 휠이 없는 패키지가 있어
#   (scikit-network 등) 소스 빌드로 넘어가면 C++ 컴파일러가 필요해진다.
#   3.12 는 이 스택 전체에 manylinux 휠이 있어 컴파일러 없이 설치된다.
FROM python:3.12-slim

# 파이썬이 .pyc 를 쓰지 않고 로그를 버퍼링하지 않게 한다(컨테이너 로그 즉시 확인).
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

WORKDIR /app

# 의존성을 먼저 깔아 레이어 캐시를 살린다. 소스만 바뀌면 이 레이어는 재사용된다.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY data/ ./data/
COPY evaluation/ ./evaluation/
COPY SERVICE.md README.md ./

# 더미 데이터를 생성해 이미지에 넣어 둔다. 시드가 고정이라 항상 같은 결과다.
RUN python data/make_data.py

EXPOSE 8000

# 색인은 빌드가 아니라 실행 시점에 한다. Titan 임베딩 호출에 자격증명이
# 필요한데, 빌드 때 키를 넣으면 이미지에 남기 때문이다.
# 이미 색인돼 있으면 retriever 가 알아서 건너뛴다.
CMD ["sh", "-c", "python -m src.retriever && exec uvicorn src.api:app --host 0.0.0.0 --port 8000"]
