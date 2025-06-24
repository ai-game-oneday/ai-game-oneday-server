FROM python:3.11-slim

WORKDIR /app

# 시스템 패키지 업데이트 및 필요한 패키지 설치
RUN apt-get update && apt-get install -y \
  gcc \
  g++ \
  libgl1-mesa-glx \
  libglib2.0-0 \
  libgomp1 \
  libgcc-s1 \
  libc6-dev \
  pkg-config \
  && rm -rf /var/lib/apt/lists/*

# Python 의존성 복사 및 설치
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# 애플리케이션 코드 복사
COPY . .

# 포트 설정
EXPOSE 8080

# 환경변수 설정
ENV PORT=8080
ENV PYTHONPATH=/app

# 서버 실행 (수정된 부분)
CMD ["python", "server.py"]