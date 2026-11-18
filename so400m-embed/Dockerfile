# 젯슨(JetPack 6, L4T r36) GPU용. 모델 가중치는 이미지에 넣지 않고 호스트 HF 캐시를 읽기 전용으로 붙여 씀.
#   docker build -t fashion-agent/so400m-embed:local so400m-embed
ARG BASE=dustynv/l4t-pytorch:r36.4.0
FROM ${BASE}
ENV PYTHONUNBUFFERED=1 TZ=Asia/Seoul HF_HUB_OFFLINE=1 PIP_NO_CACHE_DIR=1
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY app.py .
EXPOSE 8096
USER 1000
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s CMD python3 -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8096/health', timeout=4)"
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8096", "--workers", "1"]
