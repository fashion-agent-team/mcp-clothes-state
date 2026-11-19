# 젯슨(JetPack L4T R39, CUDA 13)용. torch는 젯슨 나노에서 GPU로 확인한 것과 같은 cu132 aarch64 휠을 씀.
# GPU 드라이버는 nvidia 런타임이 호스트에서 붙여 줌(compose: runtime: nvidia).
# 모델 가중치는 이미지에 넣지 않고 호스트 폴더를 읽기 전용으로 붙여 씀(SO400M_WEIGHTS).
#   docker build -t fashion-agent/so400m-embed:local so400m-embed
FROM python:3.12-slim-bookworm
ENV PYTHONUNBUFFERED=1 TZ=Asia/Seoul HF_HUB_OFFLINE=1 PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /app
RUN pip install torch==2.15.0 --index-url https://download.pytorch.org/whl/cu132
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY app.py .
EXPOSE 8096
USER 1000
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s CMD python3 -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8096/health', timeout=4)"
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8096", "--workers", "1"]
