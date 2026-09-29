FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    HF_HOME=/srv/hf USE_TF=0 TOKENIZERS_PARALLELISM=false
WORKDIR /srv
# CPU-only torch keeps the image ~1.5 GB smaller than the default CUDA wheels.
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt .
RUN pip install -r requirements.txt
# Bake the English Laya checkpoint (~850 MB) into the image so containers start without downloading.
RUN python -c "import laya; laya.load('convaiinnovations/laya', device='cpu')"
COPY app ./app
COPY cli.py .
# Hugging Face Spaces uses port 7860; Render/Cloud Run pass $PORT.
ENV PORT=7860 DB_PATH=/srv/data/research.db HF_HUB_OFFLINE=1 LAYA_DEVICE=cpu
RUN mkdir -p /srv/data && useradd -m appuser && chown -R appuser /srv
USER appuser
EXPOSE 7860
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT} --proxy-headers --forwarded-allow-ips='*'"]
