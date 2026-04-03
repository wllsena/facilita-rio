FROM python:3.11-slim

WORKDIR /app

# Install system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps (extracted from pyproject.toml)
RUN pip install --no-cache-dir \
    fastapi "uvicorn[standard]" jinja2 python-multipart rank-bm25 \
    faiss-cpu sentence-transformers scikit-learn openai structlog \
    prometheus-fastapi-instrumentator unidecode nltk numpy cachetools \
    ranx httpx

# Pre-download NLTK data and ML models (cached before code COPY)
RUN python -c "import nltk; nltk.download('rslp', quiet=True)"
RUN python -c "from sentence_transformers import SentenceTransformer, CrossEncoder; \
    SentenceTransformer('intfloat/multilingual-e5-small'); \
    CrossEncoder('cross-encoder/mmarco-mMiniLMv2-L12-H384-v1')"

# Non-root user
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Copy application code (changes here don't invalidate model cache)
COPY --chown=appuser:appuser servicos_selecionados.json .
COPY --chown=appuser:appuser app/ app/
COPY --chown=appuser:appuser evaluation/ evaluation/

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
