# API d'interprétation seule (Hugging Face Spaces « Docker » ou Render) — port 7860
FROM python:3.11-slim
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY core ./core
COPY api ./api
COPY artifacts/bundle.joblib ./artifacts/bundle.joblib
ENV MODEL_BUNDLE=/app/artifacts/bundle.joblib
EXPOSE 7860
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "7860"]
