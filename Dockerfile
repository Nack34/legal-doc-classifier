FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY data ./data
COPY classifier ./classifier
COPY api ./api

# Train at build time so the image is self-contained and reproducible (fixed seeds).
RUN python -m data.generate && python -m classifier.train

RUN useradd --create-home appuser
USER appuser
EXPOSE 8000
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
