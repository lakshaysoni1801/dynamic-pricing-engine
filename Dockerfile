FROM python:3.12-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
ENV PYTHONPATH=/app/src PE_REGISTRY_DIR=/app/artifacts/registry

# Train + register a model at build time so the image ships with a production model,
# then serve it. In a real deployment training and serving would be separate stages;
# this keeps the demo image self-contained.
RUN python eval/run_eval.py

EXPOSE 8000
CMD ["uvicorn", "pricing_engine.service:app", "--host", "0.0.0.0", "--port", "8000"]
