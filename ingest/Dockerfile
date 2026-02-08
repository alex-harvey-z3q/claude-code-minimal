FROM python:3.11-slim
WORKDIR /app
COPY pyproject.toml ./
COPY src/ ./src/
RUN pip install --no-cache-dir .
CMD ["sh", "-c", "python -m ingest.main; rc=$?; sleep 10; exit $rc"]
