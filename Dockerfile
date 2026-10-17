FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    JUDGE_MODE=auto \
    RANDOM_SEED=42 \
    BOOTSTRAP_ITERATIONS=1000

WORKDIR /app

# Install dependencies first so this layer is cached across source edits.
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy the pipeline. docker-compose additionally bind-mounts the repo over
# /app so generated artifacts land back on the host's results/ directory.
COPY . .

# Default: reproduce the frozen-test evaluation. With no OPENAI_API_KEY the
# judge runs in deterministic mock mode, so this always exits 0 offline.
CMD ["python", "evaluate.py", "--split", "test"]
