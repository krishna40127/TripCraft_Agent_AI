# TripCraft -- single-container setup: the FastAPI app and the weather MCP
# server run in the same container (mcp_client.py spawns the MCP server as
# a subprocess over stdio, so both need to live together, same as the
# reference project's layout).
FROM python:3.11-slim

WORKDIR /app

# System deps: none beyond what pip needs -- httpx/mcp/langgraph are pure
# Python + a couple of C-extension wheels available for slim images.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV HOST=0.0.0.0 \
    PORT=8000 \
    PYTHONUNBUFFERED=1

EXPOSE 8000

# .env is not copied in (see .dockerignore) -- pass config via `docker run
# --env-file .env` or your platform's env var settings instead.
CMD ["python", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
