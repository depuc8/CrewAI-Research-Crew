FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    CREWAI_DISABLE_TELEMETRY=true \
    OTEL_SDK_DISABLED=true

WORKDIR /app

# Dependencies first so source edits do not invalidate this layer.
COPY pyproject.toml README.md ./
RUN mkdir research_crew && touch research_crew/__init__.py \
    && pip install ".[frontend]" \
    && pip uninstall -y research-crew

COPY research_crew ./research_crew
COPY frontend ./frontend
COPY docker/entrypoint.sh ./entrypoint.sh
RUN pip install --no-deps . && chmod +x entrypoint.sh \
    && useradd --create-home app && chown -R app /app
USER app

# One image, two services: SERVICE=api (default) or SERVICE=frontend.
ENV SERVICE=api PORT=8080
EXPOSE 8080
ENTRYPOINT ["./entrypoint.sh"]
