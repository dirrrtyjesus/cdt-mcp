FROM python:3.12-slim AS build
WORKDIR /src
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip install --no-cache-dir build && python -m build --wheel

FROM python:3.12-slim
RUN useradd --create-home --uid 1000 cdt
COPY --from=build /src/dist/*.whl /tmp/
RUN pip install --no-cache-dir /tmp/*.whl && rm /tmp/*.whl
USER cdt
VOLUME ["/state"]
ENV CDT_MCP_TRANSPORT=streamable-http \
    CDT_MCP_HOST=0.0.0.0 \
    CDT_MCP_PORT=8000 \
    CDT_MCP_STATE_DIR=/state
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s CMD python -c "import socket;s=socket.create_connection(('127.0.0.1',8000),2);s.close()"
ENTRYPOINT ["cdt-mcp"]
