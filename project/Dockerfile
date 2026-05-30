FROM python:3.11-slim

WORKDIR /app

COPY . .

RUN pip install uv
RUN uv sync

EXPOSE 8000

CMD ["uv", "run", "uvicorn", "trashnet.service.api:app", "--host", "0.0.0.0", "--port", "8000"]