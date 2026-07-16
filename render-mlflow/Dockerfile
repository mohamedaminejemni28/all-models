FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1

RUN pip install --no-cache-dir \
    "mlflow-skinny[auth]" \
    psycopg2-binary \
    sqlalchemy \
    alembic \
    numpy \
    pandas \
    aiohttp \
    flask-cors

WORKDIR /app
COPY start-mlflow.sh /app/start-mlflow.sh
RUN chmod +x /app/start-mlflow.sh

EXPOSE 5000

CMD ["/app/start-mlflow.sh"]
