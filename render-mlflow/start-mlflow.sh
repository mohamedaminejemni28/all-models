#!/bin/sh
set -eu

: "${PORT:=5000}"
: "${MLFLOW_BACKEND_STORE_URI:?MLFLOW_BACKEND_STORE_URI is required}"
: "${MLFLOW_ARTIFACT_ROOT:?MLFLOW_ARTIFACT_ROOT is required}"
: "${MLFLOW_ADMIN_USERNAME:=admin}"
: "${MLFLOW_ADMIN_PASSWORD:?MLFLOW_ADMIN_PASSWORD is required}"
: "${MLFLOW_WORKERS:=1}"

cat > /app/basic_auth.ini <<EOF
[mlflow]
admin_username = ${MLFLOW_ADMIN_USERNAME}
admin_password = ${MLFLOW_ADMIN_PASSWORD}
default_permission = READ
database_uri = ${MLFLOW_AUTH_DATABASE_URI:-${MLFLOW_BACKEND_STORE_URI}}
authorization_function = mlflow.server.auth:authenticate_request_basic
EOF

export MLFLOW_AUTH_CONFIG_PATH=/app/basic_auth.ini

mlflow server \
  --app-name basic-auth \
  --host 0.0.0.0 \
  --port "${PORT}" \
  --backend-store-uri "${MLFLOW_BACKEND_STORE_URI}" \
  --default-artifact-root "${MLFLOW_ARTIFACT_ROOT}" \
  --workers "${MLFLOW_WORKERS}"
