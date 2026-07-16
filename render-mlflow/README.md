# MLflow on Render

This folder contains a small Docker deployment for running MLflow on Render with:

- PostgreSQL backend metadata storage
- MLflow basic authentication
- configurable admin username/password through Render environment variables

## Render setup

1. Create a PostgreSQL database on Render.
2. Copy the database's **Internal Database URL**.
3. Create a new Render **Web Service** from this repository.
4. Set **Runtime** to `Docker`.
5. If this folder is not the repository root, set Render's **Root Directory** to:

   ```text
   render-mlflow
   ```

6. Add these environment variables:

   ```text
   MLFLOW_BACKEND_STORE_URI=<Render Internal Database URL>
   MLFLOW_ADMIN_USERNAME=admin
   MLFLOW_ADMIN_PASSWORD=<choose a strong password>
   ```

Optional:

```text
MLFLOW_AUTH_DATABASE_URI=<same PostgreSQL URL or a separate PostgreSQL URL>
MLFLOW_ARTIFACT_ROOT=<persistent artifact location>
```

For a first test, you can omit `MLFLOW_AUTH_DATABASE_URI`; the auth database will use the same PostgreSQL URL as the MLflow backend store.

## Important artifact note

This deployment persists MLflow experiment metadata and login data in PostgreSQL.
It does not yet configure durable artifact storage for model files, plots, or other large logged files.
For production use, configure `MLFLOW_ARTIFACT_ROOT` with S3, Google Cloud Storage, Azure Blob Storage, or another persistent artifact store.
