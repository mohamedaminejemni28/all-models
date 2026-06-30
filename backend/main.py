"""
main.py

FastAPI application entry point.
- Registers all API routers
- Configures CORS for the React dev server
- Sets up logging
"""

import logging
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from routers import models, comparison, datasets, agent

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)

app = FastAPI(
    title="Gait ML Results Platform",
    description="API for visualising biomechanical gait analysis ML model results.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:3000",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(models.router)
app.include_router(comparison.router)
app.include_router(datasets.router)
app.include_router(agent.router)


@app.get("/")
def root():
    return {
        "message": "Gait ML Results Platform API",
        "docs": "/docs",
        "endpoints": [
            "/api/models",
            "/api/models/{model_name}/metrics?dataset=autism",
            "/api/models/{model_name}/features?dataset=autism",
            "/api/models/{model_name}/figures",
            "/api/comparison?dataset=autism",
            "/api/datasets",
            "/api/agent/chat",
            "/api/experiments",
        ],
    }