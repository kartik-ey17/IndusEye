"""FastAPI entry point for the local SentinelAI MVP."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.api import router
from backend.config import settings
from backend.db import IncidentRepository
from backend.services import SentinelRuntime

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


@asynccontextmanager
async def lifespan(app: FastAPI):
    repository = IncidentRepository(settings.database_path)
    repository.initialize()
    runtime = SentinelRuntime(settings, repository)
    runtime.load_model()  # Model failure becomes /api/health degraded, never an import-time crash.
    app.state.runtime = runtime
    yield
    runtime.close()


app = FastAPI(title="SentinelAI API", version="0.1.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000", "http://localhost:3010", "http://127.0.0.1:3010", "http://localhost:3011", "http://127.0.0.1:3011"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)
app.include_router(router)
