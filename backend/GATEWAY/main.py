from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.GATEWAY.observation_api import (
    router as observation_router,
)

from backend.GATEWAY.calidad_api import (
    router as calidad_router,
)


app = FastAPI(
    title="SIA Intelligence API",
    version="0.1.0",
)


# -------------------------------------------------
# CORS - Desarrollo local
# -------------------------------------------------
# Permite que Vite funcione desde localhost o
# 127.0.0.1 independientemente del puerto asignado.
# -------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=(
        r"^http://(localhost|127\.0\.0\.1):\d+$"
    ),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=[
        "Content-Disposition",
    ],
)


# -------------------------------------------------
# Routers
# -------------------------------------------------

app.include_router(
    observation_router
)

app.include_router(
    calidad_router
)


@app.get("/")
def root():
    return {
        "status": "ok",
        "message": "SIA Intelligence API funcionando",
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
    }