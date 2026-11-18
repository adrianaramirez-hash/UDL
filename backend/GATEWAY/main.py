from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from backend.CONFIG.auth import auth_settings
from backend.GATEWAY.auth_api import router as auth_router

from backend.GATEWAY.observation_api import (
    router as observation_router,
)

from backend.GATEWAY.calidad_api import (
    router as calidad_router,
)
from backend.GATEWAY.institutional_services_api import router as institutional_services_router
from backend.GATEWAY.academic_services_api import router as academic_services_router
from backend.GATEWAY.facilities_api import router as facilities_router
from backend.GATEWAY.school_environment_api import router as school_environment_router
from backend.GATEWAY.satisfaction_api import router as satisfaction_router
from backend.GATEWAY.language_center_api import router as language_center_router
from backend.GATEWAY.continuing_education_api import router as continuing_education_router
from backend.GATEWAY.results_report_api import router as results_report_router


app = FastAPI(
    title="SIA Intelligence API",
    version="0.1.0",
)


# -------------------------------------------------
# CORS para el origen configurado de la aplicación
# -------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=[auth_settings.public_origin] if auth_settings.public_origin else [],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
    expose_headers=[
        "Content-Disposition",
    ],
)


@app.middleware("http")
async def prevent_api_caching(request: Request, call_next):
    response = await call_next(request)
    if request.url.path.startswith("/api/"):
        response.headers["Cache-Control"] = "no-store"
        response.headers["Referrer-Policy"] = "no-referrer"
    return response


# -------------------------------------------------
# Routers
# -------------------------------------------------

app.include_router(auth_router)
app.include_router(
    observation_router
)

app.include_router(
    calidad_router
)
app.include_router(institutional_services_router)
app.include_router(academic_services_router)
app.include_router(facilities_router)
app.include_router(school_environment_router)
app.include_router(satisfaction_router)
app.include_router(language_center_router)
app.include_router(continuing_education_router)
app.include_router(results_report_router)


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
