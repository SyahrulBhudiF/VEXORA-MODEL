from fastapi import FastAPI
from .api.routes import router

app = FastAPI(
    title="Emotion Detection API",
    description="API for detecting emotions from facial images",
    version="1.0.0"
)

app.include_router(router)
