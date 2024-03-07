"""Main FastAPI entry point."""
import logging

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware

from src.api.app.middleware import BackendMiddleware
from src.api.app.endpoint import router


# LOGGING CONFIG SETTING
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)
logging.info("Running speech2text models.")


# APP CONFIG SET
app = FastAPI(
    title="Backend API",
    docs_url="/docs",
    openapi_url="/openapi.json",
    openapi_tags=[{
        "name": "Backend API",
        "description": "API for speech2text models factory method."
    }]
)


# MIDDLEWARE CONFIG SET
app.add_middleware(BackendMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)
app.include_router(router, prefix="/api", tags=["ai_models"])


# GREETING SITE
@app.get("/")
async def root() -> str:
    """Return a HTTP greeting.

    Returns:
        str: small html page with microphone.
    """
    html_file_path = "docs/index.html"
    
    try:
        with open(html_file_path, "r") as file:
            file_content = file.read()
        return HTMLResponse(content=file_content)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Documentation file not found.")