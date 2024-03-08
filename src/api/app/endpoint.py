"""Main FastAPI entry point."""
from fastapi import APIRouter, File, UploadFile
from fastapi.responses import JSONResponse

from src.ai_models.speech2text import MODELS_FACTORY


router = APIRouter()


@router.get("/get-model-names/", response_model=list)
def get_model_names() -> list:
    """Return a list of model names."""
    return MODELS_FACTORY.get_model_names()


@router.get("/get-current-model/", response_model=str)
def get_current_model() -> str:
    """Return the name of the current model."""
    return MODELS_FACTORY.get_model().get_model_name()


@router.post("/speech-to-text/", response_model=str)
def speech_to_text(audio: UploadFile = File(...)) -> str:
    """Predict function."""
    result = MODELS_FACTORY.get_model()(audio.file)
    return JSONResponse(
        status_code=200,
        content={"result": result}
    )
