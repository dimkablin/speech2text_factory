"""Main FastAPI entry point."""
from typing import Any, Dict
from fastapi import APIRouter, File, UploadFile
from fastapi.responses import JSONResponse

from src.ai_models.speech2text import MODELS_FACTORY
from src.ai_models.ocr_models.easyocr import EasyOCRDefault
from src.ai_models.ocr_models.image_process import save_upload_file

router = APIRouter()


@router.get("/get-model-names/", response_model=list)
def get_model_names() -> list:
    """Return a list of model names."""
    return MODELS_FACTORY.get_model_names()

#Initialize easyOCR
@router.get("/get-ocr-model-name/", response_model=str, tags=['OCR'])
def get_model_names() -> list:
    """Return a list of model names."""
    model = EasyOCRDefault()
    return model.get_model_type()

@router.post("/ocr-model-inference/", response_model=list, tags=['OCR'])
def image_to_result(image: UploadFile = File(...)) -> list:
    """Return OCR model inference"""
    inputs = save_upload_file(image)
    model = EasyOCRDefault()
    return model(inputs)

@router.get("/get-current-model/", response_model=str)
def get_current_model() -> str:
    """Return the name of the current model."""
    return MODELS_FACTORY.get_model().get_model_name()

@router.post("/speech-to-text/", response_model=str)
def speech_to_text(audio: UploadFile = File(...)) -> str:
    """Predict function."""
    result = MODELS_FACTORY(audio.file)
    return JSONResponse(
        status_code=200,
        content={"result": result}
    )

@router.get("/get-model-config/", response_model=Dict[str, Any])
def get_model_config(model_name: str) -> Dict[str, Any]:
    """Return the config of the model"""
    return MODELS_FACTORY.get_model_config(model_name)

@router.post("/change-model/")
def change_model(model_name: str, config: dict):
    """Change the model"""
    MODELS_FACTORY.change_model(model_name, config)
    return JSONResponse(
        status_code=200,
        content={"message": f"Model changed to {model_name} with config {config}"}
    )
