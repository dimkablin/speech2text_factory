"""Main FastAPI entry point."""
from io import BytesIO
import os
from typing import Any, Dict
from fastapi import APIRouter, File, UploadFile
from fastapi.responses import JSONResponse

from ai_models.speech2text import MODELS_FACTORY
from api.app.models import GetCResponse


router = APIRouter()


@router.get("/get-model-names", response_model=list)
async def get_model_names() -> list:
    """Return a list of model names."""
    return await MODELS_FACTORY.get_model_names()


@router.get("/get-current-model", response_model=str)
async def get_current_model() -> str:
    """Return the name of the current model."""
    return (await MODELS_FACTORY.get_model()).get_model_name()


@router.post("/speech-to-text", response_model=str)
async def speech_to_text(audio: UploadFile = File(...)) -> str:
    """Predict function."""
    path = os.path.join("data", audio.filename)

    # read and save wav file
    with open(path, "wb") as f:
        f.write(await audio.read())

    result = await MODELS_FACTORY(path)

    # delete file
    os.remove(path)

    return JSONResponse(
        status_code=200,
        content={"result": result}
    )


@router.get("/get-config", response_model=GetCResponse)
async def get_model_config(model_name: str) -> GetCResponse:
    """Return the config of the model"""
    return await MODELS_FACTORY.get_config(model_name)


@router.get("/get-current-config", response_model=GetCResponse)
async def get_model_config() -> GetCResponse:
    """Return the config of the model"""
    return await MODELS_FACTORY.get_cur_config()


@router.post("/change-model")
async def change_model(model_name: str, config: dict):
    """Change the model"""
    await MODELS_FACTORY.change_model(model_name, config)
    return JSONResponse(
        status_code=200,
        content={"message": f"Model changed to {model_name} with config {config}"}
    )
