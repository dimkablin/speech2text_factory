""" Factory Method for speech to text models """
import torch
from src.ai_models.whisper.model import Whisper
from src.ai_models.speech2text_interface import Speech2TextInterface


class Speech2TextFactory:
    """ Factory Method for speech to text models """
    MODEL_MAP = {
        Whisper.get_model_name(): Whisper
    }
    # get first model
    MODEL = MODEL_MAP[MODEL_MAP.keys()[0]]()

    @classmethod
    def get_model(cls) -> Speech2TextInterface:
        """Return the speech to text model by name."""
        return cls.MODEL

    @classmethod
    def get_model_names(cls) -> list:
        """Return a list of model names"""
        return list(cls.MODEL_MAP.keys())

    @classmethod
    def change_model(cls, model_name: str) -> None:
        """Change the model"""
        if model_name == cls.MODEL.get_model_name:
            return None

        # delete models from DEVICE       
        del cls.MODEL
        torch.cuda.empty_cache()

        cls.MODEL = cls.MODEL_MAP[model_name]()


MODELS_FACTORY = Speech2TextFactory()
