""" Factory Method for speech to text models """
from src.ai_models.whisper.model import Whisper
from src.ai_models.speech2text_interface.model import Speech2TextInterface


class Speech2TextFactory:
    """ Factory Method for speech to text models """
    MODEL_MAP = {
        Whisper.get_model_name(): Whisper
    }
    MODEL = MODEL_MAP[MODEL_MAP.keys()[0]] # get first model

    @classmethod
    def get_model(cls, model_name: str) -> Speech2TextInterface:
        """Return the speech to text model by name."""
        return Speech2TextFactory.MODEL_MAP[model_name]()

    @classmethod
    def get_model_names(cls) -> list:
        """Return a list of model names"""
        return list(Speech2TextFactory.MODEL_MAP.keys())
