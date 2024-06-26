""" Factory Method for speech to text models """
import logging
from ai_models.whisper.model import Whisper
from ai_models.stt.model import Stt
from ai_models.speech2text_interface import Speech2TextInterface


class Speech2TextFactory:
    """ Factory Method for speech to text models """
    MODEL_MAP = {
        Whisper.get_model_name(): Whisper,
        Stt.get_model_name(): Stt
    }
    # get first model
    MODEL = MODEL_MAP[Whisper.get_model_name()]()

    @classmethod
    async def __call__(cls, *args, **kwargs) -> str:
        """Call the current model."""
        result = cls.MODEL(*args, **kwargs)

        # grabage collector will delete this objects from CPU memory
        args, kwargs = None, None
        return result

    @classmethod
    async def get_model(cls) -> Speech2TextInterface:
        """Return the speech to text model by name."""
        return cls.MODEL

    @classmethod
    async def get_model_names(cls) -> list:
        """Return a list of model names"""
        return list(cls.MODEL_MAP.keys())

    @classmethod
    async def get_config(cls, model_name) -> dict:
        """Return the config of the model"""
        return cls.MODEL_MAP[model_name].get_config()

    @classmethod
    async def get_cur_config(cls) -> dict:
        """Return the config of the model"""
        return cls.MODEL.get_cur_config()

    @classmethod
    async def change_model(cls, model_name: str, config = None) -> None:
        """Change the model"""
        # delete models from DEVICE
        cls.MODEL.model.to("cpu")
        del cls.MODEL.model
        try:
            if config is None:
                cls.MODEL = cls.MODEL_MAP[model_name]()
            else:
                cls.MODEL = cls.MODEL_MAP[model_name](**config)
        except TypeError as e:
            logging.warning("Got type error during the model changing.")
            cls.MODEL = cls.MODEL_MAP[next(iter(cls.MODEL_MAP.keys()))]()




MODELS_FACTORY = Speech2TextFactory()
