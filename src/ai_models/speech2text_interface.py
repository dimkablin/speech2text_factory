"""Interface for speech to text model initialization files"""
from abc import ABC, abstractmethod


class Speech2TextInterface(ABC):
    """ Interface for speech to text model"""
    @abstractmethod
    def __call__(self, *args, **kwargs) -> str:
        """Convert the audio to text.

        Returns:
            str: text of the audio.
        """

    @abstractmethod
    def __str__(self) -> str:
        """Return the description of the model."""

    @staticmethod
    def get_model_name() -> str:
        """Return the type of the model."""
