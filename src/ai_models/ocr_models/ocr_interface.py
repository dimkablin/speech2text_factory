""" The ocr interface """

from abc import ABC, abstractmethod

class OCR(ABC):
    """ Interface of ocr ai_models """
    @abstractmethod
    def __call__(self, *args, **kwargs) -> dict:
        """
        This function returns a dictionary with specific keys and annotated value types.

        Returns:
            Dict[str, Union[List[str], List[float], List[Tuple[float, float, float, float]]]]:
            A dictionary with the following keys and value types:
            - 'rec_texts': List of strings
            - 'rec_scores': List of floats
            - 'det_polygons': List of tuples with four floats each
            - 'det_scores': List of floats
        """

    @staticmethod
    def get_model_type() -> str:
        """Return model type."""

    @abstractmethod
    def __str__(self) -> str:
        """Return model description."""
