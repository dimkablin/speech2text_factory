"""Easy OCR Initialization."""
from typing import Any
import easyocr

from src.ai_models.ocr_models.ocr_interface import OCR

class EasyOCRDefault(OCR):
    """ Initialized EasyOCR model """
    def __init__(self):
        self.languages = ['ru']
        self.model = easyocr.Reader(self.languages)

    def __call__(self, inputs, *args, **kwargs) -> list[dict[str, list[Any]]]:
        results = []
        for image in inputs:
            horizontal_boxes, free_boxes = self.model.detect(image)
            outputs = self.model.recognize(image, horizontal_boxes[0], free_boxes[0])

            result = {'rec_texts': [], 'rec_scores': [], 'det_polygons': [], 'det_scores': []}

            for bbox, text, conf in outputs:
                result['det_scores'].append(1)
                result['det_polygons'].append([int(coord) for xy in bbox for coord in xy])
                result['rec_scores'].append(conf)
                result['rec_texts'].append(text)
            results.append(result)

        return results

    def __str__(self):
        return f"EasyOCR lang {self.languages}"

    @staticmethod
    def get_model_type() -> str:
        """Return model type."""
        return "EasyOCR"
