"""Speech to text model initialization file"""
from io import BytesIO
from fastapi import UploadFile
import torch
import nemo.collections.asr as nemo_asr
from transformers import AutoProcessor
from src.ai_models.speech2text_interface import Speech2TextInterface
from src.utils.features_extractor import load_audio


class Stt(Speech2TextInterface):
    """ Speech to text model initialization file."""
    DEVICES = ["cpu", "cuda:0", "cuda:1"]
    LANGUAGES = ["russian"]

    def __init__(self,
            device = None,
            model_name: str = "nvidia/stt_ru_conformer_transducer_large",
            language: str = "russian"
        ):
        self.model_name = model_name
        self.language = language
        self.device = device
        # self.path_to_model = "./src/ai_models/stt/weights/stt_ru_conformer_transducer_large.nemo"
        self.path_to_model = "nvidia/stt_ru_conformer_transducer_large"
        self.torch_dtype = torch.float32

        if device is None:
            self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

        if device == "cuda:0":
            self.torch_dtype = torch.float16

        self.model = None
        self.processor = None
        self.load_weigths(self.path_to_model)

        # move to device
        self.model.to(self.device)

    def load_weigths(self, path: str):
        """ Download the model weights."""
        self.processor = AutoProcessor.from_pretrained(
            path,
            language=self.language,
            task="transcribe"
        )
        try:
            self.model = nemo_asr.models.EncDecRNNTBPEModel.from_pretrained(path)

        # if we didnt find the model, we try to download it
        except OSError:

            self.model = nemo_asr.models.EncDecRNNTBPEModel.from_pretrained("nvidia/stt_ru_conformer_transducer_large")

            #  save the model
            self.model.save_pretrained(path)

    def __call__(self, audio: UploadFile | str) -> str:
        """ Get model output from the pipeline.

        Args:
            file_path (UploadFile | str): uploaded file or path to the wav file.

        Returns:
            str: model output.
        """

        if isinstance(audio, UploadFile):
            audio=BytesIO(audio.file.read())

        input_features = self.processor(
            load_audio(audio),
            sampling_rate=16000,
            return_tensors="pt"
        ).input_features
        input_features = input_features.to(self.device, dtype=self.torch_dtype)

        # model inference
        pred_ids = self.model.transcribe(input_features)[0][0]

        # decode the transcription
        return pred_ids 

    def __str__(self) -> str:
        return f"Model :20 stt_ru \n\
            Dtype :20 {self.torch_dtype} \n\
            Device :20 {self.device}"

    @staticmethod
    def get_model_name() -> str:
        return "nvidia/stt_ru_conformer_transducer_large"

    @staticmethod
    def get_config() -> dict:
        """Return the list of possible configuration of the model."""
        # Be sure that the cls.__dict__ contain the keys()
        result = {
            "device": Stt.DEVICES,
            "language": Stt.LANGUAGES
        }

        return result
    