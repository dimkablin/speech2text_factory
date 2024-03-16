"""Speech to text model initialization file"""
import os
import torch
import torchaudio
import numpy as np
from io import BytesIO
from fastapi import UploadFile
import nemo.collections.asr as nemo_asr
from ai_models.speech2text_interface import Speech2TextInterface

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
        self.path_to_model = "./src/ai_models/stt/weights"
        # self.path_to_model = "nvidia/stt_ru_conformer_transducer_large"
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
        try:
            self.model = nemo_asr.models.EncDecRNNTBPEModel.restore_from(path, 
                                                                         map_location=self.device)

        # if we didnt find the model, we try to download it
        except OSError:
            self.model = nemo_asr.models.EncDecRNNTBPEModel.from_pretrained("nvidia/stt_ru_conformer_transducer_large", 
                                                                            map_location=self.device)

    def resample_and_save(self, spooled_file) -> str:
        """Will save file and return path"""
        output_file_path = f"./src/ai_models/stt/buffer/{str(hash(spooled_file))[:10]}.wav"
        spooled_file.seek(0)  # Переход к началу файла
        audio_bytes = spooled_file.read()
        waveform, sample_rate = torchaudio.load(BytesIO(audio_bytes))
        resampler = torchaudio.transforms.Resample(sample_rate, 16000)
        waveform = resampler(waveform)
        torchaudio.save(output_file_path, waveform, sample_rate=16000)
        return output_file_path


    def __call__(self, audio: UploadFile) -> str:
        """ Get model output from the pipeline.

        Args:
            file_path (str): UploadFile

        Returns:
            str: model output.
        """

        audio_path = self.resample_and_save(audio)

        # model inference
        pred_ids = self.model.transcribe([audio_path])[0]

        os.remove(audio_path)

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
