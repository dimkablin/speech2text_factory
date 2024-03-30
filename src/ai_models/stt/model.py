"""Speech to text model initialization file"""
import os
import torch
import torchaudio
import numpy as np
from io import BytesIO
from fastapi import UploadFile
import nemo.collections.asr as nemo_asr
from ai_models.speech2text_interface import Speech2TextInterface
from api.app.models import ConfigItem, GetCResponse
from utils.features_extractor import load_audio

class Stt(Speech2TextInterface):
    """ Speech to text model initialization file."""
    DEVICES = ["cpu"]
    for i in range(torch.cuda.device_count()):
        DEVICES.append("cuda:" + str(i))
    LANGUAGES = ["russian"]

    def __init__(self,
            device = None,
            model_name: str = "nvidia/stt_ru_conformer_transducer_large",
            language: str = "russian"
        ):
        self.model_name = model_name
        self.language = language
        self.device = device
        self.path_to_model = "./ai_models/stt/weights"
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

    def resample_and_save(self, path) -> str:
        """Will save file and return path"""
        waveform, sample_rate = torchaudio.load(path)
        resampler = torchaudio.transforms.Resample(sample_rate, 16000)
        waveform = resampler(waveform)
        torchaudio.save(path, waveform, sample_rate=16000)
        return path


    def __call__(self, path: str) -> str:
        """ Get model output from the pipeline.

        Args:
            file_path (str): UploadFile

        Returns:
            str: model output.
        """
        path = self.resample_and_save(path)

        # model inference
        pred_ids = self.model.transcribe([path])[0]

        # os.remove(audio_path)

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
    def get_config() -> GetCResponse:
        """Return the list of possible configuration of the model."""
        # Be sure that the cls.__dict__ contain the keys()
        response = GetCResponse(
            model_desciption=str(Stt),
            items=[
                # DEVICE CONFIG
                ConfigItem(
                    name="Девайс",
                    descriptions="Процессор, на котором будут обрабатываться вычисления.",
                    attributes_name="device",
                    options=Stt.DEVICES
                ),

                # LANGUAGE CONFIG
                ConfigItem(
                    name="Язык обработки",
                    descriptions="Процессор, на котором будут обрабатываться вычисления.",
                    attributes_name="language",
                    options=Stt.LANGUAGES
                )
            ]
        )

        return response
    
    def get_cur_config(self) -> GetCResponse:
        """Return the current config of the Whisper Small"""
        response = GetCResponse(
            items=[
                ConfigItem(
                    attributes_name="device",
                    options=self.device
                ),
                ConfigItem(
                    attributes_name="language",
                    options=self.language
                )
            ]
        )

        return response

