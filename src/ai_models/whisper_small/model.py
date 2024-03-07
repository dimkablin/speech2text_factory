"""Speech to text model initialization file"""
import os
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from src.ai_models.speech2text_interface import Speech2TextInterface


class WhisperSmall(Speech2TextInterface):
    """ Speech to text model initialization file."""
    def __init__(self, device = None):
        self.model_name = "openai/whisper-small"
        self.path_to_model = "src/ai_models/whisper_small/weigths/"
        self.torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        if device is None:
            self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

        self.model = None
        self.processor = None
        self.pipe = None
        self.load_weigths(self.path_to_model)

    def load_weigths(self, path: str):
        """ Download the model weights."""
        try:
            self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
                self.model_name,
                cache_dir=path,
                torch_dtype=self.torch_dtype,
                low_cpu_mem_usage=True,
                use_safetensors=True
            )

            self.processor = AutoProcessor.from_pretrained(self.model_name, cache_dir=path)
        except OSError:
            self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
                self.model_name,
                torch_dtype=self.torch_dtype,
                low_cpu_mem_usage=True,
                use_safetensors=True
            )

            self.processor = AutoProcessor.from_pretrained(self.model_name)
            self.model.save_pretrained(path)
            self.processor.save_pretrained(path)

        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=self.model,
            tokenizer=self.processor.tokenizer,
            feature_extractor=self.processor.feature_extractor,
            max_new_tokens=128,
            chunk_length_s=30,
            batch_size=16,
            return_timestamps=True,
            torch_dtype=self.torch_dtype,
            device=self.device,
        )

    def __call__(self, file_path: str) -> str:
        """ Get model output from the pipeline.

        Args:
            file_path (str): path to the mp3 file.

        Returns:
            str: model output.
        """

        return self.pipe(file_path)['text']

    def __str__(self) -> str:
        return f"Model :20 WhisperSmall \n\
            Dtype :20 {self.torch_dtype} \n\
            Device :20 {self.device}"

    @staticmethod
    def get_model_name() -> str:
        return "openai/whisper-small"
