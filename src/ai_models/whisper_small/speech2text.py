"""Speech to text model initialization file"""
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline


class Speech2TextOpenAI:
    """ Speech to text model initialization file."""
    def __init__(self, device = None):
        self.model_name = "openai/whisper-small"
        self.path_to_model = "/mnt/u/GitHub/speech2text_model/src/ai_models/weigths/"
        self.torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        if device is None:
            self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

        self.model = None
        self.processor = None
        self.pipe = None
        self.load_weigths(self.path_to_model)


    def predict(self, file_path: str) -> str:
        """ Get model output from the pipeline.

        Args:
            file_path (str): path to the mp3 file.

        Returns:
            str: model output.
        """

        return ['text']


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



if __name__ == "__main__":
    # print(predict("audio.mp3"))
    model = Speech2TextOpenAI()
    print(model.predict("../data/SLAVA MARLOW Ты далеко.mp3"))
