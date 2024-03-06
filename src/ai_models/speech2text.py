"""Speech to text model initialization file"""
from transformers import pipeline
from huggingface_hub import hf_hub_download

pipe = pipeline("automatic-speech-recognition", model="openai/whisper-large-v3")


def predict(file_path: str) -> str:
    """ Get model output from the pipeline.

    Args:
        file_path (str): path to the mp3 file.

    Returns:
        str: model output.
    """

    return pipe(file_path)['text']


def download_weigths():
    """ Download the model weights."""
    hf_hub_download(
        repo_id="openai/whisper-large-v3",
        filename="model.safetensors",
        local_dir="/weights/"
    )

if __name__ == "__main__":
    # print(predict("audio.mp3"))
    download_weigths()
