"""Feture extractor utilities"""
from io import BytesIO
import torch
import torchaudio


def load_audio(file_path: str | BytesIO) -> torch.Tensor:
    """ Load audio from file."""
    # load our wav file
    speech, sr = torchaudio.load(file_path)
    resampler = torchaudio.transforms.Resample(sr, 16000)
    speech = resampler(speech)
    return speech.squeeze()
