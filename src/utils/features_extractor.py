"""Feture extractor utilities"""
from os import PathLike
from typing import BinaryIO
import torch
import torchaudio


def load_audio(file: BinaryIO | str | PathLike) -> torch.Tensor:
    """ Load audio from file."""
    # load our wav file
    speech, sr = torchaudio.load(file)
    resampler = torchaudio.transforms.Resample(sr, 16000)
    speech = resampler(speech)
    return speech.squeeze()
