"""Feture extractor utilities"""
from os import PathLike
from typing import BinaryIO
import numpy as np
import torch
import torchaudio
from pydub import AudioSegment


def stereo2mono(audio: torch.Tensor) -> torch.Tensor:
    """Convert audio from stereo to mono."""
    audio_mono = torch.mean(audio, dim=0)

    return audio_mono


def load_audio(path: str | PathLike) -> torch.Tensor:
    """ Load audio from file."""
    # load our wav file
    speech, sr = torchaudio.load(path)
    speech = stereo2mono(speech)
    resampler = torchaudio.transforms.Resample(sr, 16000)
    speech = resampler(speech)
    return speech.squeeze()
