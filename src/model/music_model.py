import torch
import torchaudio

from torch.nn import Module
from torchaudio.pipelines import Wav2Vec2Bundle


class Waw2Vec2MusicModel(Module):
    def __init__(self, bundle: Wav2Vec2Bundle = torchaudio.pipelines.WAV2VEC2_XLSR53):
        self.model = bundle.get_model()
        self.sample_rate = bundle.sample_rate

    def forward(self, waveform: torch.Tensor):
        # waveform must be resample to self.sample_rate
        features, _ = self.model.extract_features(waveform)

        return features
