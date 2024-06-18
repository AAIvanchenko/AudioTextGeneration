import torch

from pathlib import Path

import torchaudio
from torch.utils.data import Dataset

from src.transforms.audio import random_cropped_audio_load, audio_resample


class AudioTextGenerationDataset(Dataset):
    def __init__(
            self,
            audio_paths: list[str | Path],
            audio_captions: list[str],
            sample_rate: int = 22050,
            duration: float | None = None,
            channel: int | None = None,
    ):
        self._audio_paths = [
            file if isinstance(file, Path) else Path(file) for file in audio_paths
        ]
        self._audio_captions = audio_captions
        self._sample_rate = sample_rate
        self._duration = duration
        self._channel = channel

    def __getitem__(
            self,
            item: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if item >= len(self) or item < -len(self):
            raise KeyError(f'{item} no found in dataset [0..{len(self)}]')

        # Загрузка аудио
        if self._duration is None:
            waveform, sample_rate = torchaudio.load(self._audio_paths[item])
        # Загрузка случайного кусочка аудио
        else:
            waveform, sample_rate = random_cropped_audio_load(self._audio_paths[item], self._duration)
        if self._channel is not None:
            # Возьмём только определённый канал
            waveform = waveform[self._channel]

        # Изменим частоту дискретизации
        if self._sample_rate is not None:
            waveform = audio_resample(waveform, sample_rate, self._sample_rate)

        # Загрузка аннотации
        caption = self._audio_captions[item]

        return waveform, caption

    def __len__(self):
        return len(self._audio_paths)

    @property
    def sample_rate(self) -> int:
        return self._sample_rate
