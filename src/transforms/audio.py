import numpy as np
import random
import torch
import torchaudio

from pathlib import Path


def audio_resample(audio: torch.Tensor, audio_sample_rate: int, target_sample_rate: int):
    # Изменим частоту дискретизации
    if audio_sample_rate != target_sample_rate:
        audio = torchaudio.functional.resample(audio, audio_sample_rate, target_sample_rate)
    return audio
    

def random_cropped_audio_load(audio: str | Path, duration: float) -> torch.Tensor | int:
    # Получим путь до аудио
    audio_path = Path(audio)
    # Получим информацию об аудио
    metadata = torchaudio.info(audio_path)
    # Получим кол-во требуемых фреймов для считывания
    load_num_frames = metadata.sample_rate * duration
    if load_num_frames > metadata.num_frames:
        raise RuntimeError(
            f'Audio file "{audio_path}" duration {metadata.num_frames / metadata.sample_rate:0.2f} less than '
            f'{duration}')
    # Случайно выберем момент начала считывания
    load_frame_offset = random.randint(0, max(metadata.num_frames - load_num_frames - 1, 0))
    # Загрузим участок данных
    waveform, sample_rate = torchaudio.load(
        audio_path,
        frame_offset=load_frame_offset,
        num_frames=load_num_frames
    )
    return waveform, sample_rate


# class RandomMusicCrop:
#     def __init__(self, duration: float):
#         self.duration = duration
#
#     def __call__(self, audio: str | Path | np.ndarray | torch.Tensor):
#         # Если строка или путь - работаем с файлом
#         if isinstance(audio, (str, Path)):
#             audio, sample_rate = random_cropped_audio_load(audio, duration=self.duration)
