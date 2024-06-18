import pandas as pd

from pathlib import Path
from typing import Literal

from .base.audio_and_caption import AudioTextGenerationDataset


class SSDDataset(AudioTextGenerationDataset):
    def __init__(
            self,
            root: str | Path,
            stage: Literal["train", "valid"],
            sample_rate: int = 22050,
            duration: float | None = None,
            channel: int = 0,
    ):
        assert stage in ["train", "valid"]

        self._root = Path(root)

        # Загрузим данные аннотации
        media_info = pd.read_csv(
            Path(self._root, "song_describer.csv"),
            usecols=["caption", "is_valid_subset", "path"]
        )
        # Выберем только данные выборки
        is_valid_row = media_info["is_valid_subset"] == True
        if stage == "valid":
            media_info = media_info[is_valid_row]
        else:
            media_info = media_info[~is_valid_row]

        # Получим пути к аудиозаписям
        audio_paths: list[Path] = [Path(self._root, "audio", file) for file in media_info["path"]]
        # Добавим суффикс 2min
        for i, path in enumerate(audio_paths):
            audio_paths[i] = path.with_suffix(f".2min{path.suffix}")
        # Получим аннотации к аудиозаписям
        audio_captions = media_info["caption"].tolist()

        super().__init__(
            audio_paths,
            audio_captions,
            duration=duration,
            sample_rate=sample_rate,
            channel=channel
        )
