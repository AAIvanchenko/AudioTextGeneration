import csv
import torch

from pathlib import Path
from typing import Literal
from torch.utils.data import Dataset

from src.transforms.audio import random_cropped_audio_load, audio_resample


class MAGJDataset(Dataset):
    def __init__(
            self,
            root: str | Path,
            stage: Literal["train", "valid", "test"],
            sample_rate: int = 22050,
            duration: float | None = None,
            channel: int = 0,
            split_num: int = 0,
            label_type: Literal["top50tags"] = 'top50tags'
    ):
        assert stage in ["train", "valid", "test"]
        assert split_num in [0]
        assert label_type in ["top50tags"]

        self._root = Path(root)
        self._sample_rate = sample_rate
        self._duration = duration
        self._channel = channel

        # Загрузим данные аннотации
        media_info: list[tuple[str, list[str]]] = []  # list of (path, tags) values from tsv
        with open(Path(self._root, "splits", f"split-{split_num}", f"autotagging_{label_type}-{stage}.tsv")) as tsv:
            reader = csv.reader(
                tsv,
                delimiter='\t',
            )
            # Считаем строку заголовка
            header: list[str] = next(reader)
            path_col_id = header.index('PATH')
            tags_col_id = header.index('TAGS')
            # Сохраним путь и теги
            for row in reader:
                media_info.append((row[path_col_id], row[tags_col_id:]))
        # Загрузим теги
        tags: list[str] = []
        if label_type == "top50tags":
            with open(Path(self._root, "tags", "top50.txt"), "r") as f:
                for line in f.readlines():
                    if line:
                        tags.append(line.strip())

        # Разделим теги на тип и значение
        tag_class, tag_label = [], []
        for cls, label in (t.split('---', maxsplit=1) for t in tags):
            if cls == 'mood/theme':
                cls = 'mood'
            tag_class.append(cls)
            tag_label.append(label)
        # Сформируем запросы для CLIP
        tags_queries = [f'A {cls} of the music is {label}' for cls, label in zip(tag_class, tag_label)]

        # Сформируем mapper'ы
        self._id_2_query = {i: query for i, query in enumerate(tags_queries)}
        self._tag_2_id = {tag: i for i, tag in enumerate(tags)}
        self._id_2_tag = {i: tag for tag, i in self._tag_2_id.items()}

        # Получим файлы и теги музыки
        music_path: list[Path] = []
        music_tags: list[list[int]] = []
        for file, tags in media_info:
            # Получим путь до файла музыки
            file = Path(self._root, file)
            if not file.is_file():
                continue
            # Получим теги музыки
            tags_idx: list[int] = [self._tag_2_id[t] for t in tags]

            music_path.append(file)
            music_tags.append(tags_idx)

        self._media_paths = music_path
        self._media_tags = music_tags

    def __getitem__(
            self,
            item: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if item >= len(self) or item < -len(self):
            raise KeyError(f'{item} no found in dataset [0..{len(self)}]')

        # Загрузка аудио
        waveform, sample_rate = random_cropped_audio_load(self._media_paths[item], self._duration)
        # Возьмём только первый канал
        waveform = waveform[self._channel]

        # Изменим частоту дискретизации
        waveform = audio_resample(waveform, sample_rate, self._sample_rate)

        # Загрузка тегов
        tags_idx = self._media_tags[item]
        # Конвертируем индексы тегов в матрицу классов
        gt_tags = torch.zeros(len(self._tag_2_id), dtype=torch.int32)
        gt_tags[tags_idx] = 1

        return waveform, gt_tags

    def __len__(self):
        return len(self._media_paths)

    @property
    def queries(self) -> dict[int, str]:
        return self._id_2_query

    @property
    def tag_id_map(self) -> dict[int, str]:
        return self._id_2_tag