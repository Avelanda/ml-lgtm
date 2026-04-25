"""
PyTorch Lightning data module wiring LGTM datasets and samplers.

For licensing see accompanying LICENSE file.
Copyright (C) 2026 Apple Inc. All Rights Reserved.

For third-party code see ACKNOWLEDGMENTS file.
"""

import random
from dataclasses import dataclass
from typing import Callable, Iterator, Optional

import numpy as np
import torch
from lightning.pytorch import LightningDataModule
from torch import Generator
from torch.utils.data import DataLoader, Dataset, IterableDataset

from lgtm.dataset.data_types import DataStage, Mode
from lgtm.dataset.dataset import DatasetCfgWrapper, get_dataset
from lgtm.utils.training_utils import StepTracker


class ValidationWrapper(Dataset):
    """
    Wraps a dataset so that PyTorch Lightning's validation step can be turned
    into a visualization step.
    """

    dataset: Dataset
    dataset_iterator: Optional[Iterator]
    length: int

    def __init__(self, dataset: Dataset, length: int) -> None:
        super().__init__()
        self.dataset = dataset
        self.length = length
        self.dataset_iterator = None

    def __len__(self):
        return self.length

    def __getitem__(self, index: int):
        if isinstance(self.dataset, IterableDataset):
            if self.dataset_iterator is None:
                self.dataset_iterator = iter(self.dataset)
            try:
                item = next(self.dataset_iterator)
                return item
            except StopIteration:
                print(
                    "[ValidationWrapper] Iterator exhausted, restarting"
                    f" (index={index})"
                )
                self.dataset_iterator = iter(self.dataset)
                item = next(self.dataset_iterator)
                return item

        random_index = torch.randint(0, len(self.dataset), tuple())
        return self.dataset[random_index.item()]


@dataclass
class DataLoaderStageCfg:
    batch_size: int
    num_workers: int
    persistent_workers: bool
    seed: int | None


@dataclass
class DataLoaderCfg:
    train: DataLoaderStageCfg
    test: DataLoaderStageCfg
    val: DataLoaderStageCfg


DatasetShim = Callable[[Dataset, DataStage], Dataset]


def worker_init_fn(worker_id: int) -> None:
    random.seed(int(torch.utils.data.get_worker_info().seed) % (2**32 - 1))
    np.random.seed(int(torch.utils.data.get_worker_info().seed) % (2**32 - 1))


class DataModule(LightningDataModule):
    dataset_cfgs: list[DatasetCfgWrapper]
    data_loader_cfg: DataLoaderCfg
    mode: Mode
    step_tracker: StepTracker | None
    dataset_shim: DatasetShim
    global_rank: int

    def __init__(
        self,
        dataset_cfgs: list[DatasetCfgWrapper],
        data_loader_cfg: DataLoaderCfg,
        mode: Mode = "train",
        step_tracker: StepTracker | None = None,
        dataset_shim: DatasetShim = lambda dataset, _: dataset,
        global_rank: int = 0,
        target_image_shape: tuple[int, int] | None = None,
    ) -> None:
        super().__init__()
        self.dataset_cfgs = dataset_cfgs
        self.data_loader_cfg = data_loader_cfg
        self.mode = mode
        self.step_tracker = step_tracker
        self.dataset_shim = dataset_shim
        self.global_rank = global_rank
        self.target_image_shape = target_image_shape

    def get_persistent(self, loader_cfg: DataLoaderStageCfg) -> bool | None:
        return (
            None
            if loader_cfg.num_workers == 0
            else loader_cfg.persistent_workers
        )

    def get_generator(
        self, loader_cfg: DataLoaderStageCfg
    ) -> torch.Generator | None:
        if loader_cfg.seed is None:
            return None
        generator = Generator()
        generator.manual_seed(loader_cfg.seed + self.global_rank)
        return generator

    def train_dataloader(self):
        datasets = get_dataset(
            dataset_cfgs=self.dataset_cfgs,
            mode=self.mode,
            data_stage="train",
            step_tracker=self.step_tracker,
            target_image_shape=self.target_image_shape,
        )
        data_loaders = []
        for dataset in datasets:
            dataset = self.dataset_shim(dataset, dataset.data_stage)
            data_loaders.append(
                DataLoader(
                    dataset=dataset,
                    batch_size=self.data_loader_cfg.train.batch_size,
                    shuffle=not isinstance(dataset, IterableDataset),
                    num_workers=self.data_loader_cfg.train.num_workers,
                    generator=self.get_generator(self.data_loader_cfg.train),
                    worker_init_fn=worker_init_fn,
                    persistent_workers=self.get_persistent(
                        self.data_loader_cfg.train
                    ),
                )
            )
        return data_loaders if len(data_loaders) > 1 else data_loaders[0]

    def val_dataloader(self):
        """
        Validation dataloader for training-time validation.
        Always wraps datasets with ValidationWrapper (length 1).
        """
        datasets = get_dataset(
            dataset_cfgs=self.dataset_cfgs,
            mode=self.mode,
            data_stage="test",
            step_tracker=self.step_tracker,
            target_image_shape=self.target_image_shape,
        )
        data_loaders = []
        for dataset in datasets:
            dataset = self.dataset_shim(dataset, dataset.data_stage)
            dataset = ValidationWrapper(dataset=dataset, length=1)
            data_loaders.append(
                DataLoader(
                    dataset=dataset,
                    batch_size=self.data_loader_cfg.val.batch_size,
                    num_workers=self.data_loader_cfg.val.num_workers,
                    generator=self.get_generator(self.data_loader_cfg.val),
                    worker_init_fn=worker_init_fn,
                    persistent_workers=self.get_persistent(
                        self.data_loader_cfg.val
                    ),
                )
            )
        return data_loaders if len(data_loaders) > 1 else data_loaders[0]

    def test_dataloader(self):
        datasets = get_dataset(
            dataset_cfgs=self.dataset_cfgs,
            mode=self.mode,
            data_stage="test",
            step_tracker=self.step_tracker,
            target_image_shape=self.target_image_shape,
        )
        data_loaders = []
        for dataset in datasets:
            dataset = self.dataset_shim(dataset, dataset.data_stage)

            data_loaders.append(
                DataLoader(
                    dataset=dataset,
                    batch_size=self.data_loader_cfg.test.batch_size,
                    num_workers=self.data_loader_cfg.test.num_workers,
                    generator=self.get_generator(self.data_loader_cfg.test),
                    worker_init_fn=worker_init_fn,
                    persistent_workers=self.get_persistent(
                        self.data_loader_cfg.test
                    ),
                )
            )
        return data_loaders if len(data_loaders) > 1 else data_loaders[0]
