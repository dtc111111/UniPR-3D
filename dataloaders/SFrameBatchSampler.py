import functools
import itertools
import torch
import random
from typing import Any, Callable, Generic, Iterable, List, Optional, TypeVar, Union, Iterator
from torch.utils.data.sampler import (
    BatchSampler,
    RandomSampler,
    Sampler,
    SequentialSampler,
)

class SFrameBatchSampler(BatchSampler):
    r"""Wraps another sampler to yield a mini-batch of indices.

    Args:
        sampler (Sampler or Iterable): Base sampler. Can be any iterable object
        batch_size (int): Size of mini-batch.
        drop_last (bool): If ``True``, the sampler will drop the last batch if
            its size would be less than ``batch_size``

    Example:
        >>> list(BatchSampler(SequentialSampler(range(10)), batch_size=3, drop_last=False))
        [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]]
        >>> list(BatchSampler(SequentialSampler(range(10)), batch_size=3, drop_last=True))
        [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
    """

    def __init__(
        self,
        sampler: Union[Sampler[int], Iterable[int]],
        batch_size: int,
        drop_last: bool = False,
        sframe_range: List[int] = [0, 5],
    ) -> None:
        super().__init__(sampler, batch_size, drop_last)
        self.sframe_range = sframe_range

    def get_sframe(self) -> int:
        return random.randint(self.sframe_range[0], self.sframe_range[1])

    def __iter__(self) -> Iterator[List[int]]:
        # Implemented based on the benchmarking in https://github.com/pytorch/pytorch/pull/76951
        sframe_now = self.get_sframe()
        if self.drop_last:
            sampler_iter = iter(self.sampler)
            while True:
                try:
                    batch = [(next(sampler_iter), sframe_now) for _ in range(self.batch_size)]
                    yield batch
                    sframe_now = self.get_sframe()
                except StopIteration:
                    break
        else:
            batch = [(0, sframe_now)] * self.batch_size
            idx_in_batch = 0
            for idx in self.sampler:
                batch[idx_in_batch] = (idx, sframe_now)
                idx_in_batch += 1
                if idx_in_batch == self.batch_size:
                    yield batch
                    sframe_now = self.get_sframe()
                    idx_in_batch = 0
                    batch = [(0, sframe_now)] * self.batch_size
            if idx_in_batch > 0:
                yield batch[:idx_in_batch]
                sframe_now = self.get_sframe()

    def __len__(self) -> int:
        # Can only be called if self.sampler has __len__ implemented
        # We cannot enforce this condition, so we turn off typechecking for the
        # implementation below.
        # Somewhat related: see NOTE [ Lack of Default `__len__` in Python Abstract Base Classes ]
        if self.drop_last:
            return len(self.sampler) // self.batch_size  # type: ignore[arg-type]
        else:
            return (len(self.sampler) + self.batch_size - 1) // self.batch_size  # type: ignore[arg-type]