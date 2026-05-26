import numpy as np
from ..autograd import Tensor

from typing import Optional, List



class Dataset:
    r"""An abstract class representing a `Dataset`.

    All subclasses should overwrite :meth:`__getitem__`, supporting fetching a
    data sample for a given key. Subclasses must also overwrite
    :meth:`__len__`, which is expected to return the size of the dataset.
    """

    def __init__(self, transforms: Optional[List] = None):
        self.transforms = transforms

    def __getitem__(self, index) -> object:
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError
    
    def apply_transforms(self, x):
        if self.transforms is not None:
            # apply the transforms
            for tform in self.transforms:
                x = tform(x)
        return x


class DataLoader:
    r"""
    Data loader. Combines a dataset and a sampler, and provides an iterable over
    the given dataset.
    Args:
        dataset (Dataset): dataset from which to load the data.
        batch_size (int, optional): how many samples per batch to load
            (default: ``1``).
        shuffle (bool, optional): set to ``True`` to have the data reshuffled
            at every epoch (default: ``False``).
     """
    dataset: Dataset
    batch_size: Optional[int]

    def __init__(
        self,
        dataset: Dataset,
        batch_size: Optional[int] = 1,
        shuffle: bool = False,
    ):

        self.dataset = dataset
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.idx = -1

        batch_size = self._batch_size()
        if not self.shuffle:
            self.ordering = np.array_split(
                np.arange(len(dataset)), 
                range(batch_size, len(dataset), batch_size),
            )
        else:
            self.ordering = np.array_split(
                np.random.permutation(len(dataset)),
                range(batch_size, len(dataset), batch_size),
            )

    def _batch_size(self):
        if self.batch_size is None:
            return len(self.dataset)
        return self.batch_size

    def _make_ordering(self):
        indices = (
            np.random.permutation(len(self.dataset))
            if self.shuffle
            else np.arange(len(self.dataset))
        )
        batch_size = self._batch_size()
        return np.array_split(indices, range(batch_size, len(indices), batch_size))

    def _to_tensor(self, batch):
        if isinstance(batch, Tensor):
            return batch
        if isinstance(batch, tuple):
            return tuple(self._to_tensor(x) for x in batch)
        if isinstance(batch, list):
            return tuple(self._to_tensor(x) for x in batch)
        return Tensor(batch)

    def _collate_samples(self, samples):
        first = samples[0]
        if isinstance(first, tuple):
            return tuple(
                np.stack([sample[field] for sample in samples])
                for field in range(len(first))
            )
        return np.stack(samples)

    def __iter__(self):
        # TODO
        ### BEGIN YOUR SOLUTION
        self.idx = 0
        self.ordering = self._make_ordering()
        ### END YOUR SOLUTION
        return self

    def __next__(self):
        # TODO
        ### BEGIN YOUR SOLUTION
        if self.idx >= len(self.ordering):
            raise StopIteration

        batch_indices = self.ordering[self.idx]
        self.idx += 1

        if getattr(self.dataset, "transforms", None) is not None:
            samples = [self.dataset[int(i)] for i in batch_indices]
            batch = self._collate_samples(samples)
        else:
            try:
                batch = self.dataset[batch_indices]
            except Exception:
                samples = [self.dataset[int(i)] for i in batch_indices]
                batch = self._collate_samples(samples)
        return self._to_tensor(batch)
        ### END YOUR SOLUTION
