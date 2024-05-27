import torch
from functools import singledispatch
from torch.utils.data import Sampler, SequentialSampler, BatchSampler, DataLoader


class RandomBatchSampler(Sampler):
    """
    this sampler randomizes the batch order only once per experiment... needs fixing
    """
    def __init__(self, dataset, batch_size):
        self.batch_size = batch_size
        self.dataset_length = len(dataset)
        self.n_batches = self.dataset_length / self.batch_size
        self.batch_ids = torch.randperm(int(self.n_batches))

    def __len__(self):
        return self.batch_size

    def __iter__(self):
        for id in self.batch_ids:
            idx = torch.arange(id * self.batch_size, (id + 1) * self.batch_size)
            for index in idx:
                yield int(index)
        if int(self.n_batches) < self.n_batches:
            idx = torch.arange(int(self.n_batches) * self.batch_size, self.dataset_length)
            for index in idx:
                yield int(index)
        self.batch_ids = torch.randperm(int(self.n_batches))
 


@singledispatch
def get_sequential_dataloader(dataset, num_workers=1, pin_memory=True) -> list[DataLoader]:
    return [DataLoader(dataset,
                       batch_size=None,
                       sampler=SequentialSampler(dataset),
                       pin_memory=pin_memory,
                       num_workers=num_workers)]


@get_sequential_dataloader.register
def _(dataset: list, num_workers=1, pin_memory=True) -> list[DataLoader]:
    dataloaders = []
    for dset in dataset:
        dataloader = DataLoader(dset,
                                batch_size=None,
                                sampler=SequentialSampler(dset),
                                pin_memory=pin_memory,
                                num_workers=num_workers)
        dataloaders.append(dataloader)
    return dataloaders


def get_batch_dataloader(dataset, batch_size, num_workers=1, pin_memory=True, shuffle=True, drop_last=True):
        return [DataLoader(dataset,
                           batch_size=batch_size,
                           shuffle=shuffle,
                           pin_memory=pin_memory,
                           num_workers=num_workers,
                           drop_last=drop_last)]


def get_batch_dataloader_bugged_but_fast(dataset, batch_size, num_workers=1, pin_memory=True, shuffle=True, drop_last=True):
    """
    dont use this for now...
    """
    if shuffle:
        return [DataLoader(dataset,
                           batch_size=None,
                           sampler=BatchSampler(RandomBatchSampler(dataset, batch_size), batch_size=batch_size,
                                                drop_last=drop_last),
                           pin_memory=pin_memory,
                           num_workers=num_workers)]
    else:
        return [DataLoader(dataset,
                           batch_size=None,
                           sampler=BatchSampler(SequentialSampler(dataset), batch_size=batch_size, drop_last=drop_last),
                           pin_memory=pin_memory,
                           num_workers=num_workers)]
