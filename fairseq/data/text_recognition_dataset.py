import numpy as np
import torch
from fairseq.data.data_utils import default_loader

from . import FairseqDataset


def collate(samples):
    """collate samples of images and targets."""
    images = torch.stack([s['source'] for s in samples])
    targets = [s['target'] for s in samples]
    target_lengths = [s['target_length'] for s in samples]
    ntokens = sum(target_lengths)
    target_lengths = torch.IntTensor(target_lengths)
    # TODO: pin-memory
    return {
        'batch_size': len(samples),
        'ntokens': ntokens,
        'net_input': {
            'image': images,
        },
        'target': targets,
        'target_length': target_lengths,
    }


class TextRecognitionDataset(FairseqDataset):
    """A dataset that provides helpers for batching."""

    def __init__(
        self, src, tgt, tgt_dict, tgt_sizes=None,
        shuffle=True, transform=None, loader=default_loader,
    ):
        self.src = src
        self.tgt = tgt
        self.tgt_dict = tgt_dict
        self.tgt_sizes = np.array(tgt_sizes) if tgt_sizes is not None else None
        self.transform = transform
        self.loader = loader
        self.shuffle = shuffle

    def __len__(self):
        return len(self.src)

    def __getitem__(self, index):
        image_name = self.src[index]
        image = self.loader(image_name)

        if self.transform is not None:
            image = self.transform(image)

        target = self.tgt[index]
        target_length = self.tgt_sizes[index]
        # Convert label to a numeric ID.
        target = torch.IntTensor([self.tgt_dict.index(i) for i in target])

        return {
            'source': image,
            'target': target,
            'target_length': target_length,
        }

    def collater(self, samples):
        """Merge a list of samples to form a mini-batch."""
        return collate(samples)

    def ordered_indices(self):
        """
        Return an ordered list of indices. Batches will be constructed based
        on this order.
        """
        if self.shuffle:
            return np.random.permutation(len(self))
        else:
            return np.arange(len(self))

    def num_tokens(self, index):
        """Return the number of tokens in a sample. This value is used to
        enforce ``--max-tokens`` during batching."""
        return self.tgt_sizes[index] if self.tgt_sizes is not None else 0

    def size(self, index):
        """Return an example's size as a float or tuple. This value is used when
        filtering a dataset with ``--max-positions``."""
        return self.tgt_sizes[index] if self.tgt_sizes is not None else 0
