import torch
from torchtext import data
from torchtext import datasets

from .data_module import DataModule


class UDPOS(DataModule):
    """Implements mechanism to fetch UDPOS dataset from torch text."""

    def __init__(self) -> None:
        super().__init__()
        self.save_hyperparameters()
        self._create_fields()
        self._import_dataset()
        self._build_vocab()
        self._build_iterators()

    def _create_fields(self):
        self.TEXT = data.Field(lower=True)
        self.UD_TAGS = data.Field(unk_token=None)

    def _import_dataset(self):
        fields = (("text", self.TEXT), ("udtags", self.UD_TAGS))
        self.train, self.val, self.test = datasets.UDPOS.splits(fields)

    def _build_vocab(self):
        self.TEXT.build_vocab(self.train,
                              vectors="glove.6B.100d",
                              min_freq=2,
                              unk_init=torch.Tensor.normal_)
        self.UD_TAGS.build_vocab(self.train)

    def _build_iterators(self, batch_size: int = 64):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.train_iter, self.val_iter, self.test_iter = data.BucketIterator.splits(
            (self.train, self.val, self.test),
            batch_size=batch_size,
            device=device,
        )

    def get_dataloader(self, train: bool):
        return self.train_iter if train else self.val_iter

    def test_dataloader(self): return self.test_iter
