import torch
from torch import nn

from ..utils.progress_board import ProgressBoard
from ..utils.hyperparameters import HyperParameters


class Module(nn.Module, HyperParameters):  # @save
    """The base class of all models."""

    def __init__(self, plot_train_per_epoch=2, plot_valid_per_epoch=1):
        super().__init__()
        self.save_hyperparameters()
        self.board = ProgressBoard()

    def loss(self, y_hat, y):
        raise NotImplementedError
    
    def accuracy(self, y_hat, y):
        raise NotImplementedError

    def forward(self, X):
        assert hasattr(self, 'net'), 'Neural network is not defined'
        return self.net(X)

    def plot(self, key, value, train: bool):
        """Plot a point in animation"""
        assert hasattr(self, 'trainer'), 'Trainer is not defined'
        self.board.xlabel = 'epoch'

        if train:
            x = self.trainer.train_batch_idx / \
                self.trainer.num_train_batches
            n = self.trainer.num_train_batches / \
                self.plot_train_per_epoch
        else:
            x = self.trainer.epoch + 1
            n = self.trainer.num_val_batches / \
                self.plot_valid_per_epoch

        self.board.draw(x, value.detach().numpy(),
                        ('train_' if train else 'val_') + key,
                        every_n=int(n))

    def training_step(self, batch):
        l = self.loss(self(*batch[:-1]), batch[-1])
        self.plot('loss', l, train=True)
        return l

    def validation_step(self, batch):
        l = self.loss(self(*batch[:-1]), batch[-1])
        self.plot('loss', l, train=False)

    def configure_optimizers(self):
        raise NotImplementedError

    def apply_init(self, inputs, init=None):
        self.forward(*inputs)

        if init is not None:
            self.net.apply(init)
