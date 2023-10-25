import torch
from torch import nn

from ..utils.hyperparameters import HyperParameters


class Module(nn.Module, HyperParameters):  # @save
    """The base class of all models."""

    def __init__(self, plot_train_per_epoch=2, plot_valid_per_epoch=1):
        super().__init__()
        self.save_hyperparameters()

    def loss(self, y_hat, y):
        raise NotImplementedError

    def accuracy(self, y_hat, y):
        raise NotImplementedError

    def forward(self, X):
        assert hasattr(self, 'net'), 'Neural network is not defined'
        return self.net(X)

    def training_step(self, batch):
        # Get features and labels
        X, y = batch

        # Reset gradients to zero
        self.optim.zero_grad()

        # Make predictions
        y_hat = self(X)

        # Reshape predictions and labels to fit loss function
        y_hat = y_hat.reshape((-1, y_hat.shape[-1]))
        y = y.reshape(-1)

        # Calculate loss and accuracy of predictions
        loss = self.loss(y_hat, y)
        acc = self.accuracy(y_hat, y)

        # Back propagation
        loss.backward()

        # Update parameters
        self.optim.step()

        return loss, acc

    def validation_step(self, batch):
        # Get features and labels
        X, y = batch

        # Make predictions
        y_hat = self(X)

        # Reshape predictions and labels to fit loss function
        y_hat = y_hat.reshape((-1, y_hat.shape[-1]))
        y = y.reshape(-1)

        # Calculate loss and accuracy of predictions
        loss = self.loss(y_hat, y)
        acc = self.accuracy(y_hat, y)

        return loss, acc

    def configure_optimizers(self):
        raise NotImplementedError
