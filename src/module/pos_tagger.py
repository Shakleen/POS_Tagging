import torch
from torch import nn
from torch.nn import functional as F

from .module import Module
from .deep_lstm import DeepLSTM


class PosTagger(Module):
    """Implements part of speech tagger model."""

    def __init__(self,
                 num_inputs: int,
                 embedding_dim: int,
                 num_hiddens: int,
                 num_outputs: int,
                 padding_idx: int,
                 bidirectional: bool = False,
                 num_layers: int = 1,
                 sigma: float = 0.01):
        """Creates a Part-of-Speech tagger model.

        Args:
            num_inputs (int): Number of unique tokens in the vocabulary.
            embedding_dim (int): Word embedding size. Typically, 50, 100, 200, 
            or 300.
            num_hiddens (int): Number of units in a hidden state of an LSTM.
            num_outputs (int): Number of unique POS tokens.
            padding_idx (int): Pad token value.
            bidirectional (bool, optional): If LSTM should be bidirectional. 
            Defaults to False.
            num_layers (int, optional): How many layers deep should the LSTM be. 
            Defaults to 1.
            sigma (float, optional): For initialization of weights using normal 
            distribution. Defaults to 0.01.
        """
        super().__init__()
        self.save_hyperparameters()

        self.embedding = nn.Embedding(num_embeddings=self.num_inputs,
                                      embedding_dim=self.embedding_dim,
                                      padding_idx=self.padding_idx)

        self.lstm = DeepLSTM(num_inputs=self.embedding_dim,
                             num_hiddens=self.num_hiddens,
                             num_layers=self.num_layers,
                             bidirectional=self.bidirectional,
                             sigma=self.sigma)

        self.fc = nn.LazyLinear(self.num_outputs)

        self.configure_optimizers()

    def forward(self, X):
        """Performs forward propagation.

        Args:
            X (torch.Tensor): Input features of shape 
            [num_steps, batch_size, embedding_dim]

        Returns:
            torch.Tensor: Predictions of shape
            [num_steps, batch_size, num_outputs]
        """
        embeds = self.embedding(X)
        outputs, _ = self.lstm(embeds)
        return self.fc(outputs)

    def configure_optimizers(self):
        self.optim = torch.optim.Adam(self.parameters())

    def loss(self, y_hat, y):
        """Calculates the loss between prediction and target.

        Args:
            y_hat (torch.Tensor): Predictions.
            y (torch.Tensor): Target labels.

        Returns:
            torch.Tensor: Loss value.
        """
        return F.cross_entropy(input=y_hat,
                               target=y,
                               ignore_index=self.padding_idx,
                               reduce='mean')

    def accuracy(self, y_hat, y):
        """Calculates the accuracy of the predictions.

        Args:
            y_hat (torch.Tensor): Predictions.
            y (torch.Tensor): Target labels.

        Returns:
            torch.Tensor: Accuracy value.
        """
        y_hat = y_hat.argmax(dim=1, keepdim=True)
        non_pad_elements = (y != self.padding_idx).nonzero()
        correct = y_hat[non_pad_elements].squeeze(1).eq(y[non_pad_elements])
        return correct.sum() / y[non_pad_elements].shape[0]
