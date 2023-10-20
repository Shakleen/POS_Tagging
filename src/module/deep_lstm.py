import torch
from torch import nn

from .module import Module
from .lstm import LSTM
from .bi_lstm import BiLSTM


class DeepLSTM(LSTM):
    def __init__(self,
                 num_inputs: int,
                 num_hiddens: int,
                 num_layers: int = 1,
                 bidirectional: bool = False,
                 sigma: float = 0.01):
        """Create a bidrectional LSTM Model.

        This model is implemented using `LSTM`.

        Args:
            num_inputs (int): Embedding dimension.
            num_hiddens (int): Hidden state dimension.
            num_layers (int): Number of layers deep.
            bidirectional (bool): Whether to use bidirectional LSTMs.
            Defaults to False.
            sigma (float, optional): For normal init of weights. 
            Defaults to 0.01.
        """
        Module.__init__(self)
        self.save_hyperparameters()
        self.layers = nn.Sequential(*[self._get_lstm(i)
                                    for i in range(num_layers)])

    def _get_lstm(self, i):
        """Creates LSTM model for `i`-th layer."""
        if not self.bidirectional:
            return LSTM(num_inputs=self.num_hiddens if i > 0 else self.num_inputs,
                        num_hiddens=self.num_hiddens,
                        sigma=self.sigma)

        return BiLSTM(num_inputs=self.num_inputs if i is 0 else self.num_hiddens * 2,
                      num_hiddens=self.num_hiddens,
                      sigma=self.sigma)

    def forward(self, inputs, Hs=None):
        """Performs the forward propagation.

        Args:
            inputs (torch.Tensor): A tensor of shape [num_steps, batch_size, 
            embedding_size].
            Hs (Tuple, optional): A list of tuples containing hidden states
            and memory cells. Defaults to None.

        Returns:
            Tuple: Returns prediction for each timestep and 
            a tuple containing hidden states and memory cells.
        """
        outputs = inputs
        Hs = [None] * self.num_layers if Hs is None else Hs

        for layer in range(self.num_layers):
            outputs, Hs[layer] = self.layers[layer](outputs, Hs[layer])
            outputs = torch.stack(outputs, 0)

        return outputs, Hs
