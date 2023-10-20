import torch

from .module import Module
from .lstm import LSTM


class BiLSTM(LSTM):
    """Implements bidrectional LSTM model."""

    def __init__(self,
                 num_inputs: int,
                 num_hiddens: int,
                 sigma: float = 0.01):
        """Create a bidrectional LSTM Model.

        This model is implemented using `LSTM`.

        Args:
            num_inputs (int): Embedding dimension.
            num_hiddens (int): Hidden state dimension.
            sigma (float, optional): For normal init of weights. 
            Defaults to 0.01.
        """
        Module.__init__(self)
        self.save_hyperparameters()

        self.forward_lstm = LSTM(num_inputs, num_hiddens, sigma)
        self.backward_lstm = LSTM(num_inputs, num_hiddens, sigma)
        self.num_hiddens *= 2

    def forward(self, inputs, Hs=None):
        """Performs the forward propagation.

        Args:
            inputs (torch.Tensor): A tensor of shape [num_steps, batch_size, 
            embedding_size].
            Hs (Tuple, optional): Forward and backward hidden states and 
            memory cell tuples. Defaults to None.

        Returns:
            Tuple: Returns prediction for each timestep and a tuple containing 
            forward and backward hidden states and memory cells.
        """
        f_h, b_h = Hs if Hs is not None else None, None

        # Perform left to right pass
        f_outputs, f_h = self.forward_lstm(inputs, f_h)

        # Perform right to left pass
        b_outputs, b_h = self.backward_lstm(reversed(inputs), b_h)

        outputs = [torch.cat([f, b], -1)
                   for (f, b) in zip(f_outputs, reversed(b_outputs))]
        return outputs, (f_h, b_h)
