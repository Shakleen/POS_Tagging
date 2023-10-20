import torch
from torch import nn

from .module import Module


class LSTM(Module):
    """Implements base LSTM module."""

    def __init__(self,
                 num_inputs: int,
                 num_hiddens: int,
                 sigma: float = 0.01):
        """Create an LSTM Model.

        This model is implemented from scratch using only nn.Parameter
        module.

        Args:
            num_inputs (int): Embedding dimension.
            num_hiddens (int): Hidden state dimension.
            sigma (float, optional): For normal init of weights. 
            Defaults to 0.01.
        """
        super().__init__()
        self.save_hyperparameters()

        init_weight = lambda *shape: nn.Parameter(
            torch.randn(*shape) * sigma)

        def triple():
            # Returns W_x, W_h, and b_h
            return (init_weight(num_inputs, num_hiddens),
                    init_weight(num_hiddens, num_hiddens),
                    nn.Parameter(torch.zeros(num_hiddens)))

        self.W_xf, self.W_hf, self.b_f = triple()  # Forget Gate
        self.W_xi, self.W_hi, self.b_i = triple()  # Input Gate
        self.W_xo, self.W_ho, self.b_o = triple()  # Output Gate
        self.W_xc, self.W_hc, self.b_c = triple()  # Input Node

    def forward(self, inputs, H_C=None):
        """Performs forward propagation of the LSTM model.

        Args:
            inputs (torch.Tensor): A tensor of shape [num_steps, batch_size, 
            embedding_size].
            H_C (Tuple): Two torch.Tensor objects. The first is the hidden
            state from previous runs and the second is the memory cell. 
            Defaults to `None`.

        Returns:
            Tuple: Returns prediction for each timestep and 
            a tuple containing hidden state and memory cell state.
        """
        H, C = self._get_hidden_and_memory_cell(inputs, H_C)
        outputs = [None] * len(inputs)

        for step, X in enumerate(inputs):
            H, C = self._process_time_step(H, C, X)
            outputs[step] = H

        return outputs, (H, C)

    def _process_time_step(self, H, C, X):
        I = self._gate(torch.sigmoid, X, H, self.W_xi, self.W_hi, self.b_i)
        F = self._gate(torch.sigmoid, X, H, self.W_xf, self.W_hf, self.b_f)
        O = self._gate(torch.sigmoid, X, H, self.W_xo, self.W_ho, self.b_o)
        C_tilde = self._gate(torch.tanh, X, H, self.W_xc, self.W_hc, self.b_c)

        C = F * C + I * C_tilde
        H = torch.tanh(C) * O
        return H, C

    def _gate(self, activation_function, X, H, W_x, W_h, b):
        return activation_function(X @ W_x + H @ W_h + b)

    def _get_hidden_and_memory_cell(self, inputs, H_C):
        if H_C is None:
            # Initial state with shape: [batch_size, num_hidden]
            H = torch.zeros((inputs.shape[1], self.num_hiddens),
                            device=inputs.device)
            C = torch.zeros((inputs.shape[1], self.num_hiddens),
                            device=inputs.device)
        else:
            H, C = H_C

        return H, C
