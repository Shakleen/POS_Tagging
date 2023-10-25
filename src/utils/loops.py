from typing import Generator

from src.module.module import Module


def train_loop(model: Module, iterator: Generator):
    """Implements training loop.

    Trains `model` with data from `iterator`.

    Args:
        model (Module): The model to be trained.
        iterator (Generator): Batch iterator for training data.

    Returns:
        Tuple[float, float]: Training loss and accuracy.
    """
    epoch_loss = epoch_acc = batch_size = 0

    # Set model to training mode
    model.train()

    # Iterate over batches
    for batch in iterator:
        # Get features and labels
        X, y = batch

        # Reset gradients to zero
        model.optim.zero_grad()

        # Make predictions
        y_hat = model(X)

        # Reshape predictions and labels to fit loss function
        y_hat = y_hat.reshape((-1, y_hat.shape[-1]))
        y = y.reshape(-1)

        # Calculate loss and accuracy of predictions
        loss = model.loss(y_hat, y)
        acc = model.accuracy(y_hat, y)

        # Back propagation
        loss.backward()

        # Update parameters
        model.optim.step()

        # Add to total loss
        epoch_loss += loss.item()
        epoch_acc += acc.item()
        batch_size += 1

    return epoch_loss / batch_size, epoch_acc / batch_size
