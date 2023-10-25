import time
import torch
from typing import Generator

from src.module.module import Module
from src.data_module.data_module import DataModule


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


def evaluation_loop(model: Module, iterator: Generator):
    epoch_loss = epoch_acc = batch_size = 0

    # Set model to training mode
    model.eval()

    # Calculate no gradients for evaluation
    with torch.no_grad():
        # Iterate over batches
        for batch in iterator:
            # Get features and labels
            X, y = batch

            # Make predictions
            y_hat = model(X)

            # Reshape predictions and labels to fit loss function
            y_hat = y_hat.reshape((-1, y_hat.shape[-1]))
            y = y.reshape(-1)

            # Calculate loss and accuracy of predictions
            loss = model.loss(y_hat, y)
            acc = model.accuracy(y_hat, y)

            # Add to total loss
            epoch_loss += loss.item()
            epoch_acc += acc.item()
            batch_size += 1

    return epoch_loss / batch_size, epoch_acc / batch_size


def train_epochs(model: Module, data: DataModule, epochs: int):
    """Trains `model` on `data` for `epochs`

    Args:
        model (Module): Model to be trained.
        data (DataModule): Data to be used for training and validating.
        epochs (int): Max epoch count.

    Returns:
        Tuple: List of loss and accuracy values.
    """
    losses, accuracies = [], []

    for epoch in range(epochs):
        start_time = time.time()

        train_loss, train_acc = train_loop(model, data.train_dataloader())
        val_loss, val_acc = evaluation_loop(model, data.val_dataloader())

        elapsed = time.time() - start_time
        
        print(f"Epoch: {epoch+1:02} in ({elapsed:4.2f}) secs")
        print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
        print(f'\t Val. Loss: {val_loss:.3f} |  Val. Acc: {val_acc*100:.2f}%')
        
    return losses, accuracies
