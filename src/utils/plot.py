from matplotlib import pyplot as plt

def plot_loss_and_accuracy(losses, accuracies, legend):
    epochs = range(1, len(losses)+1, 1)

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, losses)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend(legend, loc='best')
    plt.title("Losses over epochs")

    plt.subplot(1, 2, 2)
    plt.plot(epochs, accuracies)
    plt.xlabel("Epochs")
    plt.ylabel("Accuracies")
    plt.legend(legend, loc='best')
    plt.title("Accuracies over epochs")

    plt.show()