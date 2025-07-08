import numpy as np
import matplotlib.pyplot as plt
from network import NeuralNetwork
from data_loader import load_mnist


def visualize_activations(nn, X_sample, y_sample=None):
    # Forward pass to get activations
    nn.forward(X_sample)
    hidden_activations = nn.a1[0]  # shape: (128,)
    output_activations = nn.a2[0]  # shape: (10,)
    predicted_digit = np.argmax(output_activations)
    true_digit = np.argmax(y_sample) if y_sample is not None else None

    fig, axs = plt.subplots(1, 3, figsize=(12, 4))

    # Original image
    axs[0].imshow(X_sample.reshape(28, 28), cmap='gray')
    axs[0].set_title('Input Image')
    axs[0].axis('off')
    if y_sample is not None:
        axs[0].set_xlabel(f"Label: {true_digit}")

    # Hidden layer activations heatmap
    axs[1].imshow(hidden_activations.reshape(8, 16), cmap='viridis', aspect='auto')
    axs[1].set_title('Hidden Layer Activations (ReLU)')
    axs[1].axis('off')

    # Output layer activations (softmax probabilities)
    axs[2].bar(np.arange(10), output_activations)
    axs[2].set_xticks(np.arange(10))
    axs[2].set_ylim(0, 1)
    pred_title = f'Predicted: {predicted_digit}'
    if true_digit is not None:
        pred_title += f' (True: {true_digit})'
    axs[2].set_title(pred_title)
    axs[2].set_xlabel('Digit Class')
    axs[2].set_ylabel('Probability')

    plt.tight_layout()
    plt.show(block=False)
    plt.pause(0.001)
    return fig


if __name__ == "__main__":
    # Load trained network
    nn = NeuralNetwork()
    nn.load_weights('trained_weights.npz')
    # Load MNIST data
    (_, _), (X_test, y_test) = load_mnist()
    n_samples = X_test.shape[0]
    print("Press Enter to visualize a new random digit, or 'q' then Enter to quit.")
    while True:
        idx = np.random.randint(0, n_samples)
        X_sample = X_test[idx:idx+1]
        y_sample = y_test[idx:idx+1]
        fig = visualize_activations(nn, X_sample, y_sample)
        user_input = input("[Enter] for next digit, [q] to quit: ").strip().lower()
        plt.close(fig)
        if user_input == 'q':
            break 