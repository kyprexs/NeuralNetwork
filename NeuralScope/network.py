import numpy as np

class NeuralNetwork:
    def __init__(self, input_size=784, hidden_size=128, output_size=10):
        # Weights: input to hidden
        self.W1 = np.random.randn(input_size, hidden_size) * 0.01  # (784, 128)
        self.b1 = np.zeros((1, hidden_size))  # (1, 128)
        # Weights: hidden to output
        self.W2 = np.random.randn(hidden_size, output_size) * 0.01  # (128, 10)
        self.b2 = np.zeros((1, output_size))  # (1, 10)

    def summary(self):
        print("Network architecture:")
        print(f"Input layer: {self.W1.shape[0]} neurons")
        print(f"Hidden layer: {self.W1.shape[1]} neurons")
        print(f"Output layer: {self.W2.shape[1]} neurons")
        print(f"W1 shape: {self.W1.shape}, b1 shape: {self.b1.shape}")
        print(f"W2 shape: {self.W2.shape}, b2 shape: {self.b2.shape}")

    def relu(self, x):
        return np.maximum(0, x)

    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def forward(self, X):
        """
        Forward pass: X -> hidden (ReLU) -> output (softmax)
        Stores intermediate values for backpropagation.
        Returns output probabilities.
        """
        self.z1 = np.dot(X, self.W1) + self.b1  # (batch, 128)
        self.a1 = self.relu(self.z1)            # (batch, 128)
        self.z2 = np.dot(self.a1, self.W2) + self.b2  # (batch, 10)
        self.a2 = self.softmax(self.z2)         # (batch, 10)
        return self.a2

    def cross_entropy_loss(self, y_pred, y_true):
        """
        Compute mean cross-entropy loss.
        y_pred: predicted probabilities (batch, 10)
        y_true: true one-hot labels (batch, 10)
        """
        # Avoid log(0) by adding small epsilon
        eps = 1e-12
        y_pred = np.clip(y_pred, eps, 1. - eps)
        loss = -np.sum(y_true * np.log(y_pred)) / y_pred.shape[0]
        return loss

    def backward(self, X, y_true):
        """
        Backpropagation: compute gradients for weights and biases.
        X: input batch (batch, 784)
        y_true: true one-hot labels (batch, 10)
        """
        m = X.shape[0]
        # Output layer error
        dz2 = self.a2 - y_true  # (batch, 10)
        self.dW2 = np.dot(self.a1.T, dz2) / m  # (128, 10)
        self.db2 = np.sum(dz2, axis=0, keepdims=True) / m  # (1, 10)
        # Hidden layer error
        da1 = np.dot(dz2, self.W2.T)  # (batch, 128)
        dz1 = da1 * (self.z1 > 0)  # ReLU derivative
        self.dW1 = np.dot(X.T, dz1) / m  # (784, 128)
        self.db1 = np.sum(dz1, axis=0, keepdims=True) / m  # (1, 128)

    def update_weights(self, learning_rate=0.01):
        self.W1 -= learning_rate * self.dW1
        self.b1 -= learning_rate * self.db1
        self.W2 -= learning_rate * self.dW2
        self.b2 -= learning_rate * self.db2

    def predict(self, X):
        probs = self.forward(X)
        return np.argmax(probs, axis=1)

    def accuracy(self, X, y_true):
        y_pred = self.predict(X)
        y_true_labels = np.argmax(y_true, axis=1)
        return np.mean(y_pred == y_true_labels)

    def train(self, X_train, y_train, X_val, y_val, epochs=10, batch_size=64, learning_rate=0.01):
        n_samples = X_train.shape[0]
        for epoch in range(1, epochs + 1):
            # Shuffle data
            indices = np.arange(n_samples)
            np.random.shuffle(indices)
            X_train = X_train[indices]
            y_train = y_train[indices]
            epoch_loss = 0
            for i in range(0, n_samples, batch_size):
                X_batch = X_train[i:i+batch_size]
                y_batch = y_train[i:i+batch_size]
                # Forward
                y_pred = self.forward(X_batch)
                # Loss
                loss = self.cross_entropy_loss(y_pred, y_batch)
                epoch_loss += loss * X_batch.shape[0]
                # Backward
                self.backward(X_batch, y_batch)
                # Update
                self.update_weights(learning_rate)
            epoch_loss /= n_samples
            train_acc = self.accuracy(X_train, y_train)
            val_acc = self.accuracy(X_val, y_val)
            print(f"Epoch {epoch}: Loss={epoch_loss:.4f}, Train Acc={train_acc:.4f}, Val Acc={val_acc:.4f}")

    def save_weights(self, filename):
        np.savez(filename, W1=self.W1, b1=self.b1, W2=self.W2, b2=self.b2)

    def load_weights(self, filename):
        data = np.load(filename)
        self.W1 = data['W1']
        self.b1 = data['b1']
        self.W2 = data['W2']
        self.b2 = data['b2']

if __name__ == "__main__":
    from data_loader import load_mnist
    nn = NeuralNetwork()
    nn.summary()
    # Load MNIST data
    (X_train, y_train), (X_test, y_test) = load_mnist()
    # Split off a validation set from training data
    X_val, y_val = X_train[-5000:], y_train[-5000:]
    X_train, y_train = X_train[:-5000], y_train[:-5000]
    # Train the network
    nn.train(X_train, y_train, X_val, y_val, epochs=100, batch_size=64, learning_rate=0.01)
    # Evaluate on test set
    test_acc = nn.accuracy(X_test, y_test)
    print(f"Test accuracy: {test_acc:.4f}")
    # Save trained weights
    nn.save_weights('trained_weights.npz') 