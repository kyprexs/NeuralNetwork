import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

# Load MNIST data
def load_mnist():
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
    # Flatten images to 784-length vectors
    train_images = train_images.reshape((train_images.shape[0], 28 * 28))
    test_images = test_images.reshape((test_images.shape[0], 28 * 28))
    # Normalize pixel values to [0, 1]
    train_images = train_images.astype('float32') / 255.0
    test_images = test_images.astype('float32') / 255.0
    # One-hot encode labels
    train_labels = to_categorical(train_labels, 10)
    test_labels = to_categorical(test_labels, 10)
    return (train_images, train_labels), (test_images, test_labels)

if __name__ == "__main__":
    (train_images, train_labels), (test_images, test_labels) = load_mnist()
    print(f"Train images shape: {train_images.shape}")
    print(f"Train labels shape: {train_labels.shape}")
    print(f"Test images shape: {test_images.shape}")
    print(f"Test labels shape: {test_labels.shape}") 