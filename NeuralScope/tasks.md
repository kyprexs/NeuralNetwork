# Neural Network From Scratch — Multi-class Classification: Task Breakdown

## Step 1: Data Handling
- [x] Download the MNIST dataset
- [x] Load MNIST data into Python (train/test split)
- [x] Flatten images to 784-length vectors
- [x] Normalize pixel values to [0, 1]
- [x] One-hot encode the labels

## Step 2: Network Initialization
- [x] Define network architecture (input, hidden, output layers)
- [x] Initialize weights and biases randomly for all layers

## Step 3: Activation Functions
- [x] Implement ReLU activation and its derivative
- [ ] Implement sigmoid activation and its derivative (optional)
- [x] Implement softmax activation for output layer
- [x] Implement derivatives needed for backpropagation

## Step 4: Forward Pass
- [x] Compute weighted sums and activations for each layer
- [x] Apply softmax to output layer to get class probabilities

## Step 5: Loss Function
- [x] Implement cross-entropy loss function

## Step 6: Backpropagation
- [x] Compute output error using cross-entropy and softmax
- [x] Backpropagate error through hidden layers
- [x] Compute gradients for weights and biases

## Step 7: Weight Update
- [x] Implement gradient descent weight and bias updates
- [x] (Optional) Implement mini-batch gradient descent

## Step 8: Training Loop
- [x] Set up training loop over epochs
- [x] Shuffle and batch data (if using mini-batches)
- [x] Track training loss and accuracy per epoch

## Step 9: Evaluation
- [ ] Evaluate model on MNIST test set
- [ ] Calculate and report classification accuracy
- [ ] Visualize training loss and accuracy over epochs

## Step 10: Documentation
- [ ] Document network architecture and code usage
- [ ] Provide instructions for running training and evaluation

## Optional Enhancements
- [ ] Add support for multiple hidden layers
- [ ] Implement Adam or momentum optimizer
- [ ] Add dropout or L2 regularization
- [ ] Visualize learned weights or activations
- [ ] **Interactive visualizer for neuron activations and training progress** 