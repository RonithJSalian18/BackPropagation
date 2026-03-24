import numpy as np

class ANN:

    def __init__(self, input_size, hidden_size, output_size):
        # Initialize weights (scaled)
        self.W1 = np.random.randn(input_size, hidden_size) * 0.1
        self.W2 = np.random.randn(hidden_size, output_size) * 0.1

        # Bias terms
        self.b1 = np.zeros((1, hidden_size))
        self.b2 = np.zeros((1, output_size))

    # Activation functions
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def tanh(self, x):
        return np.tanh(x)

    # 🔷 Forward Pass
    def forward(self, X):
        self.X = X

        # Input → Hidden
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.tanh(self.z1)

        # Hidden → Output
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.output = self.sigmoid(self.z2)

        return self.output

    # 🔷 Loss Function (MSE)
    def loss(self, y):
        return np.mean((y - self.output) ** 2)

    # 🔷 Backpropagation
    def backward(self, y, lr):
        # Output layer
        error = self.output - y
        d_out = error * (self.output * (1 - self.output))

        dW2 = np.dot(self.a1.T, d_out)
        db2 = np.sum(d_out, axis=0, keepdims=True)

        # Hidden layer
        d_hidden = np.dot(d_out, self.W2.T)
        d_z1 = d_hidden * (1 - np.tanh(self.z1) ** 2)

        dW1 = np.dot(self.X.T, d_z1)
        db1 = np.sum(d_z1, axis=0, keepdims=True)

        # Update weights
        self.W2 -= lr * dW2
        self.b2 -= lr * db2
        self.W1 -= lr * dW1
        self.b1 -= lr * db1


# 🔷 DATA
X = np.array([[1, 0]])
y = np.array([[1]])

# 🔷 MODEL
ann = ANN(input_size=2, hidden_size=4, output_size=1)

# 🔷 TRAINING
epochs = 25
lr = 0.3

for i in range(epochs):
    output = ann.forward(X)
    loss = ann.loss(y)

    ann.backward(y, lr)

    print(f"Epoch {i+1}, Loss: {loss:.4f}, Output: {output}")

# 🔷 FINAL OUTPUT
final_output = ann.forward(X)
print("\nFinal Prediction:", final_output)