import numpy as np

class RNN:

    def __init__(self, input_dim, hidden_dim, output_dim):
        # Weights
        self.Wxh = np.random.randn(input_dim, hidden_dim) * 0.1
        self.Whh = np.random.randn(hidden_dim, hidden_dim) * 0.1 #Previous Hidden State → Current Hidden State
        self.Why = np.random.randn(hidden_dim, output_dim) * 0.1

        # Bias
        self.bh = np.zeros((1, hidden_dim))
        self.by = np.zeros((1, output_dim))

    # Activation functions
    def tanh(self, x):
        return np.tanh(x)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    # 🔷 Forward Pass
    #Takes a sequence input
    #Processes each step using memory (hidden state)
    #Produces a final output
    def forward(self, X):
        self.inputs = X
        self.h_states = []

        h = np.zeros((1, self.Wxh.shape[1]))  # initial hidden state value will be 1

        # Process sequence step-by-step
        for t in range(len(X)):
            x_t = X[t].reshape(1, -1)

            h = self.tanh(x_t @ self.Wxh + h @ self.Whh + self.bh) #hidden value is calculated
            self.h_states.append(h) #appended to the hidden storage

        self.final_h = h

        # Output layer
        self.output = self.sigmoid(self.final_h @ self.Why + self.by)

        return self.output

    # 🔷 Loss (optional)
    def loss(self, y):
        return np.mean((y - self.output) ** 2)

    # 🔷 Backward Pass (simplified BPTT)
    def backward(self, y, lr):
        # Output gradient
        error = self.output - y
        d_out = error * (self.output * (1 - self.output))

        dWhy = self.final_h.T @ d_out
        dby = np.sum(d_out, axis=0, keepdims=True)

        # Backprop into hidden state
        d_h = d_out @ self.Why.T
        d_h_raw = d_h * (1 - self.final_h ** 2)

        # Gradients for last timestep (simplified)
        x_last = self.inputs[-1].reshape(1, -1)

        dWxh = x_last.T @ d_h_raw
        dWhh = self.final_h.T @ d_h_raw
        dbh = np.sum(d_h_raw, axis=0, keepdims=True)

        # Update
        self.Why -= lr * dWhy
        self.by -= lr * dby
        self.Wxh -= lr * dWxh
        self.Whh -= lr * dWhh
        self.bh -= lr * dbh


# 🔷 DATA (sequence input)
X = np.array([
    [1, 0],
    [0, 1],
    [1, 1]
])

y = np.array([[1]])

# 🔷 MODEL
model = RNN(input_dim=2, hidden_dim=4, output_dim=1)

# 🔷 TRAINING
epochs = 25
lr = 0.1

for epoch in range(epochs):
    pred = model.forward(X)
    loss = model.loss(y)

    model.backward(y, lr)

    print(f"Epoch {epoch+1}, Loss: {loss:.4f}, Prediction: {pred}")

# 🔷 FINAL OUTPUT
final_pred = model.forward(X)
print("\nFinal Prediction:", final_pred)
