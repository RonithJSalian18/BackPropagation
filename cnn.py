import numpy as np

class CNN:

    def __init__(self):
        self.kernel = np.random.randn(3, 3)
        self.fc_w = np.random.randn()
        self.bias = 0.0

    # 🔷 Activation functions
    def relu(self, x):
        return max(0, x)

    def relu_derivative(self, x):
        return 1 if x > 0 else 0

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    # 🔷 Forward
    def forward(self, X):
        self.X = X

        # Convolution
        self.conv = float(np.sum(X * self.kernel))
        print("Convolution output:", self.conv)

        # ReLU
        self.relu_out = self.relu(self.conv)
        print("After ReLU:", self.relu_out)

        # Output
        self.out = float(self.sigmoid(self.relu_out * self.fc_w + self.bias))
        print("Final Output:", [[self.out]])

        return self.out

    # 🔷 Backprop
    def backward(self, y, lr):
        print("\n--- Backpropagation ---")

        error = y - self.out
        print("Error:", [[error]])

        d_out = error * self.out * (1 - self.out)
        print("Change in output:", [[d_out]])

        d_fc_weight = self.relu_out * d_out
        print("Change in weight:", [[d_fc_weight]])

        d_bias = d_out
        print("Change in bias:", [[d_bias]])

        d_relu = d_out * self.fc_w
        print("Backward to previous step:", [[d_relu]])

        d_conv = d_relu * self.relu_derivative(self.conv)
        print("After ReLU check:", [[d_conv]])

        d_filter = self.X * d_conv
        print("Change in filter:\n", d_filter)

        # 🔷 Update
        self.fc_w -= lr * d_fc_weight
        self.bias -= lr * d_bias
        self.kernel -= lr * d_filter

        print("\nUpdated weight:", [[self.fc_w]])
        print("Updated bias:", [[self.bias]])
        print("Updated filter:\n", self.kernel)


# 🔷 INPUT
X = np.array([
    [1, 2, 1],
    [0, 1, 0],
    [2, 1, 2]
])

y = 1
lr = 0.1

# 🔷 MODEL
cnn = CNN()

print("=== FORWARD ===")
cnn.forward(X)

cnn.backward(y, lr)