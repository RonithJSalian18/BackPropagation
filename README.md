# 🧠 Backpropagation using ANN, CNN & RNN (From Scratch)

![Python](https://img.shields.io/badge/Python-3.x-blue.svg)  
![NumPy](https://img.shields.io/badge/Library-NumPy-orange.svg)  
![Status](https://img.shields.io/badge/Project-Completed-brightgreen.svg)  
![Level](https://img.shields.io/badge/Level-Beginner--Intermediate-yellow.svg)

---

## 👥 Team Members

- **Ronith Salian**
- **Rajath D Shetty**

---

## ✨ Overview

This project demonstrates how **Backpropagation** works in three core deep learning models:

- 🔹 Artificial Neural Network (ANN)  
- 🔹 Convolutional Neural Network (CNN)  
- 🔹 Recurrent Neural Network (RNN)  

All models are implemented **from scratch using NumPy**, focusing on **clear understanding + step-by-step computation**.

---

## 🧩 Model Intuition
Input → Hidden Layer → Output

● -------- ● -------- ●

✔ Fully connected layers  
✔ Learns non-linear relationships  

---

### 🔷 CNN (Feature Extraction)
Input → [3×3 Filter] → Feature → Output

⬛⬛⬛
⬛⬛⬛

✔ Detects patterns (edges, shapes)  
✔ Uses shared weights (kernels)  

---

### 🔷 RNN (Sequential Learning)
x₁ → [h] → x₂ → [h] → x₃ → [h]

✔ Maintains memory of previous inputs  
✔ Processes sequences step-by-step  

---

## 🔁 Backpropagation Overview

Backpropagation is used to **minimize error** by updating weights using gradients.

### 🔹 Steps:
1. Forward Pass → Compute output  
2. Loss Calculation → Measure error  
3. Backward Pass → Compute gradients  
4. Update Weights → Reduce error  

---

## 📐 Key Formula

**Gradient:**
∂L / ∂W = (∂L / ∂y) × (∂y / ∂W)

**Error term:**
δ = (y_actual − y_predicted) × f'(z)

---

## ⚙️ Weight Update Rule
W_new = W_old − η × (∂L / ∂W)
Where:
- η = Learning rate  
- ∂L/∂W = Gradient  

---

## 🧠 ANN Implementation

### 🔹 Key Features
- Hidden layer with activation (tanh/sigmoid)  
- Forward + Backward propagation  
- Mean Squared Error (MSE)  

### 🔹 Flow

Input → Hidden → Output

### 🔹 What it learns
- Non-linear patterns  
- Basic decision boundaries  

---

## 🧠 CNN Implementation

### 🔹 Key Features
- 3×3 convolution filter  
- ReLU / Leaky ReLU activation  
- Fully connected output layer  

### 🔹 Convolution Formula
Output = Σ (Input × Kernel)

### 🔹 What it learns
- Spatial features  
- Patterns in input matrix  

---

## 🧠 RNN Implementation

### 🔹 Key Features
- Sequential input processing  
- Hidden state (memory)  
- Simplified Backpropagation Through Time (BPTT)  

### 🔹 Hidden State Equation
h_t = tanh(x_tW_x + h_{t-1}W_h + b)


### 🔹 What it learns
- Sequential patterns  
- Time-dependent relationships  

---

## ⚖️ Comparison

| Feature | ANN | CNN | RNN |
|--------|-----|-----|-----|
| Data Type | Tabular | Image | Sequence |
| Structure | Layers | Filters | Time steps |
| Memory | ❌ No | ❌ No | ✅ Yes |
| Weight Sharing | ❌ | ✅ | ✅ |
| Complexity | Low | Medium | High |

---

## 📂 Project Structure

deep_learning_project/
│
├── ann.py # Artificial Neural Network
├── cnn.py # Convolutional Neural Network
├── rnn.py # Recurrent Neural Network
└── README.md


---

## ▶️ Run the Code

```bash
python ann.py
python cnn.py
python rnn.py
```

ANN → Final Prediction: 0.98
CNN → Convolution Output: 3.66
RNN → Final Prediction: 0.92

### 🔷 ANN (Feedforward Network)
