# Détection de fraude dans les transactions financières
### Projet de Machine Learning — Apprentissage Supervisé

---

## 1. Introduction

Financial fraud is one of the most critical problems in modern banking systems.
Every year, billions of euros are lost due to fraudulent transactions that go undetected.

**The Challenge:**
Traditional rule-based systems are slow to adapt and easy to bypass.
Machine Learning offers a smarter, automated solution.

**Our Solution:**
This project builds an intelligent fraud detection system using:

- **Supervised Learning** — the model learns from labelled historical transactions
- **Binary Classification** — the model decides: Fraud (1) or Legitimate (0)
- **Perceptron Model** — a single-layer neural unit using sigmoid activation
- **Gradient Descent** — the algorithm that teaches the model to improve

> The application simulates a real banking cybersecurity monitoring system
> capable of analyzing transactions and predicting fraud in real time.

---

## 2. Dataset

**Source:** [Kaggle — Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

| Property | Value |
|---|---|
| Total rows | 284,807 transactions |
| Total columns | 31 |
| Features used | Time, Amount, V1 to V28 |
| Target variable | Class (0 = Legitimate, 1 = Fraud) |
| Missing values | 0 |
| Fraudulent transactions | 492 (0.17%) |
| Legitimate transactions | 284,315 (99.83%) |

**Target Variable:**
```
Class = 0  →  Normal (legitimate) transaction
Class = 1  →  Fraudulent transaction
```

**Class Imbalance:**
The dataset is heavily imbalanced — only 0.17% of transactions are fraudulent.
This is realistic for real banking data, and it means we cannot rely on accuracy alone.
We use `class_weight='balanced'` in the model to compensate.

> For performance, we work with a stratified sample of **20,000 rows**,
> preserving the original fraud/legitimate ratio.

**Code — Dataset Loading (`preprocessing.py`):**
```python
df = pd.read_csv("data/creditcard.csv")
n_missing    = df.isnull().sum().sum()     # Check missing values
n_duplicates = df.duplicated().sum()       # Check duplicates
df.drop_duplicates(inplace=True)           # Remove duplicates
```

---

## 3. Data Preprocessing

Preprocessing is essential before training any Machine Learning model.

### Steps Applied

**Step 1 — Data Cleaning**
```python
df.isnull().sum().sum()      # 0 missing values confirmed
df.drop_duplicates()         # Remove any duplicate rows
```

**Step 2 — Feature Scaling / Normalization**

The `Amount` and `Time` columns have very large values compared to V1–V28
(which are already PCA-transformed). We normalize them with `StandardScaler`:

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
df["Amount_scaled"] = scaler.fit_transform(df[["Amount"]])
df["Time_scaled"]   = scaler.fit_transform(df[["Time"]])
df.drop(columns=["Amount", "Time"], inplace=True)
```

**Step 3 — Features and Target Separation**
```python
X = df.drop(columns=["Class"])   # Features: V1-V28 + Amount_scaled + Time_scaled
y = df["Class"]                   # Target: 0 or 1
```

**Step 4 — Train / Test Split (80% / 20%)**
```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    stratify=y,          # Preserve class ratio in both sets
    random_state=42
)
```

---

## 4. Visualization

All visualizations use **matplotlib**, following the course requirements.

| Chart | Purpose | File |
|---|---|---|
| Bar + Pie chart | Fraud vs Legitimate count | `visualizations.py` |
| Histogram | Transaction amount distribution | `visualizations.py` |
| Distribution plot | Model fraud probability output | `visualizations.py` |
| Heatmap | Confusion matrix | `visualizations.py` |
| Line curve | Loss reduction (gradient descent) | `visualizations.py` |
| Area chart | Fraud evolution over time | `visualizations.py` |
| Sigmoid curve | Activation function explained | `visualizations.py` |
| Parabola + arrows | Gradient descent illustrated | `visualizations.py` |

All charts are displayed live inside the Streamlit dashboard.

---

## 5. Train / Test Split

Following the course methodology:

| Set | Proportion | Size (from 20k sample) |
|---|---|---|
| Training set | 80% | 16,000 transactions |
| Testing set | 20% | 4,000 transactions |

The split is **stratified** — both sets keep the same fraud ratio as the original data.

This ensures the model is evaluated on data it has never seen during training,
which gives a realistic measure of real-world performance.

---

## 6. The Model — Perceptron with Sigmoid Activation

### What is a Perceptron?

The perceptron is the fundamental unit of classification.
It takes all input features, multiplies each by a learned weight, sums them up,
and passes the result through the sigmoid activation function.

### Linear Combination (z)

```
z = w₁x₁ + w₂x₂ + w₃x₃ + ... + w₃₀x₃₀ + b
```

Where:
- `x₁ ... x₃₀` = input features (V1-V28, Amount_scaled, Time_scaled)
- `w₁ ... w₃₀` = learned weights
- `b` = bias term

### Sigmoid Activation — σ(z)

```
σ(z) = 1 / (1 + e^(−z))
```

| Value of z | σ(z) | Interpretation |
|---|---|---|
| z very negative | ≈ 0.0 | Legitimate transaction |
| z = 0 | = 0.5 | Uncertain |
| z very positive | ≈ 1.0 | Fraudulent transaction |

**Decision Rule:**
```
if σ(z) >= 0.5  →  Predict Fraud     (Class = 1)
if σ(z) <  0.5  →  Predict Legitimate (Class = 0)
```

### Manual Implementation (numpy)

```python
class ManualPerceptron:

    def _sigmoid(self, z):
        return 1.0 / (1.0 + np.exp(-z))   # Sigmoid activation

    def fit(self, X, y):
        self.weights = np.zeros(X.shape[1])
        self.bias    = 0.0

        for epoch in range(self.epochs):
            z     = X @ self.weights + self.bias   # Linear combination
            y_hat = self._sigmoid(z)               # Apply sigmoid
            ...                                    # Update weights (see Step 8)
```

### sklearn Implementation

```python
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced",   # Handle class imbalance
    solver="lbfgs"
)
model.fit(X_train, y_train)
```

> Logistic Regression in sklearn is mathematically equivalent to a single
> perceptron with sigmoid activation. It is the standard academic tool for
> binary classification.

---

## 7. Cost Function — Log Loss (Binary Cross-Entropy)

To measure how wrong the model's predictions are, we use **Log Loss**:

```
J(θ) = −(1/m) · Σ [ y·log(ŷ) + (1−y)·log(1−ŷ) ]
```

Where:
- `m` = number of training examples
- `y` = true label (0 or 1)
- `ŷ` = predicted probability from sigmoid

**Interpretation:**

| Situation | Cost |
|---|---|
| y=1 (fraud) and ŷ→1 (correctly predicts fraud) | Cost ≈ 0 (good) |
| y=1 (fraud) and ŷ→0 (misses the fraud) | Cost → ∞ (very bad) |
| y=0 (legit) and ŷ→0 (correctly predicts legit) | Cost ≈ 0 (good) |

**Objective:** Minimize J(θ) during training.

**Code:**
```python
eps  = 1e-9
loss = -np.mean(
    y * np.log(y_hat + eps) + (1 - y) * np.log(1 - y_hat + eps)
)
self.loss_history.append(loss)   # Saved for the loss curve chart
```

---

## 8. Learning Algorithm — Gradient Descent

Gradient Descent is the algorithm that teaches the model to improve.

It works by computing the gradient (slope) of the cost function
and updating the weights in the direction that reduces the error.

### Update Rule

```
w(t+1) = w(t) − α · (∂J / ∂w)
```

Where:
- `w` = model weights
- `α` = learning rate (step size)
- `∂J/∂w` = gradient of the cost function

### Code Implementation

```python
# Compute prediction error
error = y_hat - y                        # Difference between prediction and truth

# Compute gradients
dw = (X.T @ error) / n_samples          # Gradient for weights
db = np.mean(error)                      # Gradient for bias

# Update weights — Gradient Descent step
self.weights -= self.lr * dw             # w = w - α · ∂J/∂w
self.bias    -= self.lr * db             # b = b - α · ∂J/∂b
```

### Learning Rate (α)

| α too large | α too small |
|---|---|
| Overshoots the minimum | Converges too slowly |
| Unstable learning | Stable but slow |

We use `α = 0.05` (learning rate = 0.05) with 300 epochs.

---

## 9. Training the Model

### Two Approaches Used in This Project

**Approach A — Manual Perceptron (numpy)**

Used to generate the **loss curve** that demonstrates gradient descent learning.
Trained on a balanced sample of the training data for 300 epochs.

```python
perceptron = ManualPerceptron(lr=0.05, epochs=300)
perceptron.fit(X_sample, y_sample)
loss_history = perceptron.loss_history   # Saved for visualization
```

**Approach B — Logistic Regression (sklearn)**

Used as the main fraud prediction model for the live dashboard.

```python
model = LogisticRegression(max_iter=1000, class_weight="balanced")
model.fit(X_train, y_train)
```

Both approaches implement the same mathematical concept:
sigmoid activation + gradient descent + log loss minimization.

---

## 10. Evaluation

We do NOT rely on accuracy alone because the dataset is imbalanced.

### Metrics Used

| Metric | Formula | Value |
|---|---|---|
| Accuracy | (TP + TN) / Total | ~99% |
| Precision | TP / (TP + FP) | Displayed in app |
| Recall | TP / (TP + FN) | Displayed in app |
| Log Loss | Cross-entropy on probabilities | Displayed in app |
| AUC-ROC | Area under ROC curve | Displayed in app |

### Confusion Matrix

```
                  Predicted Legitimate   Predicted Fraud
Real Legitimate        TN (correct)         FP (false alarm)
Real Fraud             FN (missed!)         TP (detected)
```

| Term | Meaning | Importance |
|---|---|---|
| **TP** — True Positive | Fraud correctly detected | We want this HIGH |
| **TN** — True Negative | Legitimate correctly identified | We want this HIGH |
| **FP** — False Positive | Legitimate flagged as fraud | Minor inconvenience |
| **FN** — False Negative | Fraud NOT detected | **DANGEROUS — costly!** |

### Why Recall is the Most Important Metric

```
Recall = TP / (TP + FN)
```

In fraud detection, **missing a fraud (FN) is far more costly** than a false alarm (FP).

A missed fraud means:
- Real financial loss for the bank
- Customer funds stolen
- Reputation damage

Therefore, we optimize for **high Recall**, not just accuracy.

---

## 11. Results

### Model Performance on Test Set (4,000 transactions)

| Metric | Result |
|---|---|
| Accuracy | 99.15% |
| Recall | 57.14% |
| Loss Curve | Decreasing from epoch 1 to 300 |

### Loss Curve

The loss curve shows the binary cross-entropy decreasing over 300 training epochs.
This visually demonstrates that **gradient descent is working** —
the model improves with each iteration.

### Key Observations

- **False Negatives** exist because the dataset is extremely imbalanced
- Using `class_weight='balanced'` significantly improves Recall
- The sigmoid output provides calibrated fraud probabilities, not just binary labels
- The loss curve confirms the gradient descent convergence behavior taught in the course

---

## 12. Conclusion

This project demonstrates that **Machine Learning can effectively improve banking security**
by automatically flagging suspicious transactions before human review.

### What Was Achieved

- A complete supervised learning pipeline from raw data to live prediction
- Binary classification using perceptron concepts aligned with the course
- Gradient descent optimization with visual proof of learning
- A professional interactive dashboard for real-time fraud detection

### Academic Alignment

| Course Concept | Implementation |
|---|---|
| Supervised Learning | Model trained on 16,000 labelled transactions |
| Binary Classification | Output ∈ {0 = Legitimate, 1 = Fraud} |
| Perceptron | `ManualPerceptron` class with numpy |
| Sigmoid Activation | `σ(z) = 1 / (1 + e^(-z))` |
| Gradient Descent | `w = w − α · ∂J/∂w` |
| Log Loss | `J = −(1/m) · Σ [y·log(ŷ) + (1−y)·log(1−ŷ)]` |
| Train/Test Split | 80% / 20% stratified |
| Evaluation | Accuracy, Precision, Recall, Confusion Matrix, Log Loss |

### Limitations

- The Kaggle dataset uses anonymized PCA features (V1–V28), not raw banking columns
- Recall of ~57% shows there is room for improvement with more advanced techniques
- The demo interface maps simplified inputs to the model's feature space

### Future Work

- Use deeper neural networks (multi-layer perceptron) for higher recall
- Apply SMOTE oversampling to handle class imbalance more effectively
- Integrate real-time transaction stream processing
- Add more explainability to the fraud flagging system

---

## Project Structure

```
ML project/
├── data/
│   └── creditcard.csv          Kaggle fraud dataset (284,807 rows)
├── models/
│   ├── fraud_model.pkl         Trained LogisticRegression model
│   ├── scaler.pkl              StandardScaler for Amount and Time
│   ├── loss_history.npy        Loss values per epoch (gradient descent)
│   └── meta.pkl                Test set + feature names
├── preprocessing.py            Data loading, cleaning, scaling, split
├── train_model.py              Training — ManualPerceptron + sklearn
├── evaluation.py               Metrics: accuracy, recall, confusion matrix
├── visualizations.py           All matplotlib charts
├── app.py                      Streamlit dashboard (3 pages)
└── requirements.txt            Dependencies
```

---

## How to Run

```bash
# Step 1 — Install dependencies
pip install -r requirements.txt

# Step 2 — Train the model
python train_model.py

# Step 3 — Launch the dashboard
python -m streamlit run app.py
```

The app opens at: **http://localhost:8501**

---

## Libraries Used

| Library | Role |
|---|---|
| `pandas` | Dataset loading and manipulation |
| `numpy` | Manual perceptron, sigmoid, gradient descent |
| `matplotlib` | All charts and visualizations |
| `scikit-learn` | LogisticRegression, metrics, StandardScaler |
| `streamlit` | Interactive web dashboard |
| `joblib` | Saving and loading trained model |

> No TensorFlow, PyTorch, XGBoost, or any advanced deep learning framework was used.
> This project strictly follows the course content.
