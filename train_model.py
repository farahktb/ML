"""
train_model.py
--------------
Trains a Logistic Regression model (binary classifier — sigmoid activation,
gradient descent optimisation) and saves it to disk.

Academic concepts demonstrated
───────────────────────────────
• Supervised learning  → model learns from labelled examples
• Binary classification → output ∈ {0 = legit, 1 = fraud}
• Sigmoid activation   → σ(z) = 1 / (1 + e^{-z})
• Gradient Descent     → minimises binary cross-entropy loss
• Cost function        → J = −[y·log(ŷ) + (1−y)·log(1−ŷ)]
"""

import os
import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

from preprocessing import load_raw_data, preprocess

# ── Paths ──────────────────────────────────────────────────────────────────────
MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
MODEL_PATH = os.path.join(MODELS_DIR, "fraud_model.pkl")
SCALER_PATH = os.path.join(MODELS_DIR, "scaler.pkl")
LOSS_PATH = os.path.join(MODELS_DIR, "loss_history.npy")
META_PATH = os.path.join(MODELS_DIR, "meta.pkl")


# ── Manual sigmoid perceptron (educational) ────────────────────────────────────

class ManualPerceptron:
    """
    Single-layer logistic classifier implemented from scratch with numpy.
    Used only for educational visualisation in the Model Information page.
    """

    def __init__(self, lr: float = 0.1, epochs: int = 200):
        self.lr = lr
        self.epochs = epochs
        self.weights = None
        self.bias = 0.0
        self.loss_history = []

    @staticmethod
    def _sigmoid(z):
        return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0.0
        y = np.array(y, dtype=float)

        for _ in range(self.epochs):
            z = X @ self.weights + self.bias
            y_hat = self._sigmoid(z)

            # Binary cross-entropy loss
            eps = 1e-9
            loss = -np.mean(y * np.log(y_hat + eps) + (1 - y) * np.log(1 - y_hat + eps))
            self.loss_history.append(loss)

            # Gradients
            error = y_hat - y
            dw = (X.T @ error) / n_samples
            db = np.mean(error)

            # Gradient descent update
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

        return self

    def predict_proba(self, X):
        return self._sigmoid(X @ self.weights + self.bias)

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)


# ── Main training routine ──────────────────────────────────────────────────────

def train():
    os.makedirs(MODELS_DIR, exist_ok=True)

    print("[INFO] Loading and preprocessing data...")
    df = load_raw_data()
    X_train, X_test, y_train, y_test, scaler, feature_names = preprocess(df)

    print(f"  Train size : {len(X_train)} | Test size : {len(X_test)}")
    print(f"  Fraud in train : {y_train.sum()} | Legit in train : {(y_train==0).sum()}")

    # ── Logistic Regression (sklearn) ─────────────────────────────────────────
    # class_weight='balanced' compensates for the severe class imbalance
    print("\n[INFO] Training Logistic Regression...")
    model = LogisticRegression(
        max_iter=1000,
        solver="lbfgs",
        class_weight="balanced",
        random_state=42,
        C=1.0,
        tol=1e-4,
    )
    model.fit(X_train, y_train)
    print("  Training complete.")

    # ── Simulate loss curve (gradient descent convergence) ────────────────────
    # We simulate the loss reduction over 200 epochs using a manual perceptron
    # on a small stratified sample — used purely for educational visualisation.
    print("\n[INFO] Computing loss curve (manual perceptron)...")
    sample_idx = np.concatenate([
        np.where(y_train == 0)[0][:2000],
        np.where(y_train == 1)[0],
    ])
    np.random.shuffle(sample_idx)
    X_sample = X_train.iloc[sample_idx].values
    y_sample = y_train.iloc[sample_idx].values

    perceptron = ManualPerceptron(lr=0.05, epochs=300)
    perceptron.fit(X_sample, y_sample)
    loss_history = np.array(perceptron.loss_history)

    # ── Save artefacts ─────────────────────────────────────────────────────────
    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    np.save(LOSS_PATH, loss_history)
    joblib.dump(
        {
            "feature_names": feature_names,
            "X_test": X_test,
            "y_test": y_test,
        },
        META_PATH,
    )

    print(f"\n[INFO] Model saved  : {MODEL_PATH}")
    print(f"[INFO] Scaler saved : {SCALER_PATH}")
    print(f"[INFO] Loss curve   : {LOSS_PATH}")

    # ── Quick sanity check ─────────────────────────────────────────────────────
    from sklearn.metrics import accuracy_score, recall_score
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    print(f"\n[RESULT] Quick evaluation on test set:")
    print(f"  Accuracy : {acc:.4f}")
    print(f"  Recall   : {rec:.4f}")
    print("\n[DONE] Training complete!")


if __name__ == "__main__":
    train()
