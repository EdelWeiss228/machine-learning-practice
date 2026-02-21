import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

def load_mnist(path="mnist.npz"):
    mnist = np.load(path)
    X_train, y_train = mnist["x_train"].reshape(-1, 28*28), mnist["y_train"]
    X_test, y_test = mnist["x_test"].reshape(-1, 28*28), mnist["y_test"]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    return X_train, X_val, X_test, y_train, y_val, y_test

class MyBernoulliNBClassifier:
    def __init__(self, priors=None):
        self.priors = priors

    def fit(self, X, y):
        X = (X > 0.5).astype(int)  
        self.classes = np.unique(y)
        self.priors = {}
        self.conditional_probs = {}

        for c in self.classes:
            X_c = X[y == c]
            self.priors[c] = len(X_c) / len(y)
            self.conditional_probs[c] = (X_c.sum(axis=0) + 1e-6) / (len(X_c) + 1e-6)

    def predict_proba(self, X):
        X = (X > 0.5).astype(int)  
        posteriors = []

        for c in self.classes:
            prior = self.priors[c]
            likelihood = np.prod((self.conditional_probs[c] ** X) * ((1 - self.conditional_probs[c]) ** (1 - X)), axis=1)
            posterior = prior * likelihood
            posteriors.append(posterior)

        return np.array(posteriors).T

    def predict(self, X):
        proba = self.predict_proba(X)
        return self.classes[np.argmax(proba, axis=1)]

    def score(self, X, y):
        y_pred = self.predict(X)
        return np.mean(y_pred == y)

def train_and_evaluate():
    X_train, X_val, X_test, y_train, y_val, y_test = load_mnist()

    model = MyBernoulliNBClassifier()
    model.fit(X_train, y_train)

    y_pred_val = model.predict(X_val)
    val_accuracy = accuracy_score(y_val, y_pred_val)
    print(f"Точность на валидации: {val_accuracy * 100:.2f}%")

    y_pred_test = model.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred_test)
    print(f"Точность на тесте: {test_accuracy * 100:.2f}%")

if __name__ == "__main__":
    train_and_evaluate()