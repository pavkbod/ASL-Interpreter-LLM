import os
import cv2
import numpy as np

import os
import cv2
import numpy as np

def confusion_matrix(y_true, y_pred, num_classes):
    cm = np.zeros((num_classes, num_classes), dtype=int)

    for true_label, pred_label in zip(y_true, y_pred):
        cm[true_label][pred_label] += 1

    return cm


def print_confusion_matrix(cm, class_names):
    print("\nConfusion Matrix:")
    
    header = "True\\Pred".ljust(15)
    for name in class_names:
        header += name.ljust(15)
    print(header)

    for i, row in enumerate(cm):
        row_str = class_names[i].ljust(15)
        for val in row:
            row_str += str(val).ljust(15)
        print(row_str)


def print_per_class_accuracy(cm, class_names):
    print("\nPer-Class Accuracy:")
    for i, class_name in enumerate(class_names):
        total = np.sum(cm[i])
        correct = cm[i][i]
        acc = correct / total if total > 0 else 0
        print(f"{class_name}: {acc:.4f} ({correct}/{total})")

def load_dataset(data_dir, img_size=(64, 64)):
    X = []
    y = []

    class_names = sorted(
        [folder for folder in os.listdir(data_dir)
         if os.path.isdir(os.path.join(data_dir, folder))]
    )

    label_to_index = {label: i for i, label in enumerate(class_names)}

    for label in class_names:
        folder_path = os.path.join(data_dir, label)

        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)

            img = cv2.imread(file_path)
            if img is None:
                continue

            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.resize(img, img_size)
            img = img.astype(np.float32) / 255.0
            img = img.flatten()

            X.append(img)
            y.append(label_to_index[label])

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int32)

    return X, y, class_names


def train_test_split(X, y, test_ratio=0.2):
    indices = np.arange(len(X))
    np.random.shuffle(indices)

    X = X[indices]
    y = y[indices]

    split_index = int(len(X) * (1 - test_ratio))

    X_train = X[:split_index]
    y_train = y[:split_index]
    X_test = X[split_index:]
    y_test = y[split_index:]

    return X_train, y_train, X_test, y_test


def one_hot_encode(y, num_classes):
    one_hot = np.zeros((len(y), num_classes), dtype=np.float32)
    one_hot[np.arange(len(y)), y] = 1.0
    return one_hot


class SimpleNeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.W1 = np.random.randn(input_size, hidden_size) * np.sqrt(1.0 / input_size)
        self.b1 = np.zeros((1, hidden_size))

        self.W2 = np.random.randn(hidden_size, output_size) * np.sqrt(1.0 / hidden_size)
        self.b2 = np.zeros((1, output_size))

    def relu(self, Z):
        return np.maximum(0, Z)

    def relu_derivative(self, Z):
        return (Z > 0).astype(np.float32)

    def softmax(self, Z):
        Z = Z - np.max(Z, axis=1, keepdims=True)
        exp_scores = np.exp(Z)
        return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

    def forward(self, X):
        self.Z1 = np.dot(X, self.W1) + self.b1
        self.A1 = self.relu(self.Z1)

        self.Z2 = np.dot(self.A1, self.W2) + self.b2
        self.A2 = self.softmax(self.Z2)

        return self.A2

    def compute_loss(self, y_pred, y_true_one_hot):
        m = y_true_one_hot.shape[0]
        epsilon = 1e-10
        return -np.sum(y_true_one_hot * np.log(y_pred + epsilon)) / m

    def backward(self, X, y_true_one_hot, y_pred, learning_rate):
        m = X.shape[0]

        dZ2 = y_pred - y_true_one_hot
        dW2 = np.dot(self.A1.T, dZ2) / m
        db2 = np.sum(dZ2, axis=0, keepdims=True) / m

        dA1 = np.dot(dZ2, self.W2.T)
        dZ1 = dA1 * self.relu_derivative(self.Z1)
        dW1 = np.dot(X.T, dZ1) / m
        db1 = np.sum(dZ1, axis=0, keepdims=True) / m

        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2
        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1

    def train(self, X, y, epochs=50, learning_rate=0.01):
        y_one_hot = one_hot_encode(y, self.b2.shape[1])

        for epoch in range(epochs):
            y_pred = self.forward(X)
            loss = self.compute_loss(y_pred, y_one_hot)
            self.backward(X, y_one_hot, y_pred, learning_rate)

            if (epoch + 1) % 5 == 0 or epoch == 0:
                predictions = np.argmax(y_pred, axis=1)
                accuracy = np.mean(predictions == y)
                print(f"Epoch {epoch+1}/{epochs} | Loss: {loss:.4f} | Train Accuracy: {accuracy:.4f}")

    def predict(self, X):
        y_pred = self.forward(X)
        return np.argmax(y_pred, axis=1)


def accuracy_score(y_true, y_pred):
    return np.mean(y_true == y_pred)


def save_model(model, class_names, filename="asl_model.npz"):
    np.savez(
        filename,
        W1=model.W1,
        b1=model.b1,
        W2=model.W2,
        b2=model.b2,
        class_names=np.array(class_names)
    )
    print(f"Model saved to {filename}")


def main():
    np.random.seed(42)

    data_dir = "dataset"
    img_size = (64, 64)

    X, y, class_names = load_dataset(data_dir, img_size=img_size)

    if len(X) == 0:
        print("No images found.")
        return

    print(f"Loaded {len(X)} images.")
    print("Classes:", class_names)

    X_train, y_train, X_test, y_test = train_test_split(X, y, test_ratio=0.2)

    input_size = X_train.shape[1]
    hidden_size = 128
    output_size = len(class_names)

    model = SimpleNeuralNetwork(input_size, hidden_size, output_size)

    model.train(X_train, y_train, epochs=50, learning_rate=0.01)

    y_test_pred = model.predict(X_test)
    test_acc = accuracy_score(y_test, y_test_pred)

    print(f"Test Accuracy: {test_acc:.4f}")

    cm = confusion_matrix(y_test, y_test_pred, len(class_names))
    print_confusion_matrix(cm, class_names)
    print_per_class_accuracy(cm, class_names)

    save_model(model, class_names)

if __name__ == "__main__":
    main()