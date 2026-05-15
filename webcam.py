import cv2
import numpy as np
import time

class SimpleNeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.W1 = np.zeros((input_size, hidden_size))
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.zeros((hidden_size, output_size))
        self.b2 = np.zeros((1, output_size))

    def relu(self, Z):
        return np.maximum(0, Z)

    def softmax(self, Z):
        Z = Z - np.max(Z, axis=1, keepdims=True)
        exp_scores = np.exp(Z)
        return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

    def forward(self, X):
        Z1 = np.dot(X, self.W1) + self.b1
        A1 = self.relu(Z1)
        Z2 = np.dot(A1, self.W2) + self.b2
        A2 = self.softmax(Z2)
        return A2

    def predict(self, X):
        probs = self.forward(X)
        pred = np.argmax(probs, axis=1)
        conf = np.max(probs, axis=1)
        return pred, conf


def load_model(filename="asl_model.npz"):
    data = np.load(filename)
    W1 = data["W1"]
    b1 = data["b1"]
    W2 = data["W2"]
    b2 = data["b2"]
    class_names = data["class_names"].tolist()

    input_size = W1.shape[0]
    hidden_size = W1.shape[1]
    output_size = W2.shape[1]

    model = SimpleNeuralNetwork(input_size, hidden_size, output_size)
    model.W1 = W1
    model.b1 = b1
    model.W2 = W2
    model.b2 = b2

    return model, class_names


def preprocess_roi(roi, img_size=(64, 64)):
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, img_size)
    gray = gray.astype(np.float32) / 255.0
    gray = gray.flatten()
    gray = np.expand_dims(gray, axis=0)
    return gray


def main():
    model, class_names = load_model("asl_model.npz")

    camera_index = 0
    cap = cv2.VideoCapture(camera_index, cv2.CAP_AVFOUNDATION)

    if not cap.isOpened():
        print("Could not open webcam.")
        return

    time.sleep(1.0)
    for _ in range(10):
        cap.read()

    box_size = 300

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            print("Could not read frame from webcam.")
            break

        display = frame.copy()
        h, w, _ = frame.shape

        x1 = w // 2 - box_size // 2
        y1 = h // 2 - box_size // 2
        x2 = x1 + box_size
        y2 = y1 + box_size

        roi = frame[y1:y2, x1:x2]
        cv2.imshow("ROI", roi)
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray_small = cv2.resize(gray, (64, 64))
        cv2.imshow("Processed ROI", gray_small)
        X_input = preprocess_roi(roi)

        pred_index, confidence = model.predict(X_input)
        pred_label = class_names[pred_index[0]]
        conf_percent = confidence[0] * 100

        cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(display, f"Prediction: {pred_label}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(display, f"Confidence: {conf_percent:.2f}%", (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        cv2.putText(display, "Place hand inside box", (20, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(display, "Press q to quit", (20, 155),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.imshow("Live ASL Prediction", display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()