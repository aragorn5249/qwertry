import numpy as np
import cv2
import pickle


def sigmoid(x):
    # Sigmoid activation function: f(x) = 1 / (1 + e^(-x))
    return 1 / (1 + np.exp(-x))


def deriv_sigmoid(x):
    # Derivative of the sigmoid function: f'(x) = f(x) * (1 - f(x))
    fx = sigmoid(x)
    return fx * (1 - fx)


def mse_loss(y_true, y_pred, batch_size=100):
    # Mean Squared Error loss function
    num_batches = len(y_true) // batch_size
    loss_sum = 0.0

    for i in range(num_batches):
        start = i * batch_size
        end = (i + 1) * batch_size
        y_pred_batch = y_pred[start:end].reshape(-1, 1)  # Reshape y_pred_batch
        y_true_batch = y_true[start:end]
        loss_sum += np.mean((y_true_batch - y_pred_batch) ** 2)

    remaining_samples = len(y_true) % batch_size
    if remaining_samples > 0:
        y_pred_batch = y_pred[-remaining_samples:].reshape(
            -1, 1
        )  # Reshape y_pred_batch
        y_true_batch = y_true[-remaining_samples:]
        loss_sum += np.mean((y_true_batch - y_pred_batch) ** 2)

    loss = loss_sum / num_batches
    return loss


class CatDetector:
    def __init__(self):
        # Initialize weights and biases with smaller values
        self.w1 = np.random.normal() * 0.01
        self.w2 = np.random.normal() * 0.01
        self.w3 = np.random.normal() * 0.01
        self.w4 = np.random.normal() * 0.01
        self.w5 = np.random.normal() * 0.01
        self.w6 = np.random.normal() * 0.01

        self.b1 = np.random.normal() * 0.01
        self.b2 = np.random.normal() * 0.01
        self.b3 = np.random.normal() * 0.01

    def feedforward(self, x, flatten_output=True):
        # Forward pass through the network
        h1 = sigmoid(self.w1 * x[0] + self.w2 * x[1] + self.b1)
        h2 = sigmoid(self.w3 * x[0] + self.w4 * x[1] + self.b2)
        o1 = sigmoid(self.w5 * h1 + self.w6 * h2 + self.b3)

        if flatten_output:
            return o1.flatten()
        else:
            return o1

    def train(self, data, all_y_trues):
        learn_rate = 0.1
        epochs = 1000

        for epoch in range(epochs):
            for x, y_true in zip(data, all_y_trues):
                # Forward pass
                sum_h1 = self.w1 * x[0] + self.w2 * x[1] + self.b1
                h1 = sigmoid(sum_h1)

                sum_h2 = self.w3 * x[0] + self.w4 * x[1] + self.b2
                h2 = sigmoid(sum_h2)

                sum_o1 = self.w5 * h1 + self.w6 * h2 + self.b3
                o1 = sigmoid(sum_o1)
                y_pred = o1

                # Backward pass
                d_L_d_ypred = -2 * (y_true - y_pred)
                d_ypred_d_w5 = h1 * deriv_sigmoid(sum_o1)
                d_ypred_d_w6 = h2 * deriv_sigmoid(sum_o1)
                d_ypred_d_b3 = deriv_sigmoid(sum_o1)

                d_ypred_d_h1 = self.w5 * deriv_sigmoid(sum_o1)
                d_ypred_d_h2 = self.w6 * deriv_sigmoid(sum_o1)

                d_h1_d_w1 = x[0] * deriv_sigmoid(sum_h1)
                d_h1_d_w2 = x[1] * deriv_sigmoid(sum_h1)
                d_h1_d_b1 = deriv_sigmoid(sum_h1)

                d_h2_d_w3 = x[0] * deriv_sigmoid(sum_h2)
                d_h2_d_w4 = x[1] * deriv_sigmoid(sum_h2)
                d_h2_d_b2 = deriv_sigmoid(sum_h2)

                # Update weights and biases
                self.w1 -= learn_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_w1
                self.w2 -= learn_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_w2
                self.b1 -= learn_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_b1

                self.w3 -= learn_rate * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_w3
                self.w4 -= learn_rate * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_w4
                self.b2 -= learn_rate * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_b2

                self.w5 -= learn_rate * d_L_d_ypred * d_ypred_d_w5
                self.w6 -= learn_rate * d_L_d_ypred * d_ypred_d_w6
                self.b3 -= learn_rate * d_L_d_ypred * d_ypred_d_b3

            if epoch % 10 == 0:
                y_preds = np.apply_along_axis(self.feedforward, 1, data)
                loss = mse_loss(all_y_trues, y_preds)
                print("Epoch %d loss: %.3f" % (epoch, loss))

    def detect_cat(self, image_path):
        image = cv2.imread(image_path)
        resized_image = cv2.resize(image, (2, 1), interpolation=cv2.INTER_AREA)
        grayscale_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
        normalized_image = grayscale_image / 255.0

        prediction = self.feedforward(normalized_image.flatten())
        if prediction >= 0.5:
            return "Cat"
        else:
            return "Not a cat"


# Load the pickled data
with open("catvsnotcat_small.pkl", "rb") as file:
    all_data = pickle.load(file)

IMG_SIZE = 64
label_dict = {"cat": 1, "not-cat": 0}
all_data_processed = []
number_of_examples = len(all_data)
let_know = int(number_of_examples / 10)

for idx, example in enumerate(all_data):
    if (idx + 1) % let_know == 0:
        print(f"Processing {idx + 1}")
    resized_down = cv2.resize(
        example["X"], (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR
    )

    all_data_processed.append(
        {"X": np.array(resized_down), "Y": label_dict[example["Y"]]}
    )


# Convert processed data to numpy arrays
data = np.array([item["X"] for item in all_data_processed])
labels = np.array([item["Y"] for item in all_data_processed])

# Check if data and labels are not empty
if data.size > 0 and labels.size > 0:
    # Reshape data to 2-dimensional if needed
    if data.ndim == 1:
        data = data.reshape(-1, 1)
    if labels.ndim == 1:
        labels = labels.reshape(-1, 1)

    # Train the cat detector
    cat_detector = CatDetector()
    cat_detector.train(data, labels)
else:
    print("No data or labels available to train the cat detector.")
# Test the cat detector on a new image
image_path = "D:\ML\test\test.jpg"
prediction = cat_detector.detect_cat(image_path)
print("Prediction:", prediction)
