import numpy as np
import pickle
import cv2

# Define the necessary functions


def preprocess_images(images):
    normalized_images = images / 255.0
    return normalized_images


def one_hot_encode(labels, num_classes):
    num_labels = len(labels)
    unique_labels = list(set(labels))
    label_to_int = {label: i for i, label in enumerate(unique_labels)}
    encoded_labels = np.zeros((num_labels, num_classes))
    for i, label in enumerate(labels):
        encoded_labels[i, label_to_int[label]] = 1
    return encoded_labels


def split_dataset(dataset, test_ratio):
    np.random.shuffle(dataset)
    split_index = int(len(dataset) * (1 - test_ratio))
    train_data = dataset[:split_index]
    test_data = dataset[split_index:]
    return train_data, test_data


# Load the dataset

try:
    with open("catvsnotcat_small.pkl", "rb") as f:
        dataset = pickle.load(f)
except FileNotFoundError:
    print("Dataset file not found.")
    exit()
except Exception as e:
    print("Error loading dataset:", str(e))
    exit()

# Resize the images

resized_images = []
for data in dataset:
    resized_image = cv2.resize(data["X"], (32, 32))
    resized_images.append(resized_image)

# Extract images and labels

images = np.array(resized_images)
labels = np.array([data["Y"] for data in dataset])

# Preprocess images and encode labels

preprocessed_images = preprocess_images(images)
encoded_labels = one_hot_encode(labels, num_classes=2)

# Split the dataset

train_data, test_data = split_dataset(dataset, test_ratio=0.2)

# Resize the images for training data

resized_train_images = []
for data in train_data:
    resized_image = cv2.resize(data["X"], (32, 32))
    resized_train_images.append(resized_image)

# Convert resized train images and labels to NumPy arrays

train_images = np.array(resized_train_images)
train_labels = np.array([data["Y"] for data in train_data])

# Reshape the images to have the correct shape

train_images = train_images.reshape(len(train_images), -1)

# Adjust hyperparameters

input_size = train_images.shape[1]
hidden_size = 1000  # Increase the number of hidden units
output_size = 2
learning_rate = 0.001  # Adjust the learning rate
num_epochs = 1000  # Increase the number of epochs
batch_size = 32  # Specify the batch size
regularization_strength = 0.01  # Adjust the regularization strength

# Create MLP instance


class MLP:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # Initialize weights and biases
        self.weights1 = np.random.randn(input_size, hidden_size) * np.sqrt(
            2 / input_size
        )
        self.bias1 = np.zeros(hidden_size)
        self.weights2 = np.random.randn(hidden_size, output_size) * np.sqrt(
            2 / hidden_size
        )
        self.bias2 = np.zeros(output_size)

    def relu(self, x):
        return np.maximum(0, x)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def relu_derivative(self, x):
        return np.where(x > 0, 1, 0)

    def forward(self, X):
        # Perform forward propagation
        self.z1 = np.dot(X, self.weights1) + self.bias1
        self.a1 = self.relu(self.z1)
        self.z2 = np.dot(self.a1, self.weights2) + self.bias2
        self.a2 = self.sigmoid(self.z2)
        return self.a2

    def backward(self, X, y, learning_rate, regularization_strength):
        # Perform backward propagation
        m = X.shape[0]  # Number of training examples

        # Compute gradients
        self.dz2 = (self.a2 - y) / m
        self.dw2 = np.dot(self.a1.T, self.dz2) + regularization_strength * self.weights2
        self.db2 = np.mean(self.dz2, axis=0)

        self.dz1 = np.dot(self.dz2, self.weights2.T) * self.relu_derivative(self.a1)
        self.dw1 = np.dot(X.T, self.dz1) + regularization_strength * self.weights1
        self.db1 = np.mean(self.dz1, axis=0)

        # Update parameters
        self.weights2 -= learning_rate * self.dw2
        self.bias2 -= learning_rate * self.db2
        self.weights1 -= learning_rate * self.dw1
        self.bias1 -= learning_rate * self.db1


# Create an instance of MLP

mlp = MLP(input_size, hidden_size, output_size)

# Training loop

num_batches = len(train_images) // batch_size

for epoch in range(num_epochs):
    # Shuffle the training data
    indices = np.random.permutation(len(train_images))
    train_images = train_images[indices]
    train_labels = train_labels[indices]

    # Mini-batch training
for batch in range(num_batches):
    start = batch * batch_size
    end = start + batch_size
    batch_images = train_images[start:end]
    batch_labels = encoded_labels[start:end]

    # Forward and backward pass
    batch_predictions = mlp.forward(batch_images)
    mlp.backward(batch_images, batch_labels, learning_rate, regularization_strength)

    # Evaluate on validation set
    resized_val_images = []
    for data in test_data:
        resized_image = cv2.resize(data["X"], (32, 32))
        resized_val_images.append(resized_image)

    val_images = np.array(resized_val_images)
    val_labels = np.array([data["Y"] for data in test_data])
    val_images = val_images.reshape(len(val_images), -1)

    val_predictions = mlp.forward(val_images)
    val_loss = np.mean(
        np.square(val_predictions - one_hot_encode(val_labels, num_classes=2))
    )
    print(f"Epoch {epoch+1}/{num_epochs}, Validation Loss: {val_loss}")

# Evaluate on test set
resized_test_images = []
for data in test_data:
    resized_image = cv2.resize(data["X"], (32, 32))
    resized_test_images.append(resized_image)

test_images = np.array(resized_test_images)
test_labels = np.array([data["Y"] for data in test_data])
test_images = test_images.reshape(len(test_images), -1)

test_predictions = mlp.forward(test_images)
test_loss = np.mean(
    np.square(test_predictions - one_hot_encode(test_labels, num_classes=2))
)
print(f"Test Loss: {test_loss}")
random_image = Image.open(r"D:\ML\Project\test.jpg")
resized_image = random_image.resize((32, 32))
preprocessed_image = np.array(resized_image) / 255.0
flattened_image = preprocessed_image.reshape(1, -1)
prediction = mlp.forward(flattened_image)

# Interpret the prediction
probability_cat = prediction[0, 0]  # Probability of the image being a cat
probability_not_cat = prediction[0, 1]  # Probability of the image not being a cat

# Print the probabilities
print(f"Probability of being a cat: {probability_cat:.4f}")
print(f"Probability of not being a cat: {probability_not_cat:.4f}")

