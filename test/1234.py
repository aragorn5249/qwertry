# Load the dataset
with open("catvsnotcat_small.pkl", "rb") as f:
    dataset = pickle.load(f)

# Extract the data and labels from the dataset
data = np.array(dataset[0])
labels = np.array(dataset[1])

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


# # Test the cat detector on a new image
# image_path = "path/to/your/image.jpg"
# prediction = cat_detector.detect_cat(image_path)
# print("Prediction:", prediction)
