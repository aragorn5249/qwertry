import cv2
import numpy as np


def greyscale_converter(dataset: dict) -> dict:
    """
    Converts color images to grayscale images
    """
    print("- converting rbg images to grayscale images ...")
    dataset_processed = [
        {"X": cv2.cvtColor(image["X"], cv2.COLOR_BGR2GRAY), "Y": image["Y"]}
        for image in dataset
    ]
    return dataset_processed


def image_resizer(dataset: dict, DESIRED_IMAGE_SIZE: tuple) -> dict:
    """
    Resizes all images to user defined size
    """
    print(
        f"- resizing images to {DESIRED_IMAGE_SIZE[0]} x {DESIRED_IMAGE_SIZE[1]} x 3 pixels ..."
    )
    dataset_processed = [
        {
            "X": cv2.resize(
                image["X"], DESIRED_IMAGE_SIZE, interpolation=cv2.INTER_LANCZOS4
            ),
            "Y": image["Y"],
        }
        for image in dataset
    ]
    return dataset_processed


def image_normalizer(dataset: dict) -> dict:
    """
    Normalizes the pixel values of all images to be in the range [0,1]
    """
    print("- normalizing pixel values ...")
    dataset_processed = [
        {
            "X": cv2.normalize(
                image["X"],
                None,
                alpha=0,
                beta=1,
                norm_type=cv2.NORM_MINMAX,
                dtype=cv2.CV_32F,
            ),
            "Y": image["Y"],
        }
        for image in dataset
    ]
    return dataset_processed


def image_augmenter(dataset: dict) -> dict:
    """
    Enlarges dataset by mirroring all images in all directions
    WARNING: The dataset might be too large then to for weaker maschines!
    """
    # Flip images along x-axis, y-axis and both axis
    print("- augmenting images ...")
    dataset_extension: list[dict] = []
    for flip_direction in [-1, 0, 1]:
        dataset_extension.extend(
            [
                {"X": cv2.flip(image["X"], flip_direction), "Y": image["Y"]}
                for image in dataset
            ]
        )
    dataset_processed = dataset + dataset_extension
    print(f"   -> new size of dataset: {len(dataset_processed)}")
    return dataset_processed


def image_flattener(dataset: dict) -> dict:
    """
    Flattens all images to 1D arrays
    """
    print("- flattening images to 1-D array...")
    dataset_processed = [
        {"X": np.asarray(image["X"].flatten()), "Y": image["Y"]} for image in dataset
    ]
    return dataset_processed


def convert_labels_to_binary(dataset: dict) -> dict:
    """
    Converts the label of the images in the following way:
        -> "cat": 0
        -> "not-cat": 1
    """
    print(f"- renaming labels of the dataset to binary values ...")
    print(f"   -> cat: 0, not-cat: 1")
    dataset_processed = [
        {"X": labeled_image["X"], "Y": 0}
        if labeled_image["Y"] == "cat"
        else {"X": labeled_image["X"], "Y": 1}
        for labeled_image in dataset
    ]
    return dataset_processed


def split_to_training_and_test_set(dataset: dict) -> tuple[dict, dict]:
    """
    Splits dataset into training set (= 90% of the images) and test set (= 10% of the images)
    in such a way that both contain same ratio of cat and no-cat images

    Parameters:
    -----------
    dataset : dict
        dataset containing labeled images

    Returns:
    --------
    training_set : np.ndarray
        array containing training set
    test_set : np.ndarray
        array containing test set
    """
    print(f"- splitting dataset into training set and test set ...")
    split_percentage_training_dataset = 0.9
    cat_images: list[dict] = []
    no_cat_images: list[dict] = []
    [
        cat_images.append(labeled_image)
        if labeled_image["Y"] == "cat"
        else no_cat_images.append(labeled_image)
        for labeled_image in dataset
    ]
    training_dataset = (
        cat_images[: int(split_percentage_training_dataset * len(cat_images))]
        + no_cat_images[: int(split_percentage_training_dataset * len(no_cat_images))]
    )
    test_dataset = (
        cat_images[int(split_percentage_training_dataset * len(cat_images)) :]
        + no_cat_images[int(split_percentage_training_dataset * len(no_cat_images)) :]
    )
    np.random.shuffle(training_dataset)
    np.random.shuffle(test_dataset)
    print(f"   -> size of training set: {len(training_dataset)}")
    print(f"   -> size of test set: {len(test_dataset)}")

    return training_dataset, test_dataset


def dataset_label_splitter(dataset: dict) -> tuple[dict, dict]:
    """
    This function separates the images from the corresponding labels

    Parameters:
    -----------
    dataset : dict
        dataset which images and labels shall be separated

    Returns:
    --------
    images : np.ndarray
        array containing all images
    labels : np.ndarray
        array containing all labels
    """
    images = np.asarray([labeled_images["X"] for labeled_images in dataset])
    labels = np.asarray([[labeled_images["Y"] for labeled_images in dataset]])
    return images, labels


def preprocess(
    dataset: list[dict], DESIRED_IMAGE_SIZE: tuple[int], TRAIN_NETWORK: bool
) -> list[dict]:
    """
    This function preprocesses the dataset by performing the following steps:
    1) Convert images to grayscale images
    2) Resize images
    3) Normalize images -> all pixel values between 0 and 1
    4) Perform image augmentation to enlarge datasset -> mirroring the images along the axes
    5) Apply stratified k-fold cross-validation

    Parameters:
    -----------
    dataset: list(dict)
        dataset containing the labeled images

    Returns:
    ----------------------
    training_dataset_images: list(dict)
        list containing the images to train the network
    training_dataset_labels: list(int)
        list containing the labels corresponding to the training images
    test_dataset: list(dict)
        list containing the images to test the network
    test_dataset_labels: list(int)
        list containing the labels corresponding to the test images
    """

    # Preprocessing images in dataset
    print("\nPreprocessing dataset")

    # Convert images to grayscale images
    dataset_preprocessed = greyscale_converter(dataset)

    # Resize images to desired size
    dataset_preprocessed = image_resizer(dataset_preprocessed, DESIRED_IMAGE_SIZE)

    # Normalize images
    dataset_preprocessed = image_normalizer(dataset_preprocessed)

    """
    # Perform image augmentation
    # WARNING: The dataset might be too large then to for weaker maschines!
    dataset_preprocessed = image_augmenter(dataset_preprocessed)
    """

    # Flatten all images
    dataset_preprocessed = image_flattener(dataset_preprocessed)

    # Rename labels: "cat" -> 0, "not-cat" -> 1
    dataset_preprocessed = convert_labels_to_binary(dataset_preprocessed)

    # Split dataset into training set (= 90% of the images) and test set (= 10% of the images)
    if TRAIN_NETWORK:
        training_dataset, test_dataset = split_to_training_and_test_set(
            dataset_preprocessed
        )

    # Split images and labels in training_dataset and test_dataset
    if TRAIN_NETWORK:
        print("\nSplitting training and test dataset in image and label vector ...")
        training_dataset_images, training_dataset_labels = dataset_label_splitter(
            training_dataset
        )
        print(f"training set - image vector: {training_dataset_images.shape}")
        print(f"training set - label vector: {training_dataset_labels.shape}")
        test_dataset_images, test_dataset_labels = dataset_label_splitter(test_dataset)
        print(f"test set - image vector: {test_dataset_images.shape}")
        print(f"test set - label vector: {test_dataset_labels.shape}")
        return (
            training_dataset_images,
            training_dataset_labels,
            test_dataset_images,
            test_dataset_labels,
        )
    else:
        print("\nSplitting dataset in image and label vector ...")
        dataset_images, dataset_labels = dataset_label_splitter(dataset_preprocessed)
        print(f"data set - image vector: {dataset_images.shape}")
        print(f"data set - label vector: {dataset_labels.shape}")
        return dataset_images, dataset_labels
