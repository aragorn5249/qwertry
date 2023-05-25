import cv2


def preprocess(dataset: list[dict]) -> list[dict]:
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

    Calculated Parameters:
    ----------------------
    data_fold_dict: dict
        dictionary containing all folds of the preprocessed dataset to be applied to the neural network
    """
    

    # Preprocessing images in dataset
    print("\nPreprocessing the dataset ...")

    # Convert images to grayscale images
    print("- converting rbg images to grayscale images ...")
    dataset_preprocessed = [
        {"X": cv2.cvtColor(image["X"], cv2.COLOR_BGR2GRAY), "Y": image["Y"]}
        for image in dataset
    ]

    # Resize images to 128x128x3 pixels
    print("- resizing images to 128x128x3 pixels ...")
    dataset_preprocessed = [
        {
            "X": cv2.resize(image["X"], (128, 128), interpolation=cv2.INTER_LANCZOS4),
            "Y": image["Y"],
        }
        for image in dataset_preprocessed
    ]

    # Normalize images
    print("- normalizing pixel values ...")
    dataset_preprocessed = [
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
        for image in dataset_preprocessed
    ]

    """
    print("- center images ...")
    # Bad solution but otherwise pc freezes
    for i in range(int(len(dataset)/1000)):
        dataset_preprocessed[i*1000:(i+1)*1000] = [{"X":image["X"] - np.mean(image["X"]), "Y":image["Y"]} for image in dataset[i*1000:(i+1)*1000]]
    print(dataset_preprocessed[0]["X"])
    print(len(dataset_preprocessed))
    """
    """
    print("- standarize images ...")
    # Bad solution but otherwise pc freezes
    for i in range(int(len(dataset)/1000)):
        dataset_preprocessed[i*1000:(i+1)*1000] = [{"X":(image["X"] - np.mean(image["X"])) / np.std(image["X"]), "Y":image["Y"]} for image in dataset[i*1000:(i+1)*1000]]
    print(dataset_preprocessed[0]["X"])
    print(len(dataset_preprocessed))
    """

    # Perform image augmentation

    # Flip images along x-axis, y-axis and both axis
    print("- augmenting images ...")
    dataset_extension: list[dict] = []
    for flip_direction in [-1, 0, 1]:
        dataset_extension.extend(
            [
                {"X": cv2.flip(image["X"], flip_direction), "Y": image["Y"]}
                for image in dataset_preprocessed
            ]
        )

    dataset_preprocessed = dataset_preprocessed + dataset_extension
    print(f"   -> new size of dataset: {len(dataset_preprocessed)}")

    # Split the dataset for stratified k-fold cross-validation (with k=5)
    k = 5
    print(f"\nSplitting dataset into {k} folds ...")
    cat_images: list[dict] = []
    no_cat_images: list[dict] = []
    [
        cat_images.append(labeled_image)
        if labeled_image["Y"] == "cat"
        else no_cat_images.append(labeled_image)
        for labeled_image in dataset_preprocessed
    ]

    dataset_fold_dict: dict[str, list[dict]] = {
        f"fold_{i+1}": cat_images[
            int(i * len(cat_images) / k) : int((i + 1) * len(cat_images) / k)
        ]
        + no_cat_images[
            int((i * len(no_cat_images) / k)) : int((i + 1) * len(no_cat_images) / k)
        ]
        for i in range(k)
    }
    print(list(dataset_fold_dict.keys()))
    print(
        f"with every fold containing: \n- {int(len(cat_images)/k)} cat_images\n- {int(len(no_cat_images)/k)} no_cat_images"
    )

    return dataset_fold_dict


"""
# Using cv2.imshow() method
# Displaying the image
cv2.imshow("a", dataset_extension[0]["X"])
cv2.imshow("b", dataset_extension[1]["X"])
cv2.imshow("c", dataset_extension[2]["X"])
cv2.imshow("d", dataset_extension[-3]["X"])
cv2.imshow("e", dataset_extension[-2]["X"])
cv2.imshow("f", dataset_extension[-1]["X"])
# waits for user to press any key
# (this is necessary to avoid Python kernel form crashing)
cv2.waitKey(0)
# closing all open windows
cv2.destroyAllWindows()
"""
