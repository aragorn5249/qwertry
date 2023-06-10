import itertools
import numpy as np

from typing import Iterator


def analyze(dataset: list[dict]) -> None:
    """
    This function analyzes the dataset regarding
    - number of images contained
    - label bias -> nr of cat images and nr of no-cat images
    - image shapes

    Input Parameter:
    ----------------
    dataset : list[dict]
        dataset containing the labeled images

    Calculated Parameters:
    ----------------------
    nr_of_images : int
        number of images in dataset
    label_bias_dict:  dict[str, Iterator[int]]
        dictionary containing the counts of cat and no cat images
    image_shape_ranges_dict : dict[str, tuple[int, int]]
        dictionary containing the min-max values for every dimension (heigt, width, channels)
    """
    # Examine the dataset
    nr_of_images: int = len(dataset)

    label_bias_dict: dict[str, Iterator[int]] = {
        "nr_cat_images": itertools.count(0),
        "nr_no_cat_images": itertools.count(0),
    }
    [
        next(label_bias_dict["nr_cat_images"])
        if labeled_image["Y"] == "cat"
        else next(label_bias_dict["nr_no_cat_images"])
        for labeled_image in dataset
    ]

    image_shapes = [np.shape(np.array(labeled_image["X"])) for labeled_image in dataset]
    image_shape_ranges_dict: dict[str, tuple[int, int]] = {
        "width_ranges": (
            min(image_shapes, key=lambda x: x[0])[0],
            max(image_shapes, key=lambda x: x[0])[0],
        ),
        "height_ranges": (
            min(image_shapes, key=lambda x: x[1])[1],
            max(image_shapes, key=lambda x: x[1])[1],
        ),
        "channel_ranges": (
            min(image_shapes, key=lambda x: x[2])[2],
            max(image_shapes, key=lambda x: x[2])[2],
        ),
    }
    print("\nDataset metrics")
    print(f"number of images: {nr_of_images}")
    print(f"image distribution: {label_bias_dict}")
    print(f"image shape ranges: {image_shape_ranges_dict}")
