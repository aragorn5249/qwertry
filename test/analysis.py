import itertools
import numpy as np

from typing import Iterator


def analyze(dataset) -> list[dict]:

    # Examine the dataset
    nr_of_images: int = len(dataset)

    image_distribution: dict[str, Iterator[int]] = {
        "nr_cat_images": itertools.count(0),
        "nr_no_cat_images": itertools.count(0),
    }
    [
        next(image_distribution["nr_cat_images"])
        if labeled_image["Y"] == "cat"
        else next(image_distribution["nr_no_cat_images"])
        for labeled_image in dataset
    ]

    image_shapes = [np.shape(np.array(labeled_image["X"])) for labeled_image in dataset]
    image_shape_ranges: dict = {
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
    print(f"image distribution: {image_distribution}")
    print(f"image shape ranges: {image_shape_ranges}")