import struct
import gzip
from typing import List, Optional
from ..data_basic import Dataset
import numpy as np

def parse_mnist(image_filesname, label_filename):
    """ Read an images and labels file in MNIST format.  See this page:
    http://yann.lecun.com/exdb/mnist/ for a description of the file format.

    Args:
        image_filename (str): name of gzipped images file in MNIST format
        label_filename (str): name of gzipped labels file in MNIST format

    Returns:
        Tuple (X,y):
            X (numpy.ndarray[np.float32]): 2D numpy array containing the loaded
                data.  The dimensionality of the data should be
                (num_examples x input_dim) where 'input_dim' is the full
                dimension of the data, e.g., since MNIST images are 28x28, it
                will be 784.  Values should be of type np.float32, and the data
                should be normalized to have a minimum value of 0.0 and a
                maximum value of 1.0.

            y (numpy.ndarray[dypte=np.int8]): 1D numpy array containing the
                labels of the examples.  Values should be of type np.int8 and
                for MNIST will contain the values 0-9.
    """
    # TODO
    ### BEGIN YOUR SOLUTION
    with gzip.open(image_filesname, "rb") as image_file:
        magic, num_images, rows, cols = struct.unpack(">IIII", image_file.read(16))
        if magic != 2051:
            raise ValueError(f"invalid MNIST image file magic number: {magic}")
        image_data = image_file.read(rows * cols * num_images)
        X = np.frombuffer(image_data, dtype=np.uint8).reshape(num_images, rows * cols)

    with gzip.open(label_filename, "rb") as label_file:
        magic, num_labels = struct.unpack(">II", label_file.read(8))
        if magic != 2049:
            raise ValueError(f"invalid MNIST label file magic number: {magic}")
        label_data = label_file.read(num_labels)
        y = np.frombuffer(label_data, dtype=np.uint8)

    if num_images != num_labels:
        raise ValueError(
            f"image/label count mismatch: {num_images} images, {num_labels} labels"
        )

    return X.astype(np.float32) / 255.0, y.astype(np.int8)
    ### END YOUR SOLUTION


class MNISTDataset(Dataset):
    def __init__(
        self,
        image_filename: str,
        label_filename: str,
        transforms: Optional[List] = None,
    ):
        # TODO
        ### BEGIN YOUR SOLUTION
        super().__init__(transforms)
        result = parse_mnist(image_filename, label_filename)
        self.images = result[0]
        self.labels = result[1]
        ### END YOUR SOLUTION

    def __getitem__(self, index) -> object:
        # TODO
        ### BEGIN YOUR SOLUTION
        images = self.images[index]
        labels = self.labels[index]

        if self.transforms is not None and np.isscalar(index):
            image = images.reshape(28, 28, 1)
            image = self.apply_transforms(image)
            images = image.reshape(28 * 28)
        elif self.transforms is not None:
            images = np.stack(
                [
                    self.apply_transforms(image.reshape(28, 28, 1)).reshape(28 * 28)
                    for image in images
                ]
            )

        return images, labels
        ### END YOUR SOLUTION

    def __len__(self) -> int:
        # TODO
        ### BEGIN YOUR SOLUTION
        return self.images.shape[0]
        ### END YOUR SOLUTION


class FashionMNISTDataset(MNISTDataset):
    pass
