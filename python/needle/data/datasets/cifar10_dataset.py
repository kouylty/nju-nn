import os
import pickle
from typing import Optional, List
import numpy as np
from ..data_basic import Dataset

class CIFAR10Dataset(Dataset):
    def __init__(
        self,
        base_folder: str,
        train: bool,
        p: Optional[int] = 0.5,
        transforms: Optional[List] = None
    ):
        """
        Parameters:
        base_folder - cifar-10-batches-py folder filepath
        train - bool, if True load training dataset, else load test dataset
        Divide pixel values by 255. so that images are in 0-1 range.
        Attributes:
        X - numpy array of images
        y - numpy array of labels
        """
        # TODO
        ### BEGIN YOUR SOLUTION
        super().__init__(transforms)
        self.base_folder = base_folder
        self.train = train
        self.p = p

        filenames = (
            [f"data_batch_{i}" for i in range(1, 6)]
            if train
            else ["test_batch"]
        )

        images = []
        labels = []
        for filename in filenames:
            path = os.path.join(base_folder, filename)
            with open(path, "rb") as f:
                batch = pickle.load(f, encoding="latin1")

            data = batch["data"] if "data" in batch else batch[b"data"]
            batch_labels = (
                batch["labels"]
                if "labels" in batch
                else batch.get(b"labels", batch.get("fine_labels", batch.get(b"fine_labels")))
            )
            images.append(data)
            labels.extend(batch_labels)

        self.X = np.concatenate(images, axis=0).reshape((-1, 3, 32, 32))
        self.X = self.X.astype(np.float32) / 255.0
        self.y = np.array(labels, dtype=np.int8)
        ### END YOUR SOLUTION

    def __getitem__(self, index) -> object:
        """
        Returns the image, label at given index
        Image should be of shape (3, 32, 32)
        """
        # TODO
        ### BEGIN YOUR SOLUTION
        image = self.X[index]
        label = self.y[index]

        if self.transforms is not None and np.isscalar(index):
            image = image.transpose((1, 2, 0))
            image = self.apply_transforms(image)
            image = image.transpose((2, 0, 1))
        elif self.transforms is not None:
            image = np.stack(
                [
                    self.apply_transforms(img.transpose((1, 2, 0))).transpose((2, 0, 1))
                    for img in image
                ]
            )

        return image, label
        ### END YOUR SOLUTION

    def __len__(self) -> int:
        """
        Returns the total number of examples in the dataset
        """
        # TODO
        ### BEGIN YOUR SOLUTION
        return self.X.shape[0]
        ### END YOUR SOLUTION
