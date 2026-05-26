import os
from typing import List, Optional

import numpy as np

from ..data_basic import Dataset


class IrisDataset(Dataset):
    label_map = {
        "Iris-setosa": 0,
        "Iris-versicolor": 1,
        "Iris-virginica": 2,
    }

    def __init__(self, filename: str, transforms: Optional[List] = None):
        super().__init__(transforms)

        if os.path.splitext(filename)[1] == ".npz":
            data = np.load(filename)
            self.features = data["X"].astype(np.float32)
            self.labels = data["y"].astype(np.int8)
        else:
            features = []
            labels = []
            with open(filename, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    fields = line.split(",")
                    if len(fields) != 5:
                        continue
                    features.append([float(x) for x in fields[:4]])
                    labels.append(self.label_map[fields[4]])
            self.features = np.array(features, dtype=np.float32)
            self.labels = np.array(labels, dtype=np.int8)

    def __getitem__(self, index) -> object:
        features = self.features[index]
        labels = self.labels[index]

        if self.transforms is not None and np.isscalar(index):
            features = self.apply_transforms(features)

        return features, labels

    def __len__(self) -> int:
        return self.features.shape[0]
