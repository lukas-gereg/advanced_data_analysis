from collections.abc import Sequence

from torch.utils.data import Subset, Dataset


class CustomSubset(Subset):
    def __init__(self, dataset: Dataset, indices: Sequence[int], classes: dict):
        super().__init__(dataset, indices)
        self.classes = classes
