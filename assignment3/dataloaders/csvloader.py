import torch
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset


class CsvLoader(Dataset):
    def __init__(self, data):
        self.x = torch.tensor(data.iloc[:, :-1].values, dtype=torch.float32)
        self.y = torch.tensor(data.iloc[:, -1].values, dtype=torch.float32).unsqueeze(1)
        self.classes = dict()

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

    def set_classes(self, classes):
        self.classes = classes

        return self
