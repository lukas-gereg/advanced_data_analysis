import torch
import torch.nn as nn

class BaseModel(nn.Module):
    def __init__(self, default_params: dict) -> None:
        super().__init__()
        self.defaults = default_params

    def load_weights(self, path: str) -> None:
        self.load_state_dict(torch.load(path))