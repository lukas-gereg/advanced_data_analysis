from torch import nn

from .base_model import BaseModel


def assemble_layer(input_size, output_size, last_layer: bool) -> nn.Module:
    return nn.Sequential(
        nn.Linear(in_features=input_size, out_features=output_size),
        nn.LeakyReLU() if not last_layer else nn.Sigmoid(),
    )


class SonarModel(BaseModel):
    def __init__(self, default_params: dict):
        super().__init__(default_params)

        assert isinstance(default_params.get("input_size"), int)
        assert isinstance(default_params.get("size"), list)
        assert len(default_params.get("size")) > 0

        size = default_params.get("size")
        input_size = default_params.get("input_size")

        self.classifier = nn.Sequential(
            assemble_layer(input_size, size[0], len(size) == 1),
            *[assemble_layer(size[idx], size[idx + 1], idx == len(size) - 2) for idx in range(len(size) - 1)]
        )

    def forward(self, x):
        return self.classifier(x)