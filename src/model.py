# src/model.py
from physicsnemo.models.mlp.fully_connected import FullyConnected


def build_model(num_layers: int, layer_size: int, device):
    # Input: (r,z) -> Output: T
    return FullyConnected(
        in_features=2,
        out_features=1,
        num_layers=num_layers,
        layer_size=layer_size,
    ).to(device)