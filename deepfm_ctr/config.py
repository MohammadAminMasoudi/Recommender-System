from dataclasses import dataclass

@dataclass
class DeepFMConfig:
    topk: int = 10
    neg_per_pos_train: int = 5
    neg_per_pos_test: int = 100
    embed_dim: int = 16
    epochs: int = 3
    batch_size: int = 4096
    lr: float = 1e-3
    seed: int = 42
