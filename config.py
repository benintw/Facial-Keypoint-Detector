

import torch
from dataclasses import dataclass

@dataclass
class CONFIG:
    DEVICE: str = "mps" if torch.backends.mps.is_built() else "cpu"
    LR: float = 1e-4
    WEIGHT_DECAY: float = 5e-4
    BATCH_SIZE: int = 64
    NUM_EPOCHS: int = 5
    CHECKPOINT_FILE: str = "b0_4.pth.tar"
    SAVE_MODEL: bool = True


config = CONFIG()
