from dataclasses import dataclass
import os
from pathlib import Path

def _default_data_root() -> str:
    """
    Auto-detect dataset location.
    Priority:
      1) CASSAVA_DATA_ROOT env var (explicit override)
      2) Kaggle cassava competition mount
      3) Local fallback (for later)
    """
    env_root = os.getenv("CASSAVA_DATA_ROOT")
    if env_root:
        return env_root

    kaggle_root = "/kaggle/input/cassava-leaf-disease-classification"
    if Path(kaggle_root).exists():
        return kaggle_root

    # local fallback (if you ever download locally)
    return "data/cassava"

@dataclass
class Config:
    data_root: str = _default_data_root()

    @property
    def train_csv(self) -> str:
        return str(Path(self.data_root) / "train.csv")

    @property
    def train_images_dir(self) -> str:
        return str(Path(self.data_root) / "train_images")

    @property
    def label_map_json(self) -> str:
        return str(Path(self.data_root) / "label_num_to_disease_map.json")

    # training defaults
    img_size: int = 224
    num_classes: int = 5
    batch_size: int = 32
    epochs: int = 10
    lr: float = 3e-4
    seed: int = 42

    # models to compare
    model_names: tuple = ("EfficientNetB0", "EfficientNetB2")

CFG = Config()