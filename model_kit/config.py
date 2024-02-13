from dataclasses import dataclass


@dataclass(frozen=True)
class TrainingConfig:
    batch_size: int = 32
    epochs: int = 35
    learning_rate: float = 0.0001


@dataclass(frozen=True)
class DatasetConfig:
    data_root: str = 'datasets/scut_data'


@dataclass(frozen=True)
class ModelConfig:
    model_name: str = 'microsoft/trocr-small-printed'
