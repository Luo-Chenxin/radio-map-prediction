import yaml
from pydantic import BaseModel, ValidationError, Field, model_validator
from typing import Tuple
from typing_extensions import Self

class _LoadConfig(BaseModel):
    train_ratio: float = Field(gt=0.0, lt=1.0, description="Range is (0.0, 1.0)")
    val_ratio: float = Field(gt=0.0, lt=1.0, description="Range is (0.0, 1.0)")
    @model_validator(mode='after')
    def _check_split_ratio(self) -> Self:
        test_ration = 1.0 - (self.train_ratio + self.val_ratio)
        if not (test_ration > 0.0):
            raise ValueError(f"The test ration needs to be greater than 0.0, check train_ratio and val_ratio")
        return self
    train_batch_size: int = Field(gt=0, le=128, description="Range is (0, 128]")
    val_batch_size: int = Field(gt=0, le=128, description="Range is (0, 128]")
    test_batch_size: int = Field(gt=0, le=128, description="Range is (0, 128]")
    num_workers: int = Field(ge=0, le=64, description="Range is [0, 64]")

class _SchedulerConfig(BaseModel):
    step_size: int = Field(gt=0, description="Step size needs to be greater than 0")
    gamma: float = Field(gt=0.0, lt=1.0, description="Range is (0.0, 1.0)")

class _EarlyStopConfig(BaseModel):
    patience: int = Field(gt=0, description="patience needs to be greater than 0")
    delta: float = Field(ge=0.0, lt=0.1, description="Range is [0.0, 0.1)")

class _TrainConfig(BaseModel):
    epoch: int = Field(gt=0, description="Epoch needs to be greater than 0")
    learning_rate: float = Field(gt=0.0, description="Learning rate needs to be greater than 0.0")
    scheduler: _SchedulerConfig
    early_stop: _EarlyStopConfig
    out_dir: str

_ImgSize = Tuple[int, int]

class _DataConfig(BaseModel):
    root_dir: str
    DPM_dir: str
    DPM_cars_dir: str
    IRT2_dir: str
    IRT2_cars_dir: str
    IRT4_dir: str
    IRT4_cars_dir: str
    buildings_complete_dir: str
    buildings_missing_dir: str
    antennas_dir: str
    cars_dir: str

    simulation: str
    IRT2_weight: float = Field(gt=0.0, lt=1.0, description="Range is (0.0, 1.0)")
    city_map: str
    missing: int = Field(ge=1, le=4, description="Range is [1, 4]")
    sparse_IRT4_number: int = Field(ge=0, description="The number of IRT4 points needs to be greater than or equal to 0")
    @model_validator(mode='after')
    def _check_sparse_IRT4_number(self) -> Self:
        total_img_size = self.img_size[0] * self.img_size[1]
        if not (self.sparse_IRT4_number <= total_img_size):
            raise ValueError(f"The number of sparse IRT4 points needs to be less than or equal to total image size {total_img_size}")
        return self
    samples_number: int = Field(ge=0, description="Inputing samples number needs to be greater than or equal to 0")
    @model_validator(mode='after')
    def _check_samples_number(self) -> Self:
        total_img_size = self.img_size[0] * self.img_size[1]
        if self.sparse_IRT4_number > 0:
            if not (self.samples_number <= self.sparse_IRT4_number):
                raise ValueError(f"Inputing samples number needs to be less than or equal to sparse_IRT4_number {self.sparse_IRT4_number}")
        else:
            if not (self.samples_number <= total_img_size):
                raise ValueError(f"Inputing samples number needs to be less than or equal to total image size {total_img_size}")
        return self
    cars_exist: bool
    maps_number: int = Field(ge=1, le=700, description="Range is [1, 700]")
    transmitters_number: int
    @model_validator(mode='after')
    def _check_transmitters_number(self) -> Self:
        if self.sparse_IRT4_number > 0:
            total_data_size = self.transmitters_number * self.maps_number
            if not (1 <= self.transmitters_number <= 2):
                raise ValueError(f"transmitters_number range is [1, 2]")
        else:
            if not (1 <= self.transmitters_number <= 80):
                raise ValueError(f"transmitters_number range is [1, 80]")
        return self

    threshold: float = Field(ge=0.0, lt=1.0, description="Range is [0, 1]")
    img_size: _ImgSize

class _Config(BaseModel):
    seed: int = Field(ge=0, description="Seed needs to be greater than 0")
    load: _LoadConfig
    train: _TrainConfig 
    data: _DataConfig

class _ParisDataConfig(BaseModel):
    h5_path: str

class _ParisConfig(BaseModel):
    seed: int = Field(ge=0, description="Seed needs to be greater than 0")
    load: _LoadConfig
    train: _TrainConfig
    data: _ParisDataConfig

def load_config_strict(config_path):
    """
    Load, check and return the contents of the YAML configuration file.
    """
    with config_path.open(mode='r', encoding='utf-8') as f:
        raw_config = yaml.safe_load(f)
    
    config = _Config(**raw_config) 
    return config

def load_paris_config_strict(config_path):
    with config_path.open(mode='r', encoding='utf-8') as f:
        raw_config = yaml.safe_load(f)

    config = _ParisConfig(**raw_config)
    return config
