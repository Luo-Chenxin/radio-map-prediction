import yaml
from pydantic import BaseModel, ValidationError, Field, model_validator
from typing import Tuple, Literal, Union, Annotated
from typing_extensions import Self
from src.datamodule import RadioSeerDataModule, H5DataModule
from src.dataset import RadioSeerDataset, H5Dataset

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

class _RadioMapSeerDataConfig(BaseModel):
    type: Literal["radiomapseer"] = "radiomapseer"
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

    def get_desc(self) -> str:
        """
        Get description of dataset
        """
        simulationStr = self.simulation if self.sparse_IRT4_number == 0 else f"IRT4_Adapter_{self.sparse_IRT4_number}" 
        carsStr = "Cars_Exist" if self.cars_exist else "No_Cars_Exist"
        if self.city_map == 'complete':
            cityMapStr = 'Complete_City_Map'
        elif self.city_map == 'missing':
            cityMapStr = f'City_Map_With_{self.missing}_Missing_Buildings'
        elif self.city_map == 'rand':
            cityMapStr = 'City_Map_With_Random_Missing_Buildings'
        else: 
            cityMapStr = f'Unknown_{self.city_map}'
        samplesStr = f"Input_Samples_{self.samples_number}"

        return f"{simulationStr}|{carsStr}|{cityMapStr}|{samplesStr}"
    
    def build_datamodule(self, config_load, seed):
        return RadioSeerDataModule(RadioSeerDataset, config_load, self, seed)
    
    def get_in_channels(self) -> int:
        in_channels = 1 + 1  # buildings + transmitters
        if self.samples_number > 0:
            in_channels += 1
        if self.cars_exist:
            in_channels += 1
        return in_channels

class _H5DataConfig(BaseModel):
    type: Literal["h5"] = "h5"
    h5_path: str
    threshold: float = Field(ge=0.0, lt=1.0, description="Range is [0, 1]")
    
    def get_desc(self) -> str:
        return self.h5_path
    
    def build_datamodule(self, config_load, seed):
        return H5DataModule(H5Dataset, config_load, self, seed)
    
    def get_in_channels(self) -> int:
        return 2

_DataConfig = Annotated[
    Union[_RadioMapSeerDataConfig, _H5DataConfig],
    Field(discriminator="type"),
]

class _Config(BaseModel):
    seed: int = Field(ge=0, description="Seed needs to be greater than 0")
    load: _LoadConfig
    train: _TrainConfig 
    data: _DataConfig 

    def build_datamodule(self):
        return self.data.build_datamodule(self.load, self.seed)

    def dataset_desc(self) -> str:
        return self.data.get_desc()

def load_config_strict(config_path):
    """
    Load, check and return the contents of the YAML configuration file.
    """
    with config_path.open(mode='r', encoding='utf-8') as f:
        raw_config = yaml.safe_load(f)
    
    config = _Config(**raw_config) 
    return config