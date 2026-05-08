import yaml
from pydantic import BaseModel, ValidationError, Field, model_validator
from typing import Annotated, Tuple
from typing_extensions import Self
import sys

_CityMapIndexes = Tuple[
    Annotated[int, Field(ge=0, lt=700)],
    Annotated[int, Field(gt=0, le=700)]
]

_ImgSize = Tuple[int, int]

# Define the configuration template (specifying what fields must be included and their types).
class _DataConfig(BaseModel):
    root_dir: str
    DPM_dir: str
    DPM_cars_dir: str
    IRT2_dir: str
    IRT2_cars_dir: str
    buildings_complete_dir: str
    buildings_missing_dir: str

    simulation: str
    IRT2_weight: float = Field(gt=0.0, lt=1.0, description="Range is (0, 1)")
    city_map: str
    missing: int = Field(ge=1, le=4, description="Range is [1, 4]")
    sparse_IRT4: bool
    samples_input: bool
    cars_input: bool
    cars_simulation: bool

    city_map_inds: _CityMapIndexes
    @model_validator(mode='after')
    def _check_city_map_inds(self) -> Self:
        start, end = self.city_map_inds
        if start >= end:
            raise ValueError(f"The start index must be smaller than the end index")
        return self
    
    transmitters_number: int
    @model_validator(mode='after')
    def _check_transmitters_number(self) -> Self:
        if self.sparse_IRT4:
            if not (1 <= self.transmitters_number <= 2):
                raise ValueError(f"Range is [1, 2]")
        else:
            if not (1 <= self.transmitters_number <= 80):
                raise ValueError(f"Range is [1, 80]")
        return self

    threshold: float = Field(ge=0.0, le=1.0, description="Range is [0, 1]")
    img_size: _ImgSize
    batch_size: int = Field(gt=0, description="Batch size > 0")
    rand_seed: int = Field(ge=0, description="Random seed >= 0")

class _Config(BaseModel):
    data: _DataConfig

def load_config_strict(config_path="config/default.yaml"):
    """
    Load, check and return the contents of the YAML configuration file.
    """
    try:
        with open(file=config_path, mode='r', encoding='utf-8') as f:
            raw_config = yaml.safe_load(f)
    
        config = _Config(**raw_config) 
        return config
    except ValidationError as e:
        print("\n[Configuration verification failed] Please check the following configuration items:", file=sys.stderr)
        for error in e.errors():
            field = " -> ".join(str(x) for x in error['loc'])
            msg = error['msg']
            print(f"  - Field [{field}]: {msg}", file=sys.stderr)
        sys.exit(1)