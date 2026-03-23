"""Configuration management for SubCellPortable."""

from dataclasses import dataclass, field
from typing import Optional, Literal
import logging


@dataclass
class SubCellConfig:
    """Configuration for SubCell inference.

    This class manages all configuration parameters with validation and defaults.
    Configuration sources are merged in priority order:
    1. Default values (defined here)
    2. config.yaml file
    3. Command-line arguments (highest priority)
    """

    # Input CSV file
    path_list: Optional[str] = None

    # Model configuration
    model_channels: Literal["rybg", "rbg", "ybg", "bg"] = "rybg"
    model_type: Literal["mae_contrast_supcon_model", "vit_supcon_model"] = "mae_contrast_supcon_model"
    update_model: bool = False

    # Output configuration
    output_dir: Optional[str] = None  # Output directory for all results (required for new CSV format)
    create_csv: bool = False
    save_attention_maps: bool = False
    output_format: Literal["individual", "combined"] = "combined"
    embeddings_only: bool = False

    # Performance configuration
    gpu: int = -1  # -1 for CPU, 0+ for GPU ID
    batch_size: int = 128
    num_workers: int = 4
    prefetch_factor: int = 2
    async_saving: bool = False

    # Logging
    quiet: bool = False
    log: Optional[logging.Logger] = field(default=None, repr=False)

    def __post_init__(self):
        """Validate configuration after initialization."""
        self._validate()

    def _validate(self):
        """Validate configuration parameters."""
        # Validate batch_size
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}")

        # Validate num_workers
        if self.num_workers < 0:
            raise ValueError(f"num_workers must be non-negative, got {self.num_workers}")

        # Validate prefetch_factor
        if self.prefetch_factor < 1:
            raise ValueError(f"prefetch_factor must be at least 1, got {self.prefetch_factor}")

        # Validate gpu
        if self.gpu < -1:
            raise ValueError(f"gpu must be -1 (CPU) or non-negative GPU ID, got {self.gpu}")

        # Validate channel/model combinations exist
        valid_channels = ["rybg", "rbg", "ybg", "bg"]
        if self.model_channels not in valid_channels:
            raise ValueError(f"model_channels must be one of {valid_channels}, got {self.model_channels}")

        valid_models = ["mae_contrast_supcon_model", "vit_supcon_model"]
        if self.model_type not in valid_models:
            raise ValueError(f"model_type must be one of {valid_models}, got {self.model_type}")

    @classmethod
    def from_dict(cls, config_dict: dict) -> "SubCellConfig":
        """Create config from dictionary, ignoring unknown keys.

        Args:
            config_dict: Dictionary with configuration values

        Returns:
            SubCellConfig instance
        """
        # Filter to only known fields
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered_dict = {k: v for k, v in config_dict.items() if k in valid_fields}
        return cls(**filtered_dict)

    def to_dict(self) -> dict:
        """Convert config to dictionary, excluding logger."""
        result = {}
        for key, value in self.__dict__.items():
            if key != 'log':
                result[key] = value
        return result


# Constants
NUM_CLASSES = 31
EMBEDDING_DIM = 1536
PATH_LIST_CSV = "path_list.csv"
MODELS_URLS_FILE = "models_urls.yaml"
MODEL_CONFIG_FILE = "model_config.yaml"
RESULT_CSV_FILE = "result.csv"
LOG_FILE = "log.txt"
