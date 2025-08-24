import os
import warnings
from functools import cached_property
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

load_dotenv()


def safe_load_env_var_path(env_variable: str) -> Optional[Path]:
    value: Optional[str] = os.getenv(env_variable)
    if value is not None:
        return Path(value)
    else:
        warnings.warn(f"Warning: {env_variable} is not defined. Returning None.")
        return None


class DefaultPath:
    _instance = None

    # singleton
    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super().__new__(cls, *args, **kwargs)
        return cls._instance

    @cached_property
    def original_data_dir(self) -> Optional[Path]:
        return safe_load_env_var_path("ORIGINAL_DATA_DIR")

    @cached_property
    def project_root(self) -> Optional[Path]:
        return safe_load_env_var_path("KCAT_PREDICTION_PROJECT_ROOT")

    @cached_property
    def build(self) -> Optional[Path]:
        return self.project_root and self.project_root / 'build'
