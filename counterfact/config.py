"""Local configuration. Importing this module never reads or prints credentials."""

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    data_dir: Path

    @classmethod
    def from_env(cls) -> "Settings":
        load_dotenv(Path.cwd() / ".env", override=False)
        return cls(data_dir=Path(os.getenv("COUNTERFACT_DATA_DIR", "data")).resolve())

    @property
    def database(self) -> Path:
        return self.data_dir / "counterfact.db"
