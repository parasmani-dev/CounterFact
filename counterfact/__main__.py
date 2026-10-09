"""Generate input pairs locally. This command does not perform model inference."""

import argparse
import json
import sqlite3
import sys
from pathlib import Path

from pydantic import ValidationError

from counterfact.config import Settings
from counterfact.schemas import PairSpec
from counterfact.storage import PairStore


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("spec", type=Path, help="Validated chart-pair JSON input")
    args = parser.parse_args()
    try:
        spec = PairSpec.model_validate_json(args.spec.read_text(encoding="utf-8"))
        store = PairStore(Settings.from_env().data_dir)
        store.initialize()
        result = store.create(spec)
    except (ValidationError, OSError, ValueError, sqlite3.Error):
        print(
            "Invalid input or unavailable file/storage. Check the JSON schema and paths.",
            file=sys.stderr,
        )
        return 2
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
