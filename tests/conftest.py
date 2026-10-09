import json
from pathlib import Path

import pytest


@pytest.fixture
def pair_payload():
    return json.loads((Path(__file__).parent / "fixtures" / "bar_pair.json").read_text())
