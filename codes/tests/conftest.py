import os
import pytest
import pandas as pd
from pathlib import Path

@pytest.fixture(scope="session")
def test_data_dir():
    return Path(os.path.dirname(__file__)) / "test_data"

@pytest.fixture(scope="session")
def sample_lp_data():
    data_path = Path("~/attachments/c7227f29-de12-45f0-90d6-9bd2bb2d450c/0x88e6a0c2ddd26feeb64f039a2c41296fcb3f5640_fixed.csv").expanduser()
    return pd.read_csv(data_path)

@pytest.fixture(scope="session")
def test_pool_addr():
    return "0x88e6a0c2ddd26feeb64f039a2c41296fcb3f5640"
