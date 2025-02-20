import os
import pytest
import pandas as pd
from datetime import datetime, date

from codes.shared_library.position_utils import (
    get_relevant_position_ids_for_increase_only,
    get_relevant_position_ids_for_decrease
)

@pytest.fixture
def sample_helper_df():
    return pd.DataFrame({
        'position_id': ['1', '2', '3'],
        'nf_position_manager_address': ['addr1', 'addr1', 'addr2'],
        'date_min': [date(2023, 1, 1), date(2023, 1, 5), date(2023, 1, 10)],
        'date_max': [date(2023, 1, 10), date(2023, 1, 15), date(2023, 1, 20)]
    })

def test_get_relevant_position_ids_for_increase_only(sample_helper_df):
    x = pd.Series({
        'nf_position_manager_address': 'addr1',
        'block_timestamp': datetime(2023, 1, 7)
    })
    result = get_relevant_position_ids_for_increase_only(x, sample_helper_df)
    assert len(result) == 2
    assert '1' in result
    assert '2' in result

def test_get_relevant_position_ids_for_decrease(sample_helper_df):
    x = pd.Series({
        'nf_position_manager_address': 'addr1',
        'block_timestamp_min': datetime(2023, 1, 3),
        'block_timestamp_max': datetime(2023, 1, 12)
    })
    result = get_relevant_position_ids_for_decrease(x, sample_helper_df)
    assert len(result) == 2
    assert '1' in result
    assert '2' in result
