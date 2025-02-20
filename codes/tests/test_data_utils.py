import pytest
import pandas as pd
import numpy as np
from gmpy2 import mpz

from codes.shared_library.data_utils import (
    group_weighted_mean_factory,
    cumsum_mpz
)

@pytest.fixture
def sample_df():
    return pd.DataFrame({
        'value': [1, 2, 3, 4],
        'weight': [0.1, 0.2, 0.3, 0.4],
        'liquidity_mpz': [mpz(100), mpz(200), mpz(300), mpz(-200)]
    })

def test_group_weighted_mean_factory(sample_df):
    weighted_mean_func = group_weighted_mean_factory(sample_df, 'weight')
    result = weighted_mean_func(sample_df['value'])
    expected = np.average(sample_df['value'], weights=sample_df['weight'])
    assert abs(result - expected) < 1e-10

def test_cumsum_mpz(sample_df):
    result = cumsum_mpz(sample_df)
    assert 'net_liquidity' in result.columns
    assert result['net_liquidity'].tolist() == [100, 300, 600, 400]
