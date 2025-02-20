import os
import pytest
import pandas as pd
from datetime import datetime

from codes.create.user_level_analyzer import UserLevelAnalyzer

@pytest.fixture
def sample_pool_addr():
    return '0x88e6a0c2ddd26feeb64f039a2c41296fcb3f5640'

def test_user_level_analyzer_init(sample_pool_addr):
    analyzer = UserLevelAnalyzer(sample_pool_addr, cumulative_mode=False)
    assert analyzer.pool_addr == sample_pool_addr
    assert not analyzer.cumulative_mode
    assert not analyzer.debug

def test_user_level_analyzer_cumulative_mode(sample_pool_addr):
    analyzer = UserLevelAnalyzer(sample_pool_addr, cumulative_mode=True)
    assert analyzer.cumulative_mode

def test_user_level_analyzer_with_sample_data(sample_pool_addr, sample_lp_data):
    analyzer = UserLevelAnalyzer(sample_pool_addr, cumulative_mode=False)
    
    # Basic data validation
    assert not sample_lp_data.empty
    assert 'nf_token_id' in sample_lp_data.columns
    assert 'liquidity_provider' in sample_lp_data.columns
    
    # Test data processing
    result = analyzer.process_positions()
    assert isinstance(result, pd.DataFrame)
    assert not result.empty
    assert 'amt_roi' in result.columns
    assert 'fee_roi' in result.columns
    assert 'overall_roi' in result.columns
