import os
import pytest
import pandas as pd
from datetime import datetime

from codes.01_create.user_level_analyzer import UserLevelAnalyzer

@pytest.fixture
def sample_pool_addr():
    return '0x4e68ccd3e89f51c3074ca5072bbac773960dfa36'

def test_user_level_analyzer_init(sample_pool_addr):
    analyzer = UserLevelAnalyzer(sample_pool_addr, cumulative_mode=False)
    assert analyzer.pool_addr == sample_pool_addr
    assert not analyzer.cumulative_mode
    assert not analyzer.debug

def test_user_level_analyzer_cumulative_mode(sample_pool_addr):
    analyzer = UserLevelAnalyzer(sample_pool_addr, cumulative_mode=True)
    assert analyzer.cumulative_mode
