"""
Tests the utilities of the test suite.
"""

import pytest
from hypothesis import given
from hypothesis import strategies as st

from .strategies import MockDataFrame, mock_dataframes
from .wrappers import LibraryInfo


@given(mock_dataframes())
def test_mock_dataframes(mock_df):
    assert isinstance(mock_df, MockDataFrame)


@pytest.mark.parametrize(
    "func_name", ["mock_dataframes", "toplevel_dataframes", "interchange_dataframes"]
)
@given(data=st.data())
def test_strategy(libinfo: LibraryInfo, func_name: str, data: st.DataObject):
    func = getattr(libinfo, func_name)
    strat = func()
    data.draw(strat, label="example")


@given(data=st.data())
def test_frame_equal(libinfo: LibraryInfo, data: st.DataObject):
    df = data.draw(libinfo.toplevel_dataframes(), label="df")
    assert libinfo.frame_equal(df, df)
