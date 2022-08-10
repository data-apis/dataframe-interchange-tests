"""
Tests the utilities of the test suite.
"""

import pytest
from hypothesis import given
from hypothesis import strategies as st

from .strategies import MockDataFrame, mock_dataframes, utf8_strings
from .wrappers import LibraryInfo


@given(utf8_strings())
def test_utf8_strings(string):
    assert isinstance(string, str)
    assert string[-1:] != "\0"


@given(mock_dataframes())
def test_mock_dataframes(mock_df):
    assert isinstance(mock_df, MockDataFrame)


@pytest.mark.parametrize(
    "func_name",
    [
        "mock_dataframes",
        "toplevel_dataframes",
        "interchange_dataframes",
        "mock_single_col_dataframes",
        "columns",
        "columns_and_mock_columns",
        "buffers",
    ],
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
