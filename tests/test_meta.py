"""
Tests the utilities of the test suite.
"""

import pytest
from hypothesis import given
from hypothesis import strategies as st

from .strategies import mock_dataframes
from .wrappers import LibraryInfo


@given(mock_dataframes())
def test_mock_dataframes(_):
    pass


@pytest.mark.parametrize(
    "func_name", ["mock_dataframes", "toplevel_dataframes", "compliant_dataframes"]
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
