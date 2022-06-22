from typing import Callable

from hypothesis import given
from hypothesis import strategies as st

from .strategies import data_dicts
from .typing import DataDict
from .wrappers import LibraryInfo


def _test_dunder_dataframe(df):
    assert hasattr(df, "__dataframe__")
    assert isinstance(df.__dataframe__, Callable)
    out = df.__dataframe__()
    assert isinstance(out, dict)
    assert hasattr(out, "dataframe")
    assert hasattr(out, "version")


@given(data=st.data())
def test_toplevel_dunder_dataframe(libinfo: LibraryInfo, data: st.DataObject):
    df = data.draw(libinfo.toplevel_strategy, label="df")
    _test_dunder_dataframe(df)


@given(data=st.data())
def test_dunder_dataframe(libinfo: LibraryInfo, data: st.DataObject):
    df = data.draw(libinfo.compliant_strategy, label="df")
    _test_dunder_dataframe(df)


@given(data_dict=data_dicts())
def test_num_columns(libinfo: LibraryInfo, data_dict: DataDict):
    df = libinfo.data_to_compliant(data_dict)
    assert df.num_columns() == len(data_dict)
