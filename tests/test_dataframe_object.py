from typing import Callable

from hypothesis import given
from hypothesis import strategies as st

from .wrappers import LibraryInfo


def _test_dunder_dataframe(df):
    assert hasattr(df, "__dataframe__")
    assert isinstance(df.__dataframe__, Callable)
    out = df.__dataframe__()
    assert isinstance(out, dict)
    assert hasattr(out, "dataframe")
    assert hasattr(out, "version")


@given(data=st.data())
def test_toplevel_dunder_dataframe(linfo: LibraryInfo, data: st.DataObject):
    df = data.draw(linfo.toplevel_strategy, label="df")
    _test_dunder_dataframe(df)


@given(data=st.data())
def test_dunder_dataframe(linfo: LibraryInfo, data: st.DataObject):
    df = data.draw(linfo.compliant_strategy, label="df")
    _test_dunder_dataframe(df)
