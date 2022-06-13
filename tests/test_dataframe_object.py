from typing import Callable

from hypothesis import given
from hypothesis import strategies as st

from .wrappers import LibraryInfo


@given(data=st.data())
def test_top_level_dunder_dataframe(linfo: LibraryInfo, data: st.DataObject):
    df = data.draw(linfo.toplevel_strategy, label="df")
    assert hasattr(df, "__dataframe__")
    assert isinstance(df.__dataframe__, Callable)
    out = df.__dataframe__()
    assert isinstance(out, dict)
    assert hasattr(out, "dataframe")
    assert hasattr(out, "version")
