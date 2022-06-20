from typing import Callable

import pandas as pd
from hypothesis import given
from hypothesis import strategies as st

from .strategies import pandas_dataframes
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


@given(_df=pandas_dataframes())
def test_num_columns(libinfo: LibraryInfo, _df: pd.DataFrame):
    df = libinfo.pandas_to_compliant(_df)
    assert df.num_columns() == len(_df.columns)
