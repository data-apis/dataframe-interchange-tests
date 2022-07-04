from typing import Callable, Iterable

import numpy as np
import pytest
from hypothesis import assume, given
from hypothesis import strategies as st

from .strategies import MockColumn, MockDataFrame, NominalDtypeEnum, mock_dataframes
from .wrappers import LibraryInfo


def test_library_supports_zero_cols(libinfo: LibraryInfo):
    if not libinfo.allow_zero_cols:
        pytest.xfail("library doesn't support zero cols")
    mock_df = MockDataFrame({})
    df = libinfo.mock_to_toplevel(mock_df)
    # Just initialising a dataframe might not catch that a library doesn't
    # support zero-column dataframes - using a method like repr might!
    repr(df)


def test_library_supports_zero_rows(libinfo: LibraryInfo):
    if not libinfo.allow_zero_rows:
        pytest.xfail("library doesn't support zero rows")
    mock_df = MockDataFrame(
        {"foo_col": MockColumn(np.asarray([], dtype=np.int64), NominalDtypeEnum.INT64)}
    )
    df = libinfo.mock_to_toplevel(mock_df)
    # See above comment
    repr(df)


def _test_dunder_dataframe(df):
    assert hasattr(df, "__dataframe__")
    assert isinstance(df.__dataframe__, Callable)
    out = df.__dataframe__()
    assert isinstance(out, dict)
    assert hasattr(out, "dataframe")
    assert hasattr(out, "version")


@given(data=st.data())
def test_toplevel_dunder_dataframe(libinfo: LibraryInfo, data: st.DataObject):
    df = data.draw(libinfo.toplevel_dataframes(), label="df")
    _test_dunder_dataframe(df)


@given(data=st.data())
def test_dunder_dataframe(libinfo: LibraryInfo, data: st.DataObject):
    df = data.draw(libinfo.compliant_dataframes(), label="df")
    _test_dunder_dataframe(df)


@given(data=st.data())
def test_num_columns(libinfo: LibraryInfo, data: st.DataObject):
    mock_df = data.draw(libinfo.mock_dataframes(), label="mock_df")
    df = libinfo.mock_to_compliant(mock_df)
    out = df.num_columns()
    assert isinstance(out, int)
    assert out == mock_df.num_columns()


@given(data=st.data())
def test_num_rows(libinfo: LibraryInfo, data: st.DataObject):
    mock_df = data.draw(
        mock_dataframes(**{**libinfo.mock_dataframes_kwargs, "allow_zero_cols": False}),
        label="mock_df",
    )
    df = libinfo.mock_to_compliant(mock_df)
    out = df.num_rows()
    assume(out is not None)
    assert isinstance(out, int)
    assert out == mock_df.num_rows()


@given(data=st.data())
def test_num_chunks(libinfo: LibraryInfo, data: st.DataObject):
    mock_df = data.draw(libinfo.mock_dataframes(), label="mock_df")
    df = libinfo.mock_to_compliant(mock_df)
    out = df.num_chunks()
    assert isinstance(out, int)
    # result is implementation-dependant


@given(data=st.data())
def test_column_names(libinfo: LibraryInfo, data: st.DataObject):
    mock_df = data.draw(libinfo.mock_dataframes(), label="mock_df")
    df = libinfo.mock_to_compliant(mock_df)
    out = df.column_names()
    assert isinstance(out, Iterable)
    assert len(list(out)) == len(mock_df)
    for name, expected_name in zip(out, mock_df.keys()):
        assert isinstance(name, str)
        assert name == expected_name


@given(data=st.data())
def test_get_column(libinfo: LibraryInfo, data: st.DataObject):
    mock_df = data.draw(
        mock_dataframes(**{**libinfo.mock_dataframes_kwargs, "allow_zero_cols": False}),
        label="mock_df",
    )
    df = libinfo.mock_to_compliant(mock_df)
    for i in range(len(mock_df)):
        df.get_column(i)


@given(data=st.data())
def test_select_columns(libinfo: LibraryInfo, data: st.DataObject):
    mock_df = data.draw(
        mock_dataframes(**{**libinfo.mock_dataframes_kwargs, "allow_zero_cols": False}),
        label="mock_df",
    )
    df = libinfo.mock_to_compliant(mock_df)
    indices = data.draw(
        st.lists(st.integers(0, len(mock_df) - 1), min_size=1, unique=True),
        label="indices",
    )
    df.select_columns(indices)


@given(data=st.data())
def test_select_columns_by_name(libinfo: LibraryInfo, data: st.DataObject):
    mock_df = data.draw(
        mock_dataframes(**{**libinfo.mock_dataframes_kwargs, "allow_zero_cols": False}),
        label="mock_df",
    )
    df = libinfo.mock_to_compliant(mock_df)
    names = data.draw(
        st.lists(st.sampled_from(list(mock_df.keys())), min_size=1, unique=True),
        label="names",
    )
    df.select_columns_by_name(names)


@given(data=st.data())
def test_get_chunks(libinfo: LibraryInfo, data: st.DataObject):
    df = data.draw(libinfo.compliant_dataframes(), label="df")
    _n_chunks = df.num_chunks()
    assert isinstance(_n_chunks, int)  # sanity check
    n_chunks = data.draw(
        st.none() | st.integers(1, 2).map(lambda n: n * _n_chunks), label="n_chunks"
    )
    if n_chunks is None and not data.draw(st.booleans(), label="pass n_chunks"):
        args = []
    else:
        args = [n_chunks]
    df.get_chunks(*args)
