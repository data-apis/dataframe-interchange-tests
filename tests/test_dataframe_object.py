from typing import Iterable

from hypothesis import assume, given
from hypothesis import strategies as st

from .strategies import mock_dataframes
from .wrappers import LibraryInfo


@given(data=st.data())
def test_toplevel_dunder_dataframe(libinfo: LibraryInfo, data: st.DataObject):
    df = data.draw(libinfo.toplevel_dataframes(), label="df")
    assert hasattr(df, "__dataframe__")
    df.__dataframe__()


@given(data=st.data())
def test_dunder_dataframe(libinfo: LibraryInfo, data: st.DataObject):
    df = data.draw(libinfo.interchange_dataframes(), label="df")
    assert hasattr(df, "__dataframe__")
    df.__dataframe__()


@given(data=st.data())
def test_num_columns(libinfo: LibraryInfo, data: st.DataObject):
    mock_df = data.draw(libinfo.mock_dataframes(), label="mock_df")
    df = libinfo.mock_to_interchange(mock_df)
    out = df.num_columns()
    assert isinstance(out, int)
    assert out == mock_df.num_columns()


@given(data=st.data())
def test_num_rows(libinfo: LibraryInfo, data: st.DataObject):
    mock_df = data.draw(
        mock_dataframes(**{**libinfo.mock_dataframes_kwargs, "allow_zero_cols": False}),
        label="mock_df",
    )
    df = libinfo.mock_to_interchange(mock_df)
    out = df.num_rows()
    assume(out is not None)
    assert isinstance(out, int)
    assert out == mock_df.num_rows()


@given(data=st.data())
def test_num_chunks(libinfo: LibraryInfo, data: st.DataObject):
    mock_df = data.draw(libinfo.mock_dataframes(), label="mock_df")
    df = libinfo.mock_to_interchange(mock_df)
    out = df.num_chunks()
    assert isinstance(out, int)
    # result is implementation-dependant


@given(data=st.data())
def test_column_names(libinfo: LibraryInfo, data: st.DataObject):
    mock_df = data.draw(libinfo.mock_dataframes(), label="mock_df")
    df = libinfo.mock_to_interchange(mock_df)
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
    df = libinfo.mock_to_interchange(mock_df)
    for i in range(len(mock_df)):
        df.get_column(i)


@given(data=st.data())
def test_get_column_by_name(libinfo: LibraryInfo, data: st.DataObject):
    mock_df = data.draw(
        mock_dataframes(**{**libinfo.mock_dataframes_kwargs, "allow_zero_cols": False}),
        label="mock_df",
    )
    df = libinfo.mock_to_interchange(mock_df)
    for name in mock_df.keys():
        df.get_column_by_name(name)


@given(data=st.data())
def test_select_columns(libinfo: LibraryInfo, data: st.DataObject):
    mock_df = data.draw(
        mock_dataframes(**{**libinfo.mock_dataframes_kwargs, "allow_zero_cols": False}),
        label="mock_df",
    )
    df = libinfo.mock_to_interchange(mock_df)
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
    df = libinfo.mock_to_interchange(mock_df)
    names = data.draw(
        st.lists(st.sampled_from(list(mock_df.keys())), min_size=1, unique=True),
        label="names",
    )
    df.select_columns_by_name(names)


@given(data=st.data())
def test_get_chunks(libinfo: LibraryInfo, data: st.DataObject):
    df = data.draw(libinfo.interchange_dataframes(), label="df")
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
