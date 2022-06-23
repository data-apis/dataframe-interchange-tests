from typing import Callable, Iterable

import numpy as np
from hypothesis import assume, given
from hypothesis import strategies as st

from .strategies import data_dicts
from .wrappers import LibraryInfo


def test_library_supports_zero_cols(libinfo: LibraryInfo):
    df = libinfo.data_to_toplevel({})
    # Just initialising a dataframe might not catch that a library doesn't
    # support zero-column dataframes - using a method like repr might!
    repr(df)


def test_library_supports_zero_rows(libinfo: LibraryInfo):
    df = libinfo.data_to_toplevel({"foo_col": np.asarray([])})
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
    df = data.draw(libinfo.toplevel_strategy, label="df")
    _test_dunder_dataframe(df)


@given(data=st.data())
def test_dunder_dataframe(libinfo: LibraryInfo, data: st.DataObject):
    df = data.draw(libinfo.compliant_strategy, label="df")
    _test_dunder_dataframe(df)


@given(data=st.data())
def test_num_columns(libinfo: LibraryInfo, data: st.DataObject):
    data_dict = data.draw(data_dicts(**libinfo.data_dicts_kwargs), label="data_dict")
    df = libinfo.data_to_compliant(data_dict)
    out = df.num_columns()
    assert isinstance(out, int)
    assert out == len(data_dict)


@given(data=st.data())
def test_num_rows(libinfo: LibraryInfo, data: st.DataObject):
    data_dict = data.draw(
        data_dicts(**{"allow_zero_cols": False, **libinfo.data_dicts_kwargs}),
        label="data_dict",
    )
    df = libinfo.data_to_compliant(data_dict)
    nrows = next(x for x in data_dict.values()).size
    out = df.num_rows()
    assume(out is not None)
    assert isinstance(out, int)
    assert out == nrows


@given(data=st.data())
def test_num_chunks(libinfo: LibraryInfo, data: st.DataObject):
    data_dict = data.draw(data_dicts(**libinfo.data_dicts_kwargs), label="data_dict")
    df = libinfo.data_to_compliant(data_dict)
    out = df.num_chunks()
    assert isinstance(out, int)
    # result is implementation-dependant


@given(data=st.data())
def test_column_names(libinfo: LibraryInfo, data: st.DataObject):
    data_dict = data.draw(data_dicts(**libinfo.data_dicts_kwargs), label="data_dict")
    df = libinfo.data_to_compliant(data_dict)
    out = df.column_names()
    assert isinstance(out, Iterable)
    assert len(list(out)) == len(data_dict)
    for name, expected_name in zip(out, data_dict.keys()):
        assert isinstance(name, str)
        assert name == expected_name


@given(data=st.data())
def test_get_column(libinfo: LibraryInfo, data: st.DataObject):
    data_dict = data.draw(
        data_dicts(**{"allow_zero_cols": False, **libinfo.data_dicts_kwargs}),
        label="data_dict",
    )
    df = libinfo.data_to_compliant(data_dict)
    for i in range(len(data_dict)):
        df.get_column(i)


@given(data=st.data())
def test_select_columns(libinfo: LibraryInfo, data: st.DataObject):
    data_dict = data.draw(
        data_dicts(**{"allow_zero_cols": False, **libinfo.data_dicts_kwargs}),
        label="data_dict",
    )
    df = libinfo.data_to_compliant(data_dict)
    indices = data.draw(
        st.lists(st.integers(0, len(data_dict) - 1), min_size=1, unique=True),
        label="indices",
    )
    df.select_columns(indices)


@given(data=st.data())
def test_select_columns_by_name(libinfo: LibraryInfo, data: st.DataObject):
    data_dict = data.draw(
        data_dicts(**{"allow_zero_cols": False, **libinfo.data_dicts_kwargs}),
        label="data_dict",
    )
    df = libinfo.data_to_compliant(data_dict)
    names = data.draw(
        st.lists(st.sampled_from(list(data_dict.keys())), min_size=1, unique=True),
        label="names",
    )
    df.select_columns_by_name(names)


@given(data=st.data())
def test_get_chunks(libinfo: LibraryInfo, data: st.DataObject):
    data_dict = data.draw(data_dicts(**libinfo.data_dicts_kwargs), label="data_dict")
    df = libinfo.data_to_compliant(data_dict)
    _n_chunks = df.num_chunks()
    assert isinstance(_n_chunks, int)  # sanity check
    n_chunks = data.draw(
        st.none() | st.integers(1, 2).map(lambda n: n * _n_chunks), label="n_chunks"
    )
    if n_chunks is None and data.draw(st.booleans()):
        args = []
    else:
        args = [n_chunks]
    df.get_chunks(*args)
