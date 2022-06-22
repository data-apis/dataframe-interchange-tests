from typing import Callable, Iterable

from hypothesis import assume, given
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
    out = df.num_columns()
    assert isinstance(out, int)
    assert out == len(data_dict)


@given(data_dict=data_dicts())
def test_num_rows(libinfo: LibraryInfo, data_dict: DataDict):
    df = libinfo.data_to_compliant(data_dict)
    nrows = next(iter(data_dict.values())).size
    out = df.num_rows()
    assume(out is not None)
    assert isinstance(out, int)
    assert out == nrows


@given(data_dict=data_dicts())
def test_num_chunks(libinfo: LibraryInfo, data_dict: DataDict):
    df = libinfo.data_to_compliant(data_dict)
    out = df.num_chunks()
    assert isinstance(out, int)
    # result is implementation-dependant


@given(data_dict=data_dicts())
def test_column_names(libinfo: LibraryInfo, data_dict: DataDict):
    df = libinfo.data_to_compliant(data_dict)
    out = df.column_names()
    assert isinstance(out, Iterable)
    assert len(list(out)) == len(data_dict)
    for name, expected_name in zip(out, data_dict.keys()):
        assert isinstance(name, str)
        assert name == expected_name


@given(data_dict=data_dicts())
def test_get_column(libinfo: LibraryInfo, data_dict: DataDict):
    df = libinfo.data_to_compliant(data_dict)
    for i in range(len(data_dict)):
        df.get_column(i)


@given(data_dict=data_dicts(), data=st.data())
def test_select_columns(libinfo: LibraryInfo, data_dict: DataDict, data: st.DataObject):
    df = libinfo.data_to_compliant(data_dict)
    indices = data.draw(
        st.lists(st.integers(0, len(data_dict) - 1), min_size=1, unique=True),
        label="indices",
    )
    df.select_columns(indices)


@given(data_dict=data_dicts(), data=st.data())
def test_select_columns_by_name(
    libinfo: LibraryInfo, data_dict: DataDict, data: st.DataObject
):
    df = libinfo.data_to_compliant(data_dict)
    names = data.draw(
        st.lists(st.sampled_from(list(data_dict.keys())), min_size=1, unique=True),
        label="names",
    )
    df.select_columns_by_name(names)


@given(data_dict=data_dicts(), data=st.data())
def test_get_chunks(libinfo: LibraryInfo, data_dict: DataDict, data: st.DataObject):
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
    # result is implementation-dependant
