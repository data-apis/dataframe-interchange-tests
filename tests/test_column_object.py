from enum import IntEnum
from typing import Dict, Tuple

import numpy as np
import pytest
from hypothesis import given, note
from hypothesis import strategies as st

from tests.api import Column

from .strategies import MockColumn, NominalDtype, mock_dataframes
from .wrappers import LibraryInfo

# TODO: helpful assertion messages


def draw_column_and_mock(
    libinfo: LibraryInfo, data: st.DataObject
) -> Tuple[Column, MockColumn]:
    mock_df = data.draw(
        mock_dataframes(**{**libinfo.mock_dataframes_kwargs, "ncols": 1}),
        label="mock_df",
    )
    df = libinfo.mock_to_interchange(mock_df)
    name = next(iter(mock_df.keys()))
    note(f"{libinfo.mock_to_toplevel(mock_df)[name]=}")
    return df.get_column_by_name(name), mock_df[name]


@given(data=st.data())
def test_size(libinfo: LibraryInfo, data: st.DataObject):
    col, mock_col = draw_column_and_mock(libinfo, data)
    size = col.size
    if size is not None:
        assert isinstance(size, int)
        assert size == mock_col.array.size


@given(data=st.data())
def test_offset(libinfo: LibraryInfo, data: st.DataObject):
    col, _ = draw_column_and_mock(libinfo, data)
    offset = col.offset
    assert isinstance(offset, int)


INT_DTYPES = tuple(e for e in NominalDtype if e.value.startswith("int"))
UINT_DTYPES = tuple(e for e in NominalDtype if e.value.startswith("uint"))
FLOAT_DTYPES = tuple(e for e in NominalDtype if e.value.startswith("float"))


class DtypeKind(IntEnum):
    INT = 0
    UINT = 1
    FLOAT = 2
    BOOL = 20
    STRING = 21
    DATETIME = 22
    CATEGORICAL = 23


NOMINAL_TO_KIND: Dict[NominalDtype, DtypeKind] = {
    **{nd: DtypeKind.INT for nd in INT_DTYPES},
    **{nd: DtypeKind.UINT for nd in UINT_DTYPES},
    **{nd: DtypeKind.FLOAT for nd in FLOAT_DTYPES},
    NominalDtype.BOOL: DtypeKind.BOOL,
    NominalDtype.UTF8: DtypeKind.STRING,
    NominalDtype.DATETIME64NS: DtypeKind.DATETIME,
    NominalDtype.CATEGORY: DtypeKind.CATEGORICAL,
}

NOMINAL_TO_FSTRING: Dict[NominalDtype, str] = {
    NominalDtype.BOOL: "b",
    NominalDtype.INT8: "c",
    NominalDtype.INT16: "s",
    NominalDtype.INT32: "i",
    NominalDtype.INT64: "l",
    NominalDtype.UINT8: "C",
    NominalDtype.UINT16: "S",
    NominalDtype.UINT32: "I",
    NominalDtype.UINT64: "L",
    NominalDtype.FLOAT32: "f",
    NominalDtype.FLOAT64: "g",
    NominalDtype.UTF8: "u",
}


@given(data=st.data())
def test_dtype(libinfo: LibraryInfo, data: st.DataObject):
    col, mock_col = draw_column_and_mock(libinfo, data)
    dtype = col.dtype
    assert isinstance(dtype, tuple)
    assert len(dtype) == 4
    kind, bitwidth, fstring, endianness = col.dtype
    assert isinstance(kind, IntEnum)
    assert kind.value == NOMINAL_TO_KIND[mock_col.nominal_dtype].value
    assert isinstance(bitwidth, int)
    assert isinstance(fstring, str)
    if mock_col.nominal_dtype == NominalDtype.DATETIME64NS:
        assert fstring.startswith("tsn")
    # TODO: test categorical format string (i.e. using col's actual dtype)
    elif mock_col.nominal_dtype != NominalDtype.CATEGORY:
        assert fstring == NOMINAL_TO_FSTRING[mock_col.nominal_dtype]
    assert isinstance(endianness, str)
    assert len(endianness) == 1  # TODO: test actual value


@given(data=st.data())
def test_describe_categorical(libinfo: LibraryInfo, data: st.DataObject):
    # TODO: bias generation for categorical columns
    col, mock_col = draw_column_and_mock(libinfo, data)
    if mock_col.nominal_dtype == NominalDtype.CATEGORY:
        catinfo = col.describe_categorical
        assert isinstance(catinfo, dict)
        for key in ["is_ordered", "is_dictionary", "mapping"]:
            assert key in catinfo.keys()
        assert isinstance(catinfo["is_ordered"], bool)
        assert isinstance(catinfo["is_dictionary"], bool)
        mapping = catinfo["mapping"]
        if mapping is not None:
            assert isinstance(mapping, dict)
    else:
        with pytest.raises(TypeError):
            col.describe_categorical


@given(data=st.data())
def test_describe_null(libinfo: LibraryInfo, data: st.DataObject):
    col, _ = draw_column_and_mock(libinfo, data)
    nullinfo = col.describe_null
    assert isinstance(nullinfo, tuple)
    assert len(nullinfo) == 2
    kind, value = nullinfo
    assert isinstance(kind, int)
    assert kind in [0, 1, 2, 3, 4]
    if kind in [0, 1]:  # noll-nullable or NaN/NaT
        assert value is None
    elif kind in [3, 4]:  # bit or byte mask
        assert isinstance(value, int)
        assert value in [0, 1]


@given(data=st.data())
def test_null_count(libinfo: LibraryInfo, data: st.DataObject):
    col, mock_col = draw_column_and_mock(libinfo, data)
    null_count = col.null_count
    if null_count is not None:
        assert isinstance(null_count, int)
        if mock_col.nominal_dtype != NominalDtype.UTF8:  # TODO: test string cols
            assert null_count == sum(np.isnan(mock_col.array))


@given(data=st.data())
def test_num_chunks(libinfo: LibraryInfo, data: st.DataObject):
    col, _ = draw_column_and_mock(libinfo, data)
    num_chunks = col.num_chunks()
    assert isinstance(num_chunks, int)


@given(data=st.data())
def test_get_chunks(libinfo: LibraryInfo, data: st.DataObject):
    col, _ = draw_column_and_mock(libinfo, data)
    num_chunks = col.num_chunks()
    n_chunks = data.draw(
        st.none() | st.integers(1, 2).map(lambda n: n * num_chunks),
        label="n_chunks",
    )
    if n_chunks is None and not data.draw(st.booleans(), label="pass n_chunks"):
        args = []
    else:
        args = [n_chunks]
    col.get_chunks(*args)


@given(data=st.data())
def test_get_buffers(libinfo: LibraryInfo, data: st.DataObject):
    col, _ = draw_column_and_mock(libinfo, data)
    bufinfo = col.get_buffers()
    assert isinstance(bufinfo, dict)
    for key in ["data", "validity", "offsets"]:
        assert key in bufinfo.keys()
    # TODO: test returned dtypes (probably generalise it)
    data = bufinfo["data"]
    assert isinstance(data, tuple)
    assert len(data) == 2
    validity = bufinfo["validity"]
    if validity is not None:
        assert isinstance(validity, tuple)
        assert len(validity) == 2
    offsets = bufinfo["offsets"]
    if offsets is not None:
        assert isinstance(offsets, tuple)
        assert len(offsets) == 2
