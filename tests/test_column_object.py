from enum import IntEnum
from typing import Dict

import numpy as np
import pytest
from hypothesis import given, note
from hypothesis import strategies as st

from .strategies import NominalDtype, mock_single_col_dataframes
from .wrappers import LibraryInfo


@given(data=st.data())
def test_size(libinfo: LibraryInfo, data: st.DataObject):
    col, mock_col = data.draw(libinfo.columns_and_mock_columns(), label="col, mock_col")
    size = col.size()
    assert isinstance(size, int)
    assert size == mock_col.array.size


@given(data=st.data())
def test_offset(libinfo: LibraryInfo, data: st.DataObject):
    col = data.draw(libinfo.columns(), label="col")
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
    col, mock_col = data.draw(libinfo.columns_and_mock_columns(), label="col, mock_col")
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
def test_describe_categorical_on_categorical(libinfo: LibraryInfo, data: st.DataObject):
    if NominalDtype.CATEGORY not in libinfo.supported_dtypes:
        pytest.skip(f"categorical columns not generated for {libinfo.name}")
    mock_df = data.draw(
        mock_single_col_dataframes(
            dtypes={NominalDtype.CATEGORY},
            allow_zero_rows=libinfo.allow_zero_rows,
        ),
        label="mock_df",
    )
    df = libinfo.mock_to_interchange(mock_df)
    col = df.get_column(0)
    note(f"{col=}")
    catinfo = col.describe_categorical
    assert isinstance(catinfo, dict)
    for key in ["is_ordered", "is_dictionary", "categories"]:
        assert key in catinfo.keys()
    assert isinstance(catinfo["is_ordered"], bool)
    assert isinstance(catinfo["is_dictionary"], bool)
    if not catinfo["is_dictionary"]:
        assert catinfo["categories"] is None


@given(data=st.data())
def test_describe_categorical_on_non_categorical(
    libinfo: LibraryInfo, data: st.DataObject
):
    dtypes = libinfo.supported_dtypes
    if NominalDtype.CATEGORY in libinfo.supported_dtypes:
        dtypes.remove(NominalDtype.CATEGORY)
    mock_df = data.draw(
        mock_single_col_dataframes(
            dtypes=dtypes, allow_zero_rows=libinfo.allow_zero_rows
        ),
        label="mock_df",
    )
    df = libinfo.mock_to_interchange(mock_df)
    col = df.get_column(0)
    note(f"{col=}")
    with pytest.raises(TypeError):
        col.describe_categorical


@given(data=st.data())
def test_describe_null(libinfo: LibraryInfo, data: st.DataObject):
    col = data.draw(libinfo.columns(), label="col")
    nullinfo = col.describe_null
    assert isinstance(nullinfo, tuple)
    assert len(nullinfo) == 2
    kind, value = nullinfo
    assert isinstance(kind, int)
    assert kind in [0, 1, 2, 3, 4]
    if kind in [0, 1]:  # noll-nullable or NaN
        assert value is None
    elif kind in [3, 4]:  # bit or byte mask
        assert isinstance(value, int)
        assert value in [0, 1]


@given(data=st.data())
def test_null_count(libinfo: LibraryInfo, data: st.DataObject):
    col, mock_col = data.draw(libinfo.columns_and_mock_columns(), label="col, mock_col")
    null_count = col.null_count
    if null_count is not None:
        assert isinstance(null_count, int)
        if mock_col.nominal_dtype != NominalDtype.UTF8:  # TODO: test string cols
            assert null_count == sum(np.isnan(mock_col.array))


@given(data=st.data())
def test_num_chunks(libinfo: LibraryInfo, data: st.DataObject):
    col = data.draw(libinfo.columns(), label="col")
    num_chunks = col.num_chunks()
    assert isinstance(num_chunks, int)


@given(data=st.data())
def test_get_chunks(libinfo: LibraryInfo, data: st.DataObject):
    col = data.draw(libinfo.columns(), label="col")
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
    col = data.draw(libinfo.columns(), label="col")
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
