from collections.abc import Mapping
from enum import Enum
from typing import Collection, Dict, NamedTuple

import numpy as np
from hypothesis import strategies as st
from hypothesis.extra import numpy as nps

__all__ = [
    "mock_dataframes",
    "mock_single_col_dataframes",
    "MockDataFrame",
    "MockColumn",
    "NominalDtype",
]


class NominalDtype(Enum):
    BOOL = "bool"
    INT8 = "int8"
    INT16 = "int16"
    INT32 = "int32"
    INT64 = "int64"
    UINT8 = "uint8"
    UINT16 = "uint16"
    UINT32 = "uint32"
    UINT64 = "uint64"
    FLOAT32 = "float32"
    FLOAT64 = "float64"
    UTF8 = "U8"
    DATETIME64NS = "datetime64[ns]"
    CATEGORY = "category"


class MockColumn(NamedTuple):
    array: np.ndarray
    nominal_dtype: NominalDtype


class MockDataFrame(Mapping):
    def __init__(self, name_to_column: Dict[str, MockColumn]):
        if len(name_to_column) == 0:
            self.ncols = 0
            self.nrows = 0
        else:
            arrays = [x for x, _ in name_to_column.values()]
            self.ncols = len(arrays)
            self.nrows = arrays[0].size
            for x in arrays:
                # sanity checks
                assert x.ndim == 1
                assert x.size == self.nrows
        self._name_to_column = name_to_column

    def __getitem__(self, key: str):
        return self._name_to_column[key]

    def __iter__(self):
        return iter(self._name_to_column)

    def __len__(self):
        return len(self._name_to_column)

    def __repr__(self) -> str:
        col_reprs = []
        for name, col in self.items():
            col_reprs.append(f"'{name}': {col.nominal_dtype.value} = {col.array}")
        return "MockDataFrame({" + ", ".join(col_reprs) + "})"


def utf8_strings() -> st.SearchStrategy[str]:
    return st.from_regex(r"[a-zA-Z][a-zA-Z\_]{0,7}", fullmatch=True)


def mock_columns(
    nominal_dtype: NominalDtype, size: int
) -> st.SearchStrategy[MockColumn]:
    dtype = nominal_dtype.value
    elements = None
    if nominal_dtype == NominalDtype.CATEGORY:
        dtype = np.int8
        elements = st.integers(0, 15)
    elif nominal_dtype == NominalDtype.UTF8:
        # nps.arrays(dtype="U8") doesn't skip surrogates by default
        elements = utf8_strings()
    x_strat = nps.arrays(dtype=dtype, shape=size, elements=elements)
    return x_strat.map(lambda x: MockColumn(x, nominal_dtype))


@st.composite
def mock_dataframes(
    draw: st.DrawFn,
    *,
    dtypes: Collection[NominalDtype] = set(NominalDtype),
    allow_zero_cols: bool = True,
    allow_zero_rows: bool = True,
) -> MockDataFrame:
    min_ncols = 0 if allow_zero_cols else 1
    colnames = draw(
        st.lists(utf8_strings(), min_size=min_ncols, max_size=5, unique=True)
    )
    min_nrows = 0 if allow_zero_rows else 1
    nrows = draw(st.integers(min_nrows, 5))
    name_to_column = {}
    for colname in colnames:
        nominal_dtype = draw(st.sampled_from(list(dtypes)))
        name_to_column[colname] = draw(mock_columns(nominal_dtype, nrows))
    return MockDataFrame(name_to_column)


@st.composite
def mock_single_col_dataframes(
    draw: st.DrawFn,
    *,
    dtypes: Collection[NominalDtype] = set(NominalDtype),
    allow_zero_rows: bool = True,
) -> MockDataFrame:
    colname = draw(utf8_strings())
    nominal_dtype = draw(st.sampled_from(list(dtypes)))
    min_size = 0 if allow_zero_rows else 1
    size = draw(st.integers(min_size, 5))
    mock_col = draw(mock_columns(nominal_dtype, size))
    return MockDataFrame({colname: mock_col})
