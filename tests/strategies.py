from collections.abc import Mapping
from enum import Enum
from typing import Collection, Dict, NamedTuple

import numpy as np
from hypothesis import strategies as st
from hypothesis.extra import numpy as nps

__all__ = ["mock_dataframes", "MockDataFrame", "MockColumn", "NominalDtypeEnum"]


class NominalDtypeEnum(Enum):
    BOOL = "bool"
    UTF8 = "U8"
    DATETIME64NS = "datetime64[ns]"
    CATEGORY = "category"
    # Numerics
    UINT8 = "uint8"
    UINT16 = "uint16"
    UINT32 = "uint32"
    UINT64 = "uint64"
    INT8 = "int8"
    INT16 = "int16"
    INT32 = "int32"
    INT64 = "int64"
    FLOAT32 = "float32"
    FLOAT64 = "float64"


class MockColumn(NamedTuple):
    array: np.ndarray
    nominal_dtype: NominalDtypeEnum


class MockDataFrame(Mapping):
    def __init__(self, name_to_column: Dict[str, MockColumn]):
        if len(name_to_column) == 0:
            self._ncols = 0
            self._nrows = 0
        else:
            arrays = [x for x, _ in name_to_column.values()]
            self._ncols = len(arrays)
            self._nrows = arrays[0].size
            for x in arrays:
                # sanity checks
                assert x.ndim == 1
                assert x.size == self._nrows
        self._name_to_column = name_to_column

    def __getitem__(self, key: str):
        return self._name_to_column[key]

    def __iter__(self):
        return iter(self._name_to_column)

    def __len__(self):
        return len(self._name_to_column)

    def num_rows(self) -> int:
        return self._nrows

    def num_columns(self) -> int:
        return self._ncols

    def __repr__(self) -> str:
        col_reprs = []
        for name, col in self.items():
            col_reprs.append(f"'{name}': {col.nominal_dtype.value} = {col.array}")
        return "MockDataFrame({" + ", ".join(col_reprs) + "})"


@st.composite
def mock_dataframes(
    draw,
    *,
    exclude_dtypes: Collection[NominalDtypeEnum] = [],
    allow_zero_cols: bool = True,
    allow_zero_rows: bool = True,
) -> MockDataFrame:
    min_ncols = 0 if allow_zero_cols else 1
    colnames_strat = st.from_regex("[a-z]+", fullmatch=True)  # TODO: more valid names
    colnames = draw(
        st.lists(colnames_strat, min_size=min_ncols, max_size=5, unique=True)
    )
    min_nrows = 0 if allow_zero_rows else 1
    nrows = draw(st.integers(min_nrows, 5))
    name_to_column = {}
    valid_dtypes = [e for e in NominalDtypeEnum if e not in exclude_dtypes]
    for colname in colnames:
        nominal_dtype = draw(st.sampled_from(valid_dtypes))
        if nominal_dtype == NominalDtypeEnum.CATEGORY:
            x_strat = nps.arrays(
                dtype=np.int8, shape=nrows, elements=st.integers(0, 15)
            )
        else:
            x_strat = nps.arrays(dtype=nominal_dtype.value, shape=nrows)
        x = draw(x_strat)
        assert not isinstance(nominal_dtype, str)
        name_to_column[colname] = MockColumn(x, nominal_dtype)
    return MockDataFrame(name_to_column)
