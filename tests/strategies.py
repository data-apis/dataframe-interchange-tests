from collections.abc import Mapping
from enum import Enum
from typing import Collection, Dict, List, NamedTuple

import numpy as np
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra import numpy as nps

__all__ = ["mock_dataframes", "MockDataFrame", "MockColumn", "NominalDtypeEnum"]


nominal_numerics: List[str] = []
for kind in ["int", "uint"]:
    for bitwidth in [8, 16, 32, 64]:
        nominal_numerics.append(f"{kind}{bitwidth}")
for bitwidth in [32, 64]:
    nominal_numerics.append(f"float{bitwidth}")

NominalDtypeEnum = Enum(
    "NominalDtypeEnum",
    {
        "BOOL": "bool",
        **{name.upper(): name for name in nominal_numerics},
        "UTF8": "U8",
        "DATETIME64NS": "datetime64[ns]",
        "CATEGORY": "category",
    },
)


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


# ------------------------------------------------------------------------------
# Meta tests


@given(mock_dataframes())
def test_mock_dataframes(_):
    pass
