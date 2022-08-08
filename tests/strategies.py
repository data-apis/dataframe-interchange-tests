from collections.abc import Mapping
from enum import Enum
from typing import Collection, Dict, NamedTuple, Optional

import numpy as np
from hypothesis import strategies as st
from hypothesis.extra import numpy as nps

__all__ = ["mock_dataframes", "MockDataFrame", "MockColumn", "NominalDtype"]


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


utf8_strat = st.from_regex(r"[a-zA-Z\_]{1,8}", fullmatch=True).filter(
    lambda b: b[-1:] != "\0"
)


@st.composite
def mock_dataframes(
    draw,
    *,
    dtypes: Collection[NominalDtype] = set(NominalDtype),
    allow_zero_cols: bool = True,
    allow_zero_rows: bool = True,
    ncols: Optional[int] = None,
) -> MockDataFrame:
    if ncols is None:
        min_ncols = 0 if allow_zero_cols else 1
        max_ncols = 5
    else:
        if ncols == 0 and not allow_zero_cols:
            raise ValueError(f"ncols cannot be 0 when {allow_zero_cols=}")
        min_ncols = ncols
        max_ncols = ncols
    colnames = draw(
        st.lists(utf8_strat, min_size=min_ncols, max_size=max_ncols, unique=True)
    )
    min_nrows = 0 if allow_zero_rows else 1
    nrows = draw(st.integers(min_nrows, 5))
    name_to_column = {}
    for colname in colnames:
        nominal_dtype = draw(st.sampled_from(list(dtypes)))
        dtype = nominal_dtype.value
        elements = None
        if nominal_dtype == NominalDtype.CATEGORY:
            dtype = np.int8
            elements = st.integers(0, 15)
        elif nominal_dtype == NominalDtype.UTF8:
            # nps.arrays(dtype="U8") doesn't skip surrogates by default
            elements = utf8_strat
        x = draw(nps.arrays(dtype=dtype, shape=nrows, elements=elements))
        assert not isinstance(nominal_dtype, str)
        name_to_column[colname] = MockColumn(x, nominal_dtype)
    return MockDataFrame(name_to_column)
