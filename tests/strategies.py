from typing import List

from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra import numpy as nps

from .typing import DataDict

__all__ = ["data_dicts"]


valid_dtypes: List[str] = ["bool"]  # TODO: str, category, datetime
for kind in ["int", "uint"]:
    for bitwidth in [8, 16, 32, 64]:
        valid_dtypes.append(f"{kind}{bitwidth}")
for bitwidth in [32, 64]:
    valid_dtypes.append(f"float{bitwidth}")


@st.composite
def data_dicts(
    draw, *, allow_zero_cols: bool = True, allow_zero_rows: bool = True
) -> DataDict:
    min_ncols = 0 if allow_zero_cols else 1
    colnames_strat = st.from_regex("[a-z]+", fullmatch=True)  # TODO: more valid names
    colnames = draw(
        st.lists(colnames_strat, min_size=min_ncols, max_size=5, unique=True)
    )
    min_nrows = 0 if allow_zero_rows else 1
    nrows = draw(st.integers(min_nrows, 5))
    data = {}
    for colname in colnames:
        dtype = draw(st.sampled_from(valid_dtypes))
        x = draw(nps.arrays(dtype=dtype, shape=nrows))
        data[colname] = x
    return data


# ------------------------------------------------------------------------------
# Meta tests


@given(data_dicts())
def test_data_dicts(data_dict: DataDict):
    if len(data_dict) != 0:
        arrays = list(data_dict.values())
        nrows = arrays[0].size
        for x in data_dict.values():
            assert x.ndim == 1
            assert x.size == nrows
