from typing import List

from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra import numpy as nps

from .typing import DataDict

__all__ = ["data_dicts"]


valid_dtypes: List[str] = ["bool", "str", "datetime64[ns]"]  # TODO: category
for kind in ["int", "uint"]:
    for bitwidth in [8, 16, 32, 64]:
        valid_dtypes.append(f"{kind}{bitwidth}")
for bitwidth in [32, 64]:
    valid_dtypes.append(f"float{bitwidth}")


@st.composite
def data_dicts(draw) -> st.SearchStrategy[DataDict]:
    colnames_strat = st.from_regex("[a-z]+", fullmatch=True)  # TODO: more valid names
    nrows = draw(st.integers(0, 10))
    colnames = draw(st.lists(colnames_strat, max_size=10, unique=True))
    data = {}
    for colname in colnames:
        dtype = draw(st.sampled_from(valid_dtypes))
        x = draw(nps.arrays(dtype=dtype, shape=nrows))
        data[colname] = x
    return data


@given(data_dicts())
def test_data_dicts(data_dict: DataDict):
    if len(data_dict) != 0:
        arrays = list(data_dict.values())
        nrows = arrays[0].size
        for x in data_dict.values():
            assert x.ndim == 1
            assert x.size == nrows
