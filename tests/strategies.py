import numpy as np
import pandas as pd
from hypothesis import strategies as st
from hypothesis.extra import pandas as pds

__all__ = ["pandas_dataframes"]


valid_dtypes = [np.bool_]  # TODO: str, datetimes, categories
for kind in ["int", "uint"]:
    for bitwidth in [8, 16, 32, 64]:
        valid_dtypes.append(np.dtype(f"{kind}{bitwidth}"))
for bitwidth in [32, 64]:
    valid_dtypes.append(np.dtype(f"float{bitwidth}"))


@st.composite
def pandas_dataframes(draw) -> st.SearchStrategy[pd.DataFrame]:
    colnames_strat = st.from_regex("[a-z]+", fullmatch=True)
    colnames = draw(st.lists(colnames_strat, min_size=1, unique=True))
    columns = []
    for colname in colnames:
        dtype = draw(st.sampled_from(valid_dtypes))
        column = pds.column(colname, dtype=dtype)
        columns.append(column)
    df = draw(pds.data_frames(columns))
    return df
