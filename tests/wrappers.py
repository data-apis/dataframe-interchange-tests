from itertools import combinations
from typing import Any, Callable, Dict, NamedTuple

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from .api import DataFrame
from .strategies import MockDataFrame, mock_dataframes

__all__ = ["LibraryInfo", "libinfo_params"]

TopLevelDataFrame = Any


class LibraryInfo(NamedTuple):
    name: str
    mock_to_toplevel: Callable[[MockDataFrame], TopLevelDataFrame]
    from_dataframe: Callable[[TopLevelDataFrame], DataFrame]
    frame_equal: Callable[[TopLevelDataFrame, DataFrame], bool]
    toplevel_to_compliant: Callable[[TopLevelDataFrame], DataFrame] = lambda df: (
        df.__dataframe__()["dataframe"]
    )
    mock_dataframes_kwargs: Dict[str, Any] = {}

    def mock_to_compliant(self, mock_dataframe: MockDataFrame) -> DataFrame:
        return self.toplevel_to_compliant(self.mock_to_toplevel(mock_dataframe))

    def mock_dataframes(self) -> st.SearchStrategy[MockDataFrame]:
        return mock_dataframes(**self.mock_dataframes_kwargs)

    def toplevel_dataframes(self) -> st.SearchStrategy[TopLevelDataFrame]:
        return self.mock_dataframes().map(self.mock_to_toplevel)

    def compliant_dataframes(self) -> st.SearchStrategy[TopLevelDataFrame]:
        return self.toplevel_dataframes().map(self.toplevel_to_compliant)

    def __repr__(self) -> str:
        return f"LibraryInfo(<{self.name}>)"


libinfo_params = []


# pandas
# ------

try:
    import pandas as pd
    from pandas.api.exchange import from_dataframe as pandas_from_dataframe
except ImportError as e:
    libinfo_params.append(pytest.param("pandas", marks=pytest.mark.skip(reason=e.msg)))
else:

    def pandas_mock_to_toplevel(mock_df: MockDataFrame) -> pd.DataFrame:
        if len(mock_df) == 0:
            return pd.DataFrame()
        serieses = []
        for name, (array, nominal_dtype) in mock_df.items():
            if nominal_dtype == "str":
                dtype = pd.StringDtype()
            elif nominal_dtype == "datetime64":
                dtype = "datetime64[ns]"
            elif nominal_dtype == "category":
                dtype = "category"
            else:
                dtype = None
            s = pd.Series(array, name=name, dtype=dtype)
            serieses.append(s)
        df = pd.concat(serieses, axis=1)
        return df

    libinfo = LibraryInfo(
        name="pandas",
        mock_to_toplevel=pandas_mock_to_toplevel,
        from_dataframe=pandas_from_dataframe,
        frame_equal=lambda df1, df2: df1.equals(df2),
        toplevel_to_compliant=lambda df: df.__dataframe__(),
    )
    libinfo_params.append(pytest.param(libinfo, id=libinfo.name))


# vaex
# ----

try:
    import vaex
    from vaex.dataframe_protocol import from_dataframe_to_vaex as vaex_from_dataframe
except ImportError as e:
    libinfo_params.append(pytest.param("vaex", marks=pytest.mark.skip(reason=e.msg)))
else:

    def vaex_frame_equal(df1, df2) -> bool:
        same_shape = df1.shape == df2.shape
        if not same_shape:
            return False
        columns = df1.get_column_names()
        same_cols = columns == df2.get_column_names()
        if not same_cols:
            return False
        for col in columns:
            if not np.array_equal(df1[col].values, df2[col].values, equal_nan=True):
                return False
        return True

    libinfo = LibraryInfo(
        name="vaex",
        mock_to_toplevel=lambda data: vaex.from_items(*data.items()),
        from_dataframe=vaex_from_dataframe,
        frame_equal=vaex_frame_equal,
        toplevel_to_compliant=lambda df: df.__dataframe__(),
        mock_dataframes_kwargs={"allow_zero_cols": False, "allow_zero_rows": False},
    )
    libinfo_params.append(pytest.param(libinfo, id=libinfo.name))


# modin
# -----


try:
    import modin
    import ray
    from modin.config import Engine
    from modin.pandas.utils import from_dataframe as modin_from_dataframe
except ImportError as e:
    libinfo_params.append(pytest.param("modin", marks=pytest.mark.skip(reason=e.msg)))
else:
    ray.init()
    Engine.put("ray")
    libinfo = LibraryInfo(
        name="modin",
        mock_to_toplevel=modin.pandas.DataFrame,
        from_dataframe=modin_from_dataframe,
        frame_equal=lambda df1, df2: df1.equals(df2),  # NaNs considered equal
        toplevel_to_compliant=lambda df: df.__dataframe__(),
    )
    libinfo_params.append(pytest.param(libinfo, id=libinfo.name))


# cudf
# ----

try:
    import cudf
    from cudf.core.df_protocol import from_dataframe as cudf_from_dataframe
except ImportError as e:
    libinfo_params.append(pytest.param("cudf", marks=pytest.mark.skip(reason=e.msg)))
else:
    libinfo = LibraryInfo(
        name="cudf",
        mock_to_toplevel=cudf.DataFrame,
        from_dataframe=cudf_from_dataframe,
        frame_equal=lambda df1, df2: df1.equals(df2),  # NaNs considered equal
        toplevel_to_compliant=lambda df: df.__dataframe__(),
    )
    libinfo_params.append(pytest.param(libinfo, id=libinfo.name))


# ------------------------------------------------------------------------------
# Meta tests


@pytest.mark.parametrize(
    "func_name", ["mock_dataframes", "toplevel_dataframes", "compliant_dataframes"]
)
@given(data=st.data())
def test_mock_dataframes(libinfo: LibraryInfo, func_name: str, data: st.DataObject):
    func = getattr(libinfo, func_name)
    strat = func()
    data.draw(strat)


# TODO
def test_compatible_data_dicts_kwargs():
    libinfos = []
    for param in libinfo_params:
        if not any(m.name.startswith("skip") for m in param.marks):
            libinfo = param.values[0]
            libinfos.append(libinfo)
    if len(libinfos) < 2:
        pytest.skip()
    for libinfo1, libinfo2 in combinations(libinfos, 2):
        keys1 = libinfo1.mock_dataframes_kwargs.keys()
        keys2 = libinfo2.mock_dataframes_kwargs.keys()
        for k in set(keys1) | set(keys2):
            if k in keys1 and k in keys2:
                assert (
                    libinfo1.mock_dataframes_kwargs[k]
                    == libinfo2.mock_dataframes_kwargs[k]
                )
