from typing import Any, Callable, NamedTuple

import numpy as np
import pytest
from hypothesis import strategies as st

from .api import DataFrame
from .strategies import data_dicts
from .typing import DataDict

__all__ = ["LibraryInfo", "libinfo_params"]

TopLevelDataFrame = Any


class LibraryInfo(NamedTuple):
    name: str
    data_to_toplevel: Callable[[DataDict], TopLevelDataFrame]
    from_dataframe: Callable[[TopLevelDataFrame], DataFrame]
    frame_equal: Callable[[TopLevelDataFrame, DataFrame], bool]
    get_compliant_dataframe: Callable[[TopLevelDataFrame], DataFrame] = lambda df: (
        df.__dataframe__()["dataframe"]
    )

    def data_to_compliant(self, data_dict: DataDict) -> DataFrame:
        return self.get_compliant_dataframe(self.data_to_toplevel(data_dict))

    @property
    def toplevel_strategy(self) -> st.SearchStrategy[TopLevelDataFrame]:
        return data_dicts().map(self.data_to_toplevel)

    @property
    def compliant_strategy(self) -> st.SearchStrategy[TopLevelDataFrame]:
        return self.toplevel_strategy.map(self.get_compliant_dataframe)

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
    libinfo = LibraryInfo(
        name="pandas",
        data_to_toplevel=pd.DataFrame,
        from_dataframe=pandas_from_dataframe,
        frame_equal=lambda df1, df2: df1.equals(df2),
        get_compliant_dataframe=lambda df: df.__dataframe__(),
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
        data_to_toplevel=lambda data: vaex.from_items(*data.items()),
        from_dataframe=vaex_from_dataframe,
        frame_equal=vaex_frame_equal,
        get_compliant_dataframe=lambda df: df.__dataframe__(),
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
        data_to_toplevel=modin.pandas.DataFrame,
        from_dataframe=modin_from_dataframe,
        frame_equal=lambda df1, df2: df1.equals(df2),  # NaNs considered equal
        get_compliant_dataframe=lambda df: df.__dataframe__(),
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
        data_to_toplevel=cudf.DataFrame,
        from_dataframe=cudf_from_dataframe,
        frame_equal=lambda df1, df2: df1.equals(df2),  # NaNs considered equal
        get_compliant_dataframe=lambda df: df.__dataframe__(),
    )
    libinfo_params.append(pytest.param(libinfo, id=libinfo.name))
