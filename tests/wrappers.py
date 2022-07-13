import re
from copy import copy
from typing import Any, Callable, Dict, List, NamedTuple, Tuple

import numpy as np
import pytest
from hypothesis import strategies as st

from .api import DataFrame
from .strategies import MockDataFrame, NominalDtype, mock_dataframes

__all__ = ["libname_to_libinfo", "libinfo_params", "LibraryInfo"]

TopLevelDataFrame = Any


class LibraryInfo(NamedTuple):
    name: str
    mock_to_toplevel: Callable[[MockDataFrame], TopLevelDataFrame]
    from_dataframe: Callable[[TopLevelDataFrame], DataFrame]
    frame_equal: Callable[[TopLevelDataFrame, DataFrame], bool]
    toplevel_to_compliant: Callable[[TopLevelDataFrame], DataFrame] = lambda df: (
        df.__dataframe__()["dataframe"]
    )
    exclude_dtypes: List[NominalDtype] = []
    allow_zero_cols: bool = True
    allow_zero_rows: bool = True

    def mock_to_compliant(self, mock_dataframe: MockDataFrame) -> DataFrame:
        return self.toplevel_to_compliant(self.mock_to_toplevel(mock_dataframe))

    @property
    def mock_dataframes_kwargs(self) -> Dict[str, Any]:
        return {
            "exclude_dtypes": self.exclude_dtypes,
            "allow_zero_cols": self.allow_zero_cols,
            "allow_zero_rows": self.allow_zero_rows,
        }

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
        if mock_df.num_columns() == 0:
            return pd.DataFrame()
        serieses = []
        for name, (array, nominal_dtype) in mock_df.items():
            if nominal_dtype == NominalDtype.UTF8:
                dtype = pd.StringDtype()
            else:
                dtype = nominal_dtype.value
            s = pd.Series(array, name=name, dtype=dtype)
            serieses.append(s)
        df = pd.concat(serieses, axis=1)
        return df

    pandas_libinfo = LibraryInfo(
        name="pandas",
        mock_to_toplevel=pandas_mock_to_toplevel,
        from_dataframe=pandas_from_dataframe,
        frame_equal=lambda df1, df2: df1.equals(df2),
        toplevel_to_compliant=lambda df: df.__dataframe__(),
        exclude_dtypes=[NominalDtype.DATETIME64NS],
    )
    libinfo_params.append(pytest.param(pandas_libinfo, id=pandas_libinfo.name))


# vaex
# ----

try:
    import vaex
    from vaex.dataframe_protocol import from_dataframe_to_vaex as vaex_from_dataframe
except ImportError as e:
    libinfo_params.append(pytest.param("vaex", marks=pytest.mark.skip(reason=e.msg)))
else:

    def vaex_mock_to_toplevel(mock_df: MockDataFrame) -> TopLevelDataFrame:
        if mock_df.num_columns() == 0 or mock_df.num_rows() == 0:
            raise ValueError(f"{mock_df=} not supported by vaex")
        items: List[Tuple[str, np.ndarray]] = []
        for name, (array, _) in mock_df.items():
            items.append((name, array))
        df = vaex.from_items(*items)
        for name, (array, nominal_dtype) in mock_df.items():
            if nominal_dtype == NominalDtype.CATEGORY:
                if not np.issubdtype(array.dtype, np.integer):
                    raise ValueError(
                        f"Array with dtype {array.dtype} was given, "
                        "but only integers can be marked as categorical in vaex."
                    )
                df = df.categorize(name)
        return df

    def vaex_frame_equal(df1, df2) -> bool:
        same_shape = df1.shape == df2.shape
        if not same_shape:
            return False
        columns = df1.get_column_names()
        if not columns == df2.get_column_names():
            return False
        for col in columns:
            if df1[col].dtype == "string":
                if df2[col].dtype != "string":
                    return False
                equal_nan = False  # equal_nan=True not understood for string arrays
            else:
                equal_nan = True
            if not np.array_equal(
                df1[col].values, df2[col].values, equal_nan=equal_nan
            ):
                return False
        return True

    vaex_libinfo = LibraryInfo(
        name="vaex",
        mock_to_toplevel=vaex_mock_to_toplevel,
        from_dataframe=vaex_from_dataframe,
        frame_equal=vaex_frame_equal,
        toplevel_to_compliant=lambda df: df.__dataframe__(),
        exclude_dtypes=[NominalDtype.DATETIME64NS],
        # https://github.com/vaexio/vaex/issues/2094
        allow_zero_cols=False,
        allow_zero_rows=False,
    )
    libinfo_params.append(pytest.param(vaex_libinfo, id=vaex_libinfo.name))


# modin
# -----


try:
    import modin  # noqa: F401

    # One issue modin has with pandas upstream is an outdated import of an
    # exception class, so we try monkey-patching the class to the old path.
    try:
        from pandas.core import base
        from pandas.errors import DataError
    except ImportError:
        pass
    else:
        setattr(base, "DataError", DataError)

    import ray

    # Without local_mode=True, ray does not use our monkey-patched pandas
    ray.init(local_mode=True)

    from modin.config import Engine

    Engine.put("ray")

    from modin import pandas as mpd
    from modin.pandas.utils import from_dataframe as modin_from_dataframe
except ImportError as e:
    libinfo_params.append(pytest.param("modin", marks=pytest.mark.skip(reason=e.msg)))
else:

    def modin_mock_to_toplevel(mock_df: MockDataFrame) -> mpd.DataFrame:
        if mock_df.num_columns() == 0:
            return mpd.DataFrame()
        if mock_df.num_rows() == 0:
            raise ValueError(f"{mock_df=} not supported by modin")
        serieses: List[mpd.Series] = []
        for name, (array, nominal_dtype) in mock_df.items():
            if nominal_dtype == NominalDtype.UTF8:
                dtype = mpd.StringDtype()
            else:
                dtype = nominal_dtype.value
            s = mpd.Series(array, name=name, dtype=dtype)
            serieses.append(s)
        df = mpd.concat(serieses, axis=1)
        return df

    def modin_frame_equal(df1: mpd.DataFrame, df2: mpd.DataFrame) -> bool:
        # Note equals() does not treat NaNs as equal, unlike pandas
        # See https://github.com/modin-project/modin/issues/4653
        if df1.shape != df2.shape:
            return False
        columns = df1.columns
        if not columns.equals(df2.columns):
            return False
        for col in columns:
            s1 = df1[col]
            s2 = df2[col]
            null_mask = s1.isnull()
            if not null_mask.equals(s2.isnull()):
                return False
            if not s1[~null_mask].equals(s2[~null_mask]):
                return False
        return True

    modin_libinfo = LibraryInfo(
        name="modin",
        mock_to_toplevel=modin_mock_to_toplevel,
        from_dataframe=modin_from_dataframe,
        frame_equal=modin_frame_equal,
        toplevel_to_compliant=lambda df: df.__dataframe__(),
        # https://github.com/modin-project/modin/issues/4654
        # https://github.com/modin-project/modin/issues/4652
        exclude_dtypes=[
            NominalDtype.UTF8,
            NominalDtype.DATETIME64NS,
            NominalDtype.CATEGORY,
        ],
        # https://github.com/modin-project/modin/issues/4643
        allow_zero_rows=False,
    )
    libinfo_params.append(pytest.param(modin_libinfo, id=modin_libinfo.name))

try:
    # cudf has a few issues with upstream pandas that we "fix" with a few hacks
    try:
        import pandas
        import pyarrow
        from pandas._libs.tslibs.parsing import guess_datetime_format
        from pandas.core.tools import datetimes
        from pyarrow.lib import ArrowKeyError
    except ImportError:
        pass
    else:
        old_register_extension_type = copy(pyarrow.register_extension_type)
        r_existing_ext_type_msg = re.compile(
            "A type extension with name pandas.[a-z_]+ already defined"
        )

        def register_extension_type(*a, **kw):
            try:
                old_register_extension_type(*a, **kw)
            except ArrowKeyError as e:
                if r_existing_ext_type_msg.fullmatch(str(e)):
                    pass
                else:
                    raise e

        setattr(pyarrow, "register_extension_type", register_extension_type)
        setattr(datetimes, "_guess_datetime_format", guess_datetime_format)
        setattr(pandas, "__version__", "1.4.3")

    import cudf
    from cudf.core.df_protocol import from_dataframe as cudf_from_dataframe
except ImportError as e:
    libinfo_params.append(pytest.param("cudf", marks=pytest.mark.skip(reason=e.msg)))
else:

    def cudf_mock_to_toplevel(mock_df: MockDataFrame) -> cudf.DataFrame:
        if mock_df.num_columns() == 0:
            return cudf.DataFrame()
        serieses = []
        for name, (array, nominal_dtype) in mock_df.items():
            if NominalDtype.CATEGORY:
                # See https://github.com/rapidsai/cudf/issues/11256
                data = array.tolist()
            else:
                data = array
            s = cudf.Series(data, name=name, dtype=nominal_dtype.value)
            serieses.append(s)
        if len(serieses) == 1:
            # See https://github.com/rapidsai/cudf/issues/11244
            df = serieses[0].to_frame()
        else:
            df = cudf.concat(serieses, axis=1)
        return df

    cupy_libinfo = LibraryInfo(
        name="cudf",
        mock_to_toplevel=cudf_mock_to_toplevel,
        from_dataframe=cudf_from_dataframe,
        frame_equal=lambda df1, df2: df1.equals(df2),  # NaNs considered equal
        toplevel_to_compliant=lambda df: df.__dataframe__(),
    )
    libinfo_params.append(pytest.param(cupy_libinfo, id=cupy_libinfo.name))


libname_to_libinfo: Dict[str, LibraryInfo] = {}
for param in libinfo_params:
    if not any(m.name.startswith("skip") for m in param.marks):
        libinfo = param.values[0]
        assert isinstance(libinfo, LibraryInfo)  # for mypy
        libname_to_libinfo[libinfo.name] = libinfo
