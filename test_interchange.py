from typing import Any, Callable, Dict, NamedTuple

import pytest
from hypothesis import assume, given
from hypothesis import strategies as st

DataFrame = Any


class LibraryInfo(NamedTuple):
    strategy: st.SearchStrategy[DataFrame]
    from_dataframe: Callable[[DataFrame], DataFrame]
    frame_equal: Callable[[DataFrame, DataFrame], bool]


lib_params: list = []
lib_to_linfo: Dict[str, LibraryInfo] = {}

try:
    import numpy as np
    import pandas as pd
    from hypothesis.extra import pandas as pds
    from pandas.api.exchange import from_dataframe as pandas_from_dataframe
except ImportError as e:
    lib_params.append(pytest.param("pandas", marks=pytest.mark.skip(reason=e.msg)))
else:
    valid_dtypes = [np.bool_]  # TODO: str, datetimes, categories
    for kind in ["int", "uint"]:
        for bitwidth in [8, 16, 32, 64]:
            valid_dtypes.append(np.dtype(f"{kind}{bitwidth}"))
    for bitwidth in [32, 64]:
        valid_dtypes.append(np.dtype(f"float{bitwidth}"))

    @st.composite
    def dataframes(draw) -> st.SearchStrategy[pd.DataFrame]:
        colnames_strat = st.from_regex("[a-z]+", fullmatch=True)
        colnames = draw(st.lists(colnames_strat, min_size=1, unique=True))
        columns = []
        for colname in colnames:
            dtype = draw(st.sampled_from(valid_dtypes))
            column = pds.column(colname, dtype=dtype)
            columns.append(column)
        df = draw(pds.data_frames(columns))
        assume(df.shape[0] != 0)  # TODO: generate empty dataframes
        return df

    linfo = LibraryInfo(
        strategy=dataframes(),
        from_dataframe=pandas_from_dataframe,
        frame_equal=lambda df1, df2: df1.equals(df2),
    )
    lib_to_linfo["pandas"] = linfo
    lib_params.append("pandas")

try:
    import numpy as np
    import vaex
    from vaex.dataframe_protocol import from_dataframe_to_vaex as vaex_from_dataframe
except ImportError as e:
    lib_params.append(pytest.param("vaex", marks=pytest.mark.skip(reason=e.msg)))
else:

    def vaex_frame_equal(df1: DataFrame, df2: DataFrame) -> bool:
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

    linfo = LibraryInfo(
        strategy=st.just(vaex.from_dict({"n": [42]})),
        from_dataframe=vaex_from_dataframe,
        frame_equal=vaex_frame_equal,
    )
    lib_to_linfo["vaex"] = linfo
    lib_params.append("vaex")

try:
    import modin
    import ray  # noqa: F401
    from modin.config import Engine
    from modin.pandas.utils import from_dataframe as modin_from_dataframe
except ImportError as e:
    lib_params.append(pytest.param("modin", marks=pytest.mark.skip(reason=e.msg)))
else:
    Engine.put("ray")
    linfo = LibraryInfo(
        strategy=st.just(modin.pandas.DataFrame({"n": [42]})),
        from_dataframe=modin_from_dataframe,
        frame_equal=lambda df1, df2: df1.equals(df2),
    )
    lib_to_linfo["modin"] = linfo
    lib_params.append("modin")


# parametrize order is intentional for sensical pytest param ids
@pytest.mark.parametrize("dest_lib", lib_params)
@pytest.mark.parametrize("orig_lib", lib_params)
@given(data=st.data())
def test_from_dataframe_roundtrip(orig_lib: str, dest_lib: str, data: st.DataObject):
    """
    Round trip of dataframe interchange results in a dataframe identical to the
    original dataframe.
    """
    orig_linfo = lib_to_linfo[orig_lib]
    dest_linfo = lib_to_linfo[dest_lib]
    orig_df = data.draw(orig_linfo.strategy, label="df")
    dest_df = dest_linfo.from_dataframe(orig_df)
    roundtrip_df = orig_linfo.from_dataframe(dest_df)
    assert orig_linfo.frame_equal(roundtrip_df, orig_df), (
        f"Round trip of dataframe did not result in an identical dataframe.\n\n"
        f"Original dataframe ({orig_lib}):\n\n"
        f"{orig_df}\n\n"
        f"Intermediate dataframe ({dest_lib}):\n\n"
        f"{dest_df}\n\n"
        f"Round trip dataframe ({orig_lib}):\n\n"
        f"{roundtrip_df}\n"
    )
