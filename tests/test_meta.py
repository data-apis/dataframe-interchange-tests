"""
Tests the utilities of the test suite.
"""

import pytest
from hypothesis import given
from hypothesis import strategies as st

from .strategies import MockDataFrame, mock_dataframes, utf8_strings
from .wrappers import LibraryInfo, libname_to_libinfo


def test_ci_has_correct_library_params(pytestconfig):
    if not pytestconfig.getoption("--ci"):
        pytest.skip("only intended for --ci runs")
    assert set(libname_to_libinfo.keys()) == {
        "pandas",
        "vaex",
        "modin",
        "pyarrow.Table",
        "pyarrow.RecordBatch",
    }


@given(utf8_strings())
def test_utf8_strings(string):
    assert isinstance(string, str)
    assert string[-1:] != "\0"


@given(mock_dataframes())
def test_mock_dataframes(mock_df):
    assert isinstance(mock_df, MockDataFrame)


@pytest.mark.parametrize(
    "func_name",
    [
        "mock_dataframes",
        "toplevel_dataframes",
        "interchange_dataframes",
        "mock_single_col_dataframes",
        "columns",
        "columns_and_mock_columns",
        "buffers",
    ],
)
@given(data=st.data())
def test_strategy(libinfo: LibraryInfo, func_name: str, data: st.DataObject):
    func = getattr(libinfo, func_name)
    strat = func()
    data.draw(strat, label="example")


@given(data=st.data())
def test_frame_equal(libinfo: LibraryInfo, data: st.DataObject):
    df = data.draw(libinfo.toplevel_dataframes(), label="df")
    assert libinfo.frame_equal(df, df)


def test_pandas_frame_equal_string_object_columns():
    try:
        import pandas as pd

        libinfo = libname_to_libinfo["pandas"]
    except (KeyError, ImportError) as e:
        pytest.skip(e.msg)
    df1 = pd.DataFrame({"foo": ["bar"]})
    assert df1["foo"].dtype == object  # sanity check
    df2 = pd.DataFrame({"foo": pd.Series(["bar"], dtype=pd.StringDtype())})
    assert libinfo.frame_equal(df1, df2)
    assert libinfo.frame_equal(df2, df1)


@pytest.mark.parametrize("container_name", ["Table", "RecordBatch"])
def test_pyarrow_frame_equal_string_columns(container_name):
    try:
        import pyarrow as pa

        libinfo = libname_to_libinfo[f"pyarrow.{container_name}"]
    except (KeyError, ImportError) as e:
        pytest.skip(e.msg)

    container_class = getattr(pa, container_name)
    df1 = container_class.from_pydict(
        {
            "a": pa.array(["foo"]),
            "b": pa.DictionaryArray.from_arrays(pa.array([0]), pa.array(["bar"])),
        }
    )
    df2 = container_class.from_pydict(
        {
            "a": pa.array(["foo"], type=pa.large_string()),
            "b": pa.DictionaryArray.from_arrays(
                pa.array([0]), pa.array(["bar"], type=pa.large_string())
            ),
        }
    )
    assert libinfo.frame_equal(df1, df2)
    assert libinfo.frame_equal(df2, df1)
