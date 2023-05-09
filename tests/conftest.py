import re
from typing import Union

import pytest
from hypothesis import settings

from .wrappers import libinfo_params

# TODO: apply deadline=None only to modin test cases
settings.register_profile("no_deadline", deadline=None)
settings.load_profile("no_deadline")


def pytest_generate_tests(metafunc):
    if "libinfo" in metafunc.fixturenames:
        metafunc.parametrize("libinfo", libinfo_params)


def pytest_addoption(parser):
    parser.addoption(
        "--ci",
        action="store_true",
        help="xfail and skip relevant tests for ../.github/workflows/test.yml",
    )
    # See https://github.com/HypothesisWorks/hypothesis/issues/2434
    parser.addoption(
        "--max-examples",
        action="store",
        default=None,
        help="set max examples generated by Hypothesis",
    )


def pytest_configure(config):
    max_examples: Union[None, str] = config.getoption("--max-examples")
    if max_examples is not None:
        settings.register_profile("max_examples", max_examples=int(max_examples))
        settings.load_profile("max_examples")


ci_xfail_ids = [
    # https://github.com/rapidsai/cudf/issues/11320
    "test_signatures.py::test_buffer_method[cudf-__dlpack__]",
    "test_signatures.py::test_buffer_method[cudf-__dlpack_device__]",
    # https://github.com/vaexio/vaex/issues/2083
    # https://github.com/vaexio/vaex/issues/2093
    # https://github.com/vaexio/vaex/issues/2113
    "test_from_dataframe.py::test_from_dataframe_roundtrip[modin-vaex]",
    "test_from_dataframe.py::test_from_dataframe_roundtrip[vaex-pandas]",
    # https://github.com/data-apis/dataframe-interchange-tests/pull/21#issuecomment-1495914398
    "test_from_dataframe.py::test_from_dataframe_roundtrip[pyarrow.Table-vaex]",
    "test_from_dataframe.py::test_from_dataframe_roundtrip[vaex-pyarrow.Table]",
    # TODO: triage
    "test_from_dataframe.py::test_from_dataframe_roundtrip[pandas-pyarrow.Table]",
    "test_from_dataframe.py::test_from_dataframe_roundtrip[pyarrow.Table-pandas]",
    # https://github.com/rapidsai/cudf/issues/11389
    "test_column_object.py::test_dtype[cudf]",
    # Raises RuntimeError, which is technically correct, but the spec will
    # require TypeError soon.
    # See https://github.com/data-apis/dataframe-api/pull/74
    "test_column_object.py::test_describe_categorical[modin]",
    # https://github.com/vaexio/vaex/issues/2113
    "test_column_object.py::test_describe_categorical[vaex]",
    # https://github.com/modin-project/modin/issues/4687
    "test_column_object.py::test_null_count[modin]",
    # https://github.com/vaexio/vaex/issues/2121
    "test_column_object.py::test_get_chunks[vaex]",
]
ci_skip_ids = [
    # https://github.com/rapidsai/cudf/issues/11332
    "test_column_object.py::test_describe_categorical[cudf]",
    # https://github.com/vaexio/vaex/issues/2118
    # https://github.com/vaexio/vaex/issues/2139
    "test_column_object.py::test_dtype[vaex]",
    # SEGFAULT
    "test_from_dataframe.py::test_from_dataframe_roundtrip[pandas-vaex]",
    # modin flakiness
    "test_from_dataframe.py::test_from_dataframe_roundtrip[modin-pandas]",
    "test_from_dataframe.py::test_from_dataframe_roundtrip[modin-modin]",
    "test_meta.py::test_frame_equal[modin]",
]
assert not any(case in ci_xfail_ids for case in ci_skip_ids)  # sanity check

r_cudf_roundtrip = re.compile(r"test_from_dataframe_roundtrip\[.*cudf.*\]")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--ci"):
        for item in items:
            if any(id_ in item.nodeid for id_ in ci_xfail_ids):
                item.add_marker(pytest.mark.xfail(strict=True))
            elif any(id_ in item.nodeid for id_ in ci_skip_ids):
                item.add_marker(pytest.mark.skip("flaky"))
            elif r_cudf_roundtrip.search(item.nodeid):
                item.add_marker(pytest.mark.skip("crashes pytest"))
