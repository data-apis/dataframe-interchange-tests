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
        "--ci-xfail",
        action="store_true",
        help="xfail relevant tests for ../.github/workflows/test.yml",
    )


ci_failing_ids = [
    # dataframe objects return the interchange dataframe, not a dict
    "test_dataframe_object.py::test_toplevel_dunder_dataframe[pandas]",
    "test_dataframe_object.py::test_toplevel_dunder_dataframe[vaex]",
    "test_dataframe_object.py::test_toplevel_dunder_dataframe[modin]",
    "test_dataframe_object.py::test_dunder_dataframe[pandas]",
    "test_dataframe_object.py::test_dunder_dataframe[modin]",
    # vaex's interchange dataframe doesn't have __dataframe__()
    "test_dataframe_object.py::test_dunder_dataframe[vaex]",
    "test_signatures.py::test_dataframe_method[vaex-__dataframe__]",
    # https://github.com/vaexio/vaex/issues/2083
    # https://github.com/vaexio/vaex/issues/2093
    # https://github.com/vaexio/vaex/issues/2113
    "test_from_dataframe.py::test_from_dataframe_roundtrip[vaex-pandas]",
    "test_from_dataframe.py::test_from_dataframe_roundtrip[pandas-vaex]",
    "test_from_dataframe.py::test_from_dataframe_roundtrip[modin-vaex]",
]


def pytest_collection_modifyitems(config, items):
    if config.getoption("--ci-xfail"):
        for item in items:
            if any(id_ in item.nodeid for id_ in ci_failing_ids):
                item.add_marker(pytest.mark.xfail(reason="--ci-xfail"))
