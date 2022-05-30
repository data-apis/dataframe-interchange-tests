from importlib import import_module
from itertools import permutations
from types import ModuleType
from typing import Any, Callable

import pytest

example_dict = {"n": [42]}


def get_example(lib: ModuleType) -> Any:
    if lib.__name__ == "pandas":
        return lib.DataFrame(example_dict)
    elif lib.__name__ == "vaex":
        return lib.from_dict(example_dict)
    else:
        raise NotImplementedError(f"{lib.__name__=}")


def get_asserter(lib: ModuleType) -> Callable[[Any, Any], None]:
    if lib.__name__ == "pandas":

        def assert_frame_equal(a, b):
            lib.testing.assert_frame_equal(a, b)

    elif lib.__name__ == "vaex":

        def assert_frame_equal(a, b):
            assert a == b

    else:
        raise NotImplementedError(f"{lib.__name__=}")
    return assert_frame_equal


def get_from_dataframe(lib: ModuleType) -> Callable[[Any], Any]:
    if lib.__name__ == "pandas":
        from pandas.api.exchange import from_dataframe
    elif lib.__name__ == "vaex":
        from vaex.dataframe_protocol import from_dataframe_to_vaex as from_dataframe
    else:
        raise NotImplementedError(f"{lib.__name__=}")
    return from_dataframe


LIBRARY_NAMES = ["pandas", "vaex"]
libraries = []
for name in LIBRARY_NAMES:
    try:
        lib = import_module(name)
    except ImportError:
        pass
    else:
        libraries.append(lib)
roundtrip_params = []
for orig_lib, dest_lib in permutations(libraries, 2):
    id_ = f"{orig_lib.__name__}-{dest_lib.__name__}"
    p = pytest.param(orig_lib, dest_lib, id=id_)
    roundtrip_params.append(p)


@pytest.mark.parametrize("orig_lib, dest_lib", roundtrip_params)
def test_from_dataframe_roundtrip(orig_lib, dest_lib):
    asserter = get_asserter(orig_lib)
    orig_from_dataframe = get_from_dataframe(orig_lib)
    dest_from_dataframe = get_from_dataframe(dest_lib)
    orig_df = get_example(orig_lib)
    dest_df = dest_from_dataframe(orig_df)
    roundtrip_df = orig_from_dataframe(dest_df)
    asserter(roundtrip_df, orig_df)
