from inspect import getmembers, isfunction
from types import FunctionType

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from .api import DataFrame
from .common import lib_params, lib_to_linfo


@pytest.mark.parametrize("lib", lib_params)
@given(data=st.data())
@settings(max_examples=1)
def test_top_level_dunder_dataframe(lib: str, data: st.DataObject):
    linfo = lib_to_linfo[lib]
    df = data.draw(linfo.strategy, label="df")
    assert hasattr(df, "__dataframe__")
    out = df.__dataframe__()
    assert isinstance(out, dict)
    # TODO: test signature (loosely)


stub_params = []
for _, stub in getmembers(DataFrame, predicate=isfunction):
    p = pytest.param(stub, id=stub.__name__)
    stub_params.append(p)


@pytest.mark.parametrize("stub", stub_params)
@pytest.mark.parametrize("lib", lib_params)
@given(data=st.data())
@settings(max_examples=1)
def test_compliant_dataframe_method_signature(
    lib: str, stub: FunctionType, data: st.DataObject
):
    linfo = lib_to_linfo[lib]
    df = data.draw(linfo.strategy, label="df")
    _df = linfo.get_compliant_dataframe(df)
    assert hasattr(_df, stub.__name__)
    # TODO: test signature (loosely)
