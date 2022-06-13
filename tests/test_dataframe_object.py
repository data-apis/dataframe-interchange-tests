from inspect import Parameter, getmembers, isfunction, signature
from types import FunctionType
from typing import Callable

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from .api import DataFrame
from .wrappers import LibraryInfo

VAR_KINDS = (Parameter.VAR_POSITIONAL, Parameter.VAR_KEYWORD)
KIND_TO_STR = {
    Parameter.POSITIONAL_OR_KEYWORD: "pos or kw argument",
    Parameter.POSITIONAL_ONLY: "pos-only argument",
    Parameter.KEYWORD_ONLY: "keyword-only argument",
    Parameter.VAR_POSITIONAL: "star-args (i.e. *args) argument",
    Parameter.VAR_KEYWORD: "star-kwargs (i.e. **kwargs) argument",
}


def _test_signature(func, stub):
    sig = signature(func)
    stub_sig = signature(stub)
    params = list(sig.parameters.values())
    stub_params = list(stub_sig.parameters.values())
    # sanity checks
    if len(params) > 0:
        assert params[0].name != "self"
    assert stub_params[0].name == "self"
    del stub_params[0]

    non_kwonly_stub_params = [
        p for p in stub_params if p.kind != Parameter.KEYWORD_ONLY
    ]
    # sanity check
    assert non_kwonly_stub_params == stub_params[: len(non_kwonly_stub_params)]
    # We're not interested if the array module has additional arguments, so we
    # only iterate through the arguments listed in the spec.
    for i, stub_param in enumerate(non_kwonly_stub_params):
        assert (
            len(params) >= i + 1
        ), f"Argument '{stub_param.name}' missing from signature"
        param = params[i]

        # We're not interested in the name if it isn't actually used
        if stub_param.kind not in [Parameter.POSITIONAL_ONLY, *VAR_KINDS]:
            assert (
                param.name == stub_param.name
            ), f"Expected argument '{param.name}' to be named '{stub_param.name}'"

        if stub_param.kind in [Parameter.POSITIONAL_OR_KEYWORD, *VAR_KINDS]:
            f_stub_kind = KIND_TO_STR[stub_param.kind]
            assert param.kind == stub_param.kind, (
                f"{param.name} is a {KIND_TO_STR[param.kind]}, "
                f"but should be a {f_stub_kind}"
            )

    kwonly_stub_params = stub_params[len(non_kwonly_stub_params) :]
    for stub_param in kwonly_stub_params:
        assert (
            stub_param.name in sig.parameters.keys()
        ), f"Argument '{stub_param.name}' missing from signature"
        param = next(p for p in params if p.name == stub_param.name)
        f_stub_kind = KIND_TO_STR[stub_param.kind]
        assert param.kind in [stub_param.kind, Parameter.POSITIONAL_OR_KEYWORD], (
            f"{param.name} is a {KIND_TO_STR[param.kind]}, "
            f"but should be a {f_stub_kind} "
            f"(or at least a {KIND_TO_STR[Parameter.POSITIONAL_OR_KEYWORD]})"
        )


@given(data=st.data())
@settings(max_examples=1)
def test_top_level_dunder_dataframe(linfo: LibraryInfo, data: st.DataObject):
    df = data.draw(linfo.strategy, label="df")
    assert hasattr(df, "__dataframe__")
    assert isinstance(df.__dataframe__, Callable)
    _test_signature(df.__dataframe__, DataFrame.__dataframe__)
    out = df.__dataframe__()
    assert isinstance(out, dict)
    assert hasattr(out, "dataframe")
    assert hasattr(out, "version")


stub_params = []
for _, stub in getmembers(DataFrame, predicate=isfunction):
    p = pytest.param(stub, id=stub.__name__)
    stub_params.append(p)


@pytest.mark.parametrize("stub", stub_params)
@given(data=st.data())
@settings(max_examples=1)
def test_compliant_dataframe_method_signature(
    linfo: LibraryInfo, stub: FunctionType, data: st.DataObject
):
    df = data.draw(linfo.strategy, label="df")
    _df = linfo.get_compliant_dataframe(df)
    assert hasattr(_df, stub.__name__)
    method = getattr(_df, stub.__name__)
    assert isinstance(method, Callable)
    _test_signature(method, stub)
    # TODO: test result(s)?