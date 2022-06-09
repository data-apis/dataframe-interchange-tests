import pytest
from hypothesis import given
from hypothesis import strategies as st

from .common import lib_params, lib_to_linfo


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
