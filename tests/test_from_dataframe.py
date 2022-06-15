import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from .wrappers import LibraryInfo, linfo_params


# parametrize order is intentional for sensical pytest param ids
@pytest.mark.parametrize("dest_linfo", linfo_params)
@pytest.mark.parametrize("orig_linfo", linfo_params)
@given(data=st.data())
@settings(deadline=1000)  # namely for modin
def test_from_dataframe_roundtrip(
    orig_linfo: LibraryInfo, dest_linfo: LibraryInfo, data: st.DataObject
):
    """
    Round trip of dataframe interchange results in a dataframe identical to the
    original dataframe.
    """
    orig_df = data.draw(orig_linfo.toplevel_strategy, label="df")
    dest_df = dest_linfo.from_dataframe(orig_df)
    roundtrip_df = orig_linfo.from_dataframe(dest_df)
    assert orig_linfo.frame_equal(roundtrip_df, orig_df), (
        f"Round trip of dataframe did not result in an identical dataframe.\n\n"
        f"Original dataframe ({orig_linfo.name}):\n\n"
        f"{orig_df}\n\n"
        f"Intermediate dataframe ({dest_linfo.name}):\n\n"
        f"{dest_df}\n\n"
        f"Round trip dataframe ({orig_linfo.name}):\n\n"
        f"{roundtrip_df}\n"
    )
