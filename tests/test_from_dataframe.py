import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from tests.strategies import data_dicts

from .wrappers import LibraryInfo, libinfo_params


# parametrize order is intentional for sensical pytest param ids
@pytest.mark.parametrize("dest_libinfo", libinfo_params)
@pytest.mark.parametrize("orig_libinfo", libinfo_params)
@given(data=st.data())
@settings(deadline=1000)  # namely for modin
def test_from_dataframe_roundtrip(
    orig_libinfo: LibraryInfo, dest_libinfo: LibraryInfo, data: st.DataObject
):
    """
    Round trip of dataframe interchange results in a dataframe identical to the
    original dataframe.
    """
    data_dict = data.draw(
        data_dicts(
            **{**orig_libinfo.data_dicts_kwargs, **orig_libinfo.data_dicts_kwargs}
        ),
        label="data_dict",
    )
    orig_df = orig_libinfo.data_to_toplevel(data_dict)
    dest_df = dest_libinfo.from_dataframe(orig_df)
    roundtrip_df = orig_libinfo.from_dataframe(dest_df)
    assert orig_libinfo.frame_equal(roundtrip_df, orig_df), (
        f"Round trip of dataframe did not result in an identical dataframe.\n\n"
        f"Original dataframe ({orig_libinfo.name}):\n\n"
        f"{orig_df}\n\n"
        f"Intermediate dataframe ({dest_libinfo.name}):\n\n"
        f"{dest_df}\n\n"
        f"Round trip dataframe ({orig_libinfo.name}):\n\n"
        f"{roundtrip_df}\n"
    )
