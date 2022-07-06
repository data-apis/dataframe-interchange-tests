import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from tests.strategies import mock_dataframes

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
    mock_df = data.draw(
        mock_dataframes(
            exclude_dtypes=list(
                set(orig_libinfo.exclude_dtypes) | set(dest_libinfo.exclude_dtypes)
            ),
            allow_zero_cols=(
                orig_libinfo.allow_zero_cols and dest_libinfo.allow_zero_cols
            ),
            allow_zero_rows=(
                orig_libinfo.allow_zero_rows and dest_libinfo.allow_zero_rows
            ),
        ),
        label="mock_df",
    )
    orig_df = orig_libinfo.mock_to_toplevel(mock_df)
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
