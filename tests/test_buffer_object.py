from enum import IntEnum

from hypothesis import given
from hypothesis import strategies as st

from tests.api import Buffer

from .strategies import mock_dataframes
from .wrappers import LibraryInfo


def draw_buffer(libinfo: LibraryInfo, data: st.DataObject) -> Buffer:
    mock_df = data.draw(
        mock_dataframes(**{**libinfo.mock_dataframes_kwargs, "ncols": 1}),
        label="mock_df",
    )
    df = libinfo.mock_to_interchange(mock_df)
    name = next(iter(mock_df.keys()))
    col = df.get_column_by_name(name)
    bufinfo = col.get_buffers()
    buf, _ = bufinfo["data"]
    return buf


@given(data=st.data())
def test_bufsize(libinfo: LibraryInfo, data: st.DataObject):
    buf = draw_buffer(libinfo, data)
    bufsize = buf.bufsize
    assert isinstance(bufsize, int)


@given(data=st.data())
def test_ptr(libinfo: LibraryInfo, data: st.DataObject):
    buf = draw_buffer(libinfo, data)
    ptr = buf.ptr
    assert isinstance(ptr, int)


@given(data=st.data())
def test_dlpack_device(libinfo: LibraryInfo, data: st.DataObject):
    buf = draw_buffer(libinfo, data)
    dlpack_device = buf.__dlpack_device__()
    assert isinstance(dlpack_device, tuple)
    assert len(dlpack_device) == 2
    device_type, device_id = dlpack_device
    assert isinstance(device_type, IntEnum)
    if device_id is not None:
        assert isinstance(device_id, int)
