from enum import IntEnum

from hypothesis import given
from hypothesis import strategies as st

from .wrappers import LibraryInfo


@given(data=st.data())
def test_bufsize(libinfo: LibraryInfo, data: st.DataObject):
    buf = data.draw(libinfo.buffers(), label="buf")
    bufsize = buf.bufsize
    assert isinstance(bufsize, int)


@given(data=st.data())
def test_ptr(libinfo: LibraryInfo, data: st.DataObject):
    buf = data.draw(libinfo.buffers(), label="buf")
    ptr = buf.ptr
    assert isinstance(ptr, int)


@given(data=st.data())
def test_dlpack_device(libinfo: LibraryInfo, data: st.DataObject):
    buf = data.draw(libinfo.buffers(), label="buf")
    dlpack_device = buf.__dlpack_device__()
    assert isinstance(dlpack_device, tuple)
    assert len(dlpack_device) == 2
    device_type, device_id = dlpack_device
    assert isinstance(device_type, IntEnum)
    if device_id is not None:
        assert isinstance(device_id, int)
