from .wrappers import libinfo_params


def pytest_generate_tests(metafunc):
    if "libinfo" in metafunc.fixturenames:
        metafunc.parametrize("libinfo", libinfo_params)
