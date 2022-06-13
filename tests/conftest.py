from .wrappers import linfo_params


def pytest_generate_tests(metafunc):
    if "linfo" in metafunc.fixturenames:
        metafunc.parametrize("linfo", linfo_params)
