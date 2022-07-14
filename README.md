# DataFrame Interchange Protocol Compliance Suite

This suite contains tests for dataframe libraries adopting the [Python DataFrame
Interchange Protocol](https://data-apis.org/blog/dataframe_protocol_rfc/). Note
it's a **work in progress**.

### What are we testing?

* **Round trips** ([`test_from_dataframe.py`](./tests/test_from_dataframe.py)):

  1. Creates a dataframe via library 1
  2. Interchanges it via library 2 into a "destination" dataframe
  3. Interchanges that resulting dataframe via library 1 into a "roundtrip" dataframe

  We assert the roundtrip dataframe is equivalent to the original dataframe.

* **Signatures** ([`test_signatures.py`](./tests/test_signatures.py)): Assert methods have the correct signatures.

* **Basic functionality** ([`test_dataframe_object.py`](./tests/test_dataframe_object.py)): Smoke methods can take valid input, and assert they return valid output (where appropiate).

### What the heck is `LibraryInfo`?

Tests don't access the dataframe libraries directly, but wrappers defined in [wrappers.py](./tests/wrappers.py) in the form of a `LibraryInfo` object. This allows us to standardise how these libraries work, so we can ignore implementation details when writing our tests.

For test functions which take a `libinfo` argument, we use the [`pytest_generate_tests`](https://docs.pytest.org/en/6.2.x/reference.html#pytest.hookspec.pytest_generate_tests) hook in [conftest.py](./tests/conftest.py) to automatically parametrize it with every dataframe library installed in the user's environment.

### Other testing efforts

* [data-apis/dataframe-api#75](https://github.com/data-apis/dataframe-api/pull/75)
* [pandas](https://github.com/pandas-dev/pandas/tree/main/pandas/tests/exchange)
* [vaex](https://github.com/vaexio/vaex/blob/master/tests/dataframe_protocol_test.py)
* [cudf](https://github.com/rapidsai/cudf/blob/branch-22.08/python/cudf/cudf/tests/test_df_protocol.py)
* [modin](https://github.com/modin-project/modin/tree/master/modin/test/exchange/dataframe_protocol)
