# DBX: Data Experiment Management Hub

DBX is a powerful tool for managing data experiments, providing a flexible interface for data handling, remote execution, and parallel processing.

## Installation

To install the package in development mode, run:

```bash
pip install -e .
```

Ensure you have the following dependencies installed (as listed in `requirements.txt`):
- `ray`
- `numpy`
- `tqdm`
- `gitpython`
- `fsspec`
- `pandas`
- `torch`
- `scipy`
- `pyyaml`

## Running Tests

The test suite uses the standard Python `unittest` framework.

### Pre-requisites for Remote Tests
Remote tests rely on **Ray**. If you are running tests in an environment with a git repository, the tests will fail if the repository has uncommitted changes (unless `DBXGITREPO` is unset).

### Execute all tests

To run the full suite of tests from the package root:

```bash
python -m unittest discover tests
```

### Execute specific tests

To run only the remote functionality tests:

```bash
python -m unittest tests/test_remote.py
```

## Features

- **Remote execution**: Use the `remote()` function to instantiate a remote dbx interpreter via Ray.
- **Parallel processing**: Use `RemoteCallableExecutor` to execute tasks in parallel across distributed workers.
- **Data handling**: Structured datablocks for tracking experiments and results.
- **Nested Proxying**: Transparently interact with remote objects as if they were local.
