"""`import dbx` without the [streaming] extra: torch and mosaicml-streaming."""
import subprocess
import sys

import pytest


@pytest.mark.pinned
def test_import_dbx_without_torch_or_streaming():
    """The core -- blocks, journals, pipelines -- imports without the data modules' extras.

    A program that only reads journals -- `dbx.pipejournal()` from a viewer --
    must not need torch. `sys.modules[name] = None` makes a package
    uninstalled as far as both `import` and `importlib.util.find_spec` are
    concerned, in a fresh interpreter.
    """
    code = (
        "import sys\n"
        "sys.modules['torch'] = None\n"
        "sys.modules['streaming'] = None\n"
        "import dbx\n"
        "assert callable(dbx.pipejournal) and callable(dbx.datajournal)\n"
        "assert dbx.Datablock and dbx.Pipeline and dbx.DataJournal\n"
        "assert 'dbx.datatables' not in sys.modules\n"
    )
    proc = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, proc.stderr


def test_import_dbx_with_streaming_has_the_data_modules():
    pytest.importorskip('torch')
    pytest.importorskip('streaming')
    import dbx
    assert dbx.Datatable and dbx.Featuretable
