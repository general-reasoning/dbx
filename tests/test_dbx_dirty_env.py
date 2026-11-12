"""DBX_DIRTY=True stands for DBX_USE_WORK_REPO=False and DBX_DIRTY_REPO_OK=True, set when dbx.databits is imported."""
import os
import subprocess
import sys

import pytest

_PROBE_ = ("import os, dbx.databits; "
           "print(os.environ.get('DBX_USE_WORK_REPO'), os.environ.get('DBX_DIRTY_REPO_OK'))")


def _probe_(**env) -> list[str]:
    environ = {k: v for k, v in os.environ.items() if k not in ('DBX_DIRTY', 'DBX_USE_WORK_REPO', 'DBX_DIRTY_REPO_OK')}
    environ.update(env)
    out = subprocess.run([sys.executable, '-c', _PROBE_], env=environ, capture_output=True, text=True, check=True)
    return out.stdout.split()


@pytest.mark.parametrize('value', ['True', 'true', '1'])
def test_dirty_sets_both(value):
    assert _probe_(DBX_DIRTY=value) == ['False', 'True']


def test_dirty_overrides_a_profiles_work_repo():
    assert _probe_(DBX_DIRTY='True', DBX_USE_WORK_REPO='True') == ['False', 'True']


def test_without_it_nothing_changes():
    # DBX_USE_WORK_REPO=False here: True would clone the repos on import, which a dirty one refuses.
    assert _probe_(DBX_USE_WORK_REPO='False') == ['False', 'None']
    assert _probe_(DBX_DIRTY='False', DBX_USE_WORK_REPO='False') == ['False', 'None']
