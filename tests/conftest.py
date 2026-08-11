"""Suite-wide isolation from the developer's environment.

A block constructed without ``url=`` falls back to ``DBX_ROOT``, so a test
that leaves it out addresses whatever root the shell happens to point at --
which for anyone with a real one configured is a LIVE DATALAKE. Those tests
then need that lake's driver installed (``adlfs`` for ``abfss://``, absent
from an env that only has what ``dbx.yml`` asks for) and, worse, would write
to it if it were.

So the root is redirected here, before any test module is imported and
therefore before any block is constructed, to a temporary directory of this
run's own. A module that sets its own ``DBX_ROOT`` still wins; a module using
``os.environ.setdefault`` now finds this one rather than the ambient lake,
which is the point.

``DBX_DIRTY_REPO_OK`` is defaulted for the same reason: the suite is about
dbx's behaviour, not about the state of the checkout it runs from. The tests
that are about that state (``test_dirty_check``) set and unset it themselves.
"""
import os
import shutil
import tempfile

_TESTROOT = tempfile.mkdtemp(prefix='dbx-tests-')

# Assigned at import, not from a fixture: conftest is imported before the test
# modules are, so this is in place even for a block built at module scope.
os.environ['DBX_ROOT'] = _TESTROOT
os.environ.pop('DBX_URL', None)          # the alias DBX_ROOT is read ahead of
os.environ['DBX_LOCAL'] = os.path.join(_TESTROOT, 'local')
os.environ.setdefault('DBX_DIRTY_REPO_OK', '1')


def pytest_unconfigure(config):
    shutil.rmtree(_TESTROOT, ignore_errors=True)
