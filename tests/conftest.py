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


# --- pinned tests ----------------------------------------------------------
#
# `@pytest.mark.pinned` marks a test that states an INVARIANT rather than the
# shape the code happens to have. The suite holds both kinds, and they call for
# opposite responses when they fail:
#
#   an ordinary test  -- may encode a decision that has since changed, so the
#                        test is sometimes the thing to update
#   a pinned test     -- states something that must remain true, so a failure
#                        is a regression and the CODE is what changes
#
# Nothing enforces this; a marker cannot. What it does is make the distinction
# visible at the moment it matters -- in the failure output -- rather than
# leaving it to whoever is looking to guess which kind they are holding.

def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "pinned: an invariant; a failure is a regression, so fix the code not the test",
    )


def pytest_runtest_makereport(item, call):
    if call.when == 'call' and call.excinfo is not None and item.get_closest_marker('pinned'):
        item.config.stash.setdefault(_PINNED_FAILURES, []).append(item.nodeid)


_PINNED_FAILURES = __import__('pytest').StashKey[list]()


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    failed = config.stash.get(_PINNED_FAILURES, [])
    if not failed:
        return
    terminalreporter.write_sep('=', 'PINNED TEST FAILURES', red=True, bold=True)
    terminalreporter.write_line(
        "These state invariants, not current behaviour. A failure here is a "
        "regression:", red=True)
    terminalreporter.write_line(
        "fix the code -- do not edit the test to agree with it.", red=True)
    for nodeid in failed:
        terminalreporter.write_line(f"  {nodeid}")
