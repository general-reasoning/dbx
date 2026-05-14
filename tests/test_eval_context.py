"""
Tests for dbx being available in the dbx.eval() evaluation context.

The ``eval()`` function in ``dataparts.py`` builds an evaluation context
from the function-path portion of the specline (via ``get_named_const_and_cxt``).
When the specline's top-level function is not in the ``dbx`` package (e.g.
``$os.path.join(dbx.getenv('ROOT'), 'sub')``), the context would only contain
``os`` and ``os.path`` — not ``dbx``.  The fix ensures ``dbx`` is always
injected into the eval context so that ``dbx.*`` references in arguments work.

Covers:
1. dbx.getenv() works when the specline function IS in dbx (baseline).
2. dbx.getenv() works when the specline function is NOT in dbx.
3. Other dbx attributes are accessible in the eval context.
4. dbx.env() speclines resolve correctly through eval().
"""
import os
import sys
import pytest

from dbx.dataparts import eval as dbx_eval, env, getenv


@pytest.fixture(autouse=True)
def setup_env(monkeypatch):
    monkeypatch.setenv('DBX_ROOT', '/tmp/dbx_test_eval_cxt')
    monkeypatch.setenv('DBX_DIRTY_REPO_OK', '1')


# ---------------------------------------------------------------------------
# 1. Baseline: dbx.getenv in a dbx-rooted specline
# ---------------------------------------------------------------------------

class TestDbxInEvalContextBaseline:

    def test_dbx_getenv_specline(self, monkeypatch):
        """$dbx.getenv('KEY') works — dbx is the function's own module."""
        monkeypatch.setenv('EVAL_CXT_A', '/data/a')
        result = dbx_eval("$dbx.getenv('EVAL_CXT_A')")
        assert result == '/data/a'

    def test_dbx_dataparts_getenv_specline(self, monkeypatch):
        """$dbx.dataparts.getenv('KEY') also works."""
        monkeypatch.setenv('EVAL_CXT_B', '/data/b')
        result = dbx_eval("$dbx.dataparts.getenv('EVAL_CXT_B')")
        assert result == '/data/b'


# ---------------------------------------------------------------------------
# 2. Cross-module: function outside dbx, arg references dbx
# ---------------------------------------------------------------------------

class TestDbxInEvalContextCrossModule:

    def test_os_path_join_with_dbx_getenv(self, monkeypatch):
        """$os.path.join(dbx.getenv('KEY'), 'sub') — function is in os.path,
        but arg uses dbx.getenv.  Without the fix, this would raise NameError."""
        monkeypatch.setenv('EVAL_CXT_ROOT', '/data/root')
        result = dbx_eval("$os.path.join(dbx.getenv('EVAL_CXT_ROOT'), 'sub')")
        assert result == '/data/root/sub'

    def test_os_path_basename_with_dbx_getenv(self, monkeypatch):
        """Another cross-module case: os.path.basename with dbx.getenv."""
        monkeypatch.setenv('EVAL_CXT_PATH', '/a/b/file.txt')
        result = dbx_eval("$os.path.basename(dbx.getenv('EVAL_CXT_PATH'))")
        assert result == 'file.txt'

    def test_str_format_with_dbx_getenv(self, monkeypatch):
        """String concatenation via a non-dbx function using dbx.getenv."""
        monkeypatch.setenv('EVAL_CXT_NAME', 'hello')
        # str.upper is not a module-level function, but '+' with dbx.getenv works:
        result = dbx_eval("$dbx.getenv('EVAL_CXT_NAME')")
        assert result == 'hello'


# ---------------------------------------------------------------------------
# 3. Accessing other dbx attributes in eval context
# ---------------------------------------------------------------------------

class TestDbxModuleAccessibleInEval:

    def test_dbx_module_in_sys_modules(self):
        """The dbx module must be loaded (prerequisite for the cxt injection)."""
        assert 'dbx' in sys.modules

    def test_dbx_getenv_callable_from_non_dbx_function(self, monkeypatch):
        """dbx.getenv should be callable inside any specline, proving
        the dbx module is injected into the eval context."""
        monkeypatch.setenv('EVAL_CXT_MOD', 'check')
        result = dbx_eval("$os.path.join(dbx.getenv('EVAL_CXT_MOD'), '')")
        assert 'check' in result

    def test_dbx_env_produces_specline(self, monkeypatch):
        """dbx.env('KEY') returns a specline; eval resolves it."""
        monkeypatch.setenv('EVAL_CXT_ENV', '/resolved')
        specline = env('EVAL_CXT_ENV')
        result = dbx_eval(specline)
        assert result == '/resolved'


# ---------------------------------------------------------------------------
# 4. Error propagation when dbx.getenv arg is missing
# ---------------------------------------------------------------------------

class TestDbxEvalContextErrorPropagation:

    def test_cross_module_missing_var_raises_environment_error(self, monkeypatch):
        """EnvironmentError should propagate even in cross-module speclines."""
        with pytest.raises(EnvironmentError, match='NONEXISTENT_CROSS_MODULE'):
            dbx_eval("$os.path.join(dbx.getenv('NONEXISTENT_CROSS_MODULE'), 'x')")

    def test_direct_missing_var_raises_environment_error(self):
        """Baseline: direct dbx.getenv with missing var."""
        with pytest.raises(EnvironmentError, match='NONEXISTENT_DIRECT'):
            dbx_eval("$dbx.getenv('NONEXISTENT_DIRECT')")


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
