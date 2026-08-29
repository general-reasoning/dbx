"""Pre-import pin bootstrap for dbx Ray workers. MUST NOT import dbx.

This file is never imported as ``dbx._pinshim_``. :func:`dbx.datablocks.remote`
copies it into a scratch directory as ``dbxpinshim.py``, ships that directory to
the workers as ``runtime_env['working_dir']``, and points
``runtime_env['worker_process_setup_hook']`` at :func:`setup`. Ray runs the hook
in each worker process BEFORE it resolves any actor class -- which is to say
before anything imports dbx.

That ordering is the whole point. dbx cannot pin itself: reaching any dbx code
means importing dbx first, and an imported module is never dislodged by a
checkout. So the bootstrap has to live outside dbx and run before it, which also
means it cannot use dbx's own ``gitclone``/``gitcheckout`` helpers -- hence plain
``subprocess`` git and nothing but the standard library.

Configured entirely through the environment, since the hook takes no arguments:

``DBX_PIN_REVISION``
    ``'dbx_rev:project_rev'``, matching the order of the sources.
``DBX_PIN_SOURCE``
    SPACE-separated clone sources, dbx first. Space rather than ``':'`` because
    a git URL contains colons (``https://…``, ``git@host:path``), which is also
    what lets a worker pin itself with no shared filesystem at all.
``DBX_PIN_NODE_ROOT``
    Where clones are cached, shared by every worker process on the node.
    Defaults to ``<tmp>/dbx-pins``.
"""
import os
import shutil
import subprocess
import sys
import tempfile


def _clone(source, revision, root):
    """Clone *source* at *revision* under *root*; return the clone directory.

    Named for the revision so that the many worker processes on one node clone
    once between them rather than once each.
    """
    name = os.path.basename(source.rstrip('/'))
    if name.endswith('.git'):
        name = name[:-len('.git')]
    target = os.path.join(root, f"{name}-{revision}")
    if os.path.isdir(target):
        return target

    os.makedirs(root, exist_ok=True)
    staging = f"{target}.staging-{os.getpid()}"
    try:
        subprocess.run(['git', 'clone', '--quiet', source, staging], check=True)
        subprocess.run(['git', '-C', staging, 'checkout', '--quiet', revision], check=True)
        try:
            os.rename(staging, target)
        except OSError:
            # Another worker on this node got there first. Its clone is at the
            # same revision as ours, so drop ours and use it.
            if not os.path.isdir(target):
                raise
            shutil.rmtree(staging, ignore_errors=True)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return target


def setup():
    """Put the pinned checkouts at the head of ``sys.path``.

    Raises rather than degrading if anything goes wrong: a worker that quietly
    came up unpinned would return confidently wrong hashes, which is the exact
    failure this machinery exists to prevent.
    """
    revision = os.environ.get('DBX_PIN_REVISION')
    sources = os.environ.get('DBX_PIN_SOURCE')
    if not revision or not sources:
        return

    revisions = revision.split(':')
    srcs = sources.split()
    if len(srcs) != len(revisions):
        raise RuntimeError(
            f"dbxpinshim: {len(srcs)} sources but {len(revisions)} revisions "
            f"({sources!r} / {revision!r})"
        )

    root = os.environ.get('DBX_PIN_NODE_ROOT') or os.path.join(tempfile.gettempdir(), 'dbx-pins')
    for source, rev in zip(srcs, revisions):
        if not source or not rev:
            continue
        sys.path.insert(0, _clone(source, rev, root))
