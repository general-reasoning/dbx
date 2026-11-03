"""What a Kubernetes Job needs to run on an Anyscale image, as from an Anyscale workspace: `dbx.kube`'s `anyscale=True`.

An Anyscale image runs as its `ray` user (uid 1000, gid 100), keeps Ray in its own venv
(`/home/ray/venv`), and its Ray runtime-env hook snapshots the current directory as the working
directory of every task unless one is given: `working_dir` says which, `$ANYSCALE_WORKING_DIR` when
set. None of it is needed off Anyscale, which is why `dbx.kube` asks for it, never assumes it.
"""
import os


__all__ = ['IMAGE', 'RAY', 'RUN_AS', 'WORKING_DIR', 'present', 'job_options', 'working_dir']


# The image of the workspace this process runs in; None off Anyscale.
IMAGE = os.environ.get('ANYSCALE_RAY_IMAGE_URI')
# The image's Ray, which a venv layered on the image imports but has no console script of.
RAY = '/home/ray/venv/bin/ray'
# The image's `ray` user: uid, gid.
RUN_AS = (1000, 100)
# The Ray working directory a Job sets, so the runtime-env hook snapshots an empty directory, not the Job's cwd.
WORKING_DIR = '/tmp/ray_wd'


def present() -> bool:
    """Whether this process runs in an Anyscale workspace or cluster."""
    return 'ANYSCALE_WORKSPACE_ID' in os.environ or 'ANYSCALE_RAY_IMAGE_URI' in os.environ


def job_options() -> dict:
    """The `dbx.kube.job` options an Anyscale image needs: its image, user, Ray, and Ray working directory."""
    return dict(image=IMAGE, run_as=RUN_AS, ray_bin=RAY,
                env={'ANYSCALE_WORKING_DIR': WORKING_DIR},
                setup=['mkdir -p "$ANYSCALE_WORKING_DIR"'])


def working_dir(default: str | None = None) -> str:
    """The `runtime_env['working_dir']` to give Ray: `$ANYSCALE_WORKING_DIR`, else *default*, else `~/default`.

    On Anyscale, a Ray task given no working directory gets a snapshot of the driver's current one;
    an explicit one stops that.
    """
    return os.environ.get('ANYSCALE_WORKING_DIR', default or os.path.expanduser('~/default'))
