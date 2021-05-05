import logging
from pathlib import Path

from ignite.distributed import one_rank_only

try:
    from git import Repo
except ImportError:
    Repo = None

from matches.loop import Loop

from .callback import Callback

LOG = logging.getLogger(__name__)


def _is_dev_dir():
    try:
        Repo(search_parent_directories=True)
    except Exception:
        LOG.info("Git repo is not found. Assuming dev mode")
        return True
    return False


def _ensure_clean_worktree() -> str:
    r = Repo(search_parent_directories=True)

    diff_is_empty = True
    # Check nothing is staged
    for _ in r.index.diff(r.head.commit):
        diff_is_empty = False
        break

    # Check nothing in work tree
    for _ in r.index.diff(None):
        diff_is_empty = False
        break

    # Check no files untracked
    for _ in r.untracked_files:
        diff_is_empty = False
        break

    if not diff_is_empty:
        raise Exception("There are some uncommited changes! Aborting...")

    assert r.active_branch.name.startswith("exp"), "Branch name must start with 'exp'"

    return r.head.reference.commit.hexsha


def _write_git_ref(logdir, ref):
    log = Path(logdir)
    log.mkdir(parents=True, exist_ok=True)
    (log / "git-ref.txt").write_text(ref)


class EnsureWorkdirCleanOrDevMode(Callback):

    @one_rank_only()
    def on_train_start(self, loop: "Loop"):
        if Repo is None:
            raise Exception("GitPython not found. Run pip install GitPython")
        if _is_dev_dir():
            if loop._loader_override.mode == "disabled":
                raise Exception("Long experiments must be run from full_dir")
            ref = "dev-mode"
        else:
            ref = _ensure_clean_worktree()

        _write_git_ref(loop.logdir, ref)
