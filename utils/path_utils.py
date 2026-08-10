import os
from typing import Optional


def resolve_path(path: Optional[str]) -> Optional[str]:
    """Expand ``~`` and ``$VAR`` in a config path.

    Every dataset root in ``tasks/*/configs/*.py`` is a plain string, so this lets the
    same config run on a cluster and on a workstation by exporting e.g.
    ``DATA_ROOT`` and writing ``data_dir='$DATA_ROOT/LUMIR25'``.

    Args:
        path: path from a config, may be ``None``.
    Returns:
        The expanded path, or ``None`` if ``path`` is ``None``.
    """
    if path is None:
        return None
    return os.path.expanduser(os.path.expandvars(str(path)))
