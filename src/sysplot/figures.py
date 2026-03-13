import inspect
import os
import matplotlib.pyplot as plt
import re
from pathlib import Path

from .config import get_config


def get_figsize(
    nrows: int = 1, ncols: int = 1, nmax: int | None = None
) -> tuple[float, float]:
    """Calculate figure dimensions for a subplot grid.

    Scales the base figure size from the active :class:`~sysplot.SysplotConfig.figure_size`
    by the number of rows and columns, capping each dimension at ``nmax`` times
    the base size to prevent excessively large figures.

    Args:
        nrows: Number of subplot rows. Must be >= 1.
        ncols: Number of subplot columns. Must be >= 1.
        nmax: Maximum scale factor per dimension. Defaults to
            :attr:`~sysplot.SysplotConfig.figure_size_nmax`.

    Returns:
        Figure dimensions as ``(width, height)`` in inches.

    .. minigallery:: sysplot.get_figsize
        :add-heading:
    """
    nmax = nmax or get_config().figure_size_nmax

    if not isinstance(nrows, int) or nrows < 1:
        raise ValueError(f"nrows must be a positive integer, got {nrows!r}")
    if not isinstance(ncols, int) or ncols < 1:
        raise ValueError(f"ncols must be a positive integer, got {ncols!r}")
    if not isinstance(nmax, int) or nmax < 1:
        raise ValueError(f"nmax must be a positive integer, got {nmax!r}")

    FIGSIZE = get_config().figure_size

    width = min(ncols * FIGSIZE[0], nmax * FIGSIZE[0])
    height = min(nrows * FIGSIZE[1], nmax * FIGSIZE[1])
    return (width, height)


# ___________________________________________________________________
#  Save Figures as a File


def save_current_figure(
    chapter: int,
    number: int,
    language: str,
    suffix: str | int | None = None,
    folder: str | None = None,
    fmt: str | None = None,
    transparent: bool | None = None,
) -> Path | None:
    """Save the current Matplotlib figure with a standardized filename.

    Saves the active figure next to the calling script using the naming
    convention ``Bild_{chapter}_{number}_{script}{_suffix}.{fmt}``.
    The output directory ``{script_dir}/{folder}/{language}/`` is created
    automatically if it does not exist.

    Warning:
        You can disbable all filesaves by this function by setting the environment variable ``SYSPLOT_DISABLE_SAVE=1`` (or any non-empty value).
        This is useful for CI runs or test environments where writing output files is undesirable.

    Args:
        chapter: Chapter number. Must be >= 0.
        number: Figure number within the chapter. Must be >= 0.
        language: Language subdirectory (e.g., ``"de"`` or ``"en"``).
        suffix: Optional variant identifier appended to the filename.
        folder: Subdirectory name relative to the calling script's location.
            Defaults to :attr:`~sysplot.SysplotConfig.savefig_folder`.
        fmt: File format (e.g., ``"pdf"``, ``"svg"``, ``"png"``). Defaults to
            :attr:`~sysplot.SysplotConfig.figure_fmt`.
        transparent: Whether to save with a transparent background. Defaults to
            :attr:`~sysplot.SysplotConfig.savefig_transparent`.

    Returns:
        Path of the saved file, or ``None`` if saving is disabled via the
        ``SYSPLOT_DISABLE_SAVE`` environment variable.

    Note:
        Must be called from a Python script file, not an interactive shell.

        Set the environment variable ``SYSPLOT_DISABLE_SAVE=1`` (or any
        non-empty value) to skip all saves without modifying user code.
        This is useful for CI runs or test environments where writing
        output files is undesirable.

    .. minigallery:: sysplot.save_current_figure
        :add-heading:
    """
    if os.getenv("SYSPLOT_DISABLE_SAVE", False):
        return None

    if not isinstance(chapter, int) or chapter < 0:
        raise ValueError(f"'chapter' must be a non-negative integer, got {chapter!r}")
    if not isinstance(number, int) or number < 0:
        raise ValueError(f"'number' must be a non-negative integer, got {number!r}")
    if not isinstance(language, str) or not language:
        raise ValueError(f"'language' must be a non-empty string, got {language!r}")
    if suffix is not None and not isinstance(suffix, (str, int)):
        raise TypeError(
            f"'suffix' must be a str, int, or None, got {type(suffix).__name__!r}"
        )
    if not isinstance(folder, (str, type(None))) or folder is not None and not folder:
        raise ValueError(f"'folder' must be a non-empty string or None, got {folder!r}")
    if fmt is not None and (not isinstance(fmt, str) or not fmt):
        raise ValueError(f"'fmt' must be a non-empty string or None, got {fmt!r}")
    if transparent is not None and not isinstance(transparent, bool):
        raise TypeError(
            f"'transparent' must be a bool or None, got {type(transparent).__name__!r}"
        )
    if plt.get_fignums() == []:
        raise RuntimeError("No active Matplotlib figure exists to save.")

    fmt = fmt if fmt is not None else get_config().figure_fmt
    folder = folder if folder is not None else get_config().savefig_folder
    transparent = (
        transparent if transparent is not None else get_config().savefig_transparent
    )

    # Get info about the calling script
    try:
        caller_path = Path(inspect.stack()[1].filename).resolve()
    except Exception as e:
        raise RuntimeError(
            "Failed to determine the path of the calling script. "
            "This function must be called from a Python script, not an interactive shell."
        ) from e

    script_name_raw = caller_path.stem
    # Sanitize the script name to avoid invalid path characters (particularly on Windows)
    script_name = re.sub(r"[^A-Za-z0-9._-]", "_", script_name_raw) or "script"

    # Build output directory relative to the calling script's folder
    out_dir = caller_path.parent / folder / language
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build filename
    suffix_str = f"_{suffix}" if suffix else ""
    full_path = out_dir / f"Bild_{chapter}_{number}_{script_name}{suffix_str}.{fmt}"

    # Save figure
    plt.savefig(full_path, transparent=transparent, format=fmt)

    return full_path
