import os
import inspect
import matplotlib.pyplot as plt
import re

from .config import get_config


def get_figsize(nrows: int = 1, ncols: int = 1, nmax: int | None = None) -> tuple[float, float]:
    """Calculate figure dimensions for a subplot grid.

    Scales the base figure size from the active :class:`~sysplot.SysplotConfig`
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
) -> str:
    """Save the current Matplotlib figure with a standardized filename.

    Saves the active figure next to the calling script using the naming
    convention ``Bild_{chapter}_{number}_{script}{_suffix}.{fmt}``.
    The output directory ``{script_dir}/{folder}/{language}/`` is created
    automatically if it does not exist.

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
        Absolute path of the saved file.

    Note:
        Must be called from a Python script file, not an interactive shell.

    .. minigallery:: sysplot.save_current_figure
        :add-heading:
    """
    if not isinstance(chapter, int) or chapter < 0:
        raise ValueError(f"'chapter' must be a non-negative integer, got {chapter!r}")
    if not isinstance(number, int) or number < 0:
        raise ValueError(f"'number' must be a non-negative integer, got {number!r}")
    if not isinstance(language, str) or not language:
        raise ValueError(f"'language' must be a non-empty string, got {language!r}")
    if suffix is not None and not isinstance(suffix, (str, int)):
        raise TypeError(f"'suffix' must be a str, int, or None, got {type(suffix).__name__!r}")
    if not isinstance(folder, (str, type(None))) or folder is not None and not folder:
        raise ValueError(f"'folder' must be a non-empty string or None, got {folder!r}")
    if fmt is not None and (not isinstance(fmt, str) or not fmt):
        raise ValueError(f"'fmt' must be a non-empty string or None, got {fmt!r}")
    if transparent is not None and not isinstance(transparent, bool):
        raise TypeError(f"'transparent' must be a bool or None, got {type(transparent).__name__!r}")
    if plt.get_fignums() == []:
        raise RuntimeError("No active Matplotlib figure exists to save.")

    fmt = fmt if fmt is not None else get_config().figure_fmt
    folder = folder if folder is not None else get_config().savefig_folder
    transparent = transparent if transparent is not None else get_config().savefig_transparent

    # Get info about the calling script
    try:
        caller_path = inspect.stack()[1].filename
    except Exception as e:
        raise RuntimeError(
            "Failed to determine the path of the calling script. "
            "This function must be called from a Python script, not an interactive shell."
        ) from e
    
    script_name_raw = os.path.splitext(os.path.basename(caller_path))[0]
    # Sanitize the script name to avoid invalid path characters (particularly on Windows)
    script_name = re.sub(r'[^A-Za-z0-9._-]', "_", script_name_raw)
    if not script_name:
        script_name = "script"
    script_dir = os.path.dirname(os.path.abspath(caller_path))

    # Build output directory relative to the calling script's folder
    out_dir = os.path.join(script_dir, folder, language)
    os.makedirs(out_dir, exist_ok=True)

    # Build filename
    suffix_str = f"_{suffix}" if suffix else ""
    filename = f"Bild_{chapter}_{number}_{script_name}{suffix_str}.{fmt}"
    full_path = os.path.join(out_dir, filename)

    # Save figure
    plt.savefig(full_path, transparent=transparent, format=fmt)
    
    return full_path
