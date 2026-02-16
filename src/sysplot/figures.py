import os
import inspect
import matplotlib.pyplot as plt
import re

from .config import FIGURE_SIZE


def get_figsize(nrows: int = 1, ncols: int = 1, nmax: int = 2) -> tuple[float, float]:
    """Calculate figure size based on subplot grid dimensions.

    Computes the width and height for a Matplotlib figure containing a grid
    of subplots. Each subplot dimension is scaled by the base ``FIGURE_SIZE``,
    with an optional maximum scale factor to prevent excessively large figures.

    Args:
        nrows (int, optional): Number of subplot rows. Must be >= 1. Default is 1.
        ncols (int, optional): Number of subplot columns. Must be >= 1. Default is 1.
        nmax (int, optional): Maximum scale factor per dimension. Caps the
            multiplication of nrows/ncols to prevent oversized figures. Must be >= 1.
            Default is 2.

    Returns:
        tuple[float, float]: Figure dimensions as (width, height) in inches.

    Raises:
        ValueError: If nrows, ncols, or nmax is not a positive integer.

    Examples:
        >>> get_figsize(nrows=1, ncols=2)
        (14.0, 5.0)  # FIGURE_SIZE = (7, 5)
        >>> get_figsize(nrows=1, ncols=5, nmax=2)
        (14.0, 5.0)  # Capped at 2x FIGURE_SIZE
    """
    if not isinstance(nrows, int) or nrows < 1:
        raise ValueError(f"nrows must be a positive integer, got {nrows!r}")
    if not isinstance(ncols, int) or ncols < 1:
        raise ValueError(f"ncols must be a positive integer, got {ncols!r}")
    if not isinstance(nmax, int) or nmax < 1:
        raise ValueError(f"nmax must be a positive integer, got {nmax!r}")
    
    width = min(ncols * FIGURE_SIZE[0], nmax * FIGURE_SIZE[0])
    height = min(nrows * FIGURE_SIZE[1], nmax * FIGURE_SIZE[1])
    return (width, height)


# ___________________________________________________________________
#  Save Figures as a File


def save_current_figure(
    chapter: int, 
    number: int, 
    language: str, 
    suffix: str | None = None, 
    folder: str = "images", 
    fmt: str = "pdf",
    transparent: bool = False
) -> str:
    """Save the current Matplotlib figure with standardized naming.

    Saves the active figure to a file adjacent to it's calling script, using
    a structured naming convention: ``Bild_{chapter}_{number}_{script}{_suffix}.{fmt}``.
    The output directory is ``{script_dir}/{folder}/{language}/``.

    Args:
        chapter (int): Chapter number in the lecture notes. Must be >= 0.
        number (int): Figure number within the chapter. Must be >= 0.
        language (str): Language code for output directory (e.g., "de", "en").
            Must be a non-empty string.
        suffix (str, optional): Additional identifier appended to filename.
            Useful for variants of the same figure. Default is None.
        folder (str, optional): Output directory name relative to the calling
            script's location. Default is "images".
        fmt (str, optional): File format extension (e.g., "svg", "pdf", "png").
            Default is "pdf".
        transparent (bool, optional): If True, saves with transparent background.
            Default is False.

    Returns:
        str: Absolute path to the saved figure file.

    Raises:
        ValueError: If chapter, number, language, folder, or fmt are invalid.
        TypeError: If suffix is not a string or None.
        RuntimeError: If no active figure exists or if called from interactive shell.

    Note:
        - The calling script's filename is sanitized to remove invalid path
          characters (only alphanumeric, '.', '_', '-' are retained).
        - Output directories are created automatically if they don't exist.
        - This function **must** be called from a Python script file, not from
          an interactive Python/IPython shell.

    Example:
        >>> # In file: plots/myfile.py
        >>> plt.plot([1, 2, 3], [1, 4, 9])
        >>> save_current_figure(chapter=1, number=1, language="en")
        'C:/path/to/plots/images/en/Bild_1_1_myfile.pdf'
    """
    if not isinstance(chapter, int) or chapter < 0:
        raise ValueError(f"'chapter' must be a non-negative integer, got {chapter!r}")
    if not isinstance(number, int) or number < 0:
        raise ValueError(f"'number' must be a non-negative integer, got {number!r}")
    if suffix is not None and not isinstance(suffix, str):
        raise TypeError(f"'suffix' must be a string or None, got {type(suffix).__name__}")
    if not isinstance(language, str) or not language:
        raise ValueError(f"'language' must be a non-empty string, got {language!r}")
    if not isinstance(folder, str) or not folder:
        raise ValueError(f"'folder' must be a non-empty string, got {folder!r}")
    if not isinstance(fmt, str) or not fmt:
        raise ValueError(f"'fmt' must be a non-empty string, got {fmt!r}")
    if plt.get_fignums() == []:
        raise RuntimeError("No active Matplotlib figure exists to save.")

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
