"""Rewriting ``key = value`` assignments in upstream config-file text.

Used to update config files (TOML/INI-style) for tools that do not expose CLI overrides.
"""

from __future__ import annotations

import re
from collections.abc import Mapping


def apply_assignments(text: str, assignments: Mapping[str, str]) -> str:
    """Update ``key = value`` lines in ``text`` and return the result.

    Operates purely on the string (no file I/O). For each ``key`` in ``assignments``,
    the text after the ``=`` on the matching line is replaced with the given value,
    written exactly as given (no quoting is added, so the caller must include any quotes
    the field needs). Keys absent from ``text`` are left unchanged.

    Parameters
    ----------
    text : str
        The config file contents.
    assignments : Mapping[str, str]
        Maps each config key to its replacement value (the text after the ``=``).

    Examples
    --------
    >>> apply_assignments("vmec_file = old", {"vmec_file": '"new.nc"'})
    'vmec_file = "new.nc"'
    """
    for key, value in assignments.items():
        text = re.sub(
            rf"^({re.escape(key)}[ \t]*=[ \t]*)[^\r\n]*",
            lambda m, v=value: f"{m.group(1)}{v}",
            text,
            flags=re.MULTILINE,
        )
    return text
