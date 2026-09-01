"""Optional, dependency-free web dashboard for browsing and launching runs.

Built on the Python standard library (``http.server``) so it needs no extra
install. Launch with ``skill-factory ui`` or ``python -m skill_factory.ui``.
"""

from skill_factory.ui.server import serve

__all__ = ["serve"]
