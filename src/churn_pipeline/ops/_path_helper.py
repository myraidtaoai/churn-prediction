"""Add the churn_pipeline root to sys.path for shared module imports.

Databricks ``spark_python_task`` runs scripts via ``exec()`` — ``__file__``
is not defined in the executed script, and ``sys.path[0]`` may not point
to the script's directory.  This helper is a *regular* imported module,
so its ``__file__`` is always valid.  Import it as a side-effect::

    import _path_helper  # noqa: F401
"""

import os
import sys

_PKG_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PKG_ROOT not in sys.path:
    sys.path.insert(0, _PKG_ROOT)
