"""Regression tests for lazy ultralytics loading.

The blackletter package is consumed by CPU-only callers (e.g. the
scanning daemon when ``RUNPOD_ENABLED`` is true) that should not be
forced to load ``ultralytics`` and ``torch`` at import time. These
tests run each check in a fresh subprocess so a previous test's
imports can't pollute ``sys.modules`` and mask a regression.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap


def _run(script: str) -> str:
    """Execute ``script`` in a fresh Python subprocess.

    :param script: Python source to run.
    :returns: Captured stdout.
    :rtype: str
    """
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(script)],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


class TestLazyUltralyticsImport:
    """Importing blackletter must not transitively load ultralytics."""

    def test_import_package_does_not_load_ultralytics(self):
        out = _run(
            """
            import sys
            import blackletter  # noqa: F401
            print('ultralytics' in sys.modules)
            """
        )
        assert out == "False"

    def test_import_models_does_not_load_ultralytics(self):
        out = _run(
            """
            import sys
            import blackletter.models  # noqa: F401
            print('ultralytics' in sys.modules)
            """
        )
        assert out == "False"

    def test_import_api_does_not_load_ultralytics(self):
        out = _run(
            """
            import sys
            import blackletter.api  # noqa: F401
            print('ultralytics' in sys.modules)
            """
        )
        assert out == "False"

    def test_import_scanner_does_not_load_ultralytics(self):
        out = _run(
            """
            import sys
            import blackletter.scanner  # noqa: F401
            print('ultralytics' in sys.modules)
            """
        )
        assert out == "False"

    def test_import_process_does_not_load_ultralytics(self):
        out = _run(
            """
            import sys
            import blackletter.process  # noqa: F401
            print('ultralytics' in sys.modules)
            """
        )
        assert out == "False"

    def test_lazy_export_resolves_without_loading_ultralytics(self):
        """Accessing ``blackletter.validate`` should not pull in YOLO."""
        out = _run(
            """
            import sys
            import blackletter
            fn = blackletter.validate
            assert callable(fn)
            print('ultralytics' in sys.modules)
            """
        )
        assert out == "False"

    def test_unknown_attribute_raises_attribute_error(self):
        out = _run(
            """
            import blackletter
            try:
                blackletter.does_not_exist
            except AttributeError as exc:
                print('AttributeError')
            """
        )
        assert out == "AttributeError"
