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

    Every script here prints its answer, and prints it last. Only that line
    is returned, because an import can put its own lines on stdout ahead of
    it: pymupdf 1.28 announces the ``fitz`` deprecation that way, so
    importing any module that reaches for ``fitz`` prefixes the answer with
    a warning the caller never asked about.

    :param script: Python source to run.
    :returns: The last non-blank line of stdout.
    :rtype: str
    """
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(script)],
        check=True,
        capture_output=True,
        text=True,
    )
    lines = [line for line in result.stdout.strip().splitlines() if line.strip()]
    return lines[-1].strip() if lines else ""


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


class TestLazyDoctrImport:
    """docTR lives in the optional ``refine`` extra; importing blackletter
    modules must not load it, and its absence must fail with a clear error."""

    def test_import_refine_does_not_load_doctr(self):
        out = _run(
            """
            import sys
            import blackletter.refine  # noqa: F401
            print('doctr' in sys.modules)
            """
        )
        assert out == "False"

    def test_import_process_does_not_load_doctr(self):
        out = _run(
            """
            import sys
            import blackletter.process  # noqa: F401
            print('doctr' in sys.modules)
            """
        )
        assert out == "False"

    def test_missing_doctr_raises_helpful_import_error(self):
        """Without the ``refine`` extra, the error must name the fix."""
        out = _run(
            """
            import sys
            sys.modules['doctr'] = None  # force ImportError even if installed
            from blackletter.refine import _get_doctr_model
            try:
                _get_doctr_model()
            except ImportError as exc:
                assert 'blackletter[refine]' in str(exc), str(exc)
                assert 'skip_doctr' in str(exc), str(exc)
                print('ImportError')
            """
        )
        assert out == "ImportError"
