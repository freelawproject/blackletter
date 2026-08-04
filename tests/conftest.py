"""Fixtures shared across the geometry tests.

Rasterising a page of type is the slowest thing these tests do, and several
of them want the same page, so the expensive ones are built once.
"""

from __future__ import annotations

import pytest

from tests.pdf_fixtures import write_hairline_column_page


@pytest.fixture(scope="module")
def hairline_pdf(tmp_path_factory):
    """A two-column page of type with a gutter thinner than one pixel.

    :returns: ``(path, left_band, right_band)``, the bands in PDF points.
    """
    directory = tmp_path_factory.mktemp("hairline")
    path = directory / "hairline.pdf"
    left, right = write_hairline_column_page(path, gutter=0.3, tmp_dir=directory)
    return path, left, right
