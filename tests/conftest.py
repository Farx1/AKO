"""Shared pytest fixtures."""

from __future__ import annotations

import pytest

from ako.dispatcher.selector import reset_selector


@pytest.fixture(autouse=True, scope="session")
def isolated_kernel_cache(tmp_path_factory):
    """Keep the test suite away from the developer's ``~/.ako`` cache."""
    cache_dir = tmp_path_factory.mktemp("ako-cache")
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("AKO_CACHE_PATH", str(cache_dir / "kernel_cache.json"))
        reset_selector()
        yield
    reset_selector()
