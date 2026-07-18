import pytest

from src.backend.database import (
    DatabaseConfigurationError,
    normalize_database_url,
)


def test_normalizes_render_database_url() -> None:
    result = normalize_database_url(
        "postgresql://user@host:5432/geowatch"
    )

    assert result == (
        "postgresql+psycopg://"
        "user@host:5432/geowatch"
    )


def test_preserves_psycopg_database_url() -> None:
    url = (
        "postgresql+psycopg://"
        "user@host:5432/geowatch"
    )

    assert normalize_database_url(
        url
    ) == url


def test_rejects_unsupported_database_url() -> None:
    with pytest.raises(
        DatabaseConfigurationError,
        match="must use postgresql",
    ):
        normalize_database_url(
            "sqlite:///geowatch.db"
        )
