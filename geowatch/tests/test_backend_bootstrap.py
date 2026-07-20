from pathlib import Path
from typing import cast

import pytest
from sqlalchemy.engine import Engine

from src.backend.bootstrap import (
    DatabaseBootstrapError,
    initialize_database,
    load_migration_statements,
)


class FakeConnection:
    def __init__(
        self,
    ) -> None:
        self.statements: list[str] = []

    def exec_driver_sql(
        self,
        statement: str,
    ) -> None:
        self.statements.append(
            statement
        )


class FakeTransaction:
    def __init__(
        self,
        connection: FakeConnection,
    ) -> None:
        self.connection = connection

    def __enter__(
        self,
    ) -> FakeConnection:
        return self.connection

    def __exit__(
        self,
        exception_type: object,
        exception: object,
        traceback: object,
    ) -> None:
        return None


class FakeEngine:
    def __init__(
        self,
    ) -> None:
        self.connection = FakeConnection()
        self.begin_calls = 0

    def begin(
        self,
    ) -> FakeTransaction:
        self.begin_calls += 1

        return FakeTransaction(
            self.connection
        )


def test_loads_migration_statements(
    tmp_path: Path,
) -> None:
    path = tmp_path / "migration.sql"

    path.write_text(
        "CREATE TABLE first_table (id INTEGER);"
        "CREATE INDEX first_index ON first_table (id);",
        encoding="utf-8",
    )

    assert load_migration_statements(
        path
    ) == (
        "CREATE TABLE first_table (id INTEGER)",
        "CREATE INDEX first_index ON first_table (id)",
    )


def test_rejects_missing_migration(
    tmp_path: Path,
) -> None:
    with pytest.raises(
        DatabaseBootstrapError,
        match="does not exist",
    ):
        load_migration_statements(
            tmp_path / "missing.sql"
        )


def test_rejects_empty_migration(
    tmp_path: Path,
) -> None:
    path = tmp_path / "empty.sql"

    path.write_text(
        "   ",
        encoding="utf-8",
    )

    with pytest.raises(
        DatabaseBootstrapError,
        match="is empty",
    ):
        load_migration_statements(
            path
        )


def test_initializes_database_in_transaction(
    tmp_path: Path,
) -> None:
    path = tmp_path / "migration.sql"

    path.write_text(
        "CREATE TABLE first_table (id INTEGER);"
        "CREATE TABLE second_table (id INTEGER);",
        encoding="utf-8",
    )

    engine = FakeEngine()

    initialize_database(
        cast(
            Engine,
            engine,
        ),
        path,
    )

    assert engine.begin_calls == 1

    assert engine.connection.statements == [
        "CREATE TABLE first_table (id INTEGER)",
        "CREATE TABLE second_table (id INTEGER)",
    ]
