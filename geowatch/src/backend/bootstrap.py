from __future__ import annotations

from pathlib import Path

from sqlalchemy.engine import Engine


DEFAULT_MIGRATION_PATH = Path(
    "migrations/001_create_changes_table.sql"
)


class DatabaseBootstrapError(RuntimeError):
    pass


def load_migration_statements(
    path: Path = DEFAULT_MIGRATION_PATH,
) -> tuple[str, ...]:
    selected_path = path.expanduser().resolve()

    if not selected_path.is_file():
        raise DatabaseBootstrapError(
            f"Migration file does not exist: {selected_path}"
        )

    text = selected_path.read_text(
        encoding="utf-8"
    )

    statements = tuple(
        statement.strip()
        for statement in text.split(";")
        if statement.strip()
    )

    if not statements:
        raise DatabaseBootstrapError(
            f"Migration file is empty: {selected_path}"
        )

    return statements


def initialize_database(
    engine: Engine,
    migration_path: Path = DEFAULT_MIGRATION_PATH,
) -> None:
    statements = load_migration_statements(
        migration_path
    )

    with engine.begin() as connection:
        for statement in statements:
            connection.exec_driver_sql(
                statement
            )
