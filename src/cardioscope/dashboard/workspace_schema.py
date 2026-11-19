"""Additive workspace schema, separate from the verified experiment catalog."""

from __future__ import annotations

from pathlib import Path
from sqlite3 import Connection as SQLiteConnection
from typing import Any

from sqlalchemy import (
    JSON,
    BigInteger,
    CheckConstraint,
    Column,
    Float,
    ForeignKey,
    Index,
    Integer,
    MetaData,
    String,
    Table,
    UniqueConstraint,
    create_engine,
    event,
    func,
    select,
)
from sqlalchemy.engine import URL, Engine
from sqlalchemy.exc import SQLAlchemyError

from .registry import RegistryError, database_url

SCHEMA_VERSION = 1
schema = MetaData()
version = Table(
    "workspace_schema_version",
    schema,
    Column("singleton", Integer, primary_key=True),
    Column("version", Integer, nullable=False),
    CheckConstraint("singleton = 1"),
)
users = Table(
    "workspace_users",
    schema,
    Column("id", String(32), primary_key=True),
    Column("username", String(32), nullable=False, unique=True),
    Column("password_hash", String(255), nullable=False),
    Column("created_at", BigInteger, nullable=False),
)
preferences = Table(
    "workspace_preferences",
    schema,
    Column("owner_id", ForeignKey(users.c.id, ondelete="CASCADE"), primary_key=True),
    Column("settings", JSON, nullable=False),
)
sessions = Table(
    "workspace_sessions",
    schema,
    Column("token_hash", String(64), primary_key=True),
    Column("owner_id", ForeignKey(users.c.id, ondelete="CASCADE"), nullable=False, index=True),
    Column("expires_at", BigInteger, nullable=False, index=True),
)
attempts = Table(
    "workspace_auth_attempts",
    schema,
    Column("key", String(80), primary_key=True),
    Column("window", BigInteger, primary_key=True),
    Column("count", Integer, nullable=False),
    CheckConstraint("count > 0"),
)
profiles = Table(
    "workspace_fictional_profiles",
    schema,
    Column("id", String(32), primary_key=True),
    Column("owner_id", ForeignKey(users.c.id, ondelete="CASCADE"), nullable=False),
    Column("name", String(64), nullable=False),
    Column("name_key", String(192), nullable=False),
    Column("example", String(64), nullable=False),
    UniqueConstraint("owner_id", "name_key"),
)
reports = Table(
    "workspace_prediction_reports",
    schema,
    Column("id", String(32), primary_key=True),
    Column("owner_id", ForeignKey(users.c.id, ondelete="CASCADE"), nullable=False),
    Column("title", String(64), nullable=False),
    Column("profile_label", String(64), nullable=False),
    Column("example", String(64), nullable=False),
    Column("model", String(40), nullable=False),
    Column("feature_tier", String(16), nullable=False),
    Column("version", String(24), nullable=False),
    Column("artifact_digest", String(64), nullable=False),
    Column("score", Float, nullable=False),
    Column("historical_class", Integer, nullable=False),
    Column("threshold", Float, nullable=False),
    Column("created_at", BigInteger, nullable=False),
    Column("disclaimer", String(240), nullable=False),
    CheckConstraint("score >= 0 AND score <= 1"),
    CheckConstraint("threshold >= 0 AND threshold <= 1"),
    CheckConstraint("historical_class IN (0, 1)"),
)
Index("workspace_reports_owner_created", reports.c.owner_id, reports.c.created_at)
shares = Table(
    "workspace_report_shares",
    schema,
    Column("report_id", ForeignKey(reports.c.id, ondelete="CASCADE"), primary_key=True),
    Column("shared_at", BigInteger, nullable=False),
)


class WorkspaceError(RuntimeError):
    """Safe text for display without database or credential details."""


def initialize(engine: Engine) -> None:
    """Apply version 1 transactionally, including when two sessions start together."""
    with engine.begin() as conn:
        if engine.dialect.name == "sqlite":
            conn.exec_driver_sql("BEGIN IMMEDIATE")
        else:
            conn.execute(select(func.pg_advisory_xact_lock(2445202611)))
        version.create(conn, checkfirst=True)
        current = conn.scalar(select(version.c.version).where(version.c.singleton == 1))
        if current is None:
            schema.create_all(conn)
            conn.execute(version.insert().values(singleton=1, version=SCHEMA_VERSION))
        elif current != SCHEMA_VERSION:
            raise WorkspaceError("This workspace database uses an unsupported schema version.")


def open_workspace(url: URL | None = None) -> Engine:
    engine = None
    try:
        url = url if url is not None else database_url()
        sqlite = url.get_backend_name() == "sqlite"
        path = url.database
        if sqlite and path and path != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        engine = create_engine(
            url,
            hide_parameters=True,
            connect_args={"timeout": 10} if sqlite else {"connect_timeout": 5},
        )
        if sqlite:

            @event.listens_for(engine, "connect")
            def enable_foreign_keys(connection: SQLiteConnection, _: Any) -> None:
                autocommit = connection.autocommit
                connection.autocommit = True
                cursor = connection.cursor()
                try:
                    cursor.execute("PRAGMA foreign_keys = ON")
                finally:
                    cursor.close()
                    connection.autocommit = autocommit

        initialize(engine)
        return engine
    except WorkspaceError:
        if engine is not None:
            engine.dispose()
        raise
    except (SQLAlchemyError, RegistryError, OSError, ImportError) as exc:
        if engine is not None:
            engine.dispose()
        raise WorkspaceError("The personal workspace database is unavailable.") from exc
