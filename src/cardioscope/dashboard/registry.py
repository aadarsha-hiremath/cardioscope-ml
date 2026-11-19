"""Idempotent import of aggregate evidence into SQLite or PostgreSQL."""

from __future__ import annotations

import os
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import JSON, Column, DateTime, MetaData, String, Table, create_engine, select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.engine import URL, Engine, make_url
from sqlalchemy.exc import SQLAlchemyError

from cardioscope.dashboard.evidence import ROOT, Evidence
from cardioscope.experiment import digest

schema = MetaData()
experiments = Table(
    "experiments",
    schema,
    Column("id", String(64), primary_key=True),
    Column("model", String(24), nullable=False),
    Column("tier", String(24), nullable=False),
    Column("evaluation", String(24), nullable=False),
    Column("hospital", String(32), nullable=False),
    Column("metadata", JSON, nullable=False),
    Column("imported_at", DateTime(timezone=True), nullable=False),
)


class RegistryError(RuntimeError):
    """Database operation failed; messages exclude URLs and credentials."""


def database_url() -> URL:
    configured = os.environ.get("CARDIOSCOPE_DATABASE_URL")
    try:
        url = (
            make_url(configured)
            if configured
            else URL.create("sqlite", database=str(ROOT / "data/local/experiments.db"))
        )
        if url.drivername == "postgresql":
            url = url.set(drivername="postgresql+psycopg")
        if url.drivername not in {"sqlite", "postgresql+psycopg"}:
            raise ValueError("Unsupported backend")
        return url
    except (ValueError, SQLAlchemyError) as exc:
        raise RegistryError("Use a SQLite or postgresql+psycopg database URL.") from exc


def open_registry() -> Engine:
    engine = None
    try:
        url = database_url()
        if "CARDIOSCOPE_DATABASE_URL" not in os.environ:
            (ROOT / "data/local").mkdir(parents=True, exist_ok=True)
        engine = create_engine(
            url,
            echo=False,
            hide_parameters=True,
            connect_args={"connect_timeout": 5} if url.get_backend_name() == "postgresql" else {},
        )
        schema.create_all(engine)
        return engine
    except (SQLAlchemyError, OSError, ImportError) as exc:
        if engine is not None:
            engine.dispose()
        raise RegistryError(
            "Registry unavailable. Check the configured database and optional "
            "PostgreSQL driver. Connection details are hidden."
        ) from exc


def records(e: Evidence) -> list[dict[str, Any]]:
    rows = []
    meta = e.pooled["selection"]["metadata"]
    for r in e.pooled["evaluation"]:
        detail = {
            "dataset_sha256": meta["dataset_sha256"],
            "protocol": e.protocol["phase3_selection"],
            "parameters": r["parameters"],
            "metrics": r["holdout_metrics"],
            "threshold": r["threshold"],
            "seed": meta["seed"],
            "versions": meta["versions"],
            "artifact": "reports/phase3/pooled-both.json",
            "selection_sha256": e.pooled["selection_sha256"],
        }
        rows.append(
            dict(
                model=r["family"],
                tier=r["tier"],
                evaluation="pooled",
                hospital="All sites",
                metadata=detail,
            )
        )
    for site in e.transfer["sites"]:
        for r in site["runs"]:
            config = next(
                c
                for c in e.protocol["models"]
                if (c["family"], c["tier"]) == (r["family"], r["tier"])
            )
            detail = {
                "dataset_sha256": e.protocol["dataset_sha256"],
                "protocol": e.transfer["protocol_commit"],
                "parameters": config["parameters"],
                "metrics": r["metrics"],
                "threshold": r["threshold"],
                "policy": r["policy"],
                "calibration": r["mode"],
                "seed": e.protocol["seed"],
                "versions": e.protocol["software"],
                "artifact": "reports/phase4/hospital-results.json",
                "scientific_sha256": e.transfer["scientific_sha256"],
            }
            rows.append(
                dict(
                    model=r["family"],
                    tier=r["tier"],
                    evaluation="hospital",
                    hospital=site["hospital"],
                    metadata=detail,
                )
            )
    return [{"id": digest(r), **r} for r in rows]


def import_evidence(engine: Engine, e: Evidence) -> int:
    try:
        statement = sqlite_insert if engine.dialect.name == "sqlite" else pg_insert
        with engine.begin() as connection:
            for row in records(e):
                connection.execute(
                    statement(experiments)
                    .values(**row, imported_at=datetime.now(UTC))
                    .on_conflict_do_nothing(index_elements=["id"])
                )
            return len(connection.execute(select(experiments.c.id)).all())
    except SQLAlchemyError as exc:
        raise RegistryError("Aggregate import failed and the transaction was rolled back.") from exc


def listing(engine: Engine) -> list[dict[str, Any]]:
    try:
        with engine.connect() as connection:
            return [
                dict(row)
                for row in connection.execute(
                    select(experiments).order_by(
                        experiments.c.evaluation,
                        experiments.c.hospital,
                        experiments.c.tier,
                        experiments.c.model,
                        experiments.c.id,
                    )
                ).mappings()
            ]
    except (SQLAlchemyError, ValueError, TypeError) as exc:
        raise RegistryError("Could not read the aggregate experiment registry.") from exc


def verified_listing(engine: Engine, evidence: Evidence) -> tuple[list[dict[str, Any]], int]:
    """Check persisted rows against the verified source before rendering them."""
    expected = {row["id"]: row for row in records(evidence)}
    rows = listing(engine)
    for row in rows:
        payload = {key: value for key, value in row.items() if key != "imported_at"}
        if payload != expected.get(row["id"]) or not isinstance(row["imported_at"], datetime):
            raise RegistryError(
                "Registry contains a record that does not match the verified reports. "
                "Restore a trusted database or import into a fresh local SQLite database."
            )
    return rows, len(expected) - len(rows)
