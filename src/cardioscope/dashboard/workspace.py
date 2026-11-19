"""Authentication and owner-scoped storage for fictional educational work."""

from __future__ import annotations

import hashlib
import re
import secrets
import time
import unicodedata
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from functools import lru_cache
from threading import BoundedSemaphore
from typing import Any
from uuid import uuid4

from argon2 import PasswordHasher, Type
from argon2.exceptions import InvalidHashError, VerificationError
from sqlalchemy import and_, literal, select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.engine import Connection, Engine
from sqlalchemy.exc import IntegrityError, SQLAlchemyError

from .evidence import FAMILIES, Evidence, Record
from .examples import EXAMPLES, REPORT_DISCLAIMER
from .sandbox import expected_metadata, load_verified, validate_profile
from .workspace_schema import (
    WorkspaceError,
    attempts,
    preferences,
    profiles,
    reports,
    sessions,
    shares,
    users,
)

SESSION_SECONDS = 8 * 60 * 60
ATTEMPT_WINDOW = 15 * 60
LOGIN_LIMIT = 30
ACCOUNT_LIMIT = 6
REGISTER_LIMIT = 12
USER_PATTERN = re.compile(r"[a-z][a-z0-9_]{2,31}", re.ASCII)
AUTH_ERROR = "Sign-in failed. Check your username and password or try again later."
PASSWORD_HASHER = PasswordHasher(time_cost=3, memory_cost=65536, parallelism=4, type=Type.ID)
HASH_SLOTS = BoundedSemaphore(2)


@dataclass(frozen=True)
class Settings:
    show_grid: bool = True
    advanced_config: bool = False
    model: str = "logistic"
    tier: str = "early"
    hospital: str = "Cleveland"

    @classmethod
    def parse(cls, values: Mapping[str, Any]) -> Settings:
        from cardioscope.data.schema import HOSPITALS

        if (
            set(values) != set(asdict(cls()))
            or type(values["show_grid"]) is not bool
            or type(values["advanced_config"]) is not bool
            or values["model"] not in tuple(FAMILIES)
            or values["tier"] not in ("early", "extended")
            or values["hospital"] not in HOSPITALS
        ):
            raise WorkspaceError("Choose supported settings before saving.")
        return cls(**values)


@dataclass(frozen=True)
class Account:
    id: str
    username: str
    settings: Settings


@dataclass(frozen=True)
class Profile:
    id: str
    name: str
    example: str


@dataclass(frozen=True)
class PublicReport:
    """Fixed projection. The query never selects private or account fields."""

    example: str
    model: str
    feature_tier: str
    version: str
    artifact_digest: str
    score: float
    historical_class: int
    disclaimer: str


@dataclass(frozen=True)
class SavedReport(PublicReport):
    id: str
    owner_id: str
    title: str
    profile_label: str
    threshold: float
    created_at: int
    shared: bool


def _hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


@lru_cache(maxsize=1)
def _dummy_hash() -> str:
    return PASSWORD_HASHER.hash(secrets.token_urlsafe(32))


def _name(value: str) -> str:
    value = unicodedata.normalize("NFKC", value.strip())
    if not 1 <= len(value) <= 64 or not value.isprintable():
        raise WorkspaceError("Use a name with 1–64 printable characters.")
    return value


def _example(name: str) -> Record:
    if name not in EXAMPLES:
        raise WorkspaceError("Select a built-in fictional example.")
    return EXAMPLES[name].copy()


class Workspace:
    def __init__(self, engine: Engine) -> None:
        self.engine = engine
        self.insert = sqlite_insert if engine.dialect.name == "sqlite" else pg_insert

    @contextmanager
    def _transaction(self) -> Iterator[Connection]:
        try:
            with self.engine.begin() as conn:
                yield conn
        except SQLAlchemyError as exc:
            raise WorkspaceError(
                "The personal workspace database could not complete the request."
            ) from exc

    def _consume(self, conn: Connection, key: str, window: int, limit: int) -> bool:
        stmt = self.insert(attempts).values(key=key, window=window, count=1)
        result = conn.execute(
            stmt.on_conflict_do_update(
                index_elements=[attempts.c.key, attempts.c.window],
                set_={"count": attempts.c.count + 1},
                where=attempts.c.count < limit,
            )
        )
        return result.rowcount == 1

    def _reserve_attempt(self, action: str, username: str) -> bool:
        window = int(time.time()) // ATTEMPT_WINDOW
        max_global = LOGIN_LIMIT if action == "login" else REGISTER_LIMIT
        with self._transaction() as conn:
            available = self._consume(conn, f"{action}:global", window, max_global)
            if available:
                conn.execute(attempts.delete().where(attempts.c.window < window))
                if action == "login":
                    available = self._consume(
                        conn, f"login:{_hash(username)}", window, ACCOUNT_LIMIT
                    )
            return available

    def register(self, username: str, password: str) -> None:
        name = username.strip().lower()
        if not USER_PATTERN.fullmatch(name):
            raise WorkspaceError(
                "Use 3–32 ASCII letters, digits or underscores, starting with a letter."
            )
        if not 12 <= len(password) <= 128:
            raise WorkspaceError("Use a password with 12–128 characters.")
        error = "Unable to register. Try a different username or try again later."
        if not self._reserve_attempt("register", name):
            raise WorkspaceError(error)
        with HASH_SLOTS:
            hashed = PASSWORD_HASHER.hash(password)
        with self._transaction() as conn:
            try:
                conn.execute(
                    users.insert().values(
                        id=uuid4().hex,
                        username=name,
                        password_hash=hashed,
                        created_at=int(time.time()),
                    )
                )
            except IntegrityError as exc:
                raise WorkspaceError(error) from exc

    def login(self, username: str, password: str) -> str:
        name = username.strip().lower()
        if not USER_PATTERN.fullmatch(name) or not 1 <= len(password) <= 128:
            raise WorkspaceError(AUTH_ERROR)
        if not self._reserve_attempt("login", name):
            raise WorkspaceError(AUTH_ERROR)
        with self._transaction() as conn:
            row = (
                conn.execute(select(users).where(users.c.username == name)).mappings().one_or_none()
            )
        new_hash = None
        try:
            with HASH_SLOTS:
                fallback = _dummy_hash()
                PASSWORD_HASHER.verify(row["password_hash"] if row else fallback, password)
                if row and PASSWORD_HASHER.check_needs_rehash(row["password_hash"]):
                    new_hash = PASSWORD_HASHER.hash(password)
        except (InvalidHashError, VerificationError) as exc:
            raise WorkspaceError(AUTH_ERROR) from exc
        if row is None:
            raise WorkspaceError(AUTH_ERROR)
        token = secrets.token_urlsafe(32)
        now = int(time.time())
        with self._transaction() as conn:
            conn.execute(sessions.delete().where(sessions.c.expires_at <= now))
            if new_hash is not None:
                conn.execute(
                    users.update()
                    .where(users.c.id == row["id"], users.c.password_hash == row["password_hash"])
                    .values(password_hash=new_hash)
                )
            conn.execute(
                sessions.insert().values(
                    token_hash=_hash(token), owner_id=row["id"], expires_at=now + SESSION_SECONDS
                )
            )
        return token

    def logout(self, token: str) -> None:
        with self._transaction() as conn:
            conn.execute(sessions.delete().where(sessions.c.token_hash == _hash(token)))

    @staticmethod
    def _account(conn: Connection, token: str) -> Account:
        row = (
            conn.execute(
                select(users.c.id, users.c.username, preferences.c.settings)
                .select_from(
                    sessions.join(users, sessions.c.owner_id == users.c.id).outerjoin(
                        preferences, preferences.c.owner_id == users.c.id
                    )
                )
                .where(
                    sessions.c.token_hash == _hash(token), sessions.c.expires_at > int(time.time())
                )
            )
            .mappings()
            .one_or_none()
        )
        if row is None:
            raise WorkspaceError("Sign in to access your personal workspace.")
        settings = Settings() if row["settings"] is None else Settings.parse(row["settings"])
        return Account(row["id"], row["username"], settings)

    @contextmanager
    def _authorized(self, token: str) -> Iterator[tuple[Connection, Account]]:
        with self._transaction() as conn:
            yield conn, self._account(conn, token)

    def account(self, token: str) -> Account:
        with self._authorized(token) as (_, account):
            return account

    def save_settings(self, token: str, values: Mapping[str, Any]) -> Settings:
        validated = Settings.parse(values)
        with self._authorized(token) as (conn, account):
            stmt = self.insert(preferences).values(owner_id=account.id, settings=asdict(validated))
            conn.execute(
                stmt.on_conflict_do_update(
                    index_elements=[preferences.c.owner_id],
                    set_={"settings": stmt.excluded.settings},
                )
            )
        return validated

    def profiles(self, token: str) -> list[Profile]:
        with self._authorized(token) as (conn, account):
            rows = conn.execute(
                select(profiles.c.id, profiles.c.name, profiles.c.example)
                .where(profiles.c.owner_id == account.id)
                .order_by(profiles.c.name_key)
            )
            return [Profile(row.id, row.name, row.example) for row in rows]

    def create_profile(self, token: str, name: str, example: str) -> Profile:
        name = _name(name)
        _example(example)
        profile = Profile(uuid4().hex, name, example)
        with self._authorized(token) as (conn, account):
            try:
                conn.execute(
                    profiles.insert().values(
                        **asdict(profile), owner_id=account.id, name_key=name.casefold()
                    )
                )
            except IntegrityError as exc:
                raise WorkspaceError("Choose a distinct name for this fictional profile.") from exc
        return profile

    def delete_profile(self, token: str, profile_id: str) -> None:
        with self._authorized(token) as (conn, account):
            conn.execute(
                profiles.delete().where(
                    profiles.c.id == profile_id, profiles.c.owner_id == account.id
                )
            )

    @staticmethod
    def _owned(owner: str, report_id: str) -> Any:
        return and_(reports.c.owner_id == owner, reports.c.id == report_id)

    def save_report(
        self,
        token: str,
        evidence: Evidence,
        example: str,
        entered: Record,
        profile_id: str | None = None,
    ) -> SavedReport:
        original = _example(example)
        frame = validate_profile(entered)
        if entered != original:
            raise WorkspaceError(
                "Reset to the built-in example before saving. Edited values stay transient."
            )
        self.account(token)
        model = load_verified(evidence)
        meta = expected_metadata(evidence)
        score = float(model.predict_proba(frame)[0, 1])
        threshold = float(meta["threshold"])
        if not 0 <= score <= 1:
            raise WorkspaceError("The verified model did not return a valid score.")
        with self._authorized(token) as (conn, account):
            label = example
            if profile_id is not None:
                saved_label = conn.scalar(
                    select(profiles.c.name).where(
                        profiles.c.id == profile_id,
                        profiles.c.owner_id == account.id,
                        profiles.c.example == example,
                    )
                )
                if saved_label is None:
                    raise WorkspaceError("The selected fictional profile is unavailable.")
                label = saved_label
            saved = SavedReport(
                id=uuid4().hex,
                owner_id=account.id,
                title=label,
                profile_label=label,
                example=example,
                model=FAMILIES[meta["family"]],
                feature_tier=meta["tier"],
                version=meta["versions"]["scikit-learn"],
                artifact_digest=meta["holdout_probability_sha256"],
                score=score,
                historical_class=int(score >= threshold),
                threshold=threshold,
                created_at=int(time.time()),
                disclaimer=REPORT_DISCLAIMER,
                shared=False,
            )
            payload = asdict(saved)
            payload.pop("shared")
            conn.execute(reports.insert().values(**payload))
        return saved

    @staticmethod
    def _report_query() -> Any:
        return select(reports, shares.c.report_id.is_not(None).label("shared")).select_from(
            reports.outerjoin(shares)
        )

    def history(self, token: str) -> list[SavedReport]:
        with self._authorized(token) as (conn, account):
            query = self._report_query().where(reports.c.owner_id == account.id)
            query = query.order_by(reports.c.created_at.desc(), reports.c.id)
            return [SavedReport(**row) for row in conn.execute(query).mappings()]

    def report(self, token: str, report_id: str) -> SavedReport:
        with self._authorized(token) as (conn, account):
            query = self._report_query().where(self._owned(account.id, report_id))
            row = conn.execute(query).mappings().one_or_none()
            if row is None:
                raise WorkspaceError("That report is unavailable.")
            return SavedReport(**row)

    def rename_report(self, token: str, report_id: str, title: str) -> None:
        with self._authorized(token) as (conn, account):
            result = conn.execute(
                reports.update()
                .where(self._owned(account.id, report_id))
                .values(title=_name(title))
            )
            if result.rowcount != 1:
                raise WorkspaceError("That report is unavailable.")

    def delete_report(self, token: str, report_id: str) -> None:
        with self._authorized(token) as (conn, account):
            result = conn.execute(reports.delete().where(self._owned(account.id, report_id)))
            if result.rowcount != 1:
                raise WorkspaceError("That report is unavailable.")

    def clear_history(self, token: str) -> None:
        with self._authorized(token) as (conn, account):
            conn.execute(reports.delete().where(reports.c.owner_id == account.id))

    def set_sharing(self, token: str, report_id: str, *, shared: bool) -> None:
        with self._authorized(token) as (conn, account):
            owned = select(reports.c.id).where(self._owned(account.id, report_id))
            if conn.scalar(owned) is None:
                raise WorkspaceError("That report is unavailable.")
            if shared:
                # The INSERT itself also checks ownership and report existence.
                source = owned.add_columns(literal(int(time.time())))
                stmt = self.insert(shares).from_select(["report_id", "shared_at"], source)
                conn.execute(stmt.on_conflict_do_nothing(index_elements=[shares.c.report_id]))
            else:
                conn.execute(shares.delete().where(shares.c.report_id.in_(owned)))

    def shared_reports(self) -> list[PublicReport]:
        columns = [reports.c[name] for name in PublicReport.__dataclass_fields__]
        with self._transaction() as conn:
            query = (
                select(*columns)
                .select_from(reports.join(shares))
                .order_by(shares.c.shared_at.desc(), shares.c.report_id)
            )
            return [PublicReport(**row) for row in conn.execute(query).mappings()]
