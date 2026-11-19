"""HTTP and WebSocket authentication regressions against the real workspace store."""

from __future__ import annotations

import asyncio
import hashlib
import secrets
import time
from http.cookies import SimpleCookie
from urllib.parse import urlencode, urlsplit

import pytest
from sqlalchemy import URL, select
from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.responses import PlainTextResponse
from starlette.routing import Route, WebSocketRoute

from cardioscope.dashboard.auth import AuthGate, AuthPages, CookiePolicy
from cardioscope.dashboard.workspace import (
    REMEMBER_SECONDS,
    SESSION_SECONDS,
    InvalidSession,
    Workspace,
)
from cardioscope.dashboard.workspace_schema import open_workspace, sessions

LOCAL = "http://127.0.0.1:8501"
PUBLIC = "https://cardioscope.example"
PASSWORD = "a fictional test-only passphrase"


@pytest.fixture
def db(tmp_path):
    url = URL.create("sqlite", database=str(tmp_path / "browser.db"))
    engine = open_workspace(url)
    yield Workspace(engine), url
    engine.dispose()


def web(policy, store):
    async def dashboard(request):
        return PlainTextResponse("authenticated workspace")

    async def websocket(connection):
        assert len(connection.scope["subprotocols"]) <= 2
        await connection.accept("streamlit")
        await connection.close()

    return Starlette(
        routes=[
            *AuthPages(policy, lambda: store).routes(),
            Route("/{path:path}", dashboard),
            WebSocketRoute("/_stcore/stream", websocket),
        ],
        middleware=[Middleware(AuthGate, policy=policy, get_store=lambda: store)],
    )


def asgi(app, path, *, base=LOCAL, method="GET", origin=None, cookie="", form=None, socket=False):
    """Drive one ASGI request with HTTP headers as the browser would send them."""
    address, target = urlsplit(base), urlsplit(path)
    headers = [(b"host", address.netloc.encode())]
    if origin is not None:
        headers.append((b"origin", origin.encode()))
    if cookie:
        headers.append((b"cookie", cookie.encode()))
    if form is not None:
        headers.append((b"content-type", b"application/x-www-form-urlencoded"))
    scheme = "wss" if address.scheme == "https" else "ws"
    scope = {
        "type": "websocket" if socket else "http",
        "scheme": scheme if socket else address.scheme,
        "method": method,
        "http_version": "1.1",
        "path": target.path,
        "query_string": target.query.encode(),
        "headers": headers,
        "client": ("127.0.0.1", 12345),
    }
    if socket:
        scope["subprotocols"] = ["streamlit", "xsrf-field", "previous-session"]
        headers.append((b"sec-websocket-protocol", b"streamlit, xsrf-field, previous-session"))
    incoming = (
        {"type": "websocket.connect"}
        if socket
        else {"type": "http.request", "body": urlencode(form or {}).encode()}
    )
    messages = []

    async def receive():
        return incoming

    async def send(message):
        messages.append(message)

    asyncio.run(app(scope, receive, send))
    if socket:
        return messages[0]["type"]
    reply = messages[0]
    body = b"".join(message.get("body", b"") for message in messages[1:])
    return reply["status"], dict(reply["headers"]), body


def credentials(username, *, remember=True, register=True):
    form = {"username": username, "password": PASSWORD}
    if register:
        form["confirmation"] = PASSWORD
    if remember:
        form["remember"] = "on"
    return form


def cookie_value(headers, policy):
    return SimpleCookie(headers[b"set-cookie"].decode())[policy.name]


def expire(store, grant):
    digest = hashlib.sha256(grant.encode()).hexdigest()
    with store.engine.begin() as conn:
        conn.execute(sessions.update().where(sessions.c.token_hash == digest).values(expires_at=1))


def test_register_issues_host_only_cookie_and_persists_without_sliding_expiry(db):
    store, url = db
    policy = CookiePolicy()
    app = web(policy, store)
    status, headers, body = asgi(app, "/welcome?mode=register")
    assert status == 200 and b"Keep me signed in on this device" in body and b"checked" in body
    assert headers[b"referrer-policy"] == b"same-origin"
    reply = asgi(
        app, "/auth/register", method="POST", origin=LOCAL, form=credentials("browser_alpha")
    )
    status, headers, body = reply
    assert status == 303 and headers[b"location"] == b"/"
    cookie = cookie_value(headers, policy)
    assert cookie["httponly"] and cookie["samesite"] == "strict" and not cookie["domain"]
    assert cookie["path"] == "/" and not cookie["secure"]
    assert int(cookie["max-age"]) == REMEMBER_SECONDS
    assert PASSWORD.encode() not in body and cookie.value.encode() not in body
    with store.engine.connect() as conn:
        digest, expiry = conn.execute(select(sessions.c.token_hash, sessions.c.expires_at)).one()
    assert digest == hashlib.sha256(cookie.value.encode()).hexdigest()
    assert int(time.time()) + REMEMBER_SECONDS - 5 <= expiry <= int(time.time()) + REMEMBER_SECONDS
    separately_opened = Workspace(open_workspace(url))
    try:
        app = web(policy, separately_opened)
        status, headers, _ = asgi(app, "/my-reports", cookie=f"{policy.name}={cookie.value}")
        assert status == 200 and headers[b"cache-control"] == b"private, no-store"
        assert b"set-cookie" not in headers
        assert not separately_opened.history(cookie.value)
        assert not separately_opened.profiles(cookie.value)
        with separately_opened.engine.connect() as conn:
            assert conn.scalar(select(sessions.c.expires_at)) == expiry
    finally:
        separately_opened.engine.dispose()


def test_routes_and_websocket_fail_closed_for_signed_out_and_invalid_state(db):
    store, _ = db
    policy = CookiePolicy()
    app = web(policy, store)
    for path in ("/", "/dataset", "/account", "/settings", "/my-reports", "/shared-reports"):
        status, headers, _ = asgi(app, path)
        assert status == 303 and headers[b"location"] == b"/welcome"
    status, headers, _ = asgi(app, "/my-reports", cookie=f"{policy.name}=invalid")
    assert status == 303 and cookie_value(headers, policy)["max-age"] == "0"
    assert asgi(app, "/_stcore/stream", socket=True, origin=LOCAL) == "websocket.close"
    grant = store.register("browser_alpha", PASSWORD, remember=True)
    cookie = f"{policy.name}={grant}"
    assert (
        asgi(app, "/_stcore/stream", socket=True, origin=LOCAL, cookie=cookie) == "websocket.accept"
    )
    assert (
        asgi(app, "/_stcore/stream", socket=True, origin=PUBLIC, cookie=cookie) == "websocket.close"
    )
    expire(store, grant)
    assert (
        asgi(app, "/_stcore/stream", socket=True, origin=LOCAL, cookie=cookie) == "websocket.close"
    )
    status, headers, _ = asgi(app, "/welcome", cookie=cookie)
    assert status == 200 and cookie_value(headers, policy)["max-age"] == "0"


def test_logout_revokes_only_active_grant_and_secure_cookie_attributes(db):
    store, _ = db
    policy = CookiePolicy(PUBLIC)
    app = web(policy, store)
    assert asgi(app, "/", base=LOCAL)[0] == 403
    assert asgi(app, "/", base=PUBLIC.replace("https:", "http:"))[0] == 403
    form = credentials("browser_alpha", remember=False)
    assert (
        asgi(app, "/auth/register", base=PUBLIC, method="POST", origin=LOCAL, form=form)[0] == 403
    )
    status, headers, _ = asgi(
        app, "/auth/register", base=PUBLIC, method="POST", origin=PUBLIC, form=form
    )
    assert status == 303
    cookie = cookie_value(headers, policy)
    assert policy.name.startswith("__Host-") and cookie["secure"] and cookie["httponly"]
    assert cookie["max-age"] == cookie["expires"] == ""  # browser-session cookie
    with store.engine.connect() as conn:
        expiry = conn.scalar(select(sessions.c.expires_at))
    assert int(time.time()) + SESSION_SECONDS - 5 <= expiry <= int(time.time()) + SESSION_SECONDS
    active = f"{policy.name}={cookie.value}"
    assert asgi(app, "/auth/logout", base=PUBLIC, cookie=active)[0] == 405
    assert asgi(app, "/auth/logout", base=PUBLIC, method="POST", cookie=active)[0] == 403
    store.create_profile(cookie.value, "Owner example", "Fictional A")
    other = store.register("browser_beta", PASSWORD, remember=True)
    assert not store.profiles(other)
    status, headers, _ = asgi(
        app, "/auth/logout", base=PUBLIC, method="POST", origin=PUBLIC, cookie=active
    )
    deleted = cookie_value(headers, policy)
    assert status == 303 and deleted["secure"] and deleted["max-age"] == "0"
    with pytest.raises(InvalidSession):
        store.account(cookie.value)
    assert asgi(app, "/my-reports", base=PUBLIC, cookie=active)[0] == 303
    assert store.account(other).username == "browser_beta"
    status, headers, _ = asgi(
        app,
        "/auth/login",
        base=PUBLIC,
        method="POST",
        origin=PUBLIC,
        form=credentials("browser_alpha", register=False),
    )
    assert status == 303
    new = cookie_value(headers, policy)
    assert new.value != cookie.value
    assert store.profiles(new.value)[0].name == "Owner example"


def test_invalid_cookie_cannot_redirect_to_dashboard_when_database_is_available(db):
    store, _ = db
    policy = CookiePolicy()
    app = web(policy, store)
    status, headers, body = asgi(
        app, "/welcome", cookie=f"{policy.name}={secrets.token_urlsafe(32)}"
    )
    assert status == 200 and b"Welcome to CardioScope ML" in body
    assert cookie_value(headers, policy)["max-age"] == "0"
