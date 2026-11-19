"""Browser authentication through the Streamlit server's ASGI boundary."""

from __future__ import annotations

import os
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from html import escape
from ipaddress import ip_address
from urllib.parse import parse_qs, urlsplit

from starlette.concurrency import run_in_threadpool
from starlette.datastructures import MutableHeaders
from starlette.requests import ClientDisconnect, HTTPConnection, Request
from starlette.responses import HTMLResponse, PlainTextResponse, RedirectResponse, Response
from starlette.routing import Route
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from .workspace import REMEMBER_SECONDS, InvalidSession, Workspace
from .workspace_schema import WorkspaceError

WELCOME = "/welcome"
LOGIN = "/auth/login"
REGISTER = "/auth/register"
LOGOUT = "/auth/logout"
FORM_PATHS = (LOGIN, REGISTER, LOGOUT)
NO_STORE = {"Cache-Control": "private, no-store"}
TOKEN_FORMAT = re.compile(r"[A-Za-z0-9_-]{43}")


@dataclass(frozen=True)
class CookiePolicy:
    public_origin: str = ""

    @classmethod
    def configured(cls) -> CookiePolicy:
        origin = os.environ.get("CARDIOSCOPE_PUBLIC_ORIGIN", "").strip()
        if not origin:
            return cls()
        parsed = urlsplit(origin)
        if (
            parsed.scheme != "https"
            or not parsed.hostname
            or parsed.username is not None
            or parsed.path
            or parsed.query
            or parsed.fragment
            or origin != f"https://{parsed.netloc.lower()}"
        ):
            raise ValueError("CARDIOSCOPE_PUBLIC_ORIGIN must be an HTTPS origin without a path.")
        return cls(origin)

    @property
    def secure(self) -> bool:
        return bool(self.public_origin)

    @property
    def name(self) -> str:
        return "__Host-cardioscope_session" if self.secure else "cardioscope_session"

    def origin(self, request: HTTPConnection) -> str | None:
        host = request.headers.get("host", "").lower()
        transport = request.scope["scheme"]
        scheme = {"ws": "http", "wss": "https"}.get(transport, transport)
        origin = f"{scheme}://{host}"
        if self.secure:
            return origin if origin == self.public_origin else None
        if scheme != "http" or not re.fullmatch(r"(127\.0\.0\.1|localhost)(:[0-9]{1,5})?", host):
            return None
        try:
            peer = request.client.host if request.client else ""
            return origin if ip_address(peer).is_loopback else None
        except ValueError:
            return None

    def set(self, response: Response, grant: str, remember: bool) -> None:
        response.set_cookie(
            self.name,
            grant,
            max_age=REMEMBER_SECONDS if remember else None,
            secure=self.secure,
            httponly=True,
            samesite="strict",
        )

    def clear(self, response: Response) -> None:
        response.delete_cookie(self.name, secure=self.secure, httponly=True, samesite="strict")


def browser_token(cookies: Mapping[str, str], policy: CookiePolicy) -> str | None:
    grant = cookies.get(policy.name, "")
    return grant if TOKEN_FORMAT.fullmatch(grant) else None


async def _signed_in(grant: str | None, get_store: Callable[[], Workspace]) -> bool:
    if grant is None:
        return False
    try:
        await run_in_threadpool(lambda: get_store().account(grant))
    except InvalidSession:
        return False
    return True


class AuthGate:
    """Guard both HTTP resources and the Streamlit WebSocket before dispatch."""

    def __init__(
        self, app: ASGIApp, *, policy: CookiePolicy, get_store: Callable[[], Workspace]
    ) -> None:
        self.app = app
        self.policy = policy
        self.get_store = get_store

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        kind = scope["type"]
        if kind not in {"http", "websocket"}:
            await self.app(scope, receive, send)
            return
        request = HTTPConnection(scope)
        origin = self.policy.origin(request)
        path = scope["path"]
        read_only = scope.get("method") in {"GET", "HEAD"}
        same_origin = origin is not None and request.headers.get("origin") == origin
        if kind == "websocket":
            grant = browser_token(request.cookies, self.policy)
            try:
                allowed = same_origin and await _signed_in(grant, self.get_store)
            except WorkspaceError:
                allowed = False
            if allowed:
                await self.app(_fresh_streamlit_connection(scope), receive, send)
            else:
                await send({"type": "websocket.close", "code": 1008})
            return
        if origin is None or (not read_only and not same_origin):
            await PlainTextResponse("Request denied.", 403, NO_STORE)(scope, receive, send)
            return
        if path in FORM_PATHS and scope["method"] != "POST":
            response = PlainTextResponse("Method not allowed.", 405, {**NO_STORE, "Allow": "POST"})
            await response(scope, receive, send)
            return
        if read_only and path == "/_stcore/health":
            await self.app(scope, receive, send)
            return
        if path not in {WELCOME, *FORM_PATHS}:
            try:
                signed_in = await _signed_in(
                    browser_token(request.cookies, self.policy), self.get_store
                )
            except WorkspaceError:
                await _unavailable()(scope, receive, send)
                return
            if not signed_in:
                denied = (
                    RedirectResponse(WELCOME, 303, NO_STORE)
                    if read_only
                    else PlainTextResponse("Sign in to continue.", 401, NO_STORE)
                )
                if self.policy.name in request.cookies:
                    self.policy.clear(denied)
                await denied(scope, receive, send)
                return

        async def prevent_caching(message: Message) -> None:
            if message["type"] == "http.response.start" and not path.startswith("/static/"):
                headers = MutableHeaders(scope=message)
                headers.update(NO_STORE)
                headers["X-Frame-Options"] = "DENY"
            await send(message)

        await self.app(scope, receive, prevent_caching)


def _fresh_streamlit_connection(scope: Scope) -> Scope:
    # Streamlit uses the third offered subprotocol as a client-supplied resume ID.
    # Keep protocol negotiation and Streamlit's XSRF field, and start a fresh session.
    name = b"sec-websocket-protocol"
    headers = scope.get("headers", [])
    offered = b",".join(value for key, value in headers if key == name).split(b",", 2)
    return {
        **scope,
        "headers": [(key, value) for key, value in headers if key != name]
        + [(name, b",".join(offered[:2]))],
        "subprotocols": scope.get("subprotocols", [])[:2],
    }


async def _form(request: Request) -> dict[str, str]:
    media_type = request.headers.get("content-type", "").split(";", 1)[0]
    if media_type != "application/x-www-form-urlencoded":
        raise ValueError("Unsupported form encoding.")
    body = bytearray()
    async for chunk in request.stream():
        if len(body) + len(chunk) > 8192:
            raise ValueError("Form too large.")
        body.extend(chunk)
    parsed = parse_qs(
        body.decode("utf-8"), keep_blank_values=True, max_num_fields=4, errors="strict"
    )
    if any(len(values) != 1 for values in parsed.values()):
        raise ValueError("Duplicate fields.")
    return {key: values[0] for key, values in parsed.items()}


def _unavailable() -> Response:
    return PlainTextResponse("Workspace temporarily unavailable. Try again shortly.", 503, NO_STORE)


class AuthPages:
    def __init__(self, policy: CookiePolicy, get_store: Callable[[], Workspace]) -> None:
        self.policy = policy
        self.get_store = get_store

    def routes(self) -> list[Route]:
        return [
            Route(WELCOME, self.welcome),
            Route(LOGIN, self.credentials, methods=["POST"]),
            Route(REGISTER, self.credentials, methods=["POST"]),
            Route(LOGOUT, self.logout, methods=["POST"]),
        ]

    async def welcome(self, request: Request) -> Response:
        try:
            if await _signed_in(browser_token(request.cookies, self.policy), self.get_store):
                return RedirectResponse("/", 303, NO_STORE)
        except WorkspaceError:
            return _unavailable()
        response = _welcome(register=request.query_params.get("mode") == "register")
        if self.policy.name in request.cookies:
            self.policy.clear(response)
        return response

    async def credentials(self, request: Request) -> Response:
        grant = browser_token(request.cookies, self.policy)
        try:
            if await _signed_in(grant, self.get_store):
                return RedirectResponse("/", 303, NO_STORE)
        except WorkspaceError:
            return _unavailable()
        register = request.scope["path"] == REGISTER
        try:
            data = await _form(request)
        except (ValueError, ClientDisconnect):
            return _welcome(register=register, error="Invalid form submission.", status=400)
        username, password = data.get("username", ""), data.get("password", "")
        remember = data.get("remember") == "on"
        if register and password != data.get("confirmation"):
            return _welcome(
                register=register, error="The passwords do not match.", remember=remember
            )
        try:
            if register:
                issued = await run_in_threadpool(
                    lambda: self.get_store().register(username, password, remember=remember)
                )
            else:
                issued = await run_in_threadpool(
                    lambda: self.get_store().login(username, password, remember=remember)
                )
        except WorkspaceError as exc:
            return _welcome(register=register, error=str(exc), remember=remember)
        if issued is None:
            return _unavailable()
        response = RedirectResponse("/", 303, NO_STORE)
        self.policy.set(response, issued, remember)
        return response

    async def logout(self, request: Request) -> Response:
        grant = browser_token(request.cookies, self.policy)
        if grant is not None:
            try:
                await run_in_threadpool(lambda: self.get_store().logout(grant))
            except WorkspaceError:
                return _unavailable()
        response = RedirectResponse(WELCOME, 303, NO_STORE)
        self.policy.clear(response)
        return response


def _welcome(
    *, register: bool, error: str = "", remember: bool = True, status: int = 200
) -> HTMLResponse:
    heading = "Create your account" if register else "Welcome to CardioScope ML"
    action = REGISTER if register else LOGIN
    submit = "Create account" if register else "Sign in"
    autocomplete = "new-password" if register else "current-password"
    confirmation = (
        '<label>Confirm password<input name="confirmation" type="password" '
        'autocomplete="new-password" maxlength="128" required></label>'
        if register
        else ""
    )
    switch = (
        '<a href="/welcome">Sign in to an existing account</a>'
        if register
        else '<a href="/welcome?mode=register">Create an account</a>'
    )
    return HTMLResponse(
        f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="color-scheme" content="dark">
<title>{heading} · CardioScope ML</title>
<style>
  * {{ box-sizing: border-box; }}
  body {{ margin: 0; padding: 24px; min-height: 100svh; display: grid; place-items: center;
    background: radial-gradient(ellipse at 50% 0%, #152640, #0b1120 58%); color: #f1f5f9;
    font: 16px/1.5 system-ui, sans-serif; }}
  main {{ width: 100%; max-width: 470px; border: 1px solid #334155; border-radius: 18px;
    background: #111c30; padding: clamp(22px, 6vw, 40px); box-shadow: 0 24px 80px #02061780; }}
  .brand {{ margin: 0; color: #7dd3fc; font-weight: 700; font-size: .875rem; }}
  h1 {{ font-size: clamp(1.5rem, 5vw, 2rem); line-height: 1.16; margin: 16px 0 10px;
    text-wrap: balance; }}
  p {{ margin: 0 0 22px; color: #cbd5e1; }}
  form {{ display: grid; gap: 15px; }}
  label {{ display: grid; gap: 5px; font-size: .9375rem; }}
  input:not([type=checkbox]) {{ width: 100%; min-height: 43px; padding: 8px 10px;
    background: #0b1120; color: inherit; border: 1px solid #52637b; border-radius: 8px;
    font: inherit; }}
  input[type=checkbox] {{ width: 18px; height: 18px; margin: 0; accent-color: #38bdf8; }}
  label.remember {{ display: flex; align-items: center; gap: 8px; min-height: 32px; }}
  button {{ border: 0; min-height: 44px; border-radius: 8px; background: #38bdf8;
    color: #0b1120; font: inherit; font-weight: 700; cursor: pointer; }}
  a {{ color: #7dd3fc; text-underline-offset: 3px; }}
  :focus-visible {{ outline: 2px solid #7dd3fc; outline-offset: 3px; }}
  footer {{ margin-top: 20px; }}
  .help {{ color: #cbd5e1; font-size: .825rem; }}
  .error {{ color: #fecaca; background: #4c2232; padding: 10px 12px; border-radius: 8px; }}
</style>
</head>
<body>
<main>
<div class="brand">CardioScope ML · Research Workspace</div>
<h1>{heading}</h1>
<p>Explore verified research and manage your saved prediction reports.</p>
{f'<p class="error" role="alert">{escape(error)}</p>' if error else ""}
<form method="post" action="{action}">
<label>Username<input name="username" autocomplete="username" maxlength="32" required></label>
<label>Password<input name="password" type="password" autocomplete="{autocomplete}"
maxlength="128" required></label>
{confirmation}
{('<span class="help">Use 12–128 characters and a unique password.</span>' if register else "")}
<label class="remember">
<input type="checkbox" name="remember" {"checked" if remember else ""}>
Keep me signed in on this device
</label>
<button type="submit">{submit}</button>
</form>
<footer>{switch}</footer>
</main>
</body>
</html>""",
        status,
        {
            **NO_STORE,
            "Content-Security-Policy": "default-src 'none'; style-src 'unsafe-inline'; "
            "form-action 'self'; frame-ancestors 'none'; base-uri 'none'",
            "Referrer-Policy": "same-origin",
        },
    )
