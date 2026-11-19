"""Launch CardioScope from the repository root with its authentication gate."""

from pathlib import Path

import streamlit as st
from starlette.middleware import Middleware

from cardioscope.dashboard.auth import AuthGate, AuthPages, CookiePolicy
from cardioscope.dashboard.workspace_ui import store

policy = CookiePolicy.configured()
app = st.App(
    Path(__file__).with_name("streamlit_app.py"),
    routes=AuthPages(policy, store).routes(),
    middleware=[Middleware(AuthGate, policy=policy, get_store=store)],
)
