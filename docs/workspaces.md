# Personal workspaces and authentication

The default workflow is a trusted, loopback-only Streamlit instance. Follow the [README setup](../README.md#explore-the-research-dashboard) with the dashboard extra and launch `app/serve.py`. All dashboard pages require a valid account session. The verified research files remain independently available in the repository. SQLite works locally without a database service. Accounts use the same database configuration as the experiment registry. The original 102 verified aggregate experiments stay in the `experiments` table and appear only in the research catalog.

Existing Phase 6.1 users can sign in with their current credentials. The first sign-in after upgrading issues a browser cookie and opens the existing workspace. The database schema and saved records carry forward.

## Use the workspace

1. From **Welcome**, sign in or create an account. Registration signs you in automatically and opens **Research Overview**. A new account has default presentation settings, an empty profile and report history, and no selected sandbox example. The sandbox waits for a selection and explicit submission.
2. Save **Settings** for grid lines, the advanced model panel, and the default model family, feature tier, and held-out hospital. These defaults take effect on the relevant research pages. View filters can still be changed for the current session.
3. In **My Reports**, choose a built-in fictional example and give it a private name. Choose a fictional name that contains no personal information. The saved profile can then be selected in **Prediction Sandbox**. It points to one of three built-in examples and contains no feature vector.
4. Click **Validate & run fictional profile** for a transient result, or **Run & save private report** for the unedited built-in example. The verified pipeline scores the example. Edited values remain transient even when you have an account. If the trusted model is missing or incompatible, inference remains disabled and history remains available.
5. Open a result under **My Reports** to see its title, historical class, score, threshold, timestamp and model provenance. Rename or delete it there. The clear-history action has a separate confirmation and affects only your account.
6. Review the sharing preview and explicitly opt in to show a result to other signed-in users in **Shared Fictional Reports**. The listing shows only the built-in example label, model, software version, feature tier, prediction digest, score, class and educational disclaimer. It omits the account, username, saved title, private profile label, IDs, timestamps, and input values. Sharing persists through logout. Revocation and deletion remove the row from subsequent listings. A viewer who copied a result can retain that copy.

## Authentication and ownership

Usernames are normalized to lowercase and must consist of 3–32 ASCII letters, digits or underscores, starting with a letter. Registration requires a password of 12–128 characters. Argon2id uses a fresh salt with 64 MiB of memory, three iterations and four lanes. Authentication never writes the password. Invalid logins return the same message for unknown accounts and wrong passwords, and unknown accounts incur a dummy Argon2id check. Registration failure does not disclose an existing account record. A successful registration naturally confirms a name was available.

Sign-in issues a random 256-bit grant in a host-only, HttpOnly, SameSite=Strict cookie. The database stores only the SHA-256 digest, owner and expiration. The default **Keep me signed in on this device** option gives the cookie and database grant a 30-day absolute lifetime. Refresh, activity and reopening the browser never extend that deadline. Sign in again to start a new period. Clearing the option sets a browser-session cookie with an eight-hour absolute server limit. A browser's session restore feature can preserve browser-session cookies within that limit.

The Starlette middleware used by `st.App` checks HTTP routes and WebSocket upgrades before dispatch. The dashboard reads the handshake cookie on the server and resolves it again on each rerun. Private operations check the grant and query by the authenticated account ID. The app disables resuming disconnected Streamlit widget sessions across socket connections. Settings, profiles, and saved reports persist in the database.

**Logout** in the sidebar sends a same-origin POST. It revokes that grant, clears its cookie, and opens Welcome. It leaves the account and its saved records available after the next sign-in. Private HTML and data responses carry no-store headers. Other open tabs recheck their own grant periodically and return to Welcome when it has been revoked or has expired. An invalid credential is cleared on the next HTTP visit. The app uses cookies for browser credentials and keeps them out of Streamlit session state, URLs, and browser script storage.

The database limits login attempts to six per normalized account and 30 across the instance per 15-minute window. Registration allows 12 attempts per instance per window. The counters apply to all sessions and persist across restarts. Two concurrent Argon2 jobs can run per process. These limits suit a small local demo. Shared-instance quotas can inconvenience legitimate users during bursts and need an upstream per-client protection plan before hosting.

Reports store their server-generated identifier, owner identifier, private title and profile label, canonical example, model family, scikit-learn version, feature tier, the verified Phase 3 holdout probability digest, exact score, class, threshold, UTC creation time and disclaimer. Saved reports start private. Shared fields are selected directly from a fixed database projection. Raw prompt data, copied medical records, and arbitrary uploaded models are outside the workspace design.

## Database schema and migration

| Table | Purpose |
| --- | --- |
| `experiments` | Existing verified aggregate catalog. Its schema and rows are outside workspace migration. |
| `workspace_schema_version` | Singleton migration version. The current additive schema is version 1. |
| `workspace_users` | Stable account ID, unique normalized username, salted password hash and creation time. |
| `workspace_sessions` | Session-grant digest, owner and expiration. |
| `workspace_auth_attempts` | Bounded window counters for login and registration. |
| `workspace_preferences` | Validated settings keyed by account. |
| `workspace_fictional_profiles` | Owner, stable ID, unique per-owner normalized label and built-in example reference. |
| `workspace_prediction_reports` | Owner, stable ID, immutable inference facts and a private editable title. |
| `workspace_report_shares` | Explicit share permission keyed by report. Deleting a report cascades to its permission. |

The first workspace database access applies the versioned schema transactionally. SQLite acquires a write reservation while creating the schema and enables foreign keys on each workspace connection. The PostgreSQL path uses a transaction advisory lock and dialect-specific upserts. Opening a previously initialized database is idempotent. A higher, unsupported schema version fails closed. The registry can still be read or initialized through its own page.

### SQLite backup and restore

The default is `data/local/experiments.db`. Replace the source below if you configured a different local SQLite path. This uses SQLite’s backup API instead of copying the live database bytes. Backups contain account hashes and private report history. Store them privately, keep them outside Git, and use encryption and appropriate access controls for any off-device backup.

```powershell
@'
from pathlib import Path
from contextlib import closing
import sqlite3

source = Path("data/local/experiments.db").resolve()
backup = Path("data/local/workspace-backup.db").resolve()
if not source.is_file() or backup.exists():
    raise SystemExit("Source missing or backup destination already exists.")
with closing(sqlite3.connect(source.as_uri() + "?mode=ro", uri=True)) as src:
    with closing(sqlite3.connect(backup)) as dst:
        src.backup(dst)
        if dst.execute("PRAGMA integrity_check").fetchone() != ("ok",):
            raise SystemExit("Backup failed integrity check.")
'@ | .\.venv\Scripts\python.exe -
```

To restore, stop the app first. Copy a verified backup to a **new** path and point the app at it. Keep the original database until you verify the restored catalog and accounts. Start in a fresh browser and sign in again.

```powershell
$restored = "data/local/workspace-restored.db"
if (Test-Path $restored) { throw "Choose an unused restoration path." }
Copy-Item -LiteralPath "data/local/workspace-backup.db" -Destination $restored
$dbPath = (Resolve-Path -LiteralPath $restored).Path.Replace("\", "/")
$env:CARDIOSCOPE_DATABASE_URL = "sqlite:///$dbPath"
.\.venv\Scripts\python.exe -m streamlit run app/serve.py
```

For a code rollback, stop the app and switch back to the earlier branch. The earlier registry ignores the added workspace tables. Use a restored pre-migration backup if you need to remove the schema entirely. This preserves the current database for recovery. For PostgreSQL, use your normal snapshot/restore tooling with restricted credentials and test it on a separate server.

## Hosting readiness

**Local loopback.** Use `http://127.0.0.1:8501` on a trusted machine. Local mode also accepts `localhost`. Use the same hostname each time because cookies are host scoped. The loopback-only HTTP cookie uses HttpOnly and SameSite=Strict. It omits Secure for local HTTP compatibility. Run each mutually untrusted application on a separate hostname because cookies are scoped by host, independent of port.

**Production prerequisites.** Set `CARDIOSCOPE_PUBLIC_ORIGIN` to the exact external HTTPS origin, such as `https://cardioscope.example`. Serve the app at the origin root and keep the Streamlit listener private behind a TLS-terminating reverse proxy. The proxy must preserve the external Host, forward WebSocket upgrades and overwrite incoming forwarded headers. Configure the ASGI server to accept forwarded scheme information only from the trusted proxy, then verify that the app sees HTTPS for both requests and upgrades. The HTTPS cookie is Secure and uses the `__Host-` prefix. Test the combined proxy and Streamlit origin and XSRF controls before exposure.

Use durable database storage and a restricted service identity. A single instance can use a private persistent SQLite volume with verified encrypted backups. A multi-instance deployment needs a shared database, tested backups and a reviewed Streamlit routing strategy. Keep database credentials in a secret store and strip cookies and form bodies from proxy and application logs. Public hosting also needs per-client rate limits, account recovery or managed sign-in, retention and monitoring, and an independent deployment security review. Live PostgreSQL and internet hosting are unverified. The workspace is for fictional educational examples.

The [Phase 6 audit](release-audit.md) describes the earlier, transient-only sandbox. This guide covers the current authenticated dashboard and optional fictional report persistence.
