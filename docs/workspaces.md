# Phase 6.1 personal workspaces

The default workflow is a trusted, loopback-only Streamlit instance. Follow the [README setup](../README.md#explore-the-research-dashboard) with the dashboard extra. The eight research pages remain available to guests. SQLite works locally without a database service. Accounts use the same database configuration as the experiment registry. The original 102 verified aggregate experiments stay in the `experiments` table and appear only in the research catalog.

## Use the workspace

1. Open **Account**, register a unique username, and sign in. A new account has default presentation settings, an empty profile and report history, and no selected sandbox example. The sandbox waits for a selection and explicit submission.
2. Save **Settings** for grid lines, the advanced model panel, and the default model family, feature tier, and held-out hospital. These defaults take effect on the relevant research pages. View filters can still be changed for the current session.
3. In **My Reports**, choose a built-in fictional example and give it a private name. Choose a fictional name that contains no personal information. The saved profile can then be selected in **Prediction Sandbox**. It points to one of three built-in examples and contains no feature vector.
4. Click **Validate & run fictional profile** for a transient result, or **Run & save private report** for the unedited built-in example. The verified pipeline scores the example. Edited values remain transient even when you have an account. If the trusted model is missing or incompatible, inference remains disabled and history remains available.
5. Open a result under **My Reports** to see its title, historical class, score, threshold, timestamp and model provenance. Rename or delete it there. The clear-history action has a separate confirmation and affects only your account.
6. Review the sharing preview and explicitly opt in to show a result in **Shared Fictional Reports**. Anyone with access to this app, including signed-out viewers, can see those fields. The listing shows only the built-in example label, model, software version, feature tier, prediction digest, score, class and educational disclaimer. It omits the account, username, saved title, private profile label, IDs, timestamps, and input values. Sharing persists through sign-out. Revocation and deletion remove the row from subsequent listings. A viewer who copied a result can retain that copy.

## Authentication and ownership

Usernames are normalized to lowercase and must consist of 3–32 ASCII letters, digits or underscores, starting with a letter. Registration requires a password of 12–128 characters. Argon2id uses a fresh salt with 64 MiB of memory, three iterations and four lanes. Authentication never writes the password. Invalid logins return the same message for unknown accounts and wrong passwords, and unknown accounts incur a dummy Argon2id check. Registration failure does not disclose an existing account record. A successful registration naturally confirms a name was available.

Sign-in creates a random 256-bit grant kept in server-side Streamlit session state. The database stores its SHA-256 digest with an eight-hour absolute expiry. Each private operation resolves this grant to an account on the server and scopes queries by that account ID. Sign-out invalidates the database grant and clears that tab’s server-held workspace state. A refresh or process restart requires a fresh sign-in when Streamlit loses the session. Settings, profiles and reports persist independently of any one browser session.

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
.\.venv\Scripts\python.exe -m streamlit run app/streamlit_app.py
```

For a code rollback, stop the app and switch back to the earlier branch. The earlier registry ignores the added workspace tables. Use a restored pre-migration backup if you need to remove the schema entirely. This preserves the current database for recovery. For PostgreSQL, use your normal snapshot/restore tooling with restricted credentials and test it on a separate server.

## Hosting readiness

This is verified for trusted local operation. Public hosting is outside that verification. Before exposing it, use HTTPS and review proxy trust, WebSocket origin checks, XSRF, Secure/HttpOnly/SameSite handling for applicable cookies, session lifecycle across replicas, and an identity provider or equivalent account lifecycle controls. Add per-client rate limiting, password reset or federated sign-in, operational monitoring and retention procedures, a dedicated least-privilege database role, tested encrypted backups, credential rotation and separate storage from any other deployment. Evaluate abusive content in profile names and shared-result volume within your hosting policy. Live PostgreSQL and public hosting still require a deployment-specific security review. The app does not accept medical data for clinical use.

The current user workspaces extend the locally audited Phase 6 dashboard. The [Phase 6 audit](release-audit.md) describes the earlier, transient-only sandbox. This guide describes the optional persistence now available for explicitly saved fictional examples.
