from contextlib import closing, contextmanager
from hashlib import sha256
import json
import secrets
import sqlite3
from time import time

from backend.CONFIG.auth import auth_settings
from backend.CORE.auth.policy import Identity


def _key(token: str) -> str:
    return sha256(token.encode()).hexdigest()


class SessionStore:
    def __init__(self, path, clock=time):
        self.path = path
        self.clock = clock

    @contextmanager
    def _db(self):
        self.path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        with closing(sqlite3.connect(self.path, timeout=5)) as connection, connection:
            connection.execute(
                "CREATE TABLE IF NOT EXISTS sessions (token TEXT PRIMARY KEY, payload TEXT NOT NULL, expiry REAL NOT NULL)"
            )
            connection.execute(
                "CREATE TABLE IF NOT EXISTS login_flows (token TEXT PRIMARY KEY, payload TEXT NOT NULL, expiry REAL NOT NULL)"
            )
            yield connection

    def begin_login(self, destination: str, lifetime: int) -> tuple[str, str, str]:
        state, nonce, verifier = (secrets.token_urlsafe(32) for _ in range(3))
        now = self.clock()
        with self._db() as db:
            db.execute("DELETE FROM login_flows WHERE expiry <= ?", (now,))
            db.execute(
                "INSERT INTO login_flows VALUES (?, ?, ?)",
                (_key(state), json.dumps([destination, nonce, verifier]), now + lifetime),
            )
        return state, nonce, verifier

    def consume_login(self, state: str) -> tuple[str, str, str] | None:
        with self._db() as db:
            row = db.execute(
                "DELETE FROM login_flows WHERE token = ? AND expiry > ? RETURNING payload",
                (_key(state), self.clock()),
            ).fetchone()
        return tuple(json.loads(row[0])) if row else None

    def create(self, identity: Identity, lifetime: int, previous: str = "") -> str:
        token = secrets.token_urlsafe(32)
        now = self.clock()
        with self._db() as db:
            db.execute("DELETE FROM sessions WHERE expiry <= ? OR token = ?", (now, _key(previous)))
            db.execute(
                "INSERT INTO sessions VALUES (?, ?, ?)",
                (_key(token), json.dumps(vars(identity)), now + lifetime),
            )
        return token

    def identity(self, token: str) -> Identity | None:
        with self._db() as db:
            row = db.execute(
                "SELECT payload FROM sessions WHERE token = ? AND expiry > ?",
                (_key(token), self.clock()),
            ).fetchone()
        return Identity(**json.loads(row[0])) if row else None

    def revoke(self, token: str) -> None:
        with self._db() as db:
            db.execute("DELETE FROM sessions WHERE token = ?", (_key(token),))


session_store = SessionStore(auth_settings.session_db)
