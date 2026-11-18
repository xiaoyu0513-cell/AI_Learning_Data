"""SQLite manifest: prepare external snapshots first, atomically publish last.

This is a single-host manifest, not a distributed transaction. Unpublished
external resources may remain after a crash; the previous serving version stays.
"""
import hashlib
import json
import sqlite3
import os
from contextlib import contextmanager
from pathlib import Path
from uuid import uuid4


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":")).encode()).hexdigest()


class IndexStore:
    def __init__(self, directory):
        self.path = Path(directory) / "indexes.sqlite3"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as c:
            c.executescript("""
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS versions (
                    id TEXT PRIMARY KEY, fingerprint TEXT, collection TEXT,
                    payload TEXT, documents TEXT, status TEXT, error TEXT);
                CREATE TABLE IF NOT EXISTS active (id INTEGER PRIMARY KEY CHECK(id=1), version TEXT);
                CREATE TABLE IF NOT EXISTS vectors (key TEXT PRIMARY KEY, value TEXT);
            """)

    @contextmanager
    def connect(self):
        c = sqlite3.connect(self.path, timeout=30)
        c.row_factory = sqlite3.Row
        try:
            with c:
                yield c
        finally:
            c.close()

    @contextmanager
    def build_lock(self):
        # OS lock is released on crash, unlike a persistent boolean lock row.
        lock_path = self.path.with_suffix(".lock")
        with lock_path.open("a+b") as f:
            if os.fstat(f.fileno()).st_size == 0:
                f.write(b"0")
                f.flush()
            f.seek(0)
            try:
                if os.name == "nt":
                    import msvcrt
                    msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as e:
                raise RuntimeError("另一个进程正在构建索引") from e
            try:
                yield
            finally:
                f.seek(0)
                if os.name == "nt":
                    msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    fcntl.flock(f, fcntl.LOCK_UN)

    def active(self):
        with self.connect() as c:
            row = c.execute("SELECT v.* FROM versions v JOIN active a ON a.version=v.id WHERE a.id=1").fetchone()
        return self._decode(row)

    @staticmethod
    def _decode(row):
        if not row:
            return None
        value = dict(row)
        for field in ("payload", "documents"):
            value[field] = json.loads(value[field])
        return value

    def prepare(self, fingerprint, collection_prefix, payload, documents):
        version = uuid4().hex
        collection = f"{collection_prefix}_v_{version}"
        with self.connect() as c:
            c.execute("INSERT INTO versions VALUES (?,?,?,?,?,?,?)", (
                version, fingerprint, collection, json.dumps(payload, ensure_ascii=False),
                json.dumps(documents, ensure_ascii=False), "building", None))
        return version, collection

    def publish(self, version):
        with self.connect() as c:
            row = c.execute("SELECT status FROM versions WHERE id=?", (version,)).fetchone()
            if not row or row[0] not in ("building", "ready"):
                raise ValueError("无法发布不存在或失败的版本")
            c.execute("UPDATE versions SET status='ready', error=NULL WHERE id=?", (version,))
            c.execute("INSERT INTO active VALUES(1,?) ON CONFLICT(id) DO UPDATE SET version=excluded.version", (version,))

    def fail(self, version, error):
        with self.connect() as c:
            c.execute("UPDATE versions SET status='failed',error=? WHERE id=?", (type(error).__name__, version))

    def get_vector(self, key):
        with self.connect() as c:
            row = c.execute("SELECT value FROM vectors WHERE key=?", (key,)).fetchone()
        return json.loads(row[0]) if row else None

    def put_vector(self, key, vector):
        with self.connect() as c:
            c.execute("INSERT OR REPLACE INTO vectors VALUES(?,?)", (key, json.dumps(vector)))

    def versions(self):
        with self.connect() as c:
            return [dict(r) for r in c.execute("SELECT id,collection,status,error FROM versions ORDER BY rowid DESC")]

    def get_version(self, version):
        with self.connect() as c:
            row = c.execute("SELECT * FROM versions WHERE id=? AND status='ready'", (version,)).fetchone()
        return self._decode(row)
