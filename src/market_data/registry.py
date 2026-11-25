from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import polars as pl


class Registry:
    def __init__(self, root: Path):
        self.path = root / "metadata" / "registry.sqlite"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.path) as db:
            db.execute("""CREATE TABLE IF NOT EXISTS datasets (
                source TEXT NOT NULL, ticker TEXT NOT NULL, interval TEXT NOT NULL,
                file_path TEXT NOT NULL, first_timestamp TEXT, last_timestamp TEXT,
                observed_count INTEGER NOT NULL, file_size INTEGER NOT NULL,
                status TEXT NOT NULL, last_successful_download TEXT,
                error_message TEXT, schema_version INTEGER NOT NULL DEFAULT 1,
                PRIMARY KEY(source, ticker, interval))""")

    def upsert(
        self,
        *,
        source: str,
        ticker: str,
        interval: str,
        file_path: str,
        frame: pl.DataFrame,
        file_size: int,
        status: str = "success",
        error_message: str | None = None,
    ) -> None:
        first = frame["timestamp"].min().isoformat() if not frame.is_empty() else None
        last = frame["timestamp"].max().isoformat() if not frame.is_empty() else None
        now = datetime.now(timezone.utc).isoformat()
        with sqlite3.connect(self.path) as db:
            db.execute(
                """INSERT INTO datasets VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1)
                ON CONFLICT(source, ticker, interval) DO UPDATE SET
                file_path=excluded.file_path, first_timestamp=excluded.first_timestamp,
                last_timestamp=excluded.last_timestamp, observed_count=excluded.observed_count,
                file_size=excluded.file_size, status=excluded.status,
                last_successful_download=excluded.last_successful_download,
                error_message=excluded.error_message""",
                (
                    source,
                    ticker,
                    interval,
                    file_path,
                    first,
                    last,
                    frame.height,
                    file_size,
                    status,
                    now if status == "success" else None,
                    error_message,
                ),
            )

    def list(self) -> list[tuple]:
        with sqlite3.connect(self.path) as db:
            return db.execute("SELECT * FROM datasets ORDER BY ticker").fetchall()
