import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, List

from src.config import SQLITE_DB_PATH


DB_PATH = Path(SQLITE_DB_PATH)


def get_connection() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(exist_ok=True)
    connection = sqlite3.connect(DB_PATH)
    connection.row_factory = sqlite3.Row
    initialize_database(connection)
    return connection


def initialize_database(connection: sqlite3.Connection) -> None:
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS query_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            request_id TEXT UNIQUE,
            timestamp TEXT NOT NULL,
            payload TEXT NOT NULL
        )
        """
    )
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS feedback_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            request_id TEXT,
            timestamp TEXT NOT NULL,
            payload TEXT NOT NULL
        )
        """
    )
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS ticket_drafts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            request_id TEXT UNIQUE,
            created_at TEXT NOT NULL,
            status TEXT NOT NULL,
            payload TEXT NOT NULL,
            integration_result TEXT NOT NULL
        )
        """
    )
    connection.commit()


def write_query_log(entry: Dict[str, Any]) -> None:
    with get_connection() as connection:
        connection.execute(
            """
            INSERT OR REPLACE INTO query_logs (request_id, timestamp, payload)
            VALUES (?, ?, ?)
            """,
            (
                entry.get("request_id"),
                entry.get("timestamp", ""),
                json.dumps(entry),
            ),
        )
        connection.commit()


def write_feedback_log(entry: Dict[str, Any]) -> None:
    with get_connection() as connection:
        connection.execute(
            """
            INSERT INTO feedback_logs (request_id, timestamp, payload)
            VALUES (?, ?, ?)
            """,
            (
                entry.get("request_id"),
                entry.get("timestamp", ""),
                json.dumps(entry),
            ),
        )
        connection.commit()


def write_ticket_draft(request_id: str, ticket_draft: Dict[str, Any], integration_result: Dict[str, Any]) -> None:
    with get_connection() as connection:
        connection.execute(
            """
            INSERT OR REPLACE INTO ticket_drafts (request_id, created_at, status, payload, integration_result)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                request_id,
                integration_result.get("timestamp", ""),
                integration_result.get("status", "prepared"),
                json.dumps(ticket_draft),
                json.dumps(integration_result),
            ),
        )
        connection.commit()


def read_table_payloads(table_name: str, limit: int = 100) -> List[Dict[str, Any]]:
    if table_name not in {"query_logs", "feedback_logs", "ticket_drafts"}:
        raise ValueError("Unsupported table name.")

    with get_connection() as connection:
        rows = connection.execute(
            f"SELECT payload FROM {table_name} ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall()

    return [json.loads(row["payload"]) for row in reversed(rows)]


def database_health() -> Dict[str, Any]:
    try:
        with get_connection() as connection:
            connection.execute("SELECT 1").fetchone()
        return {
            "status": "ok",
            "backend": "sqlite",
            "path": str(DB_PATH),
        }
    except Exception as error:
        return {
            "status": "error",
            "backend": "sqlite",
            "path": str(DB_PATH),
            "error": str(error),
        }
