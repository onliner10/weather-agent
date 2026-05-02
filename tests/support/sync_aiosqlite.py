from __future__ import annotations

import sqlite3
from collections.abc import Callable, Iterable
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import aiosqlite


class _ImmediateQueue:
    def put_nowait(self, item: tuple[Any, Callable[[], Any]]) -> None:
        future, function = item
        try:
            result = function()
        except Exception as exc:
            if future is not None and not future.done():
                future.set_exception(exc)
            return
        if future is not None and not future.done():
            future.set_result(result)


class _SyncCursor:
    def __init__(self, cursor: sqlite3.Cursor) -> None:
        self._cursor = cursor

    async def execute(self, sql: str, parameters: Iterable[Any] | None = None) -> _SyncCursor:
        self._cursor.execute(sql, [] if parameters is None else parameters)
        return self

    async def executemany(self, sql: str, parameters: Iterable[Iterable[Any]]) -> _SyncCursor:
        self._cursor.executemany(sql, parameters)
        return self

    async def executescript(self, sql_script: str) -> _SyncCursor:
        self._cursor.executescript(sql_script)
        return self

    async def fetchone(self) -> sqlite3.Row | tuple[Any, ...] | None:
        return self._cursor.fetchone()

    async def fetchmany(self, size: int | None = None) -> list[sqlite3.Row | tuple[Any, ...]]:
        return self._cursor.fetchmany(size) if size is not None else self._cursor.fetchmany()

    async def fetchall(self) -> list[sqlite3.Row | tuple[Any, ...]]:
        return self._cursor.fetchall()

    async def close(self) -> None:
        self._cursor.close()

    async def __aenter__(self) -> _SyncCursor:
        return self

    async def __aexit__(self, *_args: object) -> None:
        await self.close()

    @property
    def arraysize(self) -> int:
        return self._cursor.arraysize

    @arraysize.setter
    def arraysize(self, value: int) -> None:
        self._cursor.arraysize = value

    @property
    def description(self) -> object:
        return self._cursor.description

    @property
    def lastrowid(self) -> int | None:
        return self._cursor.lastrowid

    @property
    def rowcount(self) -> int:
        return self._cursor.rowcount

    @property
    def row_factory(self) -> Callable[..., object] | None:
        return self._cursor.row_factory

    @row_factory.setter
    def row_factory(self, factory: Callable[..., object] | None) -> None:
        self._cursor.row_factory = factory

    @property
    def connection(self) -> sqlite3.Connection:
        return self._cursor.connection


class _SyncConnection:
    def __init__(
        self,
        database: str | bytes | Path,
        *,
        iter_chunk_size: int,
        **kwargs: Any,
    ) -> None:
        del iter_chunk_size
        self._database = database
        self._kwargs = kwargs
        self._connection: sqlite3.Connection | None = None
        self._tx = _ImmediateQueue()
        self._thread = SimpleNamespace(daemon=True)

    @property
    def _conn(self) -> sqlite3.Connection:
        if self._connection is None:
            raise ValueError("no active connection")
        return self._connection

    async def _connect(self) -> _SyncConnection:
        if self._connection is None:
            self._connection = sqlite3.connect(self._database, **self._kwargs)
        return self

    def __await__(self) -> Any:
        return self._connect().__await__()

    async def __aenter__(self) -> _SyncConnection:
        return await self

    async def __aexit__(self, *_args: object) -> None:
        await self.close()

    async def _execute(self, function: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        return function(*args, **kwargs)

    async def cursor(self) -> _SyncCursor:
        return _SyncCursor(self._conn.cursor())

    async def execute(self, sql: str, parameters: Iterable[Any] | None = None) -> _SyncCursor:
        return await _SyncCursor(self._conn.cursor()).execute(sql, parameters)

    async def executemany(self, sql: str, parameters: Iterable[Iterable[Any]]) -> _SyncCursor:
        return await _SyncCursor(self._conn.cursor()).executemany(sql, parameters)

    async def executescript(self, sql_script: str) -> _SyncCursor:
        return await _SyncCursor(self._conn.cursor()).executescript(sql_script)

    async def execute_insert(
        self, sql: str, parameters: Iterable[Any] | None = None
    ) -> sqlite3.Row | tuple[Any, ...] | None:
        cursor = self._conn.execute(sql, [] if parameters is None else parameters)
        cursor.execute("SELECT last_insert_rowid()")
        return cursor.fetchone()

    async def execute_fetchall(
        self, sql: str, parameters: Iterable[Any] | None = None
    ) -> list[sqlite3.Row | tuple[Any, ...]]:
        cursor = self._conn.execute(sql, [] if parameters is None else parameters)
        return cursor.fetchall()

    async def commit(self) -> None:
        self._conn.commit()

    async def rollback(self) -> None:
        self._conn.rollback()

    async def close(self) -> None:
        if self._connection is not None:
            self._connection.close()
            self._connection = None

    def stop(self) -> None:
        if self._connection is not None:
            self._connection.close()
            self._connection = None

    async def interrupt(self) -> None:
        self._conn.interrupt()

    async def create_function(
        self,
        name: str,
        num_params: int,
        func: Callable[..., object],
        deterministic: bool = False,
    ) -> None:
        self._conn.create_function(name, num_params, func, deterministic=deterministic)

    @property
    def in_transaction(self) -> bool:
        return self._conn.in_transaction

    @property
    def isolation_level(self) -> str | None:
        return self._conn.isolation_level

    @isolation_level.setter
    def isolation_level(self, value: str | None) -> None:
        self._conn.isolation_level = value

    @property
    def row_factory(self) -> Callable[..., object] | None:
        return self._conn.row_factory

    @row_factory.setter
    def row_factory(self, factory: Callable[..., object] | None) -> None:
        self._conn.row_factory = factory

    @property
    def text_factory(self) -> Callable[[bytes], object]:
        return self._conn.text_factory

    @text_factory.setter
    def text_factory(self, factory: Callable[[bytes], object]) -> None:
        self._conn.text_factory = factory

    @property
    def total_changes(self) -> int:
        return self._conn.total_changes


def install_sync_aiosqlite() -> None:
    def connect(
        database: str | bytes | Path,
        *,
        iter_chunk_size: int = 64,
        loop: object | None = None,
        **kwargs: Any,
    ) -> _SyncConnection:
        del loop
        return _SyncConnection(database, iter_chunk_size=iter_chunk_size, **kwargs)

    aiosqlite.connect = connect  # type: ignore[assignment]
