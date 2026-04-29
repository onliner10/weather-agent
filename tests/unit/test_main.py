from __future__ import annotations

import subprocess
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class TestCLIParsing:
    def test_bot_subcommand_parses(self) -> None:
        from weather_agent.__main__ import _build_parser

        parser = _build_parser()
        args = parser.parse_args(["bot"])
        assert args.command == "bot"
        assert hasattr(args, "func")

    def test_worker_subcommand_parses(self) -> None:
        from weather_agent.__main__ import _build_parser

        parser = _build_parser()
        args = parser.parse_args(["worker"])
        assert args.command == "worker"
        assert hasattr(args, "func")

    def test_missing_subcommand_raises(self) -> None:
        from weather_agent.__main__ import _build_parser

        parser = _build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args([])

    def test_invalid_subcommand_raises(self) -> None:
        from weather_agent.__main__ import _build_parser

        parser = _build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["nonexistent"])


class TestMainEntryPoint:
    def test_main_dispatches_to_subcommand(self) -> None:
        from weather_agent.__main__ import main

        mock_func = MagicMock()
        with patch("weather_agent.__main__._build_parser") as mock_parser_cls:
            mock_parser = MagicMock()
            mock_parser_cls.return_value = mock_parser
            mock_args = MagicMock()
            mock_args.func = mock_func
            mock_parser.parse_args.return_value = mock_args
            main()
            mock_func.assert_called_once_with(mock_args)

    def test_python_m_weather_agent_shows_help(self) -> None:
        result = subprocess.run(
            [sys.executable, "-m", "weather_agent", "--help"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert result.returncode == 0
        assert "weather_agent" in result.stdout
        assert "bot" in result.stdout
        assert "worker" in result.stdout


class TestRunMigrations:
    def test_run_migrations_calls_alembic_upgrade(self) -> None:
        from weather_agent.infrastructure.db.setup import run_migrations

        with patch("alembic.command.upgrade") as mock_upgrade:
            run_migrations()
            mock_upgrade.assert_called_once()
            args = mock_upgrade.call_args
            assert args[0][1] == "head"

    def test_run_migrations_propagates_errors(self) -> None:
        from weather_agent.infrastructure.db.setup import run_migrations

        with patch(
            "alembic.command.upgrade",
            side_effect=RuntimeError("migration error"),
        ):
            with pytest.raises(RuntimeError, match="migration error"):
                run_migrations()


class TestDatabaseUrlNormalization:
    def test_normalizes_postgresql_url(self) -> None:
        from weather_agent.infrastructure.db.setup import (
            normalize_database_url as _normalize_database_url,
        )

        url = "postgresql://user:pass@localhost/db"
        result = _normalize_database_url(url)
        assert result == "postgresql+psycopg_async://user:pass@localhost/db"

    def test_normalizes_postgres_url(self) -> None:
        from weather_agent.infrastructure.db.setup import (
            normalize_database_url as _normalize_database_url,
        )

        url = "postgres://user:pass@localhost/db"
        result = _normalize_database_url(url)
        assert result == "postgresql+psycopg_async://user:pass@localhost/db"

    def test_normalizes_psycopg_url(self) -> None:
        from weather_agent.infrastructure.db.setup import (
            normalize_database_url as _normalize_database_url,
        )

        url = "postgresql+psycopg://user:pass@localhost/db"
        result = _normalize_database_url(url)
        assert result == "postgresql+psycopg_async://user:pass@localhost/db"

    def test_leaves_async_url_unchanged(self) -> None:
        from weather_agent.infrastructure.db.setup import (
            normalize_database_url as _normalize_database_url,
        )

        url = "postgresql+psycopg_async://user:pass@localhost/db"
        result = _normalize_database_url(url)
        assert result == url

    def test_leaves_other_schemes_unchanged(self) -> None:
        from weather_agent.infrastructure.db.setup import (
            normalize_database_url as _normalize_database_url,
        )

        url = "sqlite+aiosqlite:///test.db"
        result = _normalize_database_url(url)
        assert result == url


class TestHomeLocationSaveHelpers:
    def test_extracts_address_from_zapamietaj_message(self) -> None:
        from weather_agent.adapters.telegram.home_location import (
            extract_home_location_request as _extract_home_location_request,
        )

        result = _extract_home_location_request(
            "Zapamiętaj moją lokalizację domową jako Rogalińska 11, Gdańsk."
        )
        assert result == "Rogalińska 11, Gdańsk"

    def test_ignores_weather_question(self) -> None:
        from weather_agent.adapters.telegram.home_location import (
            extract_home_location_request as _extract_home_location_request,
        )

        assert _extract_home_location_request("jaka pogoda za dwa dni?") is None

    @pytest.mark.asyncio
    async def test_home_location_save_persists_with_db_user_id(self) -> None:
        from sqlalchemy import select
        from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

        from weather_agent.adapters.telegram.home_location import (
            handle_home_location_save_message as _handle_home_location_save_message,
        )
        from weather_agent.domain.locations import LocationService
        from weather_agent.domain.weather import LocationRef
        from weather_agent.infrastructure.db.base import AuthorizedUser, Base, Location
        from weather_agent.infrastructure.repositories.auth_repository import AuthRepository

        engine = create_async_engine("sqlite+aiosqlite:///:memory:")
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

        geocoder = MagicMock()
        geocoder.geocode = AsyncMock(
            return_value=LocationRef(
                id="geo-1",
                name="Gdańsk",
                latitude=54.352,
                longitude=18.6466,
            )
        )

        async with factory() as session:
            auth_repo = AuthRepository(session)
            db_user_id = await auth_repo.get_or_create_authorized_user_id(7431473393)
            answer = await _handle_home_location_save_message(
                "Zapamiętaj moją lokalizację domową jako Rogalińska 11, Gdańsk.",
                db_user_id,
                LocationService(session),
                geocoder,
            )
            await session.commit()

        async with factory() as session:
            user = (
                await session.execute(
                    select(AuthorizedUser).where(AuthorizedUser.telegram_user_id == 7431473393)
                )
            ).scalar_one()
            location = (
                await session.execute(select(Location).where(Location.user_id == user.id))
            ).scalar_one()
            fallback = await LocationService(session).get_default_location(user.id)

        assert answer == "Zapamiętałem Twoją lokalizację domową jako Rogalińska 11, Gdańsk."
        assert location.name == "Rogalińska 11, Gdańsk"
        assert location.aliases == ["dom"]
        assert fallback is not None
        assert fallback.name == "Rogalińska 11, Gdańsk"
        await engine.dispose()


class TestMissingConfiguration:
    def test_missing_env_vars_exits_with_error(self) -> None:
        from weather_agent.infrastructure.services import BotServices as _BotServices

        with patch(
            "weather_agent.settings.load_settings",
            side_effect=Exception("Validation error: database_url"),
        ):
            with pytest.raises(SystemExit) as exc_info:
                _BotServices()
            assert exc_info.value.code == 1

    def test_missing_settings_produces_clear_error_message(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        from weather_agent.infrastructure.services import BotServices as _BotServices

        with patch(
            "weather_agent.settings.load_settings",
            side_effect=Exception("field required"),
        ):
            with pytest.raises(SystemExit):
                _BotServices()
            captured = capsys.readouterr()
            assert "Error loading configuration" in captured.err
            assert "environment variables" in captured.err


class TestCmdBot:
    def test_cmd_bot_calls_run_migrations(self) -> None:
        from weather_agent.cmd.bot import cmd_bot

        mock_services = MagicMock()
        mock_services.settings.telegram = MagicMock()
        mock_services.settings.telegram.allowed_user_ids = ()
        mock_services.settings.observability.enabled = False

        with (
            patch("weather_agent.cmd.bot.acquire_lock", return_value="/tmp/bot.pid"),
            patch("weather_agent.cmd.bot.BotServices", return_value=mock_services),
            patch("weather_agent.cmd.bot.run_migrations") as mock_migrate,
            patch("weather_agent.cmd.bot.asyncio") as mock_asyncio,
            patch("weather_agent.adapters.telegram.bot.TelegramBot"),
        ):
            mock_asyncio.new_event_loop.return_value = MagicMock()
            mock_asyncio.run = MagicMock()

            cmd_bot(MagicMock())
            mock_migrate.assert_called_once()

    def test_cmd_bot_creates_and_runs_telegram_bot(self) -> None:
        from weather_agent.cmd.bot import cmd_bot

        mock_services = MagicMock()
        mock_services.settings.telegram = MagicMock()
        mock_services.settings.telegram.allowed_user_ids = ()
        mock_services.settings.observability.enabled = False

        with (
            patch("weather_agent.cmd.bot.acquire_lock", return_value="/tmp/bot.pid"),
            patch("weather_agent.cmd.bot.BotServices", return_value=mock_services),
            patch("weather_agent.cmd.bot.run_migrations"),
            patch("weather_agent.cmd.bot.asyncio") as mock_asyncio,
            patch("weather_agent.adapters.telegram.bot.TelegramBot") as mock_bot_cls,
        ):
            mock_bot = MagicMock()
            mock_bot_cls.return_value = mock_bot
            mock_loop = MagicMock()
            mock_asyncio.new_event_loop.return_value = mock_loop
            mock_asyncio.run = MagicMock()

            cmd_bot(MagicMock())
            mock_bot.setup.assert_called_once()
            mock_bot.run.assert_called_once()

    def test_cmd_bot_migration_failure_is_warned(self) -> None:
        from weather_agent.cmd.bot import cmd_bot

        mock_services = MagicMock()
        mock_services.settings.telegram = MagicMock()
        mock_services.settings.telegram.allowed_user_ids = ()
        mock_services.settings.observability.enabled = False

        with (
            patch("weather_agent.cmd.bot.acquire_lock", return_value="/tmp/bot.pid"),
            patch("weather_agent.cmd.bot.BotServices", return_value=mock_services),
            patch(
                "weather_agent.cmd.bot.run_migrations",
                side_effect=RuntimeError("db not found"),
            ),
            patch("weather_agent.cmd.bot.asyncio") as mock_asyncio,
            patch("weather_agent.adapters.telegram.bot.TelegramBot") as mock_bot_cls,
        ):
            mock_bot = MagicMock()
            mock_bot_cls.return_value = mock_bot
            mock_asyncio.new_event_loop.return_value = MagicMock()
            mock_asyncio.run = MagicMock()

            cmd_bot(MagicMock())
            mock_bot.setup.assert_called_once()

    def test_cmd_bot_starts_observability_server_when_enabled(self) -> None:
        from weather_agent.cmd.bot import cmd_bot

        mock_services = MagicMock()
        mock_services.settings.telegram = MagicMock()
        mock_services.settings.telegram.allowed_user_ids = ()
        mock_services.settings.observability.enabled = True
        mock_services.settings.observability.bot_port = 9999

        with (
            patch("weather_agent.cmd.bot.acquire_lock", return_value="/tmp/bot.pid"),
            patch("weather_agent.cmd.bot.BotServices", return_value=mock_services),
            patch("weather_agent.cmd.bot.run_migrations"),
            patch("weather_agent.cmd.bot.asyncio") as mock_asyncio,
            patch("weather_agent.adapters.telegram.bot.TelegramBot"),
            patch("weather_agent.cmd.bot.start_observability_server") as mock_start_server,
        ):
            mock_asyncio.new_event_loop.return_value = MagicMock()
            mock_asyncio.run = MagicMock()

            cmd_bot(MagicMock())
            mock_start_server.assert_called_once()
            call_kwargs = mock_start_server.call_args.kwargs
            assert call_kwargs["port"] == 9999
            assert call_kwargs["host"] == "0.0.0.0"


class TestCmdWorker:
    def test_cmd_worker_calls_run_migrations(self) -> None:
        from weather_agent.cmd.worker import cmd_worker

        mock_services = MagicMock()
        mock_services.settings.scheduler = MagicMock()
        mock_services.settings.observability.enabled = False

        with (
            patch("weather_agent.cmd.worker.acquire_lock", return_value="/tmp/worker.pid"),
            patch("weather_agent.cmd.worker.BotServices", return_value=mock_services),
            patch("weather_agent.cmd.worker.run_migrations") as mock_migrate,
            patch("weather_agent.cmd.worker.asyncio") as mock_asyncio,
        ):
            mock_asyncio.run = MagicMock()
            cmd_worker(MagicMock())
            mock_migrate.assert_called_once()

    def test_cmd_worker_starts_observability_server_when_enabled(self) -> None:
        from weather_agent.cmd.worker import cmd_worker

        mock_services = MagicMock()
        mock_services.settings.scheduler = MagicMock()
        mock_services.settings.observability.enabled = True
        mock_services.settings.observability.worker_port = 9998

        with (
            patch("weather_agent.cmd.worker.acquire_lock", return_value="/tmp/worker.pid"),
            patch("weather_agent.cmd.worker.BotServices", return_value=mock_services),
            patch("weather_agent.cmd.worker.run_migrations"),
            patch("weather_agent.cmd.worker.asyncio") as mock_asyncio,
            patch("weather_agent.cmd.worker.start_observability_server") as mock_start_server,
        ):
            mock_asyncio.run = MagicMock()
            cmd_worker(MagicMock())
            mock_start_server.assert_called_once()
            call_kwargs = mock_start_server.call_args.kwargs
            assert call_kwargs["port"] == 9998
            assert call_kwargs["host"] == "0.0.0.0"


class TestCreateEngine:
    def test_create_engine_returns_async_engine(self) -> None:
        from sqlalchemy.ext.asyncio import AsyncEngine as AE

        from weather_agent.infrastructure.db.setup import create_engine as _create_engine

        engine = _create_engine("sqlite+aiosqlite:///test.db")
        assert isinstance(engine, AE)

    def test_create_session_factory_returns_sessionmaker(self) -> None:
        from weather_agent.infrastructure.db.setup import (
            create_engine as _create_engine,
        )
        from weather_agent.infrastructure.db.setup import (
            create_session_factory as _create_session_factory,
        )

        engine = _create_engine("sqlite+aiosqlite:///test.db")
        factory = _create_session_factory(engine)
        assert factory is not None


class TestSessionFactory:
    @pytest.mark.asyncio
    async def test_session_factory_creates_working_session(self) -> None:
        from weather_agent.infrastructure.db.setup import (
            create_engine as _create_engine,
        )
        from weather_agent.infrastructure.db.setup import (
            create_session_factory as _create_session_factory,
        )

        engine = _create_engine("sqlite+aiosqlite:///test.db")
        factory = _create_session_factory(engine)

        async with factory() as session:
            assert session is not None
            await session.commit()

        await engine.dispose()

    @pytest.mark.asyncio
    async def test_session_rollback_on_error(self) -> None:
        from weather_agent.infrastructure.db.setup import (
            create_engine as _create_engine,
        )
        from weather_agent.infrastructure.db.setup import (
            create_session_factory as _create_session_factory,
        )

        engine = _create_engine("sqlite+aiosqlite:///test.db")
        factory = _create_session_factory(engine)

        with pytest.raises(ValueError):
            async with factory() as session:
                assert session is not None
                raise ValueError("test error")

        await engine.dispose()
