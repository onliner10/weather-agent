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


class TestMissingConfiguration:
    def test_missing_env_vars_exits_with_error(self) -> None:
        from weather_agent.infrastructure.app_container import AppContainer

        with patch(
            "weather_agent.settings.load_settings",
            side_effect=Exception("Validation error: database_url"),
        ):
            with pytest.raises(SystemExit) as exc_info:
                AppContainer()
            assert exc_info.value.code == 1

    def test_missing_settings_produces_clear_error_message(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        from weather_agent.infrastructure.app_container import AppContainer

        with patch(
            "weather_agent.settings.load_settings",
            side_effect=Exception("field required"),
        ):
            with pytest.raises(SystemExit):
                AppContainer()
            captured = capsys.readouterr()
            assert "Error loading configuration" in captured.err
            assert "environment variables" in captured.err


class TestCmdBot:
    def _make_mock_container(self) -> AsyncMock:
        container = AsyncMock()
        container.settings.telegram = MagicMock()
        container.settings.telegram.allowed_user_ids = ()
        session_result = MagicMock()
        session_result.scalars.return_value.all.return_value = []
        session = MagicMock()
        session.execute = AsyncMock(return_value=session_result)
        session_context = MagicMock()
        session_context.__aenter__ = AsyncMock(return_value=session)
        session_context.__aexit__ = AsyncMock(return_value=None)
        container.session_factory = MagicMock(return_value=session_context)
        container.settings.observability.enabled = False
        container.settings.health = MagicMock()
        container.settings.scheduler = MagicMock()
        container.settings.model = MagicMock()
        container.__aenter__.return_value = container
        container.__aexit__.return_value = None
        return container

    def test_cmd_bot_calls_run_migrations(self) -> None:
        from weather_agent.cmd.bot import cmd_bot

        mock_container = self._make_mock_container()
        mock_settings = MagicMock(database_url="sqlite+aiosqlite:///:memory:")

        with (
            patch("weather_agent.settings.load_settings", return_value=mock_settings),
            patch("weather_agent.cmd.bot.acquire_lock", return_value="/tmp/bot.pid"),
            patch("weather_agent.cmd.bot.AppContainer", return_value=mock_container),
            patch("weather_agent.cmd.bot.run_migrations") as mock_migrate,
            patch("weather_agent.adapters.telegram.bot.TelegramBot") as mock_bot_cls,
            patch("weather_agent.adapters.telegram.handler.make_message_handler"),
        ):
            mock_bot = MagicMock()
            mock_bot.start = AsyncMock(side_effect=RuntimeError())
            mock_bot.stop = AsyncMock()
            mock_bot_cls.return_value = mock_bot

            with pytest.raises(RuntimeError):
                cmd_bot(MagicMock())
            mock_migrate.assert_called_once_with("sqlite+aiosqlite:///:memory:")

    def test_cmd_bot_creates_and_runs_telegram_bot(self) -> None:
        from weather_agent.cmd.bot import cmd_bot

        mock_container = self._make_mock_container()
        mock_settings = MagicMock(database_url="sqlite+aiosqlite:///:memory:")

        with (
            patch("weather_agent.settings.load_settings", return_value=mock_settings),
            patch("weather_agent.cmd.bot.acquire_lock", return_value="/tmp/bot.pid"),
            patch("weather_agent.cmd.bot.AppContainer", return_value=mock_container),
            patch("weather_agent.cmd.bot.run_migrations"),
            patch("weather_agent.adapters.telegram.bot.TelegramBot") as mock_bot_cls,
            patch("weather_agent.adapters.telegram.handler.make_message_handler"),
        ):
            mock_bot = MagicMock()
            mock_bot.start = AsyncMock(side_effect=RuntimeError())
            mock_bot.stop = AsyncMock()
            mock_bot_cls.return_value = mock_bot

            with pytest.raises(RuntimeError):
                cmd_bot(MagicMock())
            mock_bot.setup.assert_called_once()
            mock_bot.start.assert_called_once()
            mock_bot.stop.assert_called_once()

    def test_cmd_bot_migration_failure_fails_startup(self) -> None:
        from weather_agent.cmd.bot import cmd_bot

        mock_container = self._make_mock_container()
        mock_settings = MagicMock(database_url="sqlite+aiosqlite:///:memory:")

        with (
            patch("weather_agent.settings.load_settings", return_value=mock_settings),
            patch("weather_agent.cmd.bot.acquire_lock", return_value="/tmp/bot.pid"),
            patch("weather_agent.cmd.bot.AppContainer", return_value=mock_container),
            patch(
                "weather_agent.cmd.bot.run_migrations",
                side_effect=RuntimeError("db not found"),
            ),
            patch("weather_agent.adapters.telegram.bot.TelegramBot") as mock_bot_cls,
            patch("weather_agent.adapters.telegram.handler.make_message_handler"),
        ):
            mock_bot = MagicMock()
            mock_bot.start = AsyncMock(side_effect=RuntimeError())
            mock_bot.stop = AsyncMock()
            mock_bot_cls.return_value = mock_bot

            with pytest.raises(RuntimeError):
                cmd_bot(MagicMock())
            mock_bot.setup.assert_not_called()

    def test_cmd_bot_starts_observability_server_when_enabled(self) -> None:
        from weather_agent.cmd.bot import cmd_bot

        mock_container = self._make_mock_container()
        mock_container.settings.observability.enabled = True
        mock_container.settings.observability.bot_port = 9999
        mock_settings = MagicMock(database_url="sqlite+aiosqlite:///:memory:")

        with (
            patch("weather_agent.settings.load_settings", return_value=mock_settings),
            patch("weather_agent.cmd.bot.acquire_lock", return_value="/tmp/bot.pid"),
            patch("weather_agent.cmd.bot.AppContainer", return_value=mock_container),
            patch("weather_agent.cmd.bot.run_migrations"),
            patch("weather_agent.adapters.telegram.bot.TelegramBot") as mock_bot_cls,
            patch("weather_agent.adapters.telegram.handler.make_message_handler"),
            patch("weather_agent.cmd.bot.start_observability_server") as mock_start_server,
        ):
            mock_bot = MagicMock()
            mock_bot.start = AsyncMock(side_effect=RuntimeError())
            mock_bot.stop = AsyncMock()
            mock_bot_cls.return_value = mock_bot

            with pytest.raises(RuntimeError):
                cmd_bot(MagicMock())
            mock_start_server.assert_called_once()
            call_kwargs = mock_start_server.call_args.kwargs
            assert call_kwargs["port"] == 9999
            assert call_kwargs["host"] == "0.0.0.0"


class TestCmdWorker:
    def _make_mock_container(self) -> AsyncMock:
        container = AsyncMock()
        container.settings.scheduler = MagicMock()
        container.settings.observability.enabled = False
        container.settings.health = MagicMock()
        container.settings.model = MagicMock()
        container.rule_expression_evaluator = MagicMock()
        container.session_factory = MagicMock()
        container.__aenter__.return_value = container
        container.__aexit__.return_value = None
        return container

    def test_cmd_worker_calls_run_migrations(self) -> None:
        from weather_agent.cmd.worker import cmd_worker

        mock_container = self._make_mock_container()
        mock_session = AsyncMock()
        mock_container.session_factory.return_value.__aenter__.return_value = mock_session
        mock_settings = MagicMock(database_url="sqlite+aiosqlite:///:memory:")

        with (
            patch("weather_agent.settings.load_settings", return_value=mock_settings),
            patch("weather_agent.cmd.worker.acquire_lock", return_value="/tmp/worker.pid"),
            patch("weather_agent.cmd.worker.AppContainer", return_value=mock_container),
            patch("weather_agent.cmd.worker.run_migrations") as mock_migrate,
            patch("weather_agent.cmd.worker.trace"),
            patch(
                "weather_agent.domain.rules.service.NotificationRuleService",
            ),
            patch(
                "weather_agent.infrastructure.repositories.forecast_repository.ForecastRepository",
            ),
            patch(
                "weather_agent.infrastructure.worker.rule_evaluator.RuleEvaluationWorker",
            ) as mock_worker_cls,
        ):
            mock_worker = MagicMock()
            mock_worker.run_loop = AsyncMock()
            mock_worker_cls.return_value = mock_worker

            cmd_worker(MagicMock())
            mock_migrate.assert_called_once_with("sqlite+aiosqlite:///:memory:")

    def test_cmd_worker_starts_observability_server_when_enabled(self) -> None:
        from weather_agent.cmd.worker import cmd_worker

        mock_container = self._make_mock_container()
        mock_container.settings.observability.enabled = True
        mock_container.settings.observability.worker_port = 9998
        mock_session = AsyncMock()
        mock_container.session_factory.return_value.__aenter__.return_value = mock_session
        mock_settings = MagicMock(database_url="sqlite+aiosqlite:///:memory:")

        with (
            patch("weather_agent.settings.load_settings", return_value=mock_settings),
            patch("weather_agent.cmd.worker.acquire_lock", return_value="/tmp/worker.pid"),
            patch("weather_agent.cmd.worker.AppContainer", return_value=mock_container),
            patch("weather_agent.cmd.worker.run_migrations"),
            patch("weather_agent.cmd.worker.trace"),
            patch(
                "weather_agent.domain.rules.service.NotificationRuleService",
            ),
            patch(
                "weather_agent.infrastructure.repositories.forecast_repository.ForecastRepository",
            ),
            patch(
                "weather_agent.infrastructure.worker.rule_evaluator.RuleEvaluationWorker",
            ) as mock_worker_cls,
            patch("weather_agent.cmd.worker.start_observability_server") as mock_start_server,
        ):
            mock_worker = MagicMock()
            mock_worker.run_loop = AsyncMock()
            mock_worker_cls.return_value = mock_worker

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
