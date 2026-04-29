from __future__ import annotations

import logging

import pytest

from weather_agent.domain.auth import AuthorizationService, UnauthorizedError


class FakeRepo:
    async def add_user(self, telegram_user_id: int) -> None:
        self.added.append(telegram_user_id)

    async def remove_user(self, telegram_user_id: int) -> None:
        self.removed.append(telegram_user_id)

    async def list_user_ids(self) -> list[int]:
        return []

    def __init__(self) -> None:
        self.added: list[int] = []
        self.removed: list[int] = []


class TestIsAuthorized:
    def test_allowed_user(self) -> None:
        svc = AuthorizationService(allowed_user_ids=[42, 99])
        assert svc.is_authorized(42) is True

    def test_denied_user(self) -> None:
        svc = AuthorizationService(allowed_user_ids=[42])
        assert svc.is_authorized(999) is False

    def test_empty_allowlist(self) -> None:
        svc = AuthorizationService(allowed_user_ids=[])
        assert svc.is_authorized(1) is False


class TestCheckAuthorized:
    def test_authorized_user_passes(self) -> None:
        svc = AuthorizationService(allowed_user_ids=[42])
        svc.check_authorized(42)

    def test_unauthorized_user_raises(self) -> None:
        svc = AuthorizationService(allowed_user_ids=[42])
        with pytest.raises(UnauthorizedError) as exc_info:
            svc.check_authorized(999)
        assert exc_info.value.user_id == 999

    def test_error_message_no_secrets(self) -> None:
        svc = AuthorizationService(allowed_user_ids=[42])
        with pytest.raises(UnauthorizedError) as exc_info:
            svc.check_authorized(999)
        msg = str(exc_info.value)
        assert "bot_token" not in msg
        assert "api_key" not in msg

    def test_unauthorized_logs_user_id(self, caplog: pytest.LogCaptureFixture) -> None:
        svc = AuthorizationService(allowed_user_ids=[42])
        with caplog.at_level(logging.INFO, logger="weather_agent.domain.auth"):
            with pytest.raises(UnauthorizedError):
                svc.check_authorized(999)
        assert "999" in caplog.text


class TestAddAuthorizedUser:
    @pytest.mark.asyncio
    async def test_add_user_to_runtime(self) -> None:
        svc = AuthorizationService(allowed_user_ids=[42])
        await svc.add_authorized_user(100)
        assert svc.is_authorized(100) is True

    @pytest.mark.asyncio
    async def test_add_user_persists_to_repo(self) -> None:
        repo = FakeRepo()
        svc = AuthorizationService(allowed_user_ids=[42], repo=repo)
        await svc.add_authorized_user(100)
        assert repo.added == [100]

    @pytest.mark.asyncio
    async def test_add_already_authorized_user_idempotent(self) -> None:
        repo = FakeRepo()
        svc = AuthorizationService(allowed_user_ids=[42], repo=repo)
        await svc.add_authorized_user(42)
        assert svc.is_authorized(42) is True


class TestRemoveAuthorizedUser:
    @pytest.mark.asyncio
    async def test_remove_user_from_runtime(self) -> None:
        svc = AuthorizationService(allowed_user_ids=[42, 100])
        await svc.remove_authorized_user(100)
        assert svc.is_authorized(100) is False

    @pytest.mark.asyncio
    async def test_remove_user_persists_to_repo(self) -> None:
        repo = FakeRepo()
        svc = AuthorizationService(allowed_user_ids=[42, 100], repo=repo)
        await svc.remove_authorized_user(100)
        assert repo.removed == [100]

    @pytest.mark.asyncio
    async def test_remove_nonexistent_user_no_error(self) -> None:
        repo = FakeRepo()
        svc = AuthorizationService(allowed_user_ids=[42], repo=repo)
        await svc.remove_authorized_user(999)
        assert 42 in svc.list_authorized_users()


class TestListAuthorizedUsers:
    def test_returns_sorted_list(self) -> None:
        svc = AuthorizationService(allowed_user_ids=[99, 42, 7])
        assert svc.list_authorized_users() == [7, 42, 99]

    def test_empty_allowlist(self) -> None:
        svc = AuthorizationService(allowed_user_ids=[])
        assert svc.list_authorized_users() == []


class TestNoTelegramDependency:
    def test_service_api_uses_python_int(self) -> None:
        import inspect

        sig = inspect.signature(AuthorizationService.is_authorized)
        assert "user_id" in sig.parameters

    def test_module_has_no_telegram_imports(self) -> None:
        import inspect

        import weather_agent.domain.auth as auth_mod

        source = inspect.getsource(auth_mod)
        assert "from telegram" not in source
        assert "import telegram" not in source
