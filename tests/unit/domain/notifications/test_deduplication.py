from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import event
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from weather_agent.domain.notifications.deduplication import (
    NotificationCandidate,
    NotificationDeduplicator,
    compute_dedupe_key,
    compute_payload_hash,
    has_significant_change,
)
from weather_agent.domain.rules.models import NotificationRule
from weather_agent.infrastructure.db.base import (
    AuthorizedUser,
    Base,
    Location,
)
from weather_agent.infrastructure.db.base import (
    NotificationEvent as NotificationEventORM,
)
from weather_agent.infrastructure.db.base import (
    NotificationRule as NotificationRuleORM,
)


def _set_sqlite_foreign_keys(dbapi_connection: object, connection_record: object) -> None:
    cursor = dbapi_connection.cursor()  # type: ignore[attr-defined]
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.close()


@pytest.fixture()
async def session():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    event.listen(engine.sync_engine, "connect", _set_sqlite_foreign_keys)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    async with factory() as s:
        yield s
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    await engine.dispose()


def _make_rule(
    id: int = 1,
    cooldown_minutes: int = 60,
    snooze_until: datetime | None = None,
    dry_run: bool = False,
    location_id: int = 1,
    expression: str = 'max("wind_gusts_10m_ms", weekend()) >= 12',
) -> NotificationRule:
    return NotificationRule(
        id=id,
        short_id="RTEST",
        user_id=1,
        telegram_chat_id=12345,
        telegram_message_thread_id=None,
        location_id=location_id,
        expression_language="cel",
        expression=expression,
        schedule=None,
        lead_time_minutes=None,
        cooldown_minutes=cooldown_minutes,
        enabled=True,
        dry_run=dry_run,
        description=None,
        snooze_until=snooze_until,
        created_at=datetime.now(UTC),
        updated_at=datetime.now(UTC),
    )


async def _create_user(session: AsyncSession, user_id: int = 1) -> None:
    user = AuthorizedUser(id=user_id, telegram_user_id=user_id * 1000, role="user")
    session.add(user)
    await session.flush()


async def _create_location(session: AsyncSession, user_id: int = 1, loc_id: int = 1) -> None:
    loc = Location(
        id=loc_id,
        user_id=user_id,
        name="Test Location",
        aliases=["test"],
        latitude=52.22,
        longitude=21.01,
        enabled=True,
        created_at=datetime.now(UTC),
        updated_at=datetime.now(UTC),
    )
    session.add(loc)
    await session.flush()


async def _create_rule_orm(
    session: AsyncSession,
    rule_id: int = 1,
    short_id: str = "RTEST",
    location_id: int = 1,
    cooldown_minutes: int = 60,
    snooze_until: datetime | None = None,
    expression: str = 'max("wind_gusts_10m_ms", weekend()) >= 12',
) -> NotificationRuleORM:
    orm = NotificationRuleORM(
        id=rule_id,
        short_id=short_id,
        user_id=1,
        telegram_chat_id=12345,
        telegram_message_thread_id=None,
        location_id=location_id,
        expression_language="cel",
        expression=expression,
        enabled=True,
        dry_run=False,
        cooldown_minutes=cooldown_minutes,
        snooze_until=snooze_until,
        created_at=datetime.now(UTC),
        updated_at=datetime.now(UTC),
    )
    session.add(orm)
    await session.flush()
    return orm


async def _create_event_orm(
    session: AsyncSession,
    rule_id: int = 1,
    event_id: int = 1,
    short_id: str = "E0001",
    suppressed: bool = False,
    suppress_reason: str | None = None,
    payload_hash: str | None = None,
    sent_at: datetime | None = None,
) -> NotificationEventORM:
    delivery_status = "pending"
    if suppressed:
        delivery_status = "suppressed"
    elif sent_at is not None:
        delivery_status = "sent"
    orm = NotificationEventORM(
        id=event_id,
        short_id=short_id,
        rule_id=rule_id,
        telegram_chat_id=12345,
        telegram_message_thread_id=None,
        suppressed=suppressed,
        suppress_reason=suppress_reason,
        payload_hash=payload_hash,
        sent_at=sent_at,
        delivery_status=delivery_status,
        created_at=datetime.now(UTC),
    )
    session.add(orm)
    await session.flush()
    return orm


class TestDedupeKey:
    def test_compute_dedupe_key_basic(self) -> None:
        key = compute_dedupe_key(
            rule_id=1,
            location_id=10,
            expression="temperature_2m_c > 30",
            window_start=datetime(2025, 6, 1, 12, 0, tzinfo=UTC),
            window_end=datetime(2025, 6, 1, 18, 0, tzinfo=UTC),
        )
        assert key.rule_id == 1
        assert key.location_id == 10
        assert key.dedupe_key != ""

    def test_compute_dedupe_key_same_expression_same_hash(self) -> None:
        key1 = compute_dedupe_key(
            rule_id=1,
            location_id=1,
            expression="temp > 30",
        )
        key2 = compute_dedupe_key(
            rule_id=1,
            location_id=1,
            expression="temp > 30",
        )
        assert key1.expression_hash == key2.expression_hash

    def test_compute_dedupe_key_different_expression_different_hash(self) -> None:
        key1 = compute_dedupe_key(
            rule_id=1,
            location_id=1,
            expression="temp > 30",
        )
        key2 = compute_dedupe_key(
            rule_id=1,
            location_id=1,
            expression="wind > 10",
        )
        assert key1.expression_hash != key2.expression_hash

    def test_compute_dedupe_key_includes_window(self) -> None:
        key1 = compute_dedupe_key(
            rule_id=1,
            location_id=1,
            expression="temp > 30",
            window_start=datetime(2025, 1, 1, tzinfo=UTC),
            window_end=datetime(2025, 1, 2, tzinfo=UTC),
        )
        key2 = compute_dedupe_key(
            rule_id=1,
            location_id=1,
            expression="temp > 30",
            window_start=datetime(2025, 6, 1, tzinfo=UTC),
            window_end=datetime(2025, 6, 2, tzinfo=UTC),
        )
        assert key1.dedupe_key != key2.dedupe_key


class TestPayloadHash:
    def test_same_payload_same_hash(self) -> None:
        payload = {"temperature": 30, "wind": 10}
        assert compute_payload_hash(payload) == compute_payload_hash(payload)

    def test_different_payload_different_hash(self) -> None:
        h1 = compute_payload_hash({"temperature": 30})
        h2 = compute_payload_hash({"temperature": 25})
        assert h1 != h2

    def test_key_order_irrelevant(self) -> None:
        h1 = compute_payload_hash({"a": 1, "b": 2})
        h2 = compute_payload_hash({"b": 2, "a": 1})
        assert h1 == h2


class TestSignificantChange:
    def test_same_hash_no_change(self) -> None:
        assert has_significant_change("abc", "abc") is False

    def test_different_hash_significant(self) -> None:
        assert has_significant_change("abc", "def") is True

    def test_threshold_placeholder_ignored(self) -> None:
        assert has_significant_change("abc", "abc", threshold=0.5) is False
        assert has_significant_change("abc", "def", threshold=0.5) is True


class TestNotificationDeduplicator:
    async def test_no_suppression_when_no_prior_events(self, session: AsyncSession) -> None:
        await _create_user(session)
        await _create_location(session)
        await _create_rule_orm(session)
        rule = _make_rule()
        candidate = NotificationCandidate(
            rule_id=1,
            location_id=1,
            expression=rule.expression,
            forecast_window_start=datetime(2025, 6, 1, 12, 0, tzinfo=UTC),
            forecast_window_end=datetime(2025, 6, 1, 18, 0, tzinfo=UTC),
        )
        dedup = NotificationDeduplicator(session)
        suppressed, reason = await dedup.should_suppress(rule, candidate)
        assert suppressed is False
        assert reason is None

    async def test_snooze_active_suppresses(self, session: AsyncSession) -> None:
        await _create_user(session)
        await _create_location(session)
        snooze_until = datetime.now(UTC) + timedelta(hours=2)
        await _create_rule_orm(session, snooze_until=snooze_until)
        rule = _make_rule(snooze_until=snooze_until)
        candidate = NotificationCandidate(
            rule_id=1,
            location_id=1,
            expression=rule.expression,
        )
        dedup = NotificationDeduplicator(session)
        suppressed, reason = await dedup.should_suppress(rule, candidate)
        assert suppressed is True
        assert reason is not None
        assert "snoozed" in reason

    async def test_snooze_expired_allows(self, session: AsyncSession) -> None:
        await _create_user(session)
        await _create_location(session)
        snooze_until = datetime.now(UTC) - timedelta(hours=1)
        await _create_rule_orm(session, snooze_until=snooze_until)
        rule = _make_rule(snooze_until=snooze_until)
        candidate = NotificationCandidate(
            rule_id=1,
            location_id=1,
            expression=rule.expression,
        )
        dedup = NotificationDeduplicator(session)
        suppressed, reason = await dedup.should_suppress(rule, candidate)
        assert suppressed is False
        assert reason is None

    async def test_cooldown_not_expired_suppresses(self, session: AsyncSession) -> None:
        await _create_user(session)
        await _create_location(session)
        rule_orm = await _create_rule_orm(session, cooldown_minutes=60)
        sent_at = datetime.now(UTC) - timedelta(minutes=30)
        await _create_event_orm(
            session,
            rule_id=rule_orm.id,
            event_id=10,
            short_id="E0010",
            suppressed=False,
            sent_at=sent_at,
        )
        rule = _make_rule(cooldown_minutes=60)
        candidate = NotificationCandidate(
            rule_id=1,
            location_id=1,
            expression=rule.expression,
        )
        dedup = NotificationDeduplicator(session)
        suppressed, reason = await dedup.should_suppress(rule, candidate)
        assert suppressed is True
        assert reason is not None
        assert "cooldown" in reason

    async def test_cooldown_expired_allows(self, session: AsyncSession) -> None:
        await _create_user(session)
        await _create_location(session)
        rule_orm = await _create_rule_orm(session, cooldown_minutes=60)
        sent_at = datetime.now(UTC) - timedelta(minutes=90)
        await _create_event_orm(
            session,
            rule_id=rule_orm.id,
            event_id=10,
            short_id="E0010",
            suppressed=False,
            sent_at=sent_at,
        )
        rule = _make_rule(cooldown_minutes=60)
        candidate = NotificationCandidate(
            rule_id=1,
            location_id=1,
            expression=rule.expression,
        )
        dedup = NotificationDeduplicator(session)
        suppressed, reason = await dedup.should_suppress(rule, candidate)
        assert suppressed is False
        assert reason is None

    async def test_duplicate_dedupe_key_suppresses(self, session: AsyncSession) -> None:
        await _create_user(session)
        await _create_location(session)
        rule_orm = await _create_rule_orm(session)
        window_start = datetime(2025, 6, 1, 12, 0, tzinfo=UTC)
        window_end = datetime(2025, 6, 1, 18, 0, tzinfo=UTC)
        dedupe_key = compute_dedupe_key(
            rule_id=1,
            location_id=1,
            expression=rule_orm.expression,
            window_start=window_start,
            window_end=window_end,
        )
        import hashlib

        dedupe_hash = hashlib.sha256(dedupe_key.dedupe_key.encode()).hexdigest()

        sent_at = datetime.now(UTC) - timedelta(hours=3)
        await _create_event_orm(
            session,
            rule_id=rule_orm.id,
            event_id=10,
            short_id="E0010",
            suppressed=False,
            sent_at=sent_at,
            payload_hash=dedupe_hash,
        )

        rule = _make_rule(expression=rule_orm.expression)
        candidate = NotificationCandidate(
            rule_id=1,
            location_id=1,
            expression=rule_orm.expression,
            forecast_window_start=window_start,
            forecast_window_end=window_end,
        )
        dedup = NotificationDeduplicator(session)
        suppressed, reason = await dedup.should_suppress(rule, candidate)
        assert suppressed is True
        assert reason is not None
        assert "duplicate" in reason

    async def test_different_window_not_suppressed_as_duplicate(
        self, session: AsyncSession
    ) -> None:
        await _create_user(session)
        await _create_location(session)
        rule_orm = await _create_rule_orm(session)
        old_window_start = datetime(2025, 6, 1, 12, 0, tzinfo=UTC)
        old_window_end = datetime(2025, 6, 1, 18, 0, tzinfo=UTC)
        dedupe_key = compute_dedupe_key(
            rule_id=1,
            location_id=1,
            expression=rule_orm.expression,
            window_start=old_window_start,
            window_end=old_window_end,
        )
        import hashlib

        dedupe_hash = hashlib.sha256(dedupe_key.dedupe_key.encode()).hexdigest()

        sent_at = datetime.now(UTC) - timedelta(hours=3)
        await _create_event_orm(
            session,
            rule_id=rule_orm.id,
            event_id=10,
            short_id="E0010",
            suppressed=False,
            sent_at=sent_at,
            payload_hash=dedupe_hash,
        )

        new_window_start = datetime(2025, 6, 2, 12, 0, tzinfo=UTC)
        new_window_end = datetime(2025, 6, 2, 18, 0, tzinfo=UTC)

        rule = _make_rule(expression=rule_orm.expression)
        candidate = NotificationCandidate(
            rule_id=1,
            location_id=1,
            expression=rule_orm.expression,
            forecast_window_start=new_window_start,
            forecast_window_end=new_window_end,
        )
        dedup = NotificationDeduplicator(session)
        suppressed, reason = await dedup.should_suppress(rule, candidate)
        assert suppressed is False
        assert reason is None

    async def test_pending_duplicate_event_does_not_suppress(self, session: AsyncSession) -> None:
        await _create_user(session)
        await _create_location(session)
        rule_orm = await _create_rule_orm(session)
        window_start = datetime(2025, 6, 1, 12, 0, tzinfo=UTC)
        window_end = datetime(2025, 6, 1, 18, 0, tzinfo=UTC)
        dedupe_key = compute_dedupe_key(
            rule_id=1,
            location_id=1,
            expression=rule_orm.expression,
            window_start=window_start,
            window_end=window_end,
        )
        import hashlib

        dedupe_hash = hashlib.sha256(dedupe_key.dedupe_key.encode()).hexdigest()

        await _create_event_orm(
            session,
            rule_id=rule_orm.id,
            event_id=10,
            short_id="E0010",
            suppressed=False,
            sent_at=None,
            payload_hash=dedupe_hash,
        )

        rule = _make_rule(expression=rule_orm.expression)
        candidate = NotificationCandidate(
            rule_id=1,
            location_id=1,
            expression=rule_orm.expression,
            forecast_window_start=window_start,
            forecast_window_end=window_end,
        )
        dedup = NotificationDeduplicator(session)
        suppressed, reason = await dedup.should_suppress(rule, candidate)
        assert suppressed is False
        assert reason is None

    async def test_payload_hash_unchanged_suppresses(self, session: AsyncSession) -> None:
        await _create_user(session)
        await _create_location(session)
        rule_orm = await _create_rule_orm(session, cooldown_minutes=5)

        payload = {"temperature": 30, "wind_speed": 10}
        payload_hash = compute_payload_hash(payload)

        sent_at = datetime.now(UTC) - timedelta(minutes=30)
        await _create_event_orm(
            session,
            rule_id=rule_orm.id,
            event_id=10,
            short_id="E0010",
            suppressed=False,
            sent_at=sent_at,
            payload_hash=payload_hash,
        )

        rule = _make_rule(cooldown_minutes=5)
        candidate = NotificationCandidate(
            rule_id=1,
            location_id=1,
            expression=rule_orm.expression,
            payload=payload,
        )
        dedup = NotificationDeduplicator(session)
        suppressed, reason = await dedup.should_suppress(rule, candidate)
        assert suppressed is True
        assert reason is not None
        assert "payload unchanged" in reason

    async def test_payload_hash_changed_allows(self, session: AsyncSession) -> None:
        await _create_user(session)
        await _create_location(session)
        rule_orm = await _create_rule_orm(session, cooldown_minutes=5)

        old_payload = {"temperature": 30, "wind_speed": 10}
        old_hash = compute_payload_hash(old_payload)

        sent_at = datetime.now(UTC) - timedelta(minutes=30)
        await _create_event_orm(
            session,
            rule_id=rule_orm.id,
            event_id=10,
            short_id="E0010",
            suppressed=False,
            sent_at=sent_at,
            payload_hash=old_hash,
        )

        new_payload = {"temperature": 35, "wind_speed": 10}

        rule = _make_rule(cooldown_minutes=5)
        candidate = NotificationCandidate(
            rule_id=1,
            location_id=1,
            expression=rule_orm.expression,
            payload=new_payload,
        )
        dedup = NotificationDeduplicator(session)
        suppressed, reason = await dedup.should_suppress(rule, candidate)
        assert suppressed is False
        assert reason is None

    async def test_suppressed_events_not_counted_for_cooldown(self, session: AsyncSession) -> None:
        await _create_user(session)
        await _create_location(session)
        rule_orm = await _create_rule_orm(session, cooldown_minutes=60)

        sent_at = datetime.now(UTC) - timedelta(minutes=30)
        await _create_event_orm(
            session,
            rule_id=rule_orm.id,
            event_id=10,
            short_id="E0010",
            suppressed=True,
            suppress_reason="snoozed",
            sent_at=sent_at,
        )

        rule = _make_rule(cooldown_minutes=60)
        candidate = NotificationCandidate(
            rule_id=1,
            location_id=1,
            expression=rule.expression,
        )
        dedup = NotificationDeduplicator(session)
        suppressed, reason = await dedup.should_suppress(rule, candidate)
        assert suppressed is False
        assert reason is None

    async def test_snooze_takes_priority_over_cooldown(self, session: AsyncSession) -> None:
        await _create_user(session)
        await _create_location(session)
        snooze_until = datetime.now(UTC) + timedelta(hours=2)
        rule_orm = await _create_rule_orm(session, cooldown_minutes=5, snooze_until=snooze_until)

        sent_at = datetime.now(UTC) - timedelta(hours=3)
        await _create_event_orm(
            session,
            rule_id=rule_orm.id,
            event_id=10,
            short_id="E0010",
            suppressed=False,
            sent_at=sent_at,
        )

        rule = _make_rule(cooldown_minutes=5, snooze_until=snooze_until)
        candidate = NotificationCandidate(
            rule_id=1,
            location_id=1,
            expression=rule.expression,
        )
        dedup = NotificationDeduplicator(session)
        suppressed, reason = await dedup.should_suppress(rule, candidate)
        assert suppressed is True
        assert "snoozed" in reason

    async def test_dry_run_shows_suppression_status(self, session: AsyncSession) -> None:
        await _create_user(session)
        await _create_location(session)
        snooze_until = datetime.now(UTC) + timedelta(hours=2)
        await _create_rule_orm(session, snooze_until=snooze_until)
        rule = _make_rule(snooze_until=snooze_until)
        candidate = NotificationCandidate(
            rule_id=1,
            location_id=1,
            expression=rule.expression,
            dry_run=True,
        )
        dedup = NotificationDeduplicator(session)
        suppressed, reason = await dedup.should_suppress(rule, candidate)
        assert suppressed is True
        assert "snoozed" in reason

    async def test_dry_run_not_suppressed(self, session: AsyncSession) -> None:
        await _create_user(session)
        await _create_location(session)
        await _create_rule_orm(session)
        rule = _make_rule()
        candidate = NotificationCandidate(
            rule_id=1,
            location_id=1,
            expression=rule.expression,
            dry_run=True,
        )
        dedup = NotificationDeduplicator(session)
        suppressed, reason = await dedup.should_suppress(rule, candidate)
        assert suppressed is False
        assert reason is None

    async def test_events_with_no_sent_at_not_counted_for_cooldown(
        self, session: AsyncSession
    ) -> None:
        await _create_user(session)
        await _create_location(session)
        rule_orm = await _create_rule_orm(session, cooldown_minutes=60)

        await _create_event_orm(
            session,
            rule_id=rule_orm.id,
            event_id=10,
            short_id="E0010",
            suppressed=False,
            sent_at=None,
        )

        rule = _make_rule(cooldown_minutes=60)
        candidate = NotificationCandidate(
            rule_id=1,
            location_id=1,
            expression=rule.expression,
        )
        dedup = NotificationDeduplicator(session)
        suppressed, reason = await dedup.should_suppress(rule, candidate)
        assert suppressed is False
        assert reason is None

    async def test_dry_run_cooldown_shows_would_suppress(self, session: AsyncSession) -> None:
        await _create_user(session)
        await _create_location(session)
        rule_orm = await _create_rule_orm(session, cooldown_minutes=60)
        sent_at = datetime.now(UTC) - timedelta(minutes=30)
        await _create_event_orm(
            session,
            rule_id=rule_orm.id,
            event_id=10,
            short_id="E0010",
            suppressed=False,
            sent_at=sent_at,
        )
        rule = _make_rule(cooldown_minutes=60)
        candidate = NotificationCandidate(
            rule_id=1,
            location_id=1,
            expression=rule.expression,
            dry_run=True,
        )
        dedup = NotificationDeduplicator(session)
        suppressed, reason = await dedup.should_suppress(rule, candidate)
        assert suppressed is True
        assert "cooldown" in (reason or "")

    async def test_sequential_checks_cooldown_blocks_both(self, session: AsyncSession) -> None:
        await _create_user(session)
        await _create_location(session)
        rule_orm = await _create_rule_orm(session, cooldown_minutes=60)
        sent_at = datetime.now(UTC) - timedelta(minutes=10)
        await _create_event_orm(
            session,
            rule_id=rule_orm.id,
            event_id=10,
            short_id="E0010",
            suppressed=False,
            sent_at=sent_at,
        )

        rule = _make_rule(cooldown_minutes=60)

        dedup = NotificationDeduplicator(session)
        suppressed1, _ = await dedup.should_suppress(
            rule,
            NotificationCandidate(
                rule_id=1,
                location_id=1,
                expression=rule.expression,
            ),
        )
        assert suppressed1 is True

        suppressed2, _ = await dedup.should_suppress(
            rule,
            NotificationCandidate(
                rule_id=1,
                location_id=1,
                expression=rule.expression,
            ),
        )
        assert suppressed2 is True
