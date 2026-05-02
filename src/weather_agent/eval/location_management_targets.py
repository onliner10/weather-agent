from __future__ import annotations

from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from typing import Any, cast

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage
from langchain_core.tools import BaseTool, StructuredTool

from weather_agent.agent_factory import build_current_time_prompt_suffix, create_weather_agent
from weather_agent.domain.polish_utils import normalize_for_matching
from weather_agent.eval.location_management_schemas import (
    LocationManagementEvalOutput,
    LocationToolCallRecord,
    SeedLocation,
)
from weather_agent.eval.notification_rule_proposal_targets import RecordingRulesToolbox
from weather_agent.llm.tools.weather_tools import (
    EditLocationArgs,
    GetForecastArgs,
    GetObservationsArgs,
    ListLocationsArgs,
    RemoveLocationArgs,
    SaveLocationArgs,
)
from weather_agent.observability.logging import get_logger

logger = get_logger(__name__)


def _tool(
    coroutine: Callable[..., Awaitable[dict[str, object]]],
    *,
    name: str,
    description: str,
    args_schema: type[Any],
) -> StructuredTool:
    return StructuredTool.from_function(
        coroutine=coroutine,
        name=name,
        description=description,
        args_schema=args_schema,
    )


class RecordingLocationToolbox:
    def __init__(self, seed_locations: list[SeedLocation]) -> None:
        self.tool_calls: list[LocationToolCallRecord] = []
        self._locations = [seed.model_copy() for seed in seed_locations if seed.enabled]

    def _record(
        self,
        *,
        name: str,
        args: dict[str, object],
        error: str | None = None,
        success: str | None = None,
    ) -> dict[str, object]:
        self.tool_calls.append(
            LocationToolCallRecord(
                name=name,
                args=args,
                result_error=error,
                result_success=success,
            )
        )
        result: dict[str, object] = {}
        if error is not None:
            result["error"] = error
        if success is not None:
            result["success"] = success
        return result

    def _default_location(self) -> SeedLocation | None:
        for location in self._locations:
            if normalize_for_matching(location.name) in {"dom", "home"}:
                return location
            if any(normalize_for_matching(alias) in {"dom", "home"} for alias in location.aliases):
                return location
        return self._locations[0] if self._locations else None

    def _find_location(self, query: str) -> SeedLocation | None:
        if not query.strip():
            return self._default_location()
        query_norm = normalize_for_matching(query)
        for location in self._locations:
            if normalize_for_matching(location.name) == query_norm:
                return location
            if any(normalize_for_matching(alias) == query_norm for alias in location.aliases):
                return location
        return None

    @staticmethod
    def _location_payload(location: SeedLocation) -> dict[str, object]:
        return {
            "name": location.name,
            "aliases": location.aliases,
            "latitude": location.latitude,
            "longitude": location.longitude,
            "enabled": location.enabled,
        }

    def _missing_location(self, tool: str, args: dict[str, object]) -> dict[str, object]:
        return self._record(
            name=tool,
            args=args,
            error="Nie mam zapisanej domyślnej lokalizacji. Podaj lokalizację.",
        )

    async def get_forecast(
        self,
        location_name: str,
        start_date: str,
        end_date: str,
        variables: list[str] | None = None,
    ) -> dict[str, object]:
        location = self._find_location(location_name)
        args: dict[str, object] = {
            "location_name": location_name,
            "start_date": start_date,
            "end_date": end_date,
            "variables": variables or [],
        }
        if location is None:
            return self._missing_location("get_forecast", args)
        success = f"Prognoza dla {location.name}"
        result = self._record(name="get_forecast", args=args, success=success)
        result.update(
            {
                "location": location.name,
                "time_range": f"{start_date} – {end_date}",
                "forecast_points": [
                    {
                        "time": datetime.now(UTC).isoformat(),
                        "temperature_2m_c": 12.0,
                        "precipitation_mm": 1.0,
                    }
                ],
                "provider": "eval-fixture",
                "model": "eval-fixture",
            }
        )
        return result

    async def get_observations(self, location_name: str) -> dict[str, object]:
        location = self._find_location(location_name)
        args: dict[str, object] = {"location_name": location_name}
        if location is None:
            return self._missing_location("get_observations", args)
        result = self._record(
            name="get_observations",
            args=args,
            success=f"Obserwacje dla {location.name}",
        )
        result.update(
            {
                "location": location.name,
                "observations": [{"temperature_c": 12.0, "precipitation_mm": 0.0}],
            }
        )
        return result

    async def save_location(self, location_name: str, alias: str = "") -> dict[str, object]:
        self._locations.append(
            SeedLocation(
                name=location_name, aliases=[alias] if alias else [], latitude=54.0, longitude=18.0
            )
        )
        return self._record(
            name="save_location",
            args={"location_name": location_name, "alias": alias},
            success=f"Zapamiętałem lokalizację: {location_name}",
        )

    async def edit_location(
        self,
        location_name: str,
        new_name: str = "",
        aliases: list[str] | None = None,
        latitude: float | None = None,
        longitude: float | None = None,
        enabled: bool | None = None,
    ) -> dict[str, object]:
        location = self._find_location(location_name)
        args: dict[str, object] = {
            "location_name": location_name,
            "new_name": new_name,
            "aliases": aliases or [],
            "latitude": latitude,
            "longitude": longitude,
            "enabled": enabled,
        }
        if location is None:
            return self._record(
                name="edit_location",
                args=args,
                error=f"Nie znaleziono zapisanej lokalizacji: {location_name}",
            )
        if new_name:
            location.name = new_name
        if aliases:
            location.aliases = aliases
        if latitude is not None:
            location.latitude = latitude
        if longitude is not None:
            location.longitude = longitude
        return self._record(
            name="edit_location",
            args=args,
            success=f"Zaktualizowałem lokalizację: {location.name}",
        )

    async def remove_location(self, location_name: str) -> dict[str, object]:
        location = self._find_location(location_name)
        if location is None:
            return self._record(
                name="remove_location",
                args={"location_name": location_name},
                error=f"Nie znaleziono zapisanej lokalizacji: {location_name}",
            )
        self._locations = [item for item in self._locations if item is not location]
        return self._record(
            name="remove_location",
            args={"location_name": location_name},
            success=f"Usunąłem lokalizację: {location.name}",
        )

    async def list_locations(self, include_disabled: bool = False) -> dict[str, object]:
        locations = [self._location_payload(location) for location in self._locations]
        result = self._record(
            name="list_locations",
            args={"include_disabled": include_disabled},
            success=f"Liczba lokalizacji: {len(locations)}",
        )
        result.update({"locations": locations, "count": len(locations)})
        return result

    def to_langchain_tools(self) -> list[BaseTool]:
        specs: list[
            tuple[
                Callable[..., Awaitable[dict[str, object]]],
                str,
                str,
                type[Any],
            ]
        ] = [
            (
                self.get_forecast,
                "get_forecast",
                "Pobierz prognozę pogody dla lokalizacji i zakresu dat.",
                GetForecastArgs,
            ),
            (
                self.get_observations,
                "get_observations",
                "Pobierz aktualne obserwacje dla lokalizacji.",
                GetObservationsArgs,
            ),
            (
                self.save_location,
                "save_location",
                "Zapisz lokalizację użytkownika pod nazwą lub aliasem.",
                SaveLocationArgs,
            ),
            (
                self.edit_location,
                "edit_location",
                "Edytuj zapisaną lokalizację użytkownika po nazwie, aliasie albo ID.",
                EditLocationArgs,
            ),
            (
                self.remove_location,
                "remove_location",
                "Usuń zapisaną lokalizację użytkownika przez dezaktywację.",
                RemoveLocationArgs,
            ),
            (
                self.list_locations,
                "list_locations",
                "Wyświetl zapisane lokalizacje użytkownika.",
                ListLocationsArgs,
            ),
        ]
        return [
            _tool(coroutine, name=name, description=description, args_schema=args_schema)
            for coroutine, name, description, args_schema in specs
        ]


def build_location_management_async_target_from_factory(
    model_factory: Callable[[], BaseChatModel],
) -> Callable[[dict[str, object]], Awaitable[dict[str, Any]]]:
    async def location_management_target(inputs: dict[str, object]) -> dict[str, Any]:
        example_id = str(inputs["id"])
        question = str(inputs["question"])
        current_time = datetime.fromisoformat(str(inputs["current_time"]))
        raw_seed_locations = inputs.get("seed_locations", [])
        if not isinstance(raw_seed_locations, list):
            raw_seed_locations = []
        seed_locations = [SeedLocation.model_validate(item) for item in raw_seed_locations]
        logger.debug("location_management_eval_run", id=example_id, question=question[:80])
        location_toolbox = RecordingLocationToolbox(seed_locations)
        rules_toolbox = RecordingRulesToolbox()
        agent = create_weather_agent(
            model=model_factory(),
            tools=[*location_toolbox.to_langchain_tools(), *rules_toolbox.to_langchain_tools()],
            system_prompt_suffix=build_current_time_prompt_suffix(current_time),
        )
        result = cast(
            dict[str, Any],
            await agent.ainvoke(
                {"messages": [HumanMessage(content=question)]},
                config={"configurable": {"thread_id": example_id}},
            ),
        )
        final = result["messages"][-1]
        answer = final.content if hasattr(final, "content") else str(final)
        rule_calls = [
            LocationToolCallRecord(
                name=call.name,
                args=call.args,
                result_error=call.result_error,
                result_success=None,
            )
            for call in rules_toolbox.tool_calls
        ]
        return LocationManagementEvalOutput(
            example_id=example_id,
            answer=str(answer),
            tool_calls=[*location_toolbox.tool_calls, *rule_calls],
        ).model_dump()

    return location_management_target
