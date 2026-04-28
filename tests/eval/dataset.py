from __future__ import annotations

from pydantic import BaseModel


class EvalCase(BaseModel):
    id: str
    category: str
    input_message: str
    expected_intent: str | None = None
    expected_cel: str | None = None
    expected_location: str | None = None
    expected_time_range: str | None = None
    expected_response_pattern: str | None = None
    metadata: dict | None = None


EVAL_CASES: list[EvalCase] = [
    EvalCase(
        id="eval-001",
        category="weather_qa",
        input_message="jaka będzie pogoda w weekend nad Jeziorakiem?",
        expected_intent="weather",
        expected_location="Jeziorak",
        expected_time_range="weekend",
        expected_response_pattern=r"pogod|temperatur|wiatr|opad",
    ),
    EvalCase(
        id="eval-002",
        category="weather_qa",
        input_message="jaka będzie jutro pogoda?",
        expected_intent="weather",
        expected_time_range="jutro",
        expected_response_pattern=r"pogod|temperatur|Prognoza",
    ),
    EvalCase(
        id="eval-003",
        category="weather_qa",
        input_message="czy będzie mocno wietrznie przez następne 3 dni?",
        expected_intent="weather",
        expected_time_range="następne 3 dni",
        expected_response_pattern=r"wiatr|poryw|wietrz",
    ),
    EvalCase(
        id="eval-004",
        category="weather_qa",
        input_message="jak się ubrać na dziś wieczór?",
        expected_intent="weather",
        expected_time_range="dziś wieczorem",
        expected_response_pattern=r"temperatur|ubra|pogod",
    ),
    EvalCase(
        id="eval-005",
        category="weather_qa",
        input_message="czy w Chwarznie będzie się dało latać modelem?",
        expected_intent="weather",
        expected_response_pattern=r"wiatr|pogod|temperatur",
        metadata={"note": "No explicit time ref, defaults to today"},
    ),
    EvalCase(
        id="eval-006",
        category="rule_create",
        input_message="powiadom mnie jeśli porywy wiatru w weekend będą powyżej 12 m/s",
        expected_intent="rule",
        expected_cel="max(\"wind_gusts_10m_ms\", weekend()) > 12.0",
        expected_response_pattern=r"CEL|wyrażeni|reguł",
    ),
    EvalCase(
        id="eval-007",
        category="rule_create",
        input_message="powiadom mnie w każdy piątek o 17 wyślij prognozę dla Chwarzna",
        expected_intent="rule",
        expected_cel=None,
        expected_response_pattern=r"CEL|wyrażeni|reguł",
        metadata={"note": "recurring schedule rule without standard CEL time helper"},
    ),
    EvalCase(
        id="eval-008",
        category="rule_create",
        input_message="powiadom jak będzie padać, napisz 15 minut wcześniej",
        expected_intent="rule",
        expected_cel="any(precipitation_mm > 0.0, next_hours(minutes(15)))",
        expected_response_pattern=r"CEL|wyrażeni|reguł",
    ),
    EvalCase(
        id="eval-009",
        category="rule_create",
        input_message="powiadom mnie o nagłym pogorszeniu pogody",
        expected_intent="rule",
        expected_cel=None,
        expected_response_pattern=r"CEL|wyrażeni|reguł",
        metadata={"note": "ambiguous deterioration requires LLM interpretation"},
    ),
    EvalCase(
        id="eval-010",
        category="rule_create",
        input_message=(
            "powiadom jeśli średni wiatr będzie powyżej 7 m/s"
            " nad Jeziorakiem, daj znać dzień wcześniej"
        ),
        expected_intent="rule",
        expected_location="Jeziorak",
        expected_cel="avg(\"wind_speed_10m_ms\", weekend()) > 7.0",
        expected_response_pattern=r"CEL|wyrażeni|reguł",
    ),
    EvalCase(
        id="eval-011",
        category="rule_edit",
        input_message="powiadom dodaj temperaturę do #R7K2",
        expected_intent="rule",
        expected_response_pattern=r"CEL|wyrażeni|reguł|#R7K2",
    ),
    EvalCase(
        id="eval-012",
        category="rule_delete",
        input_message="usuń regułę #R7K2",
        expected_intent="rule",
        expected_response_pattern=r"usuń|anuluj|#R7K2|reguł",
    ),
    EvalCase(
        id="eval-013",
        category="location_resolve",
        input_message="jaka będzie pogoda w Chwarznie?",
        expected_intent="weather",
        expected_location="Chwarzno",
        expected_response_pattern=r"Chwarz|pogod|temperatur",
    ),
    EvalCase(
        id="eval-014",
        category="time_resolve",
        input_message="jaka będzie pogoda w majówkę?",
        expected_intent="weather",
        expected_time_range="majówka",
        expected_response_pattern=r"pogod|temperatur|majówk",
    ),
    EvalCase(
        id="eval-015",
        category="ambiguity",
        input_message="jaka będzie pogoda?",
        expected_intent="weather",
        expected_response_pattern=r"lokalizac|podaj|brak|kto",
        metadata={"note": "no location specified, should prompt or error"},
    ),
    EvalCase(
        id="eval-016",
        category="ambiguity",
        input_message="pogoda",
        expected_intent="weather",
        expected_response_pattern=r"pogod|lokalizac|podaj",
    ),
    EvalCase(
        id="eval-017",
        category="rule_create",
        input_message="powiadom jeśli temperatura spadnie poniżej -10 stopni",
        expected_intent="rule",
        expected_cel="min(\"temperature_2m_c\", today()) < -10.0",
        expected_response_pattern=r"CEL|wyrażeni|reguł",
    ),
    EvalCase(
        id="eval-018",
        category="rule_create",
        input_message="powiadom gdy opady przekroczą 5mm przez następne 6 godzin",
        expected_intent="rule",
        expected_cel="sum(\"precipitation_mm\", next_hours(hours(6))) > 5.0",
        expected_response_pattern=r"CEL|wyrażeni|reguł",
    ),
    EvalCase(
        id="eval-019",
        category="provider_failure",
        input_message="jaka będzie jutro pogoda?",
        expected_intent="weather",
        expected_time_range="jutro",
        expected_response_pattern=r"błęd|nie uda|przepraszam|dostawc",
        metadata={"provider_fails": True},
    ),
    EvalCase(
        id="eval-020",
        category="weather_qa",
        input_message="czy będzie padać w weekend?",
        expected_intent="weather",
        expected_time_range="weekend",
        expected_response_pattern=r"opad|deszcz|pogod|Prognoza",
    ),
    EvalCase(
        id="eval-021",
        category="command",
        input_message="/start",
        expected_intent="command",
        expected_response_pattern=r"pomoc|start|Pomoc|komend",
    ),
    EvalCase(
        id="eval-022",
        category="command",
        input_message="/help",
        expected_intent="command",
        expected_response_pattern=r"pomoc|start|Pomoc|komend|informacj",
    ),
    EvalCase(
        id="eval-023",
        category="rule_create",
        input_message="niech powiadomi gdy ciśnienie spadnie poniżej 1000 hPa",
        expected_intent="rule",
        expected_cel="min(\"pressure_msl_hpa\", today()) < 1000.0",
        expected_response_pattern=r"CEL|wyrażeni|reguł",
    ),
    EvalCase(
        id="eval-024",
        category="rule_create",
        input_message="powiadom gdy wilgotność przekroczy 90%",
        expected_intent="rule",
        expected_cel="max(\"relative_humidity_2m_pct\", today()) > 90.0",
        expected_response_pattern=r"CEL|wyrażeni|reguł",
    ),
    EvalCase(
        id="eval-025",
        category="time_resolve",
        input_message="pogoda na dzisiaj",
        expected_intent="weather",
        expected_time_range="dziś",
        expected_response_pattern=r"pogod|temperatur|Prognoza",
    ),
]