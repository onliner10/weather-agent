# Domain Vocabulary

## Authorized user
A Telegram user ID explicitly allowed to interact with the bot's weather, location, and rule workflows.

## Location
A named place with canonical coordinates and optional aliases that the system can resolve for forecasts, observations, warnings, and notifications.

## Global units
The repository-wide defaults for presenting temperature, wind, precipitation, and pressure until user-specific preferences exist.

## Forecast
A normalized weather prediction for a location and time range, produced by a `ForecastProvider` while preserving the original provider payload.

## Forecast snapshot
A persisted capture of forecast data at a specific ingest time, used as deterministic evidence for rule evaluation and later explanations.

## Observation
A normalized current or recent weather measurement, typically from IMGW synoptic data, used to enrich user-facing answers.

## Official warning
A normalized IMGW weather warning tied to a place and validity window.

## Rule
A persisted notification definition containing scope, lifecycle state, validated CEL expression, and delivery controls such as cooldown or snooze.

## Rule expression
A CEL expression proposed by the LLM or a deterministic edit flow, then validated against the project's metric and function allowlists before activation.

## Rule ID
A short stable identifier such as `#R...` shown to users when referring to an existing notification rule.

## Event ID
A short stable identifier such as `#E...` attached to an evaluation or notification outcome so the user can ask why it fired.

## Notification
A user-visible outbound message generated from a deterministic evaluation result, warning, or explicit summary workflow.

## Cooldown
A per-rule minimum interval that suppresses repeated notifications for materially identical conditions.

## Dedupe key
A stable key derived from rule scope and evidence to prevent duplicate notification sends for the same event.

## Snooze
A temporary rule suspension window after which the rule automatically becomes eligible again.

## Telegram context key
The deterministic conversation scope composed of `chat_id + message_thread_id`, falling back to `chat_id` when no topic exists.

## Dry run
An execution mode that computes what would happen, records evidence where appropriate, and avoids sending the actual user notification.
