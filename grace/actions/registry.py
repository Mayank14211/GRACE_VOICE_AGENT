"""Action registry: the ONLY layer that touches the OS.

Every action is registered by name with a typed pydantic args model and a ``risky`` flag.
``ActionRegistry.execute`` validates the arguments, refuses to run a risky action unless the caller
passes ``confirmed=True`` (only the confirmation gate does), and runs one action at a time so voice
and gesture commands never race on the OS.

The args models double as the tool schema for the LLM router (S07):
``spec.args_model.model_json_schema()``.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass, field
from typing import Any, TypeVar

from pydantic import BaseModel, ConfigDict, ValidationError

log = logging.getLogger(__name__)


class ActionArgs(BaseModel):
    """Base class for action argument models. Unknown fields are rejected."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class NoArgs(ActionArgs):
    """For actions that take no arguments."""


ArgsT = TypeVar("ArgsT", bound=ActionArgs)


@dataclass(frozen=True)
class ActionResult:
    ok: bool
    message: str
    data: Mapping[str, Any] = field(default_factory=dict)


class ActionError(Exception):
    """Raised by a handler for an expected, user-facing failure (the message is spoken)."""


class UnknownActionError(KeyError):
    pass


class ConfirmationRequiredError(PermissionError):
    """A risky action was executed without confirmation. Callers must go through the gate."""

    def __init__(self, name: str) -> None:
        super().__init__(f"action {name!r} is risky and needs confirmation")
        self.name = name


Handler = Callable[[Any], ActionResult]


@dataclass(frozen=True)
class ActionSpec:
    name: str
    args_model: type[ActionArgs]
    handler: Handler
    risky: bool = False
    description: str = ""
    # Short noun phrase used in the confirmation prompt, e.g. "shutdown" -> "Confirm shutdown?".
    confirm_label: str | None = None

    def __post_init__(self) -> None:
        if self.risky and not self.confirm_label:
            raise ValueError(f"risky action {self.name!r} needs a confirm_label")


class ActionRegistry:
    def __init__(self) -> None:
        self._specs: dict[str, ActionSpec] = {}
        self._run_lock = threading.Lock()  # one OS action at a time (executor thread lands in S05)

    # -- registration ---------------------------------------------------------------------------
    def add(self, spec: ActionSpec) -> None:
        if spec.name in self._specs:
            raise ValueError(f"action {spec.name!r} is already registered")
        self._specs[spec.name] = spec

    def register(
        self,
        name: str,
        args_model: type[ArgsT] = NoArgs,  # type: ignore[assignment]
        *,
        risky: bool = False,
        description: str = "",
        confirm_label: str | None = None,
    ) -> Callable[[Callable[[ArgsT], ActionResult]], Callable[[ArgsT], ActionResult]]:
        def decorator(handler: Callable[[ArgsT], ActionResult]) -> Callable[[ArgsT], ActionResult]:
            self.add(
                ActionSpec(
                    name=name,
                    args_model=args_model,
                    handler=handler,
                    risky=risky,
                    description=description or (handler.__doc__ or "").strip(),
                    confirm_label=confirm_label,
                )
            )
            return handler

        return decorator

    # -- lookup ---------------------------------------------------------------------------------
    def __contains__(self, name: object) -> bool:
        return name in self._specs

    def __iter__(self) -> Iterator[ActionSpec]:
        return iter(self._specs.values())

    def names(self) -> list[str]:
        return sorted(self._specs)

    def spec(self, name: str) -> ActionSpec:
        try:
            return self._specs[name]
        except KeyError:
            raise UnknownActionError(name) from None

    def is_risky(self, name: str) -> bool:
        return name in self._specs and self._specs[name].risky

    def validate(self, name: str, args: Mapping[str, Any] | None = None) -> ActionArgs:
        """Parse args against the action's model. Raises UnknownActionError / ValidationError."""
        return self.spec(name).args_model.model_validate(dict(args or {}))

    # -- execution ------------------------------------------------------------------------------
    def execute(
        self, name: str, args: Mapping[str, Any] | None = None, *, confirmed: bool = False
    ) -> ActionResult:
        """Run an action. Never raises for handler failures; returns ok=False instead.

        Raises ConfirmationRequiredError for a risky action without ``confirmed=True``: that is a
        programming error (the caller skipped the confirmation gate), not a user-facing failure.
        """
        if name not in self._specs:
            return ActionResult(False, f"I don't know how to do {name}.")
        spec = self._specs[name]
        if spec.risky and not confirmed:
            raise ConfirmationRequiredError(name)
        try:
            parsed = spec.args_model.model_validate(dict(args or {}))
        except ValidationError as exc:
            fields = ", ".join(".".join(str(p) for p in e["loc"]) or name for e in exc.errors())
            log.info("action %s rejected: invalid %s", name, fields)
            return ActionResult(False, f"I couldn't do that: invalid {fields}.")
        with self._run_lock:
            try:
                result = spec.handler(parsed)
            except ActionError as exc:
                log.info("action %s failed: %s", name, exc)
                return ActionResult(False, str(exc))
            except Exception:  # an OS call failed in an unexpected way; keep the app alive
                log.exception("action %s crashed", name)
                return ActionResult(False, "Sorry, that didn't work.")
        log.info("action %s ok=%s", name, result.ok)  # args are not logged (typed text, clipboard)
        return result


def clamp(value: int, low: int, high: int) -> int:
    return max(low, min(high, value))
