"""generate_image tool: Cloudflare Workers AI FLUX.1 [schnell], run by GRACE's code.

    POST https://api.cloudflare.com/client/v4/accounts/{account}/ai/run/@cf/black-forest-labs/flux-1-schnell
    Authorization: Bearer <CLOUDFLARE_API_TOKEN>      {"prompt": "...", "steps": 4}
    -> {"result": {"image": "<base64 JPEG>"}, "success": true, ...}

Free allowance: 10,000 neurons a day (resets 00:00 UTC). Pricing is 4.80 neurons per 512x512
tile plus 9.60 per step; the output size isn't configurable, so GRACE counts the 1024x1024 case
(4 tiles): 57.6 neurons per image at 4 steps, about 170 images a day. The quota governor refuses
before the allowance runs out.

Prompts pass ``check_prompt`` first (a whole-word filter: sexual content, minors, gore,
self-harm, hate); refused prompts never leave the machine. The image is saved to
Pictures/GRACE and opened through the action registry (screen.show_image).
"""

from __future__ import annotations

import base64
import binascii
import logging
import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import httpx

from grace.actions.screen import DEFAULT_SCREENSHOT_DIR
from grace.core.quota import NullQuota, Quota

log = logging.getLogger(__name__)

API = "https://api.cloudflare.com/client/v4"
DEFAULT_MODEL = "@cf/black-forest-labs/flux-1-schnell"
MAX_PROMPT = 2048  # the model's input schema
NEURONS_PER_TILE = 4.80
NEURONS_PER_STEP = 9.60
TILES = 4  # 1024x1024 output, counted pessimistically


def neurons(steps: int, tiles: int = TILES) -> float:
    return round(tiles * NEURONS_PER_TILE + steps * NEURONS_PER_STEP, 2)


# --------------------------------------------------------------------------- content filter
_BLOCKED: dict[str, tuple[str, ...]] = {
    "sexual content": (
        "nude", "nudes", "naked", "nudity", "porn", "porno", "pornographic", "nsfw", "sex",
        "sexual", "sexy", "erotic", "hentai", "topless", "lingerie", "fetish", "xxx", "genitals",
        "underage",
    ),
    "graphic violence": ("gore", "gory", "beheading", "decapitated", "dismembered", "mutilated",
                         "corpse", "massacre", "torture"),
    "self-harm": ("suicide", "self harm", "self-harm"),
    "hate": ("nazi", "swastika", "kkk", "white power", "racist", "genocide"),
}  # fmt: skip
_PATTERNS = {
    reason: re.compile(r"(?<![a-z])(?:" + "|".join(map(re.escape, words)) + r")(?![a-z])")
    for reason, words in _BLOCKED.items()
}


def check_prompt(prompt: str) -> str | None:
    """Why an image prompt is refused, or None if it may be sent. Whole words only, so
    "Essex" or "a happy child" pass. Pure."""
    text = " ".join(prompt.lower().split())
    for reason, pattern in _PATTERNS.items():
        if pattern.search(text):
            return reason
    return None


# --------------------------------------------------------------------------- generator
@dataclass(frozen=True)
class ImageResult:
    ok: bool
    message: str  # what to say
    images: list[Path] = field(default_factory=list)
    latency_ms: float = 0.0


class ImageGenerator(Protocol):
    name: str

    def generate(self, prompt: str) -> ImageResult: ...


class CloudflareFlux:
    name = "cloudflare"

    def __init__(
        self,
        account_id: str,
        api_token: str,
        registry: Any,
        *,
        model: str = DEFAULT_MODEL,
        steps: int = 4,
        timeout_s: float = 30.0,
        quota: Quota | None = None,
        image_dir: Path = DEFAULT_SCREENSHOT_DIR,
        transport: httpx.BaseTransport | None = None,
        now: Callable[[], float] = time.time,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        if not account_id or not api_token:
            raise ValueError("CloudflareFlux needs an account ID and an API token")
        self.model = model
        self.steps = steps
        self.timeout_s = timeout_s
        self.quota = quota or NullQuota()
        self.registry = registry
        self.image_dir = image_dir
        self.quota_key = f"cloudflare:{model}"
        self._account_url = f"{API}/accounts/{account_id}"
        self._url = f"{self._account_url}/ai/run/{model}"
        self._now = now
        self._clock = clock
        self._http = httpx.Client(
            transport=transport, headers={"Authorization": f"Bearer {api_token}"}
        )

    def __repr__(self) -> str:  # never show the token or the account
        return f"CloudflareFlux(model={self.model!r})"

    def generate(self, prompt: str) -> ImageResult:
        prompt = " ".join(prompt.split())
        if not prompt:
            return ImageResult(False, "What should the image show?")
        refused = check_prompt(prompt)
        if refused:
            log.info("image prompt refused (%s)", refused)  # the prompt itself is not logged
            return ImageResult(False, f"I can't make that image: it asks for {refused}.")
        cost = neurons(self.steps)
        if not self.quota.allow(self.quota_key, units=cost):
            return ImageResult(
                False, "Today's free image allowance is used up. It resets at 00:00 UTC."
            )
        started = self._clock()
        try:
            response = self._http.post(
                self._url,
                json={"prompt": prompt[:MAX_PROMPT], "steps": self.steps},
                timeout=self.timeout_s,
            )
        except httpx.TimeoutException:
            return ImageResult(False, "The image took too long to make.")
        except httpx.HTTPError as exc:
            log.info("cloudflare network error: %s", type(exc).__name__)
            return ImageResult(False, "I couldn't reach the image service.")
        ms = (self._clock() - started) * 1000
        data = _json(response)
        if response.status_code == 429:
            return ImageResult(False, "The image service is busy; try again in a minute.")
        if response.status_code in (401, 403):
            return ImageResult(
                False, "Cloudflare rejected the token. Check CLOUDFLARE_API_TOKEN in .env."
            )
        if response.status_code != 200 or not data.get("success", True):
            log.info("cloudflare HTTP %d: %s", response.status_code, _errors(data))
            return ImageResult(False, "The image service couldn't make that image.")
        self.quota.record(self.quota_key, units=cost)
        raw = (data.get("result") or {}).get("image") or data.get("image")
        try:
            image = base64.b64decode(raw or "", validate=True)
        except (binascii.Error, ValueError):
            image = b""
        if not image:
            return ImageResult(False, "The image service returned no image.")
        return self._save_and_show(image, ms)

    def _save_and_show(self, image: bytes, ms: float) -> ImageResult:
        stamp = time.strftime("%Y%m%d-%H%M%S", time.localtime(self._now()))
        path = self.image_dir / f"image_{stamp}.jpg"
        try:
            self.image_dir.mkdir(parents=True, exist_ok=True)
            path.write_bytes(image)
        except OSError:
            return ImageResult(False, "Couldn't save the image.")
        shown = self.registry.execute("screen.show_image", {"path": str(path)})
        message = f"Image saved as {path.name} in the {self.image_dir.name} pictures folder."
        if shown.ok:
            message += " It is open on screen."
        log.info("cloudflare image in %.0f ms", ms)
        return ImageResult(True, message, [path], ms)

    def check(self) -> tuple[bool, str]:
        """Doctor: is the token active? Token verify runs no model and spends no neurons.
        Account API tokens verify under /accounts/{id}/tokens/verify, user tokens under
        /user/tokens/verify: try the account one first."""
        last = "unknown"
        for url in (f"{self._account_url}/tokens/verify", f"{API}/user/tokens/verify"):
            try:
                response = self._http.get(url, timeout=5.0)
            except httpx.HTTPError as exc:
                return False, f"not reachable ({type(exc).__name__})"
            status = (_json(response).get("result") or {}).get("status")
            if response.status_code == 200 and status == "active":
                return True, "token active"
            last = f"HTTP {response.status_code}, token status {status or 'unknown'}"
        return False, last


def _json(response: httpx.Response) -> dict[str, Any]:
    try:
        data = response.json()
    except ValueError:
        return {}
    return data if isinstance(data, dict) else {}


def _errors(data: dict[str, Any]) -> str:
    errors = data.get("errors") or []
    return "; ".join(str(e.get("message", e)) for e in errors if isinstance(e, dict))[:200]


def build_images(
    settings: object, registry: Any, quota: Quota | None = None
) -> ImageGenerator | None:
    img = settings.images  # type: ignore[attr-defined]
    account = settings.secret("cloudflare_account_id")  # type: ignore[attr-defined]
    token = settings.secret("cloudflare_api_token")  # type: ignore[attr-defined]
    if not img.enabled or not account or not token:
        return None
    return CloudflareFlux(
        account, token, registry, model=img.model, steps=img.steps, timeout_s=img.timeout_s,
        quota=quota,
    )  # fmt: skip
