"""generate_image tool (Cloudflare FLUX.1 schnell) over a mocked transport; prompt filter."""

from __future__ import annotations

import base64
import json
from pathlib import Path

import httpx
import pytest

from grace.actions import create_fake_registry
from grace.core.quota import Limits, QuotaGovernor
from grace.intent.images import CloudflareFlux, check_prompt, neurons

TOKEN = "cf" + "ut_" + "FAKETOKEN0123456789"
JPEG = b"\xff\xd8\xff fake jpeg"


def make(tmp_path: Path, handler, **kw):
    registry, b = create_fake_registry(screenshot_dir=tmp_path)
    seen: list[httpx.Request] = []

    def record(request):
        seen.append(request)
        return handler(request)

    flux = CloudflareFlux(
        "acct123", TOKEN, registry, image_dir=tmp_path, transport=httpx.MockTransport(record),
        now=lambda: 0.0, **kw,
    )  # fmt: skip
    return flux, seen, b


def ok(request):
    return httpx.Response(200, json={"result": {"image": base64.b64encode(JPEG).decode()},
                                     "success": True, "errors": [], "messages": []})  # fmt: skip


def test_request_shape_save_and_show(tmp_path: Path) -> None:
    flux, seen, b = make(tmp_path, ok)
    result = flux.generate("  a cat   in sunglasses ")
    [path] = result.images
    assert result.ok and path.read_bytes() == JPEG and path.parent == tmp_path
    assert b.launcher.files == [path.resolve()] and "open on screen" in result.message
    req = seen[0]
    assert str(req.url) == (
        "https://api.cloudflare.com/client/v4/accounts/acct123/ai/run/"
        "@cf/black-forest-labs/flux-1-schnell"
    )
    assert req.headers["authorization"] == f"Bearer {TOKEN}"
    assert json.loads(req.content) == {"prompt": "a cat in sunglasses", "steps": 4}
    assert TOKEN not in repr(flux) and "acct123" not in repr(flux)


def test_top_level_image_field_is_accepted(tmp_path: Path) -> None:
    flux, _, _ = make(
        tmp_path, lambda r: httpx.Response(200, json={"image": base64.b64encode(JPEG).decode()})
    )
    assert flux.generate("a boat").ok


@pytest.mark.parametrize(
    ("status", "body", "fragment"),
    [
        (429, {}, "busy"),
        (401, {}, "CLOUDFLARE_API_TOKEN"),
        (400, {"success": False, "errors": [{"message": "bad"}]}, "couldn't make"),
        (200, {"success": True, "result": {}}, "no image"),
        (200, {"success": True, "result": {"image": "!!!not base64"}}, "no image"),
    ],
)
def test_failures_are_spoken(tmp_path: Path, status, body, fragment) -> None:
    flux, _, b = make(tmp_path, lambda r: httpx.Response(status, json=body))
    result = flux.generate("a boat")
    assert not result.ok and fragment in result.message and b.launcher.files == []


def test_timeout_and_network(tmp_path: Path) -> None:
    def slow(request):
        raise httpx.ReadTimeout("slow", request=request)

    assert "too long" in make(tmp_path, slow)[0].generate("a boat").message


def test_neurons_are_counted_against_the_daily_allowance(tmp_path: Path) -> None:
    assert neurons(4) == 57.6  # 4 tiles x 4.80 + 4 steps x 9.60
    quota = QuotaGovernor(
        {"cloudflare:@cf/black-forest-labs/flux-1-schnell": Limits(units_per_day=120)},
        state_path=None, clock=lambda: 1_800_000_000,
    )  # fmt: skip
    flux, seen, _ = make(tmp_path, ok, quota=quota)
    assert flux.generate("a boat").ok and flux.generate("a car").ok  # 115.2 neurons
    third = flux.generate("a plane")
    assert not third.ok and "allowance" in third.message and len(seen) == 2
    assert quota.used("cloudflare:@cf/black-forest-labs/flux-1-schnell")["upd"] == 115.2


def test_failed_generation_spends_no_neurons(tmp_path: Path) -> None:
    quota = QuotaGovernor({}, state_path=None, clock=lambda: 1_800_000_000)
    flux, _, _ = make(tmp_path, lambda r: httpx.Response(500, json={}), quota=quota)
    flux.generate("a boat")
    assert quota.used(flux.quota_key)["upd"] == 0


@pytest.mark.parametrize(
    ("prompt", "reason"),
    [
        ("a naked person on a beach", "sexual content"),
        ("NSFW anime girl", "sexual content"),
        ("a gory beheading scene", "graphic violence"),
        ("someone thinking about suicide", "self-harm"),
        ("a nazi rally", "hate"),
        ("a happy child flying a kite", None),
        ("a cottage in Essex at sunset", None),  # whole words only
        ("chicken breasts on a grill", None),
        ("a cat in sunglasses", None),
    ],
)
def test_prompt_filter(prompt: str, reason: str | None) -> None:
    assert check_prompt(prompt) == reason


def test_refused_prompts_never_leave_the_machine(tmp_path: Path) -> None:
    flux, seen, _ = make(tmp_path, ok)
    result = flux.generate("a naked person")
    assert not result.ok and "sexual content" in result.message and seen == []


def test_token_check_for_the_doctor(tmp_path: Path) -> None:
    def verify(request):
        active = "/accounts/acct123/tokens/verify" in str(request.url)
        body = {"result": {"status": "active"}, "success": True} if active else {"success": False}
        return httpx.Response(200 if active else 401, json=body)

    flux, seen, _ = make(tmp_path, verify)  # an account API token
    assert flux.check() == (True, "token active")
    assert str(seen[0].url).endswith("/accounts/acct123/tokens/verify")

    def user_only(request):
        if "/user/tokens/verify" in str(request.url):
            return httpx.Response(200, json={"result": {"status": "active"}, "success": True})
        return httpx.Response(401, json={"success": False})

    flux, seen, _ = make(tmp_path, user_only)  # a user API token
    assert flux.check() == (True, "token active") and len(seen) == 2
    bad, _, _ = make(tmp_path, lambda r: httpx.Response(401, json={"success": False}))
    assert bad.check() == (False, "HTTP 401, token status unknown")
